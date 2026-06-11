#!/usr/bin/env python3
"""SCCD v5 — FiLM coupling variant (image->structure). Identical to train_sccd_v5_ddp.py EXCEPT the
image->structure injection is FiLM (gamma*h + beta) instead of gated-additive (h + sigmoid(gate)*inj).
FiLM is identity-initialized (gamma=1, beta=0). Same F2 init, same 60 epochs, same loss & data -> fair A/B.
Launch:  torchrun --nproc_per_node=4 train_sccd_v5_film.py --epochs 60 --bs 16 --out sccd_v5_film"""
import os, sys, time, argparse, random, warnings
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
from torch_scatter import scatter
warnings.filterwarnings("ignore")
ROOT = Path("/home/ubuntu/efs/KDD/EM/stem2cif-mattergen"); PROJ = ROOT/"methods/DiffCSP"; EMD=ROOT/"methods/EMDiffuse"
sys.path.insert(0,str(PROJ)); sys.path.insert(0,str(PROJ/"scripts")); sys.path.insert(0,str(EMD)); os.environ["PROJECT_ROOT"]=str(PROJ)
import train_phase3_f2_guarded as M
from train_phase3_f2_guarded import (Phase3F2Guarded, load_f2_weights, load_evidence_weights, F2_CKPT,
    STEMCrystalDatasetWithNoise, add_scaled_lattice_prop, get_scaler_from_data_list,
    lattice_params_to_logvec, slab_logvec_to_matrix)
from diffcsp.pl_modules.diff_utils import d_log_p_wrapped_normal
EV_CKPT = PROJ/"outputs/phase1_evidence/best_evidence_unet.pt"


class V5(nn.Module):
    """FiLM coupling: image->structure via gamma*h + beta at every node-feature layer."""
    def __init__(s, m, w1_w=0.03, img_w=1.0, p_drop=0.0):
        super().__init__()
        s.m=m; s.base=m.crystal.base; s.lat_head=m.crystal.lattice_out_6d
        s.evidence_unet=m.evidence_unet; s.g_crys_to_unet=m.g_crys_to_unet
        H=s.base.coord_out.in_features; nL=s.base.num_layers; s.H=H; s.nL=nL
        s.gimg_head=nn.Sequential(nn.Linear(256,256),nn.SiLU(),nn.Linear(256,256))
        s.g_crys_proj=nn.Linear(H,256)
        # FiLM: gamma (scale) and beta (shift) per layer, identity-initialized (gamma=1, beta=0)
        s.gamma=nn.ModuleList([nn.Linear(256,H) for _ in range(nL+1)])
        for l in s.gamma: nn.init.zeros_(l.weight); nn.init.ones_(l.bias)
        s.beta=nn.ModuleList([nn.Linear(256,H) for _ in range(nL+1)])
        for l in s.beta: nn.init.zeros_(l.weight); nn.init.zeros_(l.bias)
        s.unk=nn.Parameter(torch.zeros(1,s.base.node_embedding.embedding_dim)); nn.init.normal_(s.unk,std=0.02)
        s.w1_w=w1_w; s.img_w=img_w; s.p_drop=p_drop
    def struct(s, te, atom_types, frac, lat, na, n2g, z_lat, g_img, ck, use_img=True):
        base=s.base; edges,fd=base.gen_edges(na,frac,lat,n2g); e2g=n2g[edges[0]]
        emb=base.node_embedding(atom_types-1); keep=ck[n2g][:,None]; emb=keep*emb+(1-keep)*s.unk
        nf=base.atom_latent_emb(torch.cat([emb,te.repeat_interleave(na,0)],1))
        def addimg(h,layer):
            gi=g_img[n2g]; return s.gamma[layer](gi)*h + s.beta[layer](gi)
        if use_img: nf=addimg(nf,0)
        for i in range(base.num_layers):
            h=addimg(nf,i+1) if use_img else nf
            nf=base._modules[f"csp_layer_{i}"](h,frac,lat,edges,e2g,frac_diff=fd)
        if base.ln: nf=base.final_layer_norm(nf)
        coord=base.coord_out(nf); gfeat=scatter(nf,n2g,dim=0,reduce="mean",dim_size=na.shape[0])
        return s.lat_head(torch.cat([gfeat,z_lat],-1)), coord, gfeat
    def gimg_of(s, ev_t, stem, t, g_crys):
        s.m._inject_g_crys=g_crys
        try: pred=s.evidence_unet(torch.cat([ev_t,stem],1), t)
        finally: s.m._inject_g_crys=None
        return pred, s.gimg_head(s.m._bottleneck_feat.mean(dim=[2,3]))
    def forward(s, te, atom_types, inp, lat, na, n2g, zt, rz, tar, ev_t, stem, en, ab, c0, c1, t):
        ck=(torch.rand(na.shape[0],device=zt.device)>s.p_drop).float()
        _,_,g_crys=s.struct(te,atom_types,inp,lat,na,n2g,zt,None,ck,use_img=False)
        pred_noise,g_img=s.gimg_of(ev_t,stem,t,s.g_crys_proj(g_crys))
        pz,px,_=s.struct(te,atom_types,inp,lat,na,n2g,zt,g_img,ck,use_img=True)
        l_lat=F.mse_loss(pz[:,[0,1,5]],rz[:,[0,1,5]]); l_coord=F.mse_loss(px,tar)
        z0h=(zt-c1[:,None]*pz)/c0[:,None].clamp(min=1e-6); ln=(ab>0.2)
        def sl1(p,q): return F.l1_loss(torch.sort(p)[0],torch.sort(q)[0])
        zt0=(zt-c1[:,None]*rz)/c0[:,None].clamp(min=1e-6)
        l_w1=((sl1(z0h[ln][:,0],zt0[ln][:,0])+sl1(z0h[ln][:,1],zt0[ln][:,1])+sl1(z0h[ln][:,5],zt0[ln][:,5]))/3 if ln.any() else pz.sum()*0)
        l_img=F.mse_loss(pred_noise,en)
        loss=10*l_lat+l_coord+s.w1_w*l_w1+s.img_w*l_img
        return loss, l_img.detach()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--epochs",type=int,default=60); ap.add_argument("--bs",type=int,default=16)
    ap.add_argument("--lr_struct",type=float,default=2e-5); ap.add_argument("--lr_unet",type=float,default=1e-5); ap.add_argument("--lr_head",type=float,default=2e-4)
    ap.add_argument("--img_w",type=float,default=1.0); ap.add_argument("--w1_w",type=float,default=0.03); ap.add_argument("--p_drop",type=float,default=0.0)
    ap.add_argument("--out",default="sccd_v5_film"); ap.add_argument("--limit",type=int,default=0); a=ap.parse_args()
    lr=int(os.environ.get("LOCAL_RANK",0)); world=int(os.environ.get("WORLD_SIZE",1)); rank=int(os.environ.get("RANK",0))
    dist.init_process_group("nccl"); torch.cuda.set_device(lr); dev=f"cuda:{lr}"
    torch.manual_seed(0); random.seed(0)
    out=PROJ/"outputs"/a.out; (rank==0) and out.mkdir(parents=True,exist_ok=True)
    m=Phase3F2Guarded(g_img_dropout=0.0, inject_crys_to_img=True, detach_crys_to_img=True).to(dev)
    load_f2_weights(m,str(F2_CKPT)); load_evidence_weights(m, EV_CKPT)
    m.crystal.requires_grad_(False); m.f2_ref.requires_grad_(False)
    m.crystal.base.requires_grad_(True); m.crystal.lattice_out_6d.requires_grad_(True)
    m.evidence_unet.requires_grad_(True); m.g_crys_to_unet.requires_grad_(True)
    net=V5(m,a.w1_w,a.img_w,a.p_drop).to(dev)
    ddp=DDP(net, device_ids=[lr], find_unused_parameters=True)
    base=net.base
    g=[{"params":list(net.base.parameters())+list(net.lat_head.parameters()),"lr":a.lr_struct},
       {"params":list(net.evidence_unet.parameters()),"lr":a.lr_unet},
       {"params":list(net.gimg_head.parameters())+list(net.g_crys_proj.parameters())+list(net.gamma.parameters())+list(net.beta.parameters())+list(net.g_crys_to_unet.parameters())+[net.unk],"lr":a.lr_head}]
    opt=torch.optim.Adam(g)
    bs=m.beta_scheduler; ss=m.sigma_scheduler; T=bs.timesteps
    tr=torch.load(M.DATA_ROOT/"train.pt",weights_only=False); va=torch.load(M.DATA_ROOT/"val.pt",weights_only=False)
    add_scaled_lattice_prop(tr,"scale_length"); add_scaled_lattice_prop(va,"scale_length")
    if a.limit: tr=tr[:a.limit]; va=va[:max(1,a.limit//4)]
    sc_l=get_scaler_from_data_list(tr,key="scaled_lattice"); sc_e=get_scaler_from_data_list(tr,key="formation_energy_per_atom")
    def mk(rows): d=STEMCrystalDatasetWithNoise(rows,M.STEM_ROOT,precache=False); d.lattice_scaler=sc_l; d.scaler=sc_e; return d
    tds=mk(tr); samp=DistributedSampler(tds,num_replicas=world,rank=rank,shuffle=True)
    tl=DataLoader(tds,batch_size=a.bs,sampler=samp,num_workers=4)
    vl=DataLoader(mk(va),batch_size=48,shuffle=False,num_workers=2)
    def ev_im(b): return b.evidence if b.evidence.dim()==4 else b.evidence.unsqueeze(1)
    def stem_im(b): return b.stem_image if b.stem_image.dim()==4 else b.stem_image.unsqueeze(1)
    if rank==0: print(f"[v5-FiLM-DDP] world={world} bs/gpu={a.bs} eff_bs={a.bs*world} trainable={sum(p.numel() for p in ddp.parameters() if p.requires_grad):,}",flush=True)
    best=1e9
    for ep in range(1,a.epochs+1):
        ddp.train(); samp.set_epoch(ep); t0=time.time(); tot=ti=0.0; n=0
        for b in tl:
            b=b.to(dev); B=b.num_graphs; t=torch.randint(1,T,(B,),device=dev); te=m.time_embedding(t)
            ab=bs.alphas_cumprod[t]; c0=ab.sqrt(); c1=(1-ab).sqrt()
            z0=lattice_params_to_logvec(b.lengths,b.angles,dim=6); rz=torch.randn_like(z0)
            zt=c0[:,None]*z0+c1[:,None]*rz; zt[:,3:5]=0.5; lat=slab_logvec_to_matrix(zt,project_ab=True)
            sig=ss.sigmas[t].repeat_interleave(b.num_atoms)[:,None]; sn=ss.sigmas_norm[t].repeat_interleave(b.num_atoms)[:,None]
            rx=torch.randn_like(b.frac_coords); inp=(b.frac_coords+sig*rx)%1.; tar=d_log_p_wrapped_normal(sig*rx,sig)/torch.sqrt(sn)
            ev=ev_im(b); stem=stem_im(b); en=torch.randn_like(ev); ev_t=c0[:,None,None,None]*ev+c1[:,None,None,None]*en
            loss,limg=ddp(te,b.atom_types,inp,lat,b.num_atoms,b.batch,zt,rz,tar,ev_t,stem,en,ab,c0,c1,t)
            opt.zero_grad(); loss.backward(); opt.step(); tot+=loss.item(); ti+=limg.item(); n+=1
        dt=time.time()-t0
        if rank==0 and (ep%5==0 or ep==1):
            net.eval()
            with torch.no_grad():
                ab2=bs.alphas_cumprod[200]; ce0=ab2.sqrt(); ce1=(1-ab2).sqrt()
                def run(mode):
                    sM=cnt=0.0
                    for b in vl:
                        b=b.to(dev); B=b.num_graphs; t=torch.full((B,),200,device=dev); te=m.time_embedding(t)
                        z0=lattice_params_to_logvec(b.lengths,b.angles,dim=6); zt=z0.clone(); zt[:,3:5]=0.5; lat=slab_logvec_to_matrix(zt,project_ab=True)
                        sig=ss.sigmas[t].repeat_interleave(b.num_atoms)[:,None]; sn=ss.sigmas_norm[t].repeat_interleave(b.num_atoms)[:,None]
                        rx=torch.randn_like(b.frac_coords); inp=(b.frac_coords+sig*rx)%1.; tar=d_log_p_wrapped_normal(sig*rx,sig)/torch.sqrt(sn)
                        ev=ev_im(b); stem=stem_im(b); ev_t=ce0*ev+ce1*torch.randn_like(ev); ck=torch.ones(B,device=dev)
                        _,_,gc=net.struct(te,b.atom_types,inp,lat,b.num_atoms,b.batch,zt,None,ck,use_img=False)
                        _,gimg=net.gimg_of(ev_t,stem,t,net.g_crys_proj(gc))
                        if mode=="shuf": gimg=torch.roll(gimg,1,0)
                        _,px,_=net.struct(te,b.atom_types,inp,lat,b.num_atoms,b.batch,zt,gimg,ck,use_img=True)
                        sM+=((px-tar)**2).sum().item(); cnt+=tar.numel()
                    return sM/cnt
                cor=run("cor"); shf=run("shuf")
            net.train()
            if cor<best:
                best=cor; torch.save({"base":net.base.state_dict(),"lat_head":net.lat_head.state_dict(),"evidence_unet":net.evidence_unet.state_dict(),"g_crys_to_unet":net.g_crys_to_unet.state_dict(),"gimg_head":net.gimg_head.state_dict(),"g_crys_proj":net.g_crys_proj.state_dict(),"gamma":net.gamma.state_dict(),"beta":net.beta.state_dict(),"unk":net.unk.detach()},out/"best.pt")
            print(f"ep{ep:03d} {dt:.0f}s loss={tot/n:.4f} img={ti/n:.4f} | IMG={cor:.4f} SHUF={shf:.4f} gap={shf-cor:+.4f}",flush=True)
        dist.barrier()
    if rank==0: print(f"[done] best={best:.4f}",flush=True)
    dist.destroy_process_group()


if __name__=="__main__":
    main()
