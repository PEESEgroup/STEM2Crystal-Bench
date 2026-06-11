#!/usr/bin/env python3
"""Generate + SAVE SCCD v5 test predictions as rank_*.cif (one dir per sample), so the unified harness
can score them (fixes the old single-batch no-file problem). v5 = EMDiffuse U-Net image branch + global
g_img conditioning. g_img is computed once per sample from the observed evidence (image->structure) and
reused across the 1000 structure-sampling steps (tractable; the heavy U-Net runs once, not per step).
Shardable for 4-GPU parallelism: --noise X --start a --end b  on CUDA_VISIBLE_DEVICES=g."""
import os,sys,argparse,random,warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np,torch,torch.nn as nn,torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_scatter import scatter
ROOT=Path(os.environ.get("SCCD_HOME","/home/ubuntu/efs/KDD/EM/stem2cif-mattergen")); PROJ=ROOT/"methods/DiffCSP"; EMD=ROOT/"methods/EMDiffuse"
sys.path.insert(0,str(PROJ)); sys.path.insert(0,str(PROJ/"scripts")); sys.path.insert(0,str(EMD)); os.environ["PROJECT_ROOT"]=str(PROJ)
import train_phase3_f2_guarded as M
from train_phase3_f2_guarded import (Phase3F2Guarded,load_f2_weights,load_evidence_weights,F2_CKPT,STEMCrystalDatasetWithNoise,
    add_scaled_lattice_prop,get_scaler_from_data_list,slab_logvec_to_matrix)
from train_diffcsp6d_slabprior import slab_logvec_to_lattice_params
from diffcsp.common.data_utils import chemical_symbols
from pymatgen.core import Structure,Lattice
EV_CKPT=PROJ/"outputs/phase1_evidence/best_evidence_unet.pt"
OUTROOT=ROOT/"run_eval/predictions"

def struct(base,lat_head,unk,inj,gate,te,atom_types,frac,lat,na,n2g,z_lat,g_img,ck,use_img=True):
    edges,fd=base.gen_edges(na,frac,lat,n2g); e2g=n2g[edges[0]]
    emb=base.node_embedding(atom_types-1); keep=ck[n2g][:,None]; emb=keep*emb+(1-keep)*unk
    nf=base.atom_latent_emb(torch.cat([emb,te.repeat_interleave(na,0)],1))
    def addimg(h,layer):
        gi=g_img[n2g]; c=inj[layer](gi); c=torch.sigmoid(gate[layer](torch.cat([h,gi],-1)))*c; return h+c
    if use_img: nf=addimg(nf,0)
    for i in range(base.num_layers):
        h=addimg(nf,i+1) if use_img else nf
        nf=base._modules[f"csp_layer_{i}"](h,frac,lat,edges,e2g,frac_diff=fd)
    if base.ln: nf=base.final_layer_norm(nf)
    return lat_head(torch.cat([scatter(nf,n2g,0,reduce="mean",dim_size=na.shape[0]),z_lat],-1)), base.coord_out(nf)

@torch.no_grad()
def gimg_of(m,gimg_head,ev_t,stem,t):
    m._inject_g_crys=None                                   # inference: image->structure (no circular g_crys)
    pred=m.evidence_unet(torch.cat([ev_t,stem],1),t)
    return gimg_head(m._bottleneck_feat.mean(dim=[2,3]))

@torch.no_grad()
def sample(m,base,lat_head,unk,inj,gate,g_img,batch,cfg):
    dev=batch.atom_types.device; B=batch.num_graphs; beta=m.beta_scheduler; sigma=m.sigma_scheduler
    torch.manual_seed(cfg["seed"]); torch.cuda.manual_seed(cfg["seed"])
    z=torch.randn(B,m.lattice_dim,device=dev)*m.noise_scale[None]*cfg["temp"]; z[:,3:5]=0.5+0.05*torch.randn(B,2,device=dev)
    x=torch.rand(batch.num_nodes,3,device=dev); x[:,2]=torch.clamp(cfg["z_mid"]+0.1*torch.randn(batch.num_nodes,device=dev),0,1); x=x%1.
    ck=torch.ones(B,device=dev)
    for t in range(beta.timesteps,0,-1):
        ti=torch.full((B,),t,device=dev); te=m.time_embedding(ti)
        al=beta.alphas[t];ab=beta.alphas_cumprod[t];sig=beta.sigmas[t];sx=sigma.sigmas[t];sn=sigma.sigmas_norm[t]
        c0=1/al.sqrt();c1=(1-al)/(1-ab).sqrt(); z[:,3:5]=0.5; lat=slab_logvec_to_matrix(z.clamp(-5,5),project_ab=True)
        def fwd(xc): return struct(base,lat_head,unk,inj,gate,te,batch.atom_types,xc,lat,batch.num_atoms,batch.batch,z,g_img,ck)
        rx=torch.randn_like(x) if t>1 else torch.zeros_like(x); step=1e-5*cfg["step_lr"]*(sx/sigma.sigma_begin)**2; std=(2*step).sqrt()
        _,px=fwd(x); xh=(x-step*px*sn.sqrt()+std*rx)%1.
        rz=torch.randn_like(z)*m.noise_scale[None] if t>1 else torch.zeros_like(z); rx=torch.randn_like(x) if t>1 else torch.zeros_like(x)
        adj=sigma.sigmas[t-1]; s2=sx**2-adj**2; st2=((adj**2*s2)/sx**2).sqrt()
        pz,px=fwd(xh); x=(xh-s2*px*sn.sqrt()+st2*rx)%1.; z=c0*(z-c1*pz)+sig*rz
    z[:,3:5]=0.5
    return x.cpu(), z.clamp(-5,5).cpu()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--ckpt",default="sccd_v5/best.pt"); ap.add_argument("--rows",default="test.pt")
    ap.add_argument("--noise",default="low",choices=["low","mid","high"]); ap.add_argument("--tag",default="sccd_v5")
    ap.add_argument("--start",type=int,default=0); ap.add_argument("--end",type=int,default=0); ap.add_argument("--chunk",type=int,default=64); ap.add_argument("--K",type=int,default=5); a=ap.parse_args()
    dev="cuda"; torch.manual_seed(0); random.seed(0)
    m=Phase3F2Guarded(g_img_dropout=0.0,inject_crys_to_img=True).to(dev); load_f2_weights(m,str(F2_CKPT)); load_evidence_weights(m,EV_CKPT)
    base=m.crystal.base; lat_head=m.crystal.lattice_out_6d; H=base.coord_out.in_features; nL=base.num_layers
    gimg_head=nn.Sequential(nn.Linear(256,256),nn.SiLU(),nn.Linear(256,256)).to(dev)
    inj=nn.ModuleList([nn.Linear(256,H) for _ in range(nL+1)]).to(dev); gate=nn.ModuleList([nn.Linear(H+256,H) for _ in range(nL+1)]).to(dev)
    ck=torch.load(PROJ/"outputs"/a.ckpt,map_location="cpu",weights_only=False)
    base.load_state_dict(ck["base"]); lat_head.load_state_dict(ck["lat_head"]); m.evidence_unet.load_state_dict(ck["evidence_unet"])
    m.g_crys_to_unet.load_state_dict(ck["g_crys_to_unet"]); gimg_head.load_state_dict(ck["gimg_head"]); inj.load_state_dict(ck["inj"]); gate.load_state_dict(ck["gate"]); unk=ck["unk"].to(dev)
    for p in m.parameters(): p.requires_grad_(False)
    m.eval(); base.eval(); gimg_head.eval()
    rows=torch.load(M.DATA_ROOT/a.rows,weights_only=False); add_scaled_lattice_prop(rows,"scale_length")
    a.end=a.end or len(rows); rows=rows[a.start:a.end]
    tr=torch.load(M.DATA_ROOT/"train.pt",weights_only=False); add_scaled_lattice_prop(tr,"scale_length")
    ds=STEMCrystalDatasetWithNoise(rows,M.STEM_ROOT,precache=False); ds.noises=[a.noise]; ds.views=[0]
    ds.lattice_scaler=get_scaler_from_data_list(tr,key="scaled_lattice"); ds.scaler=get_scaler_from_data_list(tr,key="formation_energy_per_atom")
    ab2=m.beta_scheduler.alphas_cumprod[200]; ce0=ab2.sqrt(); ce1=(1-ab2).sqrt()
    cfgs=[dict(seed=42+i*101,temp=1.0,step_lr=1.0,z_mid=0.5) for i in range(a.K)]
    done=0
    for b in DataLoader(ds,batch_size=a.chunk,shuffle=False,num_workers=4):
        b=b.to(dev); B=b.num_graphs; na=b.num_atoms.cpu().numpy(); off=np.concatenate([[0],np.cumsum(na)]); at=b.atom_types.cpu().numpy()
        ev=(b.evidence if b.evidence.dim()==4 else b.evidence.unsqueeze(1)); stem=(b.stem_image if b.stem_image.dim()==4 else b.stem_image.unsqueeze(1))
        t200=torch.full((B,),200,device=dev); ev_t=ce0*ev+ce1*torch.randn_like(ev)
        g_img=gimg_of(m,gimg_head,ev_t,stem,t200)                     # once per sample
        # map batch graphs to sids via order (DataLoader shuffle=False preserves order)
        for ci,cfg in enumerate(cfgs):
            x,z=sample(m,base,lat_head,unk,inj,gate,g_img,b,cfg); L,A=slab_logvec_to_lattice_params(z,project_ab=True); x=x.numpy();L=L.numpy();A=A.numpy()
            for i in range(B):
                sid=rows[done+i]["mp_id"]; d=OUTROOT/a.tag/a.noise/f"{sid}_0"; d.mkdir(parents=True,exist_ok=True)
                try:
                    st=Structure(Lattice.from_parameters(*L[i].tolist(),*A[i].tolist()),[chemical_symbols[int(zz)] for zz in at[off[i]:off[i+1]]],x[off[i]:off[i+1]]%1.0)
                    st.to(filename=str(d/f"rank_{ci+1}.cif"))
                except Exception: pass
        done+=B; print(f"[{a.noise} {a.start}:{a.end}] {done}/{len(rows)} saved",flush=True)
    print(f"[done] {a.noise} {a.start}:{a.end} -> {OUTROOT/a.tag/a.noise}",flush=True)
if __name__=="__main__": main()
