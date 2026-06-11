#!/usr/bin/env python3
"""Unified eval on Real-5 v2 (5 genuine monolayers) for diffusion methods: SCCD v5 (+img/no-img), F2.
All 5 are 2D -> monolayer GT, c-normalized matching. Reports per-sample Hit@5 + in-plane MAE_a, raw (no templating)."""
import os,sys,json,argparse,warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch_geometric.data import Data, Batch
from PIL import Image
ROOT=Path("/home/ubuntu/efs/KDD/EM/stem2cif-mattergen"); PROJ=ROOT/"methods/DiffCSP"; EMD=ROOT/"methods/EMDiffuse"
sys.path.insert(0,str(PROJ)); sys.path.insert(0,str(PROJ/"scripts")); sys.path.insert(0,str(EMD)); os.environ["PROJECT_ROOT"]=str(PROJ)
from train_phase3_f2_guarded import Phase3F2Guarded,load_f2_weights,load_evidence_weights,F2_CKPT,slab_logvec_to_matrix
from train_diffcsp6d_slabprior import slab_logvec_to_lattice_params
from gen_sccd_v5 import struct, gimg_of
from diffcsp.common.data_utils import chemical_symbols
from pymatgen.core import Structure,Lattice
from pymatgen.io.cif import CifParser
from pymatgen.analysis.structure_matcher import StructureMatcher
DS=Path("/home/ubuntu/efs/KDD/EM/real5_v2"); EV_CKPT=PROJ/"outputs/phase1_evidence/best_evidence_unet.pt"; IMG=256; CN=20.0
Z={s:i for i,s in enumerate(chemical_symbols)}
M=StructureMatcher(ltol=0.3,stol=0.3,angle_tol=15,primitive_cell=True,scale=True,attempt_supercell=True)
sids=json.load(open(DS/"split.json"))['test']
GT={}; comp={}; imgs={}
for sid in sids:
    s=CifParser(str(DS/"cifs"/f"{sid}.cif")).parse_structures(primitive=False)[0]
    GT[sid]=(s,s.composition.reduced_formula,s.lattice.a); comp[sid]=([Z[e.symbol] for e in s.species],len(s))
    p=DS/"images"/f"{sid}.png"; imgs[sid]=torch.tensor(np.array(Image.open(p).convert("L").resize((IMG,IMG),Image.BILINEAR),np.float32)/255.0)
def norm_c(st):
    L=st.lattice; cz=np.array([c[2] for c in st.frac_coords])*L.c; cz=cz-cz.mean()
    nf=st.frac_coords.copy(); nf[:,2]=0.5+cz/CN
    return Structure(Lattice.from_parameters(L.a,L.b,CN,90,90,L.gamma),st.species,nf)
@torch.no_grad()
def sample(m,base,lat_head,unk,inj,gate,g_img,batch,seed,use_img):
    dev=batch.atom_types.device; B=batch.num_graphs; beta=m.beta_scheduler; sigma=m.sigma_scheduler
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    z=torch.randn(B,m.lattice_dim,device=dev)*m.noise_scale[None]; z[:,3:5]=0.5+0.05*torch.randn(B,2,device=dev)
    x=torch.rand(batch.num_nodes,3,device=dev); x[:,2]=torch.clamp(0.5+0.1*torch.randn(batch.num_nodes,device=dev),0,1); x=x%1.
    ck=torch.ones(B,device=dev)
    for t in range(beta.timesteps,0,-1):
        ti=torch.full((B,),t,device=dev); te=m.time_embedding(ti)
        al=beta.alphas[t];ab=beta.alphas_cumprod[t];sig=beta.sigmas[t];sx=sigma.sigmas[t];sn=sigma.sigmas_norm[t]
        c0=1/al.sqrt();c1=(1-al)/(1-ab).sqrt(); z[:,3:5]=0.5; lat=slab_logvec_to_matrix(z.clamp(-5,5),project_ab=True)
        def fwd(xc): return struct(base,lat_head,unk,inj,gate,te,batch.atom_types,xc,lat,batch.num_atoms,batch.batch,z,g_img,ck,use_img=use_img)
        rx=torch.randn_like(x) if t>1 else torch.zeros_like(x); step=1e-5*(sx/sigma.sigma_begin)**2; std=(2*step).sqrt()
        _,px=fwd(x); xh=(x-step*px*sn.sqrt()+std*rx)%1.
        rz=torch.randn_like(z)*m.noise_scale[None] if t>1 else torch.zeros_like(z); rx=torch.randn_like(x) if t>1 else torch.zeros_like(x)
        adj=sigma.sigmas[t-1]; s2=sx**2-adj**2; st2=((adj**2*s2)/sx**2).sqrt()
        pz,px=fwd(xh); x=(xh-s2*px*sn.sqrt()+st2*rx)%1.; z=c0*(z-c1*pz)+sig*rz
    z[:,3:5]=0.5
    return x.cpu(), z.clamp(-5,5).cpu()
def build_batch(dev):
    return Batch.from_data_list([Data(atom_types=torch.tensor(comp[s][0]),num_atoms=torch.tensor([comp[s][1]]),num_nodes=comp[s][1]) for s in sids]).to(dev)
def to_structs(b,x,z):
    L,A=slab_logvec_to_lattice_params(z,project_ab=True); L=L.numpy();A=A.numpy(); x=x.numpy()
    na=b.num_atoms.cpu().numpy(); off=np.concatenate([[0],np.cumsum(na)]); at=b.atom_types.cpu().numpy(); out=[]
    for i in range(len(sids)):
        try: out.append(Structure(Lattice.from_parameters(*L[i].tolist(),*A[i].tolist()),[chemical_symbols[int(z2)] for z2 in at[off[i]:off[i+1]]],x[off[i]:off[i+1]]%1.0))
        except: out.append(None)
    return out
def run(name,m,base,lat_head,unk,inj,gate,g_img,use_img,K,tag=None):
    dev="cuda"; b=build_batch(dev); first={s:None for s in sids}; besta={s:None for s in sids}
    from pymatgen.io.cif import CifWriter
    OUTROOT=ROOT/"run_eval/predictions"/f"{tag}_real5v2"/"real" if tag else None
    for k in range(K):
        x,z=sample(m,base,lat_head,unk,inj,gate,g_img,b,42+k*7,use_img); sts=to_structs(b,x,z)
        for i,sid in enumerate(sids):
            c=sts[i]; gt,gf,ga=GT[sid]
            if tag and k<5 and c is not None:        # save first-5 candidates as rank_*.cif (c-normalized monolayer)
                od=OUTROOT/f"{sid}_0"; od.mkdir(parents=True,exist_ok=True)
                try: CifWriter(norm_c(c)).write_file(str(od/f"rank_{k+1}.cif"))
                except: pass
            if c is None or c.composition.reduced_formula!=gf: continue
            da=abs(c.lattice.a-ga); besta[sid]=da if besta[sid] is None else min(besta[sid],da)
            if first[sid] is None:
                try:
                    if M.fit(gt,norm_c(c)): first[sid]=k+1
                except: pass
    print(f"\n[{name}]  (K={K}, RAW no-templating)",flush=True)
    print(f"  {'material':16s} {'GT_a':>6s} {'a_best':>6s} {'MAE_a':>6s} {'Hit@5':>6s} {'first':>5s}",flush=True)
    for sid in sids:
        gt,gf,ga=GT[sid]; h5=1 if (first[sid] and first[sid]<=5) else 0
        print(f"  {sid.replace('real0','').split('_',1)[1][:16] if '_' in sid else sid:16s} {ga:6.3f} {(('%.3f'%(ga-besta[sid])) if besta[sid] is not None else '--'):>6s} {(('%.3f'%besta[sid]) if besta[sid] is not None else '--'):>6s} {h5:>6d} {str(first[sid]):>5s}",flush=True)
    h5=sum(1 for s in sids if first[s] and first[s]<=5)/len(sids)
    mae=np.mean([besta[s] for s in sids if besta[s] is not None])
    print(f"  => Hit@5={h5:.2f} | MAE_a={mae:.3f} (over {sum(besta[s] is not None for s in sids)}/{len(sids)} formula-valid)",flush=True)
def load_v5():
    dev="cuda"; m=Phase3F2Guarded(g_img_dropout=0.0,inject_crys_to_img=True).to(dev); load_f2_weights(m,str(F2_CKPT)); load_evidence_weights(m,EV_CKPT)
    base=m.crystal.base; lat_head=m.crystal.lattice_out_6d; H=base.coord_out.in_features; nL=base.num_layers
    gh=nn.Sequential(nn.Linear(256,256),nn.SiLU(),nn.Linear(256,256)).to(dev)
    inj=nn.ModuleList([nn.Linear(256,H) for _ in range(nL+1)]).to(dev); gate=nn.ModuleList([nn.Linear(H+256,H) for _ in range(nL+1)]).to(dev)
    ck=torch.load(PROJ/"outputs/sccd_v5/best.pt",map_location="cpu",weights_only=False)
    base.load_state_dict(ck["base"]); lat_head.load_state_dict(ck["lat_head"]); m.evidence_unet.load_state_dict(ck["evidence_unet"])
    m.g_crys_to_unet.load_state_dict(ck["g_crys_to_unet"]); gh.load_state_dict(ck["gimg_head"]); inj.load_state_dict(ck["inj"]); gate.load_state_dict(ck["gate"]); unk=ck["unk"].to(dev)
    for p in m.parameters(): p.requires_grad_(False)
    m.eval(); base.eval(); gh.eval(); return m,base,lat_head,unk,inj,gate,gh
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--K",type=int,default=20); a=ap.parse_args()
    dev="cuda"; m,base,lat_head,unk,inj,gate,gh=load_v5()
    ab2=m.beta_scheduler.alphas_cumprod[200]; ce0=ab2.sqrt(); ce1=(1-ab2).sqrt(); gl=[]
    for sid in sids:
        stem=imgs[sid].view(1,1,IMG,IMG).to(dev); ev_t=ce0*stem+ce1*torch.randn_like(stem)
        gl.append(gimg_of(m,gh,ev_t,stem,torch.full((1,),200,device=dev)))
    gimg=torch.cat(gl,0)
    print(f"=== Real-5 v2 unified eval (diffusion), {len(sids)} monolayers ===",flush=True)
    for sid in sids: print(f"  {sid}: {GT[sid][1]} a={GT[sid][2]:.3f}",flush=True)
    run("SCCD v5 (+img)",m,base,lat_head,unk,inj,gate,gimg,True,a.K,tag="sccd_v5img")
    run("SCCD v5 (no-img)",m,base,lat_head,unk,inj,gate,torch.zeros(len(sids),256,device=dev),False,a.K,tag="sccd_v5noimg")
    mf=Phase3F2Guarded(g_img_dropout=0.0,inject_crys_to_img=True).to(dev); load_f2_weights(mf,str(F2_CKPT))
    for p in mf.parameters(): p.requires_grad_(False)
    mf.eval(); bf=mf.crystal.base.eval(); lf=mf.crystal.lattice_out_6d; Hd=bf.coord_out.in_features; nLf=bf.num_layers
    di=nn.ModuleList([nn.Linear(256,Hd) for _ in range(nLf+1)]).to(dev); dg=nn.ModuleList([nn.Linear(Hd+256,Hd) for _ in range(nLf+1)]).to(dev)
    run("F2 (no-img)",mf,bf,lf,torch.zeros(1,bf.node_embedding.embedding_dim,device=dev),di,dg,torch.zeros(len(sids),256,device=dev),False,a.K,tag="f2")
if __name__=="__main__": main()
