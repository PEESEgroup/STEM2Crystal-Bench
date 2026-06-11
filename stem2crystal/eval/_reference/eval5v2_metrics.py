#!/usr/bin/env python3
"""Comprehensive Real-5 v2 metrics, uniform over all methods' saved predictions (monolayer GT, 2D matching).
Metrics: Form | Hit@1 | Hit@5 | MinRMS5_md/mn | Match_abs | RMSD_A | SG-Acc | W1(a/b/g) | MAE_a | relErr_a."""
import sys, json, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance
from pymatgen.io.cif import CifParser
from pymatgen.core import Structure, Lattice
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
ROOT=Path("/home/ubuntu/efs/KDD/EM/stem2cif-mattergen"); DS=Path("/home/ubuntu/efs/KDD/EM/real5_v2"); PRED=ROOT/"run_eval/predictions"
CN=20.0
MS=StructureMatcher(ltol=0.3,stol=0.3,angle_tol=15,primitive_cell=True,scale=True,attempt_supercell=True)   # scale-normalized
MA=StructureMatcher(ltol=0.3,stol=0.4,angle_tol=15,primitive_cell=True,scale=False,attempt_supercell=True)   # absolute
def load(p):
    try: return CifParser(str(p)).parse_structures(primitive=False)[0]
    except: return None
def sg(s):
    try: return SpacegroupAnalyzer(s,symprec=0.3).get_space_group_number()
    except: return 1
def norm_c(st):
    L=st.lattice; cz=np.array([c[2] for c in st.frac_coords])*L.c; cz=cz-cz.mean()
    nf=st.frac_coords.copy(); nf[:,2]=0.5+cz/CN
    try: return Structure(Lattice.from_parameters(L.a,L.b,CN,90,90,L.gamma),st.species,nf)
    except: return st
sids=json.load(open(DS/"split.json"))["test"]
GT={}
for sid in sids:
    s=load(DS/"cifs"/f"{sid}.cif"); GT[sid]=dict(s=norm_c(s),f=s.composition.reduced_formula,sg=sg(norm_c(s)),a=s.lattice.a,b=s.lattice.b,g=s.lattice.gamma,n=len(s),v=norm_c(s).volume)
def cands(method,sid):
    d=PRED/method/"real"/f"{sid}_0"
    cs=[load(p) for p in sorted(d.glob("rank_*.cif"))] if d.exists() else []
    return [norm_c(c) for c in cs if c is not None][:5]
def score(method):
    N=len(sids); hit1=hit5=mabs=sgok=nform=0; rms5=[]; pa=[];pb=[];pg=[];ga=[];gb=[];gg=[]; mae=[];rel=[]; rA=[]
    for sid in sids:
        g=GT[sid]; cs=cands(method,sid)
        if not cs: continue
        t1=cs[0]; fok=(t1.composition.reduced_formula==g["f"])
        if fok: nform+=1
        pa.append(t1.lattice.a);pb.append(t1.lattice.b);pg.append(t1.lattice.gamma); ga.append(g["a"]);gb.append(g["b"]);gg.append(g["g"])
        if fok and sg(t1)==g["sg"]: sgok+=1
        fm=[c for c in cs if c.composition.reduced_formula==g["f"]]
        if fm: mae.append(min(abs(c.lattice.a-g["a"]) for c in fm)); rel.append(min(abs(c.lattice.a-g["a"])/g["a"] for c in fm))
        try:
            if fok and MS.fit(g["s"],t1): hit1+=1
        except: pass
        h=False; mr=None; ha=False; ra=None
        for c in fm:
            try:
                if MS.fit(g["s"],c): h=True
                dd=MS.get_rms_dist(g["s"],c)
                if dd is not None: mr=dd[0] if mr is None else min(mr,dd[0])
            except: pass
            try:
                if MA.fit(g["s"],c):
                    ha=True; dd=MA.get_rms_dist(g["s"],c)
                    if dd is not None:
                        val=dd[0]*(g["v"]/g["n"])**(1/3.); ra=val if ra is None else min(ra,val)
            except: pass
        if h: hit5+=1
        if ha: mabs+=1
        if mr is not None: rms5.append(mr)
        if ra is not None: rA.append(ra)
    W=lambda p,q: wasserstein_distance(p,q) if p else None
    return dict(Form=nform/N,Hit1=hit1/N,Hit5=hit5/N,
        MinRMS_md=float(np.median(rms5)) if rms5 else None, MinRMS_mn=float(np.mean(rms5)) if rms5 else None,
        Mabs=mabs/N, RMSD=float(np.median(rA)) if rA else None, SGacc=sgok/N,
        W1a=W(pa,ga),W1b=W(pb,gb),W1g=W(pg,gg), MAEa=np.mean(mae) if mae else None, relA=np.median(rel) if rel else None)
def f(v,p=3): return "  -- " if v is None else f"{v:.{p}f}"
METH=[("AutoMat","automat_real5v2"),("MicroscopyGPT","microscopygpt_real5v2"),("DiffCSP (original)","diffcsp_real5v2"),
      ("SCCD v5 (+img)","sccd_v5img_real5v2")]
cols=["Form","Hit@1","Hit@5","MinRMS_md","MinRMS_mn","Match_abs","RMSD_A","SG-Acc","W1(a)","W1(b)","W1(g)","MAE_a","relErr_a"]
print(f"=== Real-5 v2 — comprehensive metrics (monolayer GT, RAW, no templating; {len(sids)} samples) ===")
print(f"{'Method':17s}"+"".join(f"{c:>10s}" for c in cols))
for disp,m in METH:
    r=score(m)
    vals=[f(r['Form'],2),f(r['Hit1'],2),f(r['Hit5'],2),f(r['MinRMS_md'],4),f(r['MinRMS_mn'],4),f(r['Mabs'],2),f(r['RMSD'],3),f(r['SGacc'],2),f(r['W1a']),f(r['W1b']),f(r['W1g'],2),f(r['MAEa']),f(r['relA'])]
    print(f"{disp:17s}"+"".join(f"{v:>10s}" for v in vals))
print("\n[note] Hit/MinRMS/SG/W1 via scale=TRUE (in-plane, c-normalized monolayer); Match_abs/RMSD_A via scale=FALSE (absolute, Angstrom).")
