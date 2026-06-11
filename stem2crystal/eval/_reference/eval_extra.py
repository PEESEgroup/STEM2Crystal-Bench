#!/usr/bin/env python3
"""Reviewer-requested EXTRA metrics (parallel over samples):
 #1 Top-1 (non-oracle): Hit@1, MinRMS@1 med/mean              [Edv2-W4, Ds4v]
 #2 SG accuracy (top-1 spacegroup == GT) + failure-mode mix   [Ds4v-Q5, bNEE]
 #3 complexity-stratified Hit@5 by #atoms and #Wyckoff sites  [Edv2-W2, bNEE-Q4]
Failure modes (over Hit@5 MISSES, categorised by top-1): formula / wrong_lattice / lattice_ok_wrong_SG / coord_other.
"""
import os,sys,json,argparse,warnings,signal; warnings.filterwarnings("ignore")
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
ROOT=Path("/home/ubuntu/efs/KDD/EM/stem2cif-mattergen"); DATA=ROOT/"data/all_stem_output"; PRED=ROOT/"run_eval/predictions"
SMT=StructureMatcher(stol=0.5,angle_tol=10,ltol=0.3,scale=True)
class TO(Exception): pass
def _h(s,f): raise TO()
signal.signal(signal.SIGALRM,_h)
def guard(fn,sec=1.5):
    try: signal.setitimer(signal.ITIMER_REAL,sec); r=fn(); signal.setitimer(signal.ITIMER_REAL,0); return r
    except Exception: signal.setitimer(signal.ITIMER_REAL,0); return None
def load(p):
    try: return Structure.from_file(str(p))
    except: return None
def red(s):
    try: return s.composition.reduced_formula
    except: return None
def sg(s):
    try: return SpacegroupAnalyzer(s,symprec=0.1).get_space_group_number()
    except: return None
def nwyck(s):
    try: return len(SpacegroupAnalyzer(s,symprec=0.1).get_symmetrized_structure().equivalent_sites)
    except: return None
def latt_relerr(p,g):
    try: return max(abs(p.lattice.a-g.lattice.a)/g.lattice.a, abs(p.lattice.b-g.lattice.b)/g.lattice.b, abs(p.lattice.gamma-g.lattice.gamma)/max(g.lattice.gamma,1e-6))
    except: return 9.9
_GT={}; _M=None; _NZ=None
def init(m,nz,gt):
    global _GT,_M,_NZ; _GT=gt; _M=m; _NZ=nz
def per_sid(sid):
    g=_GT.get(sid)
    if g is None: return None
    d=PRED/_M/_NZ/f"{sid}_0"
    cands=[c for c in (load(d/f"rank_{k}.cif") for k in range(1,6)) if c is not None]
    if not cands: return dict(parsed=0,natom=g["na"],nw=g["nw"])
    top1=cands[0]; f1=red(top1); fok=int(f1==g["f"])
    # Top-1 match
    hit1=0; rms1=None
    if fok:
        r=guard(lambda: SMT.get_rms_dist(g["s"],top1))
        if r is not None: hit1=1; rms1=r[0]
    # Hit@5
    fm=[c for c in cands if red(c)==g["f"]]; hit5=0
    for c in fm:
        r=guard(lambda: SMT.get_rms_dist(g["s"],c))
        if r is not None: hit5=1; break
    # SG accuracy (top-1, conditioned on formula ok)
    sg1=sg(top1) if fok else None; sgmatch=int(fok and sg1==g["sg"])
    # failure mode (only for Hit@5 misses), by top-1
    fail=None
    if not hit5:
        if not fok: fail="formula"
        elif latt_relerr(top1,g["s"])>0.20: fail="wrong_lattice"
        elif sg1!=g["sg"]: fail="lattice_ok_wrong_SG"
        else: fail="coord_other"
    return dict(parsed=1,hit1=hit1,rms1=rms1,hit5=hit5,sgmatch=sgmatch,fok=fok,fail=fail,natom=g["na"],nw=g["nw"])

ATOM_BINS=[(1,4),(4,7),(7,12),(12,20),(20,999)]; WYCK_BINS=[(1,4),(4,6),(6,10),(10,999)]
def binlabel(v,bins): return next((f"[{a},{b if b<999 else '+'})" for a,b in bins if a<=v<b),"?")
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--method",required=True); ap.add_argument("--noise",required=True)
    ap.add_argument("--nproc",type=int,default=44); ap.add_argument("--n",type=int,default=0); a=ap.parse_args()
    sids=json.load(open(DATA/"split.json"))["test"]
    if a.n: sids=sids[:a.n]
    gt={}
    for s in sids:
        st=load(DATA/"cifs"/f"{s}.cif")
        if st: gt[s]=dict(s=st,f=red(st),sg=sg(st),na=len(st),nw=nwyck(st))
    import time; t0=time.time()
    with Pool(a.nproc,initializer=init,initargs=(a.method,a.noise,gt),maxtasksperchild=200) as p:
        res=[r for r in p.map(per_sid,sids,chunksize=4) if r]
    par=[r for r in res if r.get("parsed")]; N=len(gt)
    r1=[r["rms1"] for r in par if r["rms1"] is not None]
    out=dict(method=a.method,noise=a.noise,N=N,
        Hit1=sum(r["hit1"] for r in par)/N, MinRMS1_med=float(np.median(r1)) if r1 else None, MinRMS1_mean=float(np.mean(r1)) if r1 else None,
        SGacc=sum(r["sgmatch"] for r in par)/N, Hit5=sum(r["hit5"] for r in par)/N)
    # failure-mode mix (% of all samples)
    fails=[r["fail"] for r in par if r["fail"]]
    out["fail"]={k:round(fails.count(k)/N,3) for k in ["formula","wrong_lattice","lattice_ok_wrong_SG","coord_other"]}
    # complexity-stratified Hit@5
    def strat(key,bins):
        d={}
        for a0,b0 in bins:
            sub=[r for r in par if r[key] is not None and a0<=r[key]<b0]
            lbl=f"[{a0},{b0 if b0<999 else '+'})"
            d[lbl]=dict(n=len(sub),Hit5=round(sum(x["hit5"] for x in sub)/max(len(sub),1),3))
        return d
    out["by_atoms"]=strat("natom",ATOM_BINS); out["by_wyck"]=strat("nw",WYCK_BINS); out["sec"]=time.time()-t0
    Path("/tmp/evalextra").mkdir(exist_ok=True); json.dump(out,open(f"/tmp/evalextra/{a.method}_{a.noise}.json","w"))
    fm=lambda v:"  --  " if v is None else f"{v:.4f}"
    print(f"[{a.method:22s} {a.noise:4s}] Hit@1 {out['Hit1']:.3f} MinRMS@1 {fm(out['MinRMS1_med'])}/{fm(out['MinRMS1_mean'])} SGacc {out['SGacc']:.3f} Hit@5 {out['Hit5']:.3f} | fail {out['fail']} | {out['sec']:.0f}s",flush=True)
if __name__=="__main__": main()
