#!/usr/bin/env python3
"""Merge ALL metrics into one master table per noise:
 evalfull (MinRMS@5, RMSD_A, matchF, Hit@5, FExA, W1a/b/g, KL) + evalextra (Hit@1, MinRMS@1, SG_acc, failure modes)
 + rmsd_out (RMSD_A for DiffCSP). DiffCSP/MatterGen noise-invariant (low replicated)."""
import json,os
EF="/tmp/evalfull"; EX="/tmp/evalextra"; RM="/tmp/rmsd_out"
METH=[("MatterGen","mattergen_ehull0_top5",True),("MicroscopyGPT","microscopygpt",False),
      ("DiffCSP","diffcsp",True),("SCCD v5","sccd_v5",False)]
def j(d,m,nz):
    p=f"{d}/{m}_{nz}.json"; return json.load(open(p)) if os.path.exists(p) else None
def f(v,p=4): return " --" if v is None else f"{v:.{p}f}"
COLS=["MinRMS5_md","MinRMS5_mn","Hit@1","Hit@5","MinRMS1_md","RMSD_A_md","RMSD_A_mn","matchF","SG_acc","FExA","W1(a)","W1(b)","W1(g)","KL_SG"]
for nz in ["low","mid","high"]:
    print(f"\n===================== {nz.upper()} NOISE =====================")
    print(f"{'Method':13s} "+" ".join(f"{c:>9s}" for c in COLS))
    rowsdone=True
    for disp,m,ninv in METH:
        snz="low" if ninv else nz
        e=j(EF,m,snz); x=j(EX,m,snz); r=j(RM,m,snz)
        if e is None or x is None:
            print(f"{disp:13s}   (pending: gen/eval not complete)"); rowsdone=False; continue
        rA_md=r.get("RMSD_A_med") if (r and r.get("RMSD_A_med") is not None) else e.get("RMSD_A_med")
        rA_mn=r.get("RMSD_A_mean") if (r and r.get("RMSD_A_mean") is not None) else e.get("RMSD_A_mean")
        mF   =r.get("matchF")     if (r and r.get("RMSD_A_med") is not None) else e.get("matchF")
        vals=[e.get("MinRMS_med"),e.get("MinRMS_mean"),x.get("Hit1"),e.get("Hit5"),x.get("MinRMS1_med"),
              rA_md,rA_mn,mF,x.get("SGacc"),e.get("FExA"),e.get("W1a"),e.get("W1b"),e.get("W1g"),e.get("KLSG")]
        print(f"{disp:13s} "+" ".join(f"{f(v):>9s}" for v in vals))
    # failure-mode sub-table
    print(f"  -- failure modes (% of all samples; Hit@5 misses by top-1) : formula / wrong_lattice / lattice_ok_wrong_SG / coord_other --")
    for disp,m,ninv in METH:
        x=j(EX,m,"low" if ninv else nz)
        if x is None: continue
        fl=x["fail"]; print(f"     {disp:13s} {fl['formula']:.3f} / {fl['wrong_lattice']:.3f} / {fl['lattice_ok_wrong_SG']:.3f} / {fl['coord_other']:.3f}")
print("\n[note] StructureMatcher stol=0.5/ltol=0.3/angle=10. MinRMS/Hit/SG via scale=TRUE; RMSD_A/matchF via scale=FALSE (Angstrom). W1 over all top-1 (paper def). All methods, one harness.")
