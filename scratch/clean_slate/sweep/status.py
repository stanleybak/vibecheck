#!/usr/bin/env python3
"""Rich sweep status: %done, ETA, and vc2-vs-ABC/VC1 scorecard."""
import csv, os, subprocess, time
HERE=os.path.dirname(os.path.abspath(__file__))
RES=os.path.expanduser('~/repositories/vnncomp2026_results_official')
KEY=os.path.expanduser('~/.ssh/g5-8x.pem')
HOST=open(os.path.join(HERE,'../../../scratch/aws_host.txt')).read().strip()
BOX=os.path.join(HERE,'box_results.csv')
def rel(p):
    i=p.find('/benchmarks/'); return p[i+len('/benchmarks/'):] if i>=0 else p
def ref(t):
    d={}
    for row in csv.reader(open(f'{RES}/{t}/results.csv')):
        if len(row)>=6: d[(rel(row[1]),rel(row[2]))]=row[4]
    return d
def main():
    total=sum(1 for _ in open(os.path.join(HERE,'worklist.csv')))-1
    vc,ab=ref('vibecheck'),ref('alpha_beta_crown')
    def rk(onnx,vnnlib):   # reference key: PAIR rows -> original onnx
        return (onnx.split('|')[1] if onnx.startswith('PAIR|') else onnx, vnnlib)
    box={}; refkey={}
    if os.path.exists(BOX):
        for row in csv.reader(open(BOX)):
            if len(row)>=6:
                box[(row[1],row[2])]=(row[4],float(row[5]))
                refkey[(row[1],row[2])]=rk(row[1],row[2])
    done=len(box)
    solved=[k for k,(v,t) in box.items() if v in ('sat','unsat')]
    err=[k for k,(v,t) in box.items() if v not in ('sat','unsat','timeout','unknown')]
    to=[k for k,(v,t) in box.items() if v in ('timeout','unknown')]
    # match vs each ref (only where ref is confident)
    m_vc=sum(1 for k in solved if vc.get(refkey[k])==box[k][0])
    m_ab=sum(1 for k in solved if ab.get(refkey[k])==box[k][0])
    contra=[k for k in solved if (vc.get(refkey[k]) in('sat','unsat') and vc.get(refkey[k])!=box[k][0] and ab.get(refkey[k])!=box[k][0]) or (ab.get(refkey[k]) in('sat','unsat') and ab.get(refkey[k])!=box[k][0] and vc.get(refkey[k])!=box[k][0])]
    # worklist: (key) -> (ref_time, timeout)
    wl={}
    for r in csv.DictReader(open(os.path.join(HERE,'worklist.csv'))):
        wl[(r['onnx_rel'],r['vnnlib_rel'])]=(float(r['ref_time']),float(r['timeout']))
    # SLOWDOWN model (Stan): slowdown = median(vc2_time/ref_time) over
    # completed; remaining est = min(ref_time*slowdown, timeout); sum.
    ratios=[t/wl[k][0] for k,(v,t) in box.items()
            if k in wl and wl[k][0]>0.1 and v in ('sat','unsat')]
    ratios.sort()
    slow=ratios[len(ratios)//2] if ratios else 1.0        # median
    gpu_s=sum(t for _,t in box.values())
    remain_s=0.0
    for k,(rt,tmo) in wl.items():
        if k in box: continue
        remain_s+=min(rt*slow, tmo)
    eta_h=remain_s/3600
    pct=100*done/total
    print(f'PROGRESS: {done}/{total} ({pct:.1f}%)  |  GPU-time used {gpu_s/3600:.1f}h  |  slowdown vs refs {slow:.1f}x (median)  |  ETA ~{eta_h:.0f}h (slowdown-capped-at-timeout model)')
    print(f'  verdicts: solved={len(solved)} (sat/unsat) timeout/unknown={len(to)} error={len(err)}')
    print(f'  vs references (on {len(solved)} solved): matches VC1={m_vc} ABC={m_ab}  contradictions={len(contra)}')
    if err:
        from collections import Counter
        ec=Counter(box[k][0] for k in err)
        # which categories erroring
        cc=Counter(k[0].split('/')[0] for k in err)
        print(f'  errors by category: {dict(cc)}')
    if contra:
        print(f'  !!! {len(contra)} SOUNDNESS CONTRADICTIONS -- investigate')
if __name__=='__main__': main()
