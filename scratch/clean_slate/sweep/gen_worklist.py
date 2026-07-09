"""Build the vc2 sweep work-list: every instance solved by vc1 OR abcrown,
sorted by the fastest reference solve time (easy wins first), with the
official per-instance timeout joined from instances.csv.

Output CSV columns:
  ref_time, category, version, onnx_rel, vnnlib_rel, timeout,
  vc1_verdict, vc1_time, abc_verdict, abc_time
Paths are relative to the benchmarks root (benchmarks/<cat>/<ver>/...),
so both local and the box resolve them under their own benchmarks dir.
"""
import csv
import os
import sys

RES = os.path.expanduser('~/repositories/vnncomp2026_results_official')
BENCH = os.path.expanduser('~/repositories/vnncomp2026_benchmarks/benchmarks')
OUT = os.path.join(os.path.dirname(__file__), 'worklist.csv')


def rel_after_benchmarks(p):
    # results path: vnncomp2026_benchmarks/benchmarks/<cat>/<ver>/onnx/x
    i = p.find('/benchmarks/')
    return p[i + len('/benchmarks/'):] if i >= 0 else p


def load(tool):
    d = {}
    with open(f'{RES}/{tool}/results.csv') as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            cat, onnx, vnnlib, ver, verdict, t = row[:6]
            onnx_r = rel_after_benchmarks(onnx)      # <cat>/<ver>/onnx/x
            vnnlib_r = rel_after_benchmarks(vnnlib)
            # key on the PATH (which carries the true version dir), NOT the
            # results version FIELD -- they disagree for several categories
            # (results ver 1.0 but path .../2.0/...), which silently dropped
            # adaptive_cruise/sat_relu/smart_turn/cgan2026/relusplitter.
            d[(onnx_r, vnnlib_r)] = (cat, verdict, float(t) if t else 0.0)
    return d


def timeout_index():
    """(onnx_rel, vnnlib_rel) -> official timeout; plus cat_default[cat]."""
    idx = {}
    cat_to = {}
    for cat in os.listdir(BENCH):
        catd = os.path.join(BENCH, cat)
        if not os.path.isdir(catd):
            continue
        for ver in os.listdir(catd):
            ic = os.path.join(catd, ver, 'instances.csv')
            if not os.path.exists(ic):
                continue
            with open(ic) as f:
                for row in csv.reader(f):
                    if len(row) < 3:
                        continue
                    onnx, vnnlib, to = row[0], row[1], row[2]
                    if onnx.strip().startswith('['):
                        continue        # network-pair (handled separately)
                    onnx_r = f'{cat}/{ver}/{onnx.lstrip("./")}'
                    vnnlib_r = f'{cat}/{ver}/{vnnlib.lstrip("./")}'
                    idx[(onnx_r, vnnlib_r)] = float(to)
                    cat_to.setdefault(cat, []).append(float(to))
    # category default = the most common timeout in that category
    from collections import Counter
    cat_default = {c: Counter(v).most_common(1)[0][0]
                   for c, v in cat_to.items()}
    return idx, cat_default


def main():
    vc = load('vibecheck')
    ab = load('alpha_beta_crown')
    tos, cat_default = timeout_index()
    keys = set(vc) | set(ab)
    rows = []
    missing_to = 0
    for k in keys:
        vcv = vc.get(k)      # (cat, verdict, time) | None
        abv = ab.get(k)
        cat = (vcv or abv)[0]
        # derive version dir from the path key
        ver = k[0].split('/')[1] if '/' in k[0] else '1.0'
        onnx_r, vnnlib_r = k
        solved_times = []
        if vcv and vcv[1] in ('sat', 'unsat'):
            solved_times.append(vcv[2])
        if abv and abv[1] in ('sat', 'unsat'):
            solved_times.append(abv[2])
        if not solved_times:
            continue
        ref_time = min(solved_times)
        to = tos.get(k)
        if to is None:
            # reference solved an instance not in the current instances.csv
            # (older instance set); the file exists, so sweep it with the
            # category's standard timeout rather than dropping it.
            to = cat_default.get(cat)
            if to is None:
                missing_to += 1
                continue
        rows.append((ref_time, cat, ver, onnx_r, vnnlib_r, to,
                     vcv[1] if vcv else '', vcv[2] if vcv else '',
                     abv[1] if abv else '', abv[2] if abv else ''))
    # network-pair categories (isomorphic/monotonic): instances.csv onnx is
    # a pair [('f',orig),('g',pert)]. Match to references by vnnlib_rel (the
    # reference records the ORIGINAL net path + this vnnlib). Encode the pair
    # in onnx_rel as PAIR|<f_rel>|<g_rel> for the driver's --net arg.
    import ast
    ref_by_vnnlib = {}
    for src in (vc, ab):
        for (onnx_r, vnnlib_r), (cat, verd, t) in src.items():
            if verd in ('sat', 'unsat'):
                ref_by_vnnlib.setdefault(vnnlib_r, []).append(t)
    for cat in os.listdir(BENCH):
        catd = os.path.join(BENCH, cat)
        if not os.path.isdir(catd):
            continue
        for ver in os.listdir(catd):
            ic = os.path.join(catd, ver, 'instances.csv')
            if not os.path.exists(ic):
                continue
            for row in csv.reader(open(ic)):
                if len(row) < 3 or not row[0].strip().startswith('['):
                    continue
                try:
                    pair = dict(ast.literal_eval(row[0]))
                except Exception:
                    continue
                f_rel = f'{cat}/{ver}/' + pair['f'].lstrip('./')
                g_rel = f'{cat}/{ver}/' + pair['g'].lstrip('./')
                vnnlib_r = f'{cat}/{ver}/' + row[1].lstrip('./')
                times = ref_by_vnnlib.get(vnnlib_r)
                if not times:
                    continue                 # neither ref solved this pair
                def _ex(r):
                    return (os.path.exists(os.path.join(BENCH, r))
                            or os.path.exists(os.path.join(BENCH, r + '.gz')))
                if not (_ex(f_rel) and _ex(g_rel)):
                    continue
                rows.append((min(times), cat, ver, f'PAIR|{f_rel}|{g_rel}',
                             vnnlib_r, float(row[2]), '', '', '', ''))
    rows.sort(key=lambda r: r[0])
    with open(OUT, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['ref_time', 'category', 'version', 'onnx_rel',
                    'vnnlib_rel', 'timeout', 'vc1_verdict', 'vc1_time',
                    'abc_verdict', 'abc_time'])
        w.writerows(rows)
    print(f'work-list: {len(rows)} instances -> {OUT}')
    print(f'  (dropped {missing_to} with no instances.csv timeout match)')
    # sum of official timeouts (worst-case GPU-seconds if all time out)
    total_to = sum(r[5] for r in rows)
    print(f'  worst-case GPU-time if ALL hit timeout: {total_to/3600:.0f} GPU-h')
    print(f'  (reference solved these fast; realistic << that)')
    from collections import Counter
    c = Counter(r[1] for r in rows)
    print('  by category:', dict(sorted(c.items())))


if __name__ == '__main__':
    main()
