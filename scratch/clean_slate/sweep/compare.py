#!/usr/bin/env python3
"""Compare vc2 sweep results against vc1 + abcrown references. Flags
SOUNDNESS disagreements loudly (vc2 sat vs ref unsat, or vc2 unsat vs
ref sat -- one of them is wrong), and summarizes per-category
solved/matched/timeout/error. Also emits ~/repositories/
vnncomp2026_results_official/vc2/results.csv in the reference format.

Usage: compare.py [box_results.csv]
"""
import csv
import os
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.expanduser('~/repositories/vnncomp2026_results_official')
BOX = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'box_results.csv')


def rel(p):
    i = p.find('/benchmarks/')
    return p[i + len('/benchmarks/'):] if i >= 0 else p


def load_ref(tool):
    d = {}
    with open(f'{RES}/{tool}/results.csv') as f:
        for row in csv.reader(f):
            if len(row) >= 6:
                d[(rel(row[1]), rel(row[2]))] = row[4]
    return d


def load_box():
    d = {}
    if os.path.exists(BOX):
        with open(BOX) as f:
            for row in csv.reader(f):
                if len(row) >= 6:
                    d[(row[1], row[2])] = (row[0], row[4], row[5])
    return d


def main():
    vc, ab = load_ref('vibecheck'), load_ref('alpha_beta_crown')
    box = load_box()
    unsound = []           # vc2 contradicts a reference sat/unsat
    percat = defaultdict(lambda: defaultdict(int))
    for k, (cat, verd, t) in box.items():
        percat[cat][verd] += 1
        rv, rb = vc.get(k), ab.get(k)
        refs = {x for x in (rv, rb) if x in ('sat', 'unsat')}
        if verd in ('sat', 'unsat') and refs and verd not in refs and len(refs) == 1:
            # a reference is confident and vc2 says the OPPOSITE -> one is unsound
            unsound.append((cat, k, verd, rv, rb, t))
    print(f'=== vc2 sweep: {len(box)} instances ===')
    for cat in sorted(percat):
        d = dict(percat[cat])
        print(f'  {cat:42s} {d}')
    if unsound:
        print(f'\n### SOUNDNESS ALERT: {len(unsound)} vc2/reference contradictions ###')
        for cat, k, verd, rv, rb, t in unsound[:40]:
            print(f'  {cat} {k[0].split("/")[-1]} / {k[1].split("/")[-1]}: '
                  f'vc2={verd} vc1={rv} abc={rb} ({t}s)')
    else:
        print('\nno sat/unsat contradictions vs references')


if __name__ == '__main__':
    main()
