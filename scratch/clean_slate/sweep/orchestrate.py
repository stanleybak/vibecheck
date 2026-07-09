#!/usr/bin/env python3
"""Disk-managed orchestrator for the vc2 vnncomp sweep. The box has ~6.5G
free, so files are STREAMED in size-bounded batches: push a batch -> the
durable box driver grinds it -> drain results+CEs back to local -> delete
the batch's files + CEs from the box. Sorted worklist means batches
progress fast->slow. Idempotent and resumable: the box results.csv (never
deleted) drives the done-skip, so re-pushing never re-runs.

Usage (run from local each poll):
  orchestrate.py poll    # drain finished results+CEs, then push next batch
  orchestrate.py status  # progress summary
Env: HOST (default from scratch/aws_host.txt), KEY (ssh key).
"""
import csv
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.expanduser('~/Desktop/temp/vc_copy1/vibecheck-nn')
BENCH_LOCAL = os.path.expanduser('~/repositories/vnncomp2026_benchmarks/benchmarks')
WORKLIST = os.path.join(HERE, 'worklist.csv')
RAW_LOCAL = os.path.join(HERE, 'box_results.csv')     # mirror of box results
CE_LOCAL = os.path.join(HERE, 'ce')                   # drained CEs
KEY = os.environ.get('KEY', os.path.expanduser('~/.ssh/g5-8x.pem'))
HOST = os.environ.get('HOST') or open(
    os.path.join(REPO, 'scratch/aws_host.txt')).read().strip()

BOX_BENCH = '/home/ubuntu/bench_stage'
BOX_RUN = '/home/ubuntu/persistent_runs/vc2_sweep'
BOX_RESULTS = f'{BOX_RUN}/results.csv'
BOX_CE = f'{BOX_RUN}/ce'
BOX_MANIFEST = f'{BOX_RUN}/batch.csv'
BOX_SCRIPTS = '/home/ubuntu/vc2/vnncomp_scripts_vc2'

SIZE_BUDGET = int(os.environ.get('SIZE_BUDGET', 2_500_000_000))   # 2.5 GB
COUNT_BUDGET = int(os.environ.get('COUNT_BUDGET', 600))

NO_ATTACK = os.environ.get('NO_ATTACK', '0')       # SAT soundness sweep
SUFFIX = os.environ.get('SUFFIX', '')              # results-set suffix (e.g. _sat)
if SUFFIX:
    RAW_LOCAL = os.path.join(HERE, f'box_results{SUFFIX}.csv')
    BOX_RUN = f'/home/ubuntu/persistent_runs/vc2_sweep{SUFFIX}'
    BOX_RESULTS = f'{BOX_RUN}/results.csv'
    BOX_CE = f'{BOX_RUN}/ce'
    BOX_MANIFEST = f'{BOX_RUN}/batch.csv'


def ssh(cmd, timeout=60, check=False):
    r = subprocess.run(['ssh', '-o', 'ConnectTimeout=15', '-i', KEY,
                        f'ubuntu@{HOST}', cmd], capture_output=True,
                       text=True, timeout=timeout)
    if check and r.returncode:
        raise RuntimeError(f'ssh failed: {r.stderr[:200]}')
    return r.stdout.strip()


def done_keys():
    """(onnx_rel, vnnlib_rel) already in the local results mirror."""
    keys = set()
    if os.path.exists(RAW_LOCAL):
        with open(RAW_LOCAL) as f:
            for row in csv.reader(f):
                if len(row) >= 3:
                    keys.add((row[1], row[2]))
    return keys


def worklist():
    rows = []
    with open(WORKLIST) as f:
        r = csv.DictReader(f)
        for x in r:
            rows.append(x)
    return rows


def next_batch():
    done = done_keys()
    batch, size = [], 0
    for x in worklist():
        k = (x['onnx_rel'], x['vnnlib_rel'])
        if k in done:
            continue
        op = os.path.join(BENCH_LOCAL, x['onnx_rel'])
        vp = os.path.join(BENCH_LOCAL, x['vnnlib_rel'])
        if not os.path.exists(op) or not os.path.exists(vp):
            continue
        fsz = os.path.getsize(op) + os.path.getsize(vp)
        if batch and (size + fsz > SIZE_BUDGET or len(batch) >= COUNT_BUDGET):
            break
        batch.append(x)
        size += fsz
    return batch, size


def driver_running():
    out = ssh("pgrep -f 'sweep_driver.sh' | head -1")
    return bool(out.strip())


def drain():
    ssh(f'mkdir -p {BOX_RUN} {BOX_CE}')
    os.makedirs(CE_LOCAL + SUFFIX, exist_ok=True)
    # pull results + CEs
    subprocess.run(['rsync', '-az', '-e', f'ssh -i {KEY}',
                    f'ubuntu@{HOST}:{BOX_RESULTS}', RAW_LOCAL],
                   capture_output=True)
    subprocess.run(['rsync', '-az', '-e', f'ssh -i {KEY}',
                    f'ubuntu@{HOST}:{BOX_CE}/', CE_LOCAL + SUFFIX + '/'],
                   capture_output=True)
    # delete drained CEs on box (kept locally) + staged files for done insts
    ssh(f'rm -f {BOX_CE}/* 2>/dev/null; '
        f'find {BOX_BENCH} -type f -delete 2>/dev/null; true')


def push():
    batch, size = next_batch()
    if not batch:
        print('no more instances to push')
        return 0
    # stage files: build an rsync file-list (relpaths under BENCH_LOCAL)
    filelist = HERE + '/.stage_files'
    with open(filelist, 'w') as f:
        for x in batch:
            f.write(x['onnx_rel'] + '\n')
            f.write(x['vnnlib_rel'] + '\n')
    ssh(f'mkdir -p {BOX_BENCH} {BOX_RUN} {BOX_CE}')
    subprocess.run(['rsync', '-az', '--files-from=' + filelist,
                    '-e', f'ssh -i {KEY}', BENCH_LOCAL + '/',
                    f'ubuntu@{HOST}:{BOX_BENCH}/'], check=True)
    # write manifest on box
    man = HERE + '/.batch.csv'
    with open(man, 'w') as f:
        for x in batch:
            f.write(f"{x['category']},{x['version']},{x['onnx_rel']},"
                    f"{x['vnnlib_rel']},{x['timeout']}\n")
    subprocess.run(['rsync', '-az', '-e', f'ssh -i {KEY}', man,
                    f'ubuntu@{HOST}:{BOX_MANIFEST}'], check=True)
    # launch durable driver (setsid; idempotent skip via box results.csv)
    ssh(f'sudo rm -f /tmp/idle_since; setsid bash -c '
        f'"VC2_NO_ATTACK={NO_ATTACK} VC2_SRC=/home/ubuntu/vc2/src '
        f'VNNCOMP_PYTHON_PATH=/home/ubuntu/vibe/bin '
        f'bash {HERE.replace(REPO, "/home/ubuntu/vc2")}/sweep_driver.sh '
        f'{BOX_MANIFEST} {BOX_BENCH} {BOX_RESULTS} {BOX_CE} {BOX_SCRIPTS}" '
        f'</dev/null >/dev/null 2>&1 &')
    print(f'pushed batch: {len(batch)} instances, {size/1e6:.0f} MB '
          f'({batch[0]["category"]} .. {batch[-1]["category"]})')
    return len(batch)


def status():
    total = len(worklist())
    done = len(done_keys())
    from collections import Counter
    verd = Counter()
    if os.path.exists(RAW_LOCAL):
        with open(RAW_LOCAL) as f:
            for row in csv.reader(f):
                if len(row) >= 5:
                    verd[row[4]] += 1
    run = 'RUNNING' if driver_running() else 'idle'
    print(f'done {done}/{total}  driver={run}  verdicts={dict(verd)}')


def poll():
    drain()
    if not driver_running():
        push()
    status()


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'poll'
    {'poll': poll, 'status': status, 'push': push,
     'drain': drain}[cmd]()
