<p align="center">
  <img src="https://raw.githubusercontent.com/stanleybak/vibecheck-nn/main/vibecheck.svg" alt="vibecheck logo" width="640">
</p>

The **vibecheck** formal verification tool is a high-performance, vibe-coded decision procedure for neural networks. Given an ONNX neural network and a VNNLIB specification, vibecheck tries to prove the property or find a counterexample. It solves the same open-loop neural network verification problem as established verifiers like [α,β-CROWN](https://github.com/Verified-Intelligence/alpha-beta-CROWN), [Marabou](https://github.com/NeuralNetworkVerification/Marabou), and [NNV](https://github.com/verivital/nnv), hopefully faster and on larger networks. Does your neural network pass the vibecheck?

## Install

```bash
pip install vibecheck-nn
```

This provides both the `vibecheck` command-line tool and the importable `vibecheck`
Python package (requires Python 3.10+). For an isolated, always-available CLI,
install it with [pipx](https://pipx.pypa.io/) instead:

```bash
pipx install vibecheck-nn
```

The `vibecheck` command is on your PATH whenever its install environment is active
(or globally, with pipx). Otherwise, invoke it as a module in any environment where
the package is installed:

```bash
python -m vibecheck --version
python -m vibecheck verify <query.vnnlib> --network N=<model.onnx> --timeout 60
```

vibecheck depends on PyTorch, whose default wheel is a large CUDA build (roughly
4 GB). On a CPU-only machine, install the CPU build of torch first, then vibecheck:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install vibecheck-nn
```

Certain analysis features build on Gurobi (`gurobipy`, installed automatically).
The bundled license is size-limited but fine for small models and the examples;
larger networks need a full [Gurobi license](https://www.gurobi.com/) (free for
academics).

### From source (development)

```bash
git clone https://github.com/stanleybak/vibecheck-nn
cd vibecheck-nn
uv venv --python 3.12 .venv                        # https://docs.astral.sh/uv/
VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[dev]"
```

Then use `.venv/bin/vibecheck` for the CLI and `.venv/bin/python` for the tests.

## Usage

vibecheck implements the [VNN-LIB standard](https://www.vnnlib.org/)'s solver CLI:

```bash
vibecheck verify <query.vnnlib> --network NAME=<model.onnx> [--timeout SECONDS] \
                 [--serialise-assignments DIR]
vibecheck supports <capability>        # for example --onnx-operators
vibecheck --name | --version
vibecheck --examples-dir               # path to the bundled example files
```

`verify` prints the verdict (`sat`/`unsat`/`unknown`/`timed-out`) as the first
stdout line, followed only by the satisfying assignment for `sat` (progress goes
to stderr); `--serialise-assignments DIR` writes the assignment as ONNX
TensorProtos instead. In `supports` output, a `*` after an identifier marks
partial support, with a short note on the same line.

The example below verifies the bundled ACAS-Xu network against a property that
holds. `vibecheck --examples-dir` prints where the bundled files live, so `cd`
there first to use short filenames. Everything except the final `unsat` is
progress on stderr (trimmed here):

```console
$ cd "$(vibecheck --examples-dir)"
$ vibecheck verify prop_1.vnnlib --network N=ACASXU_run2a_2_2_batch_2000.onnx --timeout 60
[vibecheck] device auto-selected: cuda
[   0.1s] [vibecheck] Net(13 ops, in=5, out=5, input:1, linmap:7, relu:6)
[   1.1s] [vibecheck/pgd] best_margin=+4.012e+00 iters=49 t=0.28s
[   1.1s] [vibecheck] crown: worst=-5215.2451 open=1/1
[   1.3s] [vibecheck] alpha-crown: worst=-1436.0988 open=1/1
[   1.3s] [vibecheck/bab] refine probe: crown=-995.688 zono=-3352.839 -> full_refine=True
[   1.6s] [vibecheck] input_split_bab (wide slice, alpha=8): unsat {'bounded': 397, 'splits': 99, 'rounds': 9}
[vibecheck] verdict: unsat  (1.60s)
unsat
```

And a violated property, where stdout is the verdict plus the satisfying assignment:

```console
$ vibecheck verify prop_2.vnnlib --network N=ACASXU_run2a_2_2_batch_2000.onnx --timeout 60  2>/dev/null
sat
X float32 [1,1,1,5]
0.61291265487670898
-0.0035594701766967773
-0.15507221221923828
0.48228561878204346
-0.46316659450531006
Y float32 [1,5]
0.051299020648002625
-0.021836085245013237
0.024599984288215637
-0.014120602980256081
0.023107599467039108
```

## Programmatic use

`vibecheck.verify()` runs a verification like the CLI and returns a `VerifyResult`:

```python
from importlib.resources import files
from vibecheck import verify

# a bundled ACAS-Xu example that has a counterexample
ex = files("vibecheck") / "examples"
r = verify(net=str(ex / "ACASXU_run2a_2_2_batch_2000.onnx"),
           spec=str(ex / "prop_2.vnnlib"), timeout=60)

print(r.verdict)              # 'sat'  (else 'unsat' / 'unknown' / 'timeout' / 'error')
print(r.counterexample["X"])  # the violating input       (numpy array)
print(r.counterexample["Y"])  # the network's output on X
```

```
sat
[[[[ 0.61291265 -0.00355947 -0.15507221  0.48228562 -0.4631666 ]]]]
[[ 0.05129902 -0.02183609  0.02459998 -0.0141206   0.0231076 ]]
```

`VerifyResult` also carries `.details` (timings and the route taken) and `.elapsed`.

Files are just one input form. `net` also accepts an in-memory `onnx.ModelProto`,
serialized ONNX bytes, or a `torch.nn.Module` (exported with `torch.onnx.export`;
pass `example_input=`), and `spec` accepts raw VNNLIB text or a `Spec` builder
(an input box plus unsafe output regions):

```python
import torch
from vibecheck import Spec, verify

model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU())
spec = Spec(x_lo=[0, 0], x_hi=[1, 1]).forbid([[1.0, 0.0]], [-100.0])  # unsafe: y0 >= 100
r = verify(model, spec, timeout=30, example_input=torch.zeros(1, 2))
print(r.verdict)   # 'unsat': y0 can never reach 100 on the unit box
```

Each `forbid(W, b)` call adds one unsafe disjunct (all rows of `W @ y + b >= 0`
holding at once); its rows must be single-output thresholds or zero-bias
differences. For anything richer, pass raw VNNLIB text instead.

## Tests

The unit tests take about 20 seconds and require no external models or data.

```bash
.venv/bin/python -m pytest tests/
```

The [vnncomp2025_benchmarks](https://github.com/VNN-COMP/vnncomp2025_benchmarks)
and [vnncomp2026_benchmarks](https://github.com/VNN-COMP/vnncomp2026_benchmarks)
repositories hold hundreds of ONNX networks and VNNLIB specs you can run as
tests. Clone one and point vibecheck at any instance:

```bash
git clone https://github.com/VNN-COMP/vnncomp2025_benchmarks.git
cd vnncomp2025_benchmarks
./setup.sh        # optional: downloads large models for benchmarks that need them

# extract row 0 (Id 0) of the acasxu instances.csv and run it
BENCH=benchmarks/acasxu_2023
IFS=, read -r ONNX VNNLIB TIMEOUT < "$BENCH/instances.csv"
vibecheck verify "$BENCH/$VNNLIB" --network N="$BENCH/$ONNX" --timeout "$TIMEOUT"   # -> unsat
```

The expected verdict for each instance is published in the
[VNN-COMP 2025 report](https://arxiv.org/abs/2512.19007) (Table 70); Id 0 is the
row run above.

## Versions

- **`vnncomp_2026`** (git tag): the exact code submitted to VNN-COMP 2026.
- **1.0.0**: the VNN-COMP 2026 engine with a reworked CLI (the VNN-LIB standard
  solver interface plus the legacy flat form) and pip packaging
  (`pip install vibecheck-nn`).
- **1.1.0** (this release): a clean-slate reimplementation of the verification
  core. Roughly
  95% as good as 1.0.0 on the VNN-COMP 2026 benchmarks (it solves about 97% as
  many instances, for about 92% of the score), with a much cleaner design in
  under a third of the code (about 18k lines versus 59k).

## Contributors

* [Stanley Bak](https://stanleybak.com) (lead)
* Doug Wehbe (testing)
