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

vibecheck depends on PyTorch, whose default wheel is a large CUDA build (roughly
4 GB). On a CPU-only machine, install the CPU build of torch first, then vibecheck:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install vibecheck-nn
```

The default `graph` mode uses Gurobi (`gurobipy`, installed automatically) for its
LP/MILP tightening. The bundled license is size-limited but fine for small models
and the examples; verifying large networks needs a full
[Gurobi license](https://www.gurobi.com/) (free for academics).

### From source (development)

```bash
git clone https://github.com/stanleybak/vibecheck-nn
cd vibecheck-nn
uv venv --python 3.12 .venv                        # https://docs.astral.sh/uv/
VIRTUAL_ENV=$PWD/.venv uv pip install -e ".[dev]"
```

Then use `.venv/bin/vibecheck` for the CLI and `.venv/bin/python` for the tests.

## Usage

vibecheck implements the VNN-LIB standard's solver CLI:

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

Example, on the bundled ACAS-Xu network with a property that holds. The example
files ship with the package; `vibecheck --examples-dir` prints their location, so
`cd` there first to run with short filenames. Everything except the final `unsat`
is progress on stderr (trimmed here):

```console
$ cd "$(vibecheck --examples-dir)"
$ vibecheck verify prop_1.vnnlib --network N=ACASXU_run2a_2_2_batch_2000.onnx --timeout 60
Auto-config: acasxu_2023.yaml | rule 5: low input-dim (<=20) FC (input-split) | Low-dimensional ReLU FC (ACAS-Xu family): batched input-split BaB, hybrid ACAS-Xu path off.
Loading network: ACASXU_run2a_2_2_batch_2000.onnx
  22 ops, 6 ReLU layers, 0 fork points, input shape: (1, 1, 1, 5)
Loading spec: prop_1.vnnlib
  1 constraint(s), 1 disjunct(s)
Running graph verification (device=gpu, impl=optimized, profile=auto:acasxu_2023.yaml(rule 5), timeout=60.0s)...
[pgd] no CE: restarts=100 iters=100/100 gap(best_margin)=+4.012e+00 elapsed=0.29s
[branch] iter=0 split X_1 (width=1.0000) leaves=1
...
[branch] iter=7 split X_2 (width=0.1250) leaves=17

Result: verified
  Time: 2.24s
unsat
```

And a violated property, where stdout is the verdict plus the satisfying assignment:

```console
$ vibecheck verify prop_2.vnnlib --network N=ACASXU_run2a_2_2_batch_2000.onnx --timeout 60  2>/dev/null
sat
X float32 [1,1,1,5]
0.6208617091178894
-0.01862180233001709
-0.024315983057022095
0.47021740674972534
-0.46439468860626221
Y float32 [1,5]
0.023801768198609352
-0.021306384354829788
0.023232858628034592
-0.016359567642211914
0.023021360859274864
```

The `Auto-config:` line shows config selection: with no `--config`, vibecheck
auto-selects a bundled per-benchmark config from the structure of the network and
spec (input dim, conv/transformer/nonlinear ops, network-pair kind) and logs which
rule fired. To override, pass your own YAML with `--config /path/to/config.yaml`
(its keys map 1:1 to the tool's settings).

The legacy flat CLI (`vibecheck --net model.onnx --spec property.vnnlib
--results-file out.txt`, the form the VNNCOMP harness drives) is unchanged; see
`vibecheck --help`.

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
[ 0.62086171 -0.0186218  -0.02431598  0.47021741 -0.46439469]
[ 0.02380177 -0.02130638  0.02323286 -0.01635957  0.02302136]
```

`VerifyResult` also carries `.details` (the verifier's verbose object) and `.elapsed`.

## Tests

```bash
# Unit tests: no external data, ~1-2 min
.venv/bin/python -m pytest tests/ -k "not vnncomp" -m "not integration"

# Per-benchmark verdict regressions (need a local benchmark clone; see below)
.venv/bin/python -m pytest tests/integration -m integration
```

The [vnncomp2025_benchmarks](https://github.com/VNN-COMP/vnncomp2025_benchmarks)
and [vnncomp2026_benchmarks](https://github.com/VNN-COMP/vnncomp2026_benchmarks)
repositories hold hundreds of ONNX networks and VNNLIB specs you can run. Clone
one, run its `setup.sh` to download the models, and point vibecheck at any instance:

```bash
git clone https://github.com/VNN-COMP/vnncomp2025_benchmarks.git
cd vnncomp2025_benchmarks
./setup.sh        # downloads and unpacks the per-benchmark onnx/vnnlib

# a quick ACAS-Xu sat case
BENCH=benchmarks/acasxu_2023
vibecheck verify "$BENCH/vnnlib/prop_2.vnnlib" \
    --network N="$BENCH/onnx/ACASXU_run2a_1_2_batch_2000.onnx" --timeout 60   # -> sat
```

The integration tests run such instances as regressions; point them at your clone
by copying `tests/paths.yaml.template` to `tests/paths.yaml` and setting
`vnncomp_benchmarks:` to the clone root.

## Contributors

* [Stanley Bak](https://stanleybak.com) (lead)
* Doug Wehbe (testing)
