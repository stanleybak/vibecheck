"""VNNLIB 2.0 counterexample validation (vibecheck implementation).

Decides whether a SAT witness would be *accepted* by the VNN-COMP 2026 scorer,
by replaying the witness input(s) through the ORIGINAL network(s) on onnxruntime
CPU — the scorer's engine — and checking the spec's assertions with the scorer's
tolerance rule. The solver's claimed output ``Y`` is IGNORED; the ORT-recomputed
``Y`` is authoritative. Replaying the *original* graph (never a merged/augmented
one) is the whole point: it catches the float32/numerical edge cases the scorer
would catch and a solver working in different arithmetic would miss.

Verdicts (``CexResult``):
  CORRECT                  assertions hold at zero tolerance.
  CORRECT_UP_TO_TOLERANCE  hold only once the INPUT box is relaxed by ``abs_tol``
                           (output comparisons stay STRICT at zero tolerance).
  SPEC_NOT_VIOLATED        witness replays, but the spec is not violated.
  NO_CE                    the counterexample file is missing.
  MALFORMED_CE             the assignment text is not well-formed for the spec.
  UNSUPPORTED              the spec/model uses a shape vibecheck can't replay.
``ACCEPTED_RESULTS`` (CORRECT + CORRECT_UP_TO_TOLERANCE) are the verdicts the
scorer awards the instance (no penalty).

Single- and multi-network specs are supported. Pass ``onnx`` as a single path
(str) for a single-network spec, or as ``[(net_name, path), ...]`` for a pair
spec: isomorphic (``g`` isometric-to ``f`` — two distinct nets) or monotonic
(``g`` equal-to ``f`` — one net, replayed for both).

Independent vibecheck reimplementation — NOT vendored from the competition code.
Accept/reject verdicts are pinned bit-identical to the scorer by golden-fixture
tests (``tests/test_cex_check_v2_golden.py``) and the equivalence tests
(``tests/test_competition_cex_equiv.py``).
"""

import gzip
import math
import os
import re

import numpy as np
import onnxruntime as ort
import vnnlib


# --------------------------------------------------------------------------- #
# Result classification (string values identical to the scorer's
# CounterexampleResult enum, so downstream comparisons stay stable).
# --------------------------------------------------------------------------- #
class CexResult:
    CORRECT = "correct"
    CORRECT_UP_TO_TOLERANCE = "correct_up_to_tolerance"
    NO_CE = "no_ce"
    SPEC_NOT_VIOLATED = "spec_not_violated"
    MALFORMED_CE = "malformed_ce"
    UNSUPPORTED = "unsupported"


# Accepted == the scorer awards the instance: a CORRECT witness, or a
# CORRECT_UP_TO_TOLERANCE one (input within abs_tol of the box, output strict).
ACCEPTED_RESULTS = frozenset({CexResult.CORRECT, CexResult.CORRECT_UP_TO_TOLERANCE})


class _MalformedCE(Exception):
    """The assignment text does not match the spec's declared variables."""


class _Unsupported(Exception):
    """The spec/model uses a construct this validator does not replay."""


# --------------------------------------------------------------------------- #
# Type tables (dictated by the vnnlib DType names and onnxruntime type strings).
# --------------------------------------------------------------------------- #
_NUMPY_DTYPES = {
    "F16": np.float16, "F32": np.float32, "F64": np.float64,
    "I8": np.int8, "I16": np.int16, "I32": np.int32, "I64": np.int64,
    "U8": np.uint8, "U16": np.uint16, "U32": np.uint32, "U64": np.uint64,
    "Bool": np.bool_, "Real": np.float64, "Unknown": np.float64,
}
_VNNLIB_TYPE_NAMES = {
    "F16": "float16", "F32": "float32", "F64": "float64",
    "I8": "int8", "I16": "int16", "I32": "int32", "I64": "int64",
    "U8": "uint8", "U16": "uint16", "U32": "uint32", "U64": "uint64",
    "Bool": "bool", "Real": "real", "Unknown": "real",
}
_FLOAT_DTYPE_KEYS = {"F16", "F32", "F64", "Real", "Unknown"}
_ONNX_RUNTIME_DTYPES = {
    "tensor(float16)": np.float16, "tensor(float)": np.float32,
    "tensor(double)": np.float64, "tensor(int8)": np.int8,
    "tensor(int16)": np.int16, "tensor(int32)": np.int32,
    "tensor(int64)": np.int64, "tensor(uint8)": np.uint8,
    "tensor(uint16)": np.uint16, "tensor(uint32)": np.uint32,
    "tensor(uint64)": np.uint64, "tensor(bool)": np.bool_,
}
_VNNLIB_TO_ONNX_TYPES = {
    "F16": "tensor(float16)", "F32": "tensor(float)", "F64": "tensor(double)",
    "I8": "tensor(int8)", "I16": "tensor(int16)", "I32": "tensor(int32)",
    "I64": "tensor(int64)", "U8": "tensor(uint8)", "U16": "tensor(uint16)",
    "U32": "tensor(uint32)", "U64": "tensor(uint64)", "Bool": "tensor(bool)",
}

# onnxruntime raises these for a model/replay problem; treat them as UNSUPPORTED
# (the witness can't be validated here) rather than crashing the verifier.
_ORT_ERRORS = tuple(
    getattr(ort.capi.onnxruntime_pybind11_state, name)
    for name in ("Fail", "InvalidArgument", "InvalidGraph", "InvalidProtobuf",
                 "NoSuchFile", "NotImplemented", "RuntimeException")
    if hasattr(ort.capi.onnxruntime_pybind11_state, name)
)


# --------------------------------------------------------------------------- #
# IO helpers.
# --------------------------------------------------------------------------- #
def _read_text(path):
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return stream.read()
    with open(path, "r", encoding="utf-8") as stream:
        return stream.read()


def _session(model_path):
    if str(model_path).endswith(".gz"):
        with gzip.open(model_path, "rb") as stream:
            return ort.InferenceSession(stream.read(), providers=["CPUExecutionProvider"])
    return ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])


# --------------------------------------------------------------------------- #
# Assignment (counterexample) parsing — VNNLIB 2.0 section 5.3 textual form.
# --------------------------------------------------------------------------- #
_ASSIGNMENT_HEADER = re.compile(r"^(\S+)\s+(\S+)\s+\[([0-9,\s]*)\]$")


def _definitions(query):
    """Every variable the assignment must supply, in declaration order:
    each network's inputs then outputs (hidden variables are not supported)."""
    return tuple(d for network in query.networks
                 for d in (*network.inputs, *network.outputs))


def _parse_scalar(text, dtype):
    if dtype == np.bool_:
        low = text.lower()
        if low not in ("true", "false", "0", "1"):
            raise ValueError(f"invalid boolean value {text!r}")
        return low in ("true", "1")
    if np.issubdtype(dtype, np.integer):
        return int(text)
    return float(text)


def _type_name_matches(dtype_key, type_name):
    low = type_name.lower()
    if low == _VNNLIB_TYPE_NAMES[dtype_key].lower():
        return True
    return dtype_key in _FLOAT_DTYPE_KEYS and low == "real"


def parse_text_assignment(content, query):
    """Parse the textual assignment into ``{var_name: np.ndarray}``.

    Raises ``_MalformedCE`` if the text does not match the spec's declared
    variables (wrong order/name/shape/type/count), or ``_Unsupported`` for a
    declared dtype this validator has no numpy mapping for.
    """
    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    if lines and lines[0] == "sat":       # optional leading 'sat' verdict line
        lines.pop(0)

    assignment = {}
    pos = 0
    for definition in _definitions(query):
        if pos >= len(lines):
            raise _MalformedCE(f"missing assignment for variable {definition.name}")
        header = _ASSIGNMENT_HEADER.fullmatch(lines[pos])
        if not header:
            raise _MalformedCE(f"invalid assignment header: {lines[pos]!r}")
        pos += 1

        name, type_name, dims = header.groups()
        if name != definition.name:
            raise _MalformedCE(f"expected variable {definition.name}, found {name}")
        shape = [] if not dims.strip() else [int(v.strip()) for v in dims.split(",")]
        if shape != list(definition.shape):
            raise _MalformedCE(f"variable {name} has shape {shape}, "
                               f"expected {list(definition.shape)}")

        dtype_key = definition.dtype.name
        if not _type_name_matches(dtype_key, type_name):
            raise _MalformedCE(f"variable {name} has type {type_name}, "
                               f"expected {_VNNLIB_TYPE_NAMES[dtype_key]}")

        count = math.prod(shape)
        if pos + count > len(lines):
            raise _MalformedCE(f"not enough values for variable {name}")
        dtype = _NUMPY_DTYPES[dtype_key]
        try:
            values = [_parse_scalar(v, dtype) for v in lines[pos:pos + count]]
            assignment[name] = np.asarray(values, dtype=dtype).reshape(shape)
        except (OverflowError, TypeError, ValueError) as error:
            raise _MalformedCE(f"invalid value for variable {name}: {error}") from error
        pos += count

    if pos != len(lines):
        raise _MalformedCE(f"unexpected content after assignments: {lines[pos]!r}")
    return assignment


# --------------------------------------------------------------------------- #
# Network -> ONNX path resolution and ORT replay.
# --------------------------------------------------------------------------- #
def _resolve_paths(query, onnx):
    """Map each declared network to an ORT model path.

    ``onnx`` is a single path (single-network spec) or ``[(net_name, path), ...]``
    (multi-network spec). A network declared ``equal-to`` another reuses that
    network's path (e.g. monotonic ``g`` equal-to ``f``); every other network
    needs its own explicit path (e.g. isomorphic ``f``/``g``).
    """
    if isinstance(onnx, str) or hasattr(onnx, "__fspath__"):
        implemented = [nw for nw in query.networks if not nw.equal_to]
        if len(implemented) != 1:
            raise _Unsupported(
                f"a single ONNX path needs exactly one implemented network, "
                f"got {len(implemented)}")
        explicit = {implemented[0].name: str(onnx)}
    else:
        explicit = {}
        for name, path in onnx:
            explicit[str(name)] = str(path)

    paths = {}
    for network in query.networks:
        if network.equal_to:
            # The vnnlib parser guarantees an equal-to target is a network
            # declared earlier, so its path is already resolved here.
            paths[network.name] = paths[network.equal_to]
        elif network.name in explicit:
            paths[network.name] = explicit[network.name]
        else:
            raise _Unsupported(f"no ONNX model provided for network {network.name}")
    return paths


def _reshape_for_onnx(value, onnx_shape, name):
    if len(value.shape) == len(onnx_shape) and all(
            not isinstance(d, int) or d <= 0 or v == d
            for v, d in zip(value.shape, onnx_shape)):
        return value
    if all(isinstance(d, int) and d > 0 for d in onnx_shape) \
            and value.size == math.prod(onnx_shape):
        return value.reshape(onnx_shape)
    raise _Unsupported(f"cannot reshape variable {name} from {value.shape} "
                       f"to ONNX shape {onnx_shape}")


def _match_positional(definitions, onnx_values, network_name, kind):
    # The vnnlib 2.0 grammar has no onnx-name annotation, so spec variables bind
    # to ONNX inputs/outputs positionally, in declaration order.
    if len(onnx_values) != len(definitions):
        raise _Unsupported(f"network {network_name} declares {len(definitions)} "
                           f"{kind}s, but ONNX has {len(onnx_values)}")
    return list(zip(definitions, onnx_values))


def _check_element_type(definition, onnx_value, network_name):
    key = definition.dtype.name
    if key in ("Real", "Unknown"):        # untyped in the spec: accept ONNX's type
        return
    expected = _VNNLIB_TO_ONNX_TYPES[key]
    if onnx_value.type != expected:
        raise _Unsupported(f"network {network_name} declares {definition.name} as "
                           f"{_VNNLIB_TYPE_NAMES[key]}, but ONNX {onnx_value.name} "
                           f"has type {onnx_value.type}")


def _run_networks(query, paths, assignment):
    """Replay every declared network on ORT-CPU; return ``{output_var: ndarray}``."""
    outputs = {}
    for network in query.networks:
        session = _session(paths[network.name])
        input_matches = _match_positional(network.inputs, session.get_inputs(),
                                          network.name, "input")
        output_matches = _match_positional(network.outputs, session.get_outputs(),
                                           network.name, "output")
        feeds = {}
        for definition, onnx_input in input_matches:
            _check_element_type(definition, onnx_input, network.name)
            if onnx_input.type not in _ONNX_RUNTIME_DTYPES:
                raise _Unsupported(f"unsupported ONNX input type {onnx_input.type} "
                                   f"for {onnx_input.name}")
            value = assignment[definition.name].astype(
                _ONNX_RUNTIME_DTYPES[onnx_input.type], copy=False)
            feeds[onnx_input.name] = _reshape_for_onnx(
                value, onnx_input.shape, definition.name)
        for definition, onnx_output in output_matches:
            _check_element_type(definition, onnx_output, network.name)

        output_names = [ov.name for _, ov in output_matches]
        results = session.run(output_names, feeds)
        for (definition, _), result in zip(output_matches, results):
            result = np.asarray(result)
            if result.shape != tuple(definition.shape):
                if result.size != math.prod(definition.shape):
                    raise _Unsupported(f"cannot reshape ONNX output for "
                                       f"{definition.name} from {result.shape} "
                                       f"to {definition.shape}")
                result = result.reshape(definition.shape)
            outputs[definition.name] = result
    return outputs


# --------------------------------------------------------------------------- #
# Assertion evaluation.
#
# The scorer walks the vnnlib boolean/arithmetic AST per assertion. That is
# ~24 s on a 1.27M-input-bound spec, almost all of which are simple box bounds
# `(>= X[i] lb)` / `(<= X[i] ub)`. We recognize exactly that shape and batch it
# into one numpy comparison, falling back to the AST walk for any output/mixed/
# complex assertion. The comparison + tolerance rule is identical either way:
#   `>` : v > rhs - tol      `>=`: v >= rhs - tol
#   `<` : v < rhs + tol      `<=`: v <= rhs + tol
# so the accept/reject verdict is identical to the scorer, only faster.
# --------------------------------------------------------------------------- #
def _eval_arithmetic(expr, assignment):
    kind = type(expr).__name__
    if kind == "Var":
        return assignment[expr.name][tuple(expr.indices)]
    if kind in ("Float", "Int", "IntExpr"):
        return expr.value
    if kind == "Literal":
        return float(expr.lexeme)
    if kind == "Negate":
        return -_eval_arithmetic(expr.expr, assignment)
    if kind == "Plus":
        return sum(_eval_arithmetic(a, assignment) for a in expr.args)
    if kind == "Minus":
        head = _eval_arithmetic(expr.head, assignment)
        return head - sum(_eval_arithmetic(a, assignment) for a in expr.rest)
    if kind == "Multiply":
        return math.prod(_eval_arithmetic(a, assignment) for a in expr.args)
    raise _Unsupported(f"unsupported arithmetic expression {kind}")


def _eval_boolean(expr, assignment, tol):
    kind = type(expr).__name__
    if kind == "And":
        return all(_eval_boolean(a, assignment, tol) for a in expr.args)
    if kind == "Or":
        return any(_eval_boolean(a, assignment, tol) for a in expr.args)
    lhs = _eval_arithmetic(expr.lhs, assignment)
    rhs = _eval_arithmetic(expr.rhs, assignment)
    if kind == "GreaterThan":
        return lhs > rhs - tol
    if kind == "LessThan":
        return lhs < rhs + tol
    if kind == "GreaterEqual":
        return lhs >= rhs - tol
    if kind == "LessEqual":
        return lhs <= rhs + tol
    if kind == "Equal":                    # `(== ...)` — e.g. monotonic input pins
        return abs(lhs - rhs) <= tol
    if kind == "NotEqual":                 # `(!= ...)`
        return abs(lhs - rhs) > tol
    raise _Unsupported(f"unsupported boolean expression {kind}")


def _expression_variables(expr):
    if type(expr).__name__ == "Var":
        return {expr.name}
    variables = set()
    for attr in ("expr", "lhs", "rhs", "head"):
        if hasattr(expr, attr):
            variables |= _expression_variables(getattr(expr, attr))
    for attr in ("args", "rest"):
        if hasattr(expr, attr):
            for child in getattr(expr, attr):
                variables |= _expression_variables(child)
    return variables


def _input_names(query):
    return {d.name for network in query.networks for d in network.inputs}


_CMP_FLIP = {"GreaterThan": "LessThan", "LessThan": "GreaterThan",
             "GreaterEqual": "LessEqual", "LessEqual": "GreaterEqual"}


def _const(node):
    kind = type(node).__name__
    if kind in ("Float", "Int", "IntExpr"):
        return float(node.value)
    if kind == "Literal":
        return float(node.lexeme)
    if kind == "Negate":
        inner = _const(node.expr)
        return None if inner is None else -inner
    return None


def _box_atoms(expr):
    """If ``expr`` is ``Var <cmp> const`` (or an ``and`` of such), return a list of
    ``(var_name, indices, cmp_kind, rhs)``; else ``None`` (not a pure box)."""
    kind = type(expr).__name__
    if kind == "And":
        out = []
        for arg in expr.args:
            sub = _box_atoms(arg)
            if sub is None:
                return None
            out.extend(sub)
        return out
    if kind in _CMP_FLIP:
        lhs, rhs = expr.lhs, expr.rhs
        if type(lhs).__name__ == "Var":
            c = _const(rhs)
            return None if c is None else [(lhs.name, tuple(lhs.indices), kind, c)]
        if type(rhs).__name__ == "Var":
            c = _const(lhs)
            return None if c is None else [(rhs.name, tuple(rhs.indices), _CMP_FLIP[kind], c)]
    return None


def _c_strides(shape):
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return strides


# Assertion partition is assignment-independent, so cache it by query identity
# (one entry; cleared on a new query so a sweep never leaks).
_PREP_CACHE = {}


def _prepare_assertions(query):
    """Partition assertions once into (a) batched pure-input box atoms as numpy
    arrays keyed by ``(var, cmp_kind)`` and (b) a fallback list of
    output/mixed/complex assertions ``(expr, is_pure_input)``."""
    inputs = _input_names(query)
    shapes = {d.name: tuple(d.shape)
              for network in query.networks for d in network.inputs}
    raw = {}
    other = []
    for assertion in query.assertions:
        atoms = _box_atoms(assertion.expr)
        if atoms is not None and all(nm in inputs for nm, _, _, _ in atoms):
            for nm, idx, cmp_kind, rhs in atoms:
                raw.setdefault((nm, cmp_kind), []).append((idx, rhs))
            continue
        variables = _expression_variables(assertion.expr)
        other.append((assertion.expr, bool(variables) and variables <= inputs))

    box = {}
    for (nm, cmp_kind), entries in raw.items():
        strides = np.asarray(_c_strides(shapes[nm]), dtype=np.int64)
        flat = np.array([e[0] for e in entries], dtype=np.int64) @ strides
        rhs = np.array([e[1] for e in entries], dtype=np.float64)
        box[(nm, cmp_kind)] = (flat, rhs)
    return box, other


def _get_prepared(query):
    entry = _PREP_CACHE.get(id(query))
    if entry is not None and entry[0] is query:
        return entry[1]
    prepared = _prepare_assertions(query)
    _PREP_CACHE.clear()
    _PREP_CACHE[id(query)] = (query, prepared)
    return prepared


def _assertions_hold(query, assignment, input_tol, output_tol=0.0):
    """True iff every assertion holds. Box atoms are pure-input (``input_tol``);
    a fallback atom uses ``input_tol`` when all its variables are inputs, else
    ``output_tol``."""
    box, other = _get_prepared(query)
    for (nm, cmp_kind), (flat, rhs) in box.items():
        vals = np.asarray(assignment[nm]).ravel()[flat].astype(np.float64)
        if cmp_kind == "GreaterThan":
            ok = vals > rhs - input_tol
        elif cmp_kind == "LessThan":
            ok = vals < rhs + input_tol
        elif cmp_kind == "GreaterEqual":
            ok = vals >= rhs - input_tol
        else:                              # LessEqual
            ok = vals <= rhs + input_tol
        if not ok.all():
            return False
    for expr, is_input in other:
        if not _eval_boolean(expr, assignment, input_tol if is_input else output_tol):
            return False
    return True


# --------------------------------------------------------------------------- #
# Public entry point.
# --------------------------------------------------------------------------- #
def _classify(query, onnx, ce_path, abs_tol):
    if not os.path.exists(ce_path):
        return CexResult.NO_CE, f"counterexample file not found: {ce_path}"
    try:
        query_obj = query
        assignment = parse_text_assignment(_read_text(ce_path), query_obj)
        paths = _resolve_paths(query_obj, onnx)
        computed = _run_networks(query_obj, paths, assignment)
    except _MalformedCE as error:
        return CexResult.MALFORMED_CE, str(error)
    except (_Unsupported, vnnlib.VNNLibException, *_ORT_ERRORS) as error:
        return CexResult.UNSUPPORTED, str(error)

    # The scorer ignores the solver's Y and uses the ORT-recomputed outputs.
    evaluated = dict(assignment)
    evaluated.update(computed)

    if _assertions_hold(query_obj, evaluated, 0.0, 0.0):
        return CexResult.CORRECT, "assertions hold at zero tolerance"
    if _assertions_hold(query_obj, evaluated, abs_tol, 0.0):
        return (CexResult.CORRECT_UP_TO_TOLERANCE,
                f"assertions hold within input tolerance {abs_tol} (output strict)")
    return CexResult.SPEC_NOT_VIOLATED, "witness does not violate the specification"


def validate_cex_v2(onnx, vnnlib_path, ce_path, abs_tol=1e-4, rel_tol=0.0):
    """Validate a VNNLIB-2.0 counterexample file the way the VNN-COMP 2026 scorer
    does. Returns ``(result_str, message)``; ``result_str`` is one of
    ``CexResult.*`` and lies in ``ACCEPTED_RESULTS`` iff the scorer would accept.

    ``onnx``: a single ONNX path (single-network spec), or ``[(net_name, path),
    ...]`` for a multi-network (pair) spec. ``rel_tol`` is accepted for
    signature compatibility with the scorer and is unused (the scorer's accept
    decision does not compare solver outputs, which we ignore).
    """
    del rel_tol
    try:
        query = vnnlib.parse_query_string(_read_text(vnnlib_path))
    except (FileNotFoundError, vnnlib.VNNLibException, OSError) as error:
        return CexResult.UNSUPPORTED, f"could not parse specification: {error}"
    return _classify(query, onnx, ce_path, abs_tol)
