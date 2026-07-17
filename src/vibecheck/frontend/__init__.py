"""vc2 front end: ONNX/VNNLIB loading, spec objects, witness validation,
counterexample emission, and the instance transpilers (network-pair merge,
nonlinear-v2 augment). Ported from v1 as part of making vibecheck
standalone; each module notes its origin and what was pruned (audited by
a 23-family coverage battery -- see the standalone-refactor commit)."""
