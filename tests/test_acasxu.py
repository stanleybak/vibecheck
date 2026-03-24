"""ACAS Xu integration tests using vnncomp benchmarks."""

from vibecheck import ComputeGraph, load_vnnlib, zonotope_verify


ACASXU_NET = "acasxu_2023/onnx/ACASXU_run2a_1_1_batch_2000.onnx.gz"


def test_acasxu_1_1_loads(vnncomp_benchmarks):
    """ACAS Xu net 1_1 loads correctly: sequential FC, 5 inputs."""
    net = str(vnncomp_benchmarks / ACASXU_NET)
    g = ComputeGraph.from_onnx(net)

    assert g.input_shape == (1, 1, 1, 5)
    assert len(g.relu_nodes()) == 6
    assert len(g.fork_points()) == 0  # sequential


def test_acasxu_1_1_prop2(vnncomp_benchmarks):
    """ACAS Xu net 1_1, property 2: zonotope analysis runs end-to-end."""
    net = str(vnncomp_benchmarks / ACASXU_NET)
    spec_path = str(vnncomp_benchmarks / "acasxu_2023/vnnlib/prop_2.vnnlib.gz")

    graph = ComputeGraph.from_onnx(net)
    spec = load_vnnlib(spec_path)

    assert len(spec.x_lo) == 5
    assert spec.n_constraints > 0

    result, details = zonotope_verify(graph, spec)

    assert result in ('verified', 'unknown')
    assert 'worst_margin' in details
