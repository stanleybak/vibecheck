"""ACAS Xu integration tests using vnncomp benchmarks."""

from vibecheck import ComputeGraph, parse_vnnlib, zonotope_verify


ACASXU_NET = "acasxu_2023/onnx/ACASXU_run2a_1_1_batch_2000.onnx.gz"


def test_acasxu_1_1_loads(vnncomp_benchmarks):
    """ACAS Xu net 1_1 loads correctly: sequential FC, 5 inputs."""
    net = str(vnncomp_benchmarks / ACASXU_NET)
    g = ComputeGraph.from_onnx(net)

    assert g.input_shape == (5,)
    assert len(g.relu_nodes()) == 6
    assert len(g.fork_points()) == 0  # sequential


def test_acasxu_1_1_prop2(vnncomp_benchmarks):
    """ACAS Xu net 1_1, property 2: zonotope analysis runs end-to-end."""
    net = str(vnncomp_benchmarks / ACASXU_NET)
    spec = str(vnncomp_benchmarks / "acasxu_2023/vnnlib/prop_2.vnnlib.gz")

    graph = ComputeGraph.from_onnx(net)
    x_lo, x_hi, pred_label, competitors = parse_vnnlib(spec)

    assert len(x_lo) == 5
    assert len(competitors) > 0

    result, details = zonotope_verify(graph, x_lo, x_hi, pred_label, competitors)

    assert result in ('verified', 'unknown')
    assert 'worst_margin' in details
    assert len(details['margins']) == len(competitors)

    print(f"\nResult: {result}")
    print(f"Worst margin: {details['worst_margin']:.6f}")
    for comp, margin in details['margins'].items():
        print(f"  Class {pred_label} vs {comp}: {margin:.6f}")
