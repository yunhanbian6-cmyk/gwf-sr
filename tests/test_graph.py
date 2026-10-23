import pytest
import torch

from gwf_sr.graph import (
    PixelGraph,
    PixelGraphBatch,
    build_batch_pixel_graphs,
    build_pixel_graph,
    graph_degree,
    graph_spectral_radius_bound,
    graph_total_weight,
    normalized_adjacency_action,
)


def test_uniform_graph_node_and_edge_counts():
    graph = build_pixel_graph((3, 4))
    assert graph.node_count == 12
    assert graph.edge_count == 3 * (4 - 1) + 4 * (3 - 1)
    assert graph.source_index.shape == (2 * graph.edge_count,)
    assert graph.target_index.shape == (2 * graph.edge_count,)
    assert graph.edge_weight.shape == (2 * graph.edge_count,)
    assert graph.source_index.dtype == torch.int64
    assert graph.edge_weight.dtype == torch.float32
    assert torch.all(graph.edge_weight == 1.0)


def test_undirected_edges_appear_twice_in_incidence_table():
    graph = build_pixel_graph((3, 3))
    forward = torch.stack((graph.source_index, graph.target_index))
    reverse = torch.stack((graph.target_index, graph.source_index))
    forward_set = {tuple(pair.tolist()) for pair in forward.T}
    reverse_set = {tuple(pair.tolist()) for pair in reverse.T}
    assert forward_set == reverse_set
    assert len(forward_set) == 2 * graph.edge_count
    undirected_set = {tuple(sorted(pair)) for pair in forward_set}
    assert len(undirected_set) == graph.edge_count
    assert graph.edge_count == 3 * (3 - 1) + 3 * (3 - 1)


def test_graph_has_no_self_loops_or_diagonal_edges():
    graph = build_pixel_graph((4, 5))
    source = graph.source_index
    target = graph.target_index
    assert torch.all(source != target)
    row_difference = torch.abs(source // 5 - target // 5)
    col_difference = torch.abs(source % 5 - target % 5)
    assert torch.all((row_difference + col_difference) == 1)


def test_node_indexing_follows_raster_order():
    graph = build_pixel_graph((2, 3))
    horizontal_source = torch.tensor([0, 1, 3, 4])
    horizontal_target = torch.tensor([1, 2, 4, 5])
    vertical_start = 4
    vertical_source = torch.tensor([0, 1, 2])
    vertical_target = torch.tensor([3, 4, 5])
    assert torch.equal(graph.source_index[:4], horizontal_source)
    assert torch.equal(graph.target_index[:4], horizontal_target)
    assert torch.equal(graph.source_index[vertical_start : vertical_start + 3], vertical_source)
    assert torch.equal(graph.target_index[vertical_start : vertical_start + 3], vertical_target)


def test_invalid_edge_weight_mode_is_rejected():
    with pytest.raises(ValueError, match="Unknown edge_weight_mode"):
        build_pixel_graph((3, 3), edge_weight_mode="gradient")
    with pytest.raises(ValueError, match="Unknown edge_weight_mode"):
        build_pixel_graph((3, 3), edge_weight_mode="unknown")


@pytest.mark.parametrize("shape", [(0, 3), (3, 0), (-1, 3)])
def test_invalid_shape_is_rejected(shape):
    with pytest.raises(ValueError):
        build_pixel_graph(shape)


def test_degree_matches_boundary_and_corner_expectations():
    graph = build_pixel_graph((3, 3))
    degree = graph_degree(graph)
    assert degree.shape == (3, 3)
    assert degree[0, 0] == 2.0
    assert degree[0, 1] == 3.0
    assert degree[1, 1] == 4.0
    assert degree.sum() == 2.0 * graph.edge_count


def test_total_weight_counts_undirected_edges_once():
    graph = build_pixel_graph((3, 4))
    total = graph_total_weight(graph)
    assert total.ndim == 0
    assert total.item() == pytest.approx(float(graph.edge_count))


def test_spectral_radius_bound_is_one():
    graph = build_pixel_graph((4, 4))
    bound = graph_spectral_radius_bound(graph)
    assert bound.ndim == 0
    assert bound.item() == pytest.approx(1.0)


def test_normalized_adjacency_preserves_constant_fields():
    graph = build_pixel_graph((3, 3))
    values = torch.full((3, 3), 0.5)
    degree_flat = graph_degree(graph).reshape(-1)
    normalizer = torch.sqrt(degree_flat[graph.source_index] * degree_flat[graph.target_index])
    normalized_weight = graph.edge_weight / normalizer
    expected_flat = torch.zeros(graph.node_count, dtype=values.dtype)
    expected_flat.index_add_(0, graph.source_index, normalized_weight * 0.5)
    expected = expected_flat.reshape(3, 3)
    result = normalized_adjacency_action(values, graph)
    assert result.shape == values.shape
    assert result.dtype == values.dtype
    assert torch.allclose(result, expected, atol=1e-6)


def test_normalized_adjacency_action_is_linear():
    graph = build_pixel_graph((3, 3))
    first = torch.rand(3, 3, generator=torch.Generator().manual_seed(3))
    second = torch.rand(3, 3, generator=torch.Generator().manual_seed(4))
    combined = normalized_adjacency_action(first + second, graph)
    separate = normalized_adjacency_action(first, graph) + normalized_adjacency_action(second, graph)
    assert torch.allclose(combined, separate, atol=1e-6)


def test_batch_graphs_are_isolated_per_sample():
    images = torch.rand(3, 2, 3)
    batch = build_batch_pixel_graphs(images)
    assert isinstance(batch, PixelGraphBatch)
    assert batch.batch_size == 3
    assert batch.node_count == 6
    assert batch.edge_count == 2 * (3 - 1) + 3 * (2 - 1)
    assert batch.source_index.dtype == torch.int64
    assert batch.edge_weight.dtype == torch.float32

    offsets = batch.source_index // batch.node_count
    assert torch.equal(offsets, batch.batch_index)
    assert torch.all(batch.source_index % batch.node_count != batch.target_index % batch.node_count)


def test_batch_graph_diagnostics_have_batch_shape():
    images = torch.rand(3, 2, 3)
    batch = build_batch_pixel_graphs(images)
    degree = graph_degree(batch)
    assert degree.shape == (3, 2, 3)
    total = graph_total_weight(batch)
    assert total.shape == (3,)
    assert torch.allclose(total, torch.full((3,), float(batch.edge_count)))


def test_batch_normalized_adjacency_matches_per_sample_action():
    images = torch.rand(3, 2, 3)
    batch = build_batch_pixel_graphs(images)
    result = normalized_adjacency_action(images, batch)
    assert result.shape == images.shape
    for index in range(3):
        single = build_pixel_graph((2, 3))
        expected = normalized_adjacency_action(images[index], single)
        assert torch.allclose(result[index], expected, atol=1e-6)


def test_metrics_reject_illegal_dtype_and_shape_mismatch():
    graph = build_pixel_graph((2, 2))
    with pytest.raises(TypeError):
        normalized_adjacency_action(torch.ones(2, 2, dtype=torch.int64), graph)
    with pytest.raises(TypeError):
        normalized_adjacency_action(torch.ones(2, 2, dtype=torch.bool), graph)
    with pytest.raises(TypeError):
        normalized_adjacency_action(torch.ones(2, 2, dtype=torch.complex64), graph)
    with pytest.raises(TypeError):
        normalized_adjacency_action(torch.ones(2, 2, dtype=torch.float16), graph)
    with pytest.raises(ValueError):
        normalized_adjacency_action(torch.ones(3, 3), graph)


def test_metrics_allow_float32_float64_mismatch():
    graph = build_pixel_graph((2, 2))
    values64 = torch.rand(2, 2, dtype=torch.float64)
    result64 = normalized_adjacency_action(values64, graph)
    assert result64.dtype == torch.float64


def test_graph_outputs_are_deterministic():
    graph = build_pixel_graph((4, 4))
    values = torch.rand(4, 4, generator=torch.Generator().manual_seed(5))
    assert torch.equal(
        normalized_adjacency_action(values, graph),
        normalized_adjacency_action(values, graph),
    )


def test_graph_preserves_dtype_and_device():
    for dtype in (torch.float32, torch.float64):
        images = torch.rand(2, 3, 3, dtype=dtype)
        batch = build_batch_pixel_graphs(images)
        assert batch.dtype == dtype
        assert batch.device == images.device
        assert batch.edge_weight.dtype == dtype
