"""PyTorch implementation of typed graph networks used by GraphCast."""

from __future__ import annotations

from typing import Any, Callable, Mapping, Optional, Union

import torch

from graphcast import typed_graph

TensorTree = Any


def _tree_map(fn: Callable[[torch.Tensor], torch.Tensor], tree: TensorTree) -> TensorTree:
  """Applies ``fn`` to every tensor in ``tree`` preserving structure."""
  if tree is None:
    return None
  if isinstance(tree, torch.Tensor):
    return fn(tree)
  if isinstance(tree, (list, tuple)):
    return type(tree)(_tree_map(fn, x) for x in tree)
  if isinstance(tree, dict):
    return {k: _tree_map(fn, v) for k, v in tree.items()}
  if tree == ():  # treat empty features as immutable sentinel
    return tree
  raise TypeError(f"Unsupported feature container type: {type(tree)!r}")


def _first_tensor(tree: TensorTree) -> Optional[torch.Tensor]:
  if isinstance(tree, torch.Tensor):
    return tree
  if isinstance(tree, (list, tuple)):
    for x in tree:
      result = _first_tensor(x)
      if result is not None:
        return result
  if isinstance(tree, dict):
    for x in tree.values():
      result = _first_tensor(x)
      if result is not None:
        return result
  return None


def _as_tensor(array_like, *, dtype=None, device=None) -> torch.Tensor:
  if isinstance(array_like, torch.Tensor):
    if device is not None and array_like.device != device:
      array_like = array_like.to(device)
    if dtype is not None and array_like.dtype != dtype:
      array_like = array_like.to(dtype=dtype)
    return array_like
  return torch.as_tensor(array_like, dtype=dtype, device=device)


def segment_sum(data: torch.Tensor,
                segment_ids: torch.Tensor,
                num_segments: int) -> torch.Tensor:
  if data.numel() == 0:
    shape = (num_segments,) + tuple(data.shape[1:])
    return torch.zeros(shape, dtype=data.dtype, device=data.device)
  out_shape = (num_segments,) + tuple(data.shape[1:])
  result = torch.zeros(out_shape, dtype=data.dtype, device=data.device)
  scatter_index = segment_ids.view(-1, *([1] * (data.dim() - 1)))
  scatter_index = scatter_index.expand_as(data)
  result.scatter_add_(0, scatter_index, data)
  return result


def segment_mean(data: torch.Tensor,
                 segment_ids: torch.Tensor,
                 num_segments: int) -> torch.Tensor:
  summed = segment_sum(data, segment_ids, num_segments)
  counts = torch.bincount(segment_ids, minlength=num_segments)
  counts = counts.clamp_min(1).to(dtype=data.dtype, device=data.device)
  counts = counts.view(-1, *([1] * (data.dim() - 1)))
  return summed / counts


def _repeat_global(features: TensorTree,
                   counts: torch.Tensor) -> TensorTree:
  if features is None or features == ():
    return features

  def _repeat(t: torch.Tensor) -> torch.Tensor:
    return torch.repeat_interleave(t, counts, dim=0)

  return _tree_map(_repeat, features)


NodeFn = Callable[[TensorTree, Mapping[str, TensorTree], Mapping[str, TensorTree], TensorTree], TensorTree]
GlobalFn = Callable[[Mapping[str, TensorTree], Mapping[str, TensorTree], TensorTree], TensorTree]


class GraphNetworkCallable:
  """Callable wrapper implementing a typed graph network."""

  def __init__(
      self,
      update_edge_fn: Mapping[str, Callable[[TensorTree, TensorTree, TensorTree, TensorTree], TensorTree]],
      update_node_fn: Mapping[str, NodeFn],
      update_global_fn: Optional[GlobalFn],
      aggregate_edges_for_nodes_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
      aggregate_nodes_for_globals_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
      aggregate_edges_for_globals_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
  ) -> None:
    self._update_edge_fn = dict(update_edge_fn)
    self._update_node_fn = dict(update_node_fn)
    self._update_global_fn = update_global_fn
    self._aggregate_edges_for_nodes_fn = aggregate_edges_for_nodes_fn
    self._aggregate_nodes_for_globals_fn = aggregate_nodes_for_globals_fn
    self._aggregate_edges_for_globals_fn = aggregate_edges_for_globals_fn

  def __call__(self, graph: typed_graph.TypedGraph) -> typed_graph.TypedGraph:
    updated_graph = graph

    updated_edges = dict(updated_graph.edges)
    for edge_set_name, edge_fn in self._update_edge_fn.items():
      edge_set_key = graph.edge_key_by_name(edge_set_name)
      updated_edges[edge_set_key] = _edge_update(
          updated_graph, edge_fn, edge_set_key)
    updated_graph = updated_graph._replace(edges=updated_edges)

    updated_nodes = dict(updated_graph.nodes)
    for node_set_key, node_fn in self._update_node_fn.items():
      updated_nodes[node_set_key] = _node_update(
          updated_graph, node_fn, node_set_key, self._aggregate_edges_for_nodes_fn)
    updated_graph = updated_graph._replace(nodes=updated_nodes)

    if self._update_global_fn:
      updated_context = _global_update(
          updated_graph,
          self._update_global_fn,
          self._aggregate_edges_for_globals_fn,
          self._aggregate_nodes_for_globals_fn)
      updated_graph = updated_graph._replace(context=updated_context)

    return updated_graph


class _EdgeFnNoGlobal:
  def __init__(self, fn: Callable[[TensorTree, TensorTree, TensorTree], TensorTree]) -> None:
    self._fn = fn

  def __call__(self, edge_features, sender_features, receiver_features, _global_features):
    return self._fn(edge_features, sender_features, receiver_features)


class _NodeFnWithSent:
  def __init__(self, fn: Callable[[TensorTree, Mapping[str, TensorTree], Mapping[str, TensorTree]], TensorTree]) -> None:
    self._fn = fn

  def __call__(self, node_features, sent_features, received_features, _global_features):
    return self._fn(node_features, sent_features, received_features)


class _NodeFnWithoutSent:
  def __init__(self, fn: Callable[[TensorTree, Mapping[str, TensorTree]], TensorTree]) -> None:
    self._fn = fn

  def __call__(self, node_features, _sent_features, received_features, _global_features):
    return self._fn(node_features, received_features)


class GraphMapFeaturesCallable:
  """Applies embedding functions independently to edges, nodes and globals."""

  def __init__(
      self,
      embed_edge_fn: Optional[Mapping[str, Callable[[TensorTree], TensorTree]]] = None,
      embed_node_fn: Optional[Mapping[str, Callable[[TensorTree], TensorTree]]] = None,
      embed_global_fn: Optional[Callable[[TensorTree], TensorTree]] = None,
  ) -> None:
    self._embed_edge_fn = dict(embed_edge_fn) if embed_edge_fn else {}
    self._embed_node_fn = dict(embed_node_fn) if embed_node_fn else {}
    self._embed_global_fn = embed_global_fn

  def __call__(self, graph: typed_graph.TypedGraph) -> typed_graph.TypedGraph:
    updated_edges = dict(graph.edges)
    for edge_set_name, embed_fn in self._embed_edge_fn.items():
      edge_set_key = graph.edge_key_by_name(edge_set_name)
      edge_set = graph.edges[edge_set_key]
      updated_edges[edge_set_key] = edge_set._replace(features=embed_fn(edge_set.features))

    updated_nodes = dict(graph.nodes)
    for node_set_key, embed_fn in self._embed_node_fn.items():
      node_set = graph.nodes[node_set_key]
      updated_nodes[node_set_key] = node_set._replace(features=embed_fn(node_set.features))

    updated_context = graph.context
    if self._embed_global_fn:
      updated_context = updated_context._replace(features=self._embed_global_fn(updated_context.features))

    return graph._replace(edges=updated_edges, nodes=updated_nodes, context=updated_context)


def GraphNetwork(
    update_edge_fn: Mapping[str, Callable[[TensorTree, TensorTree, TensorTree, TensorTree], TensorTree]],
    update_node_fn: Mapping[str, NodeFn],
    update_global_fn: Optional[GlobalFn] = None,
    aggregate_edges_for_nodes_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor] = segment_sum,
    aggregate_nodes_for_globals_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor] = segment_sum,
    aggregate_edges_for_globals_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor] = segment_sum,
) -> GraphNetworkCallable:
  return GraphNetworkCallable(
      update_edge_fn=update_edge_fn,
      update_node_fn=update_node_fn,
      update_global_fn=update_global_fn,
      aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn,
      aggregate_nodes_for_globals_fn=aggregate_nodes_for_globals_fn,
      aggregate_edges_for_globals_fn=aggregate_edges_for_globals_fn)


def InteractionNetwork(
    update_edge_fn: Mapping[str, Callable[[TensorTree, TensorTree, TensorTree], TensorTree]],
    update_node_fn: Mapping[str, Union[
        Callable[[TensorTree, Mapping[str, TensorTree], Mapping[str, TensorTree]], TensorTree],
        Callable[[TensorTree, Mapping[str, TensorTree]], TensorTree]]],
    aggregate_edges_for_nodes_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor] = segment_sum,
    include_sent_messages_in_node_update: bool = False,
) -> GraphNetworkCallable:
  wrapped_update_edge_fn = {k: _EdgeFnNoGlobal(fn) for k, fn in update_edge_fn.items()}
  if include_sent_messages_in_node_update:
    wrapped_update_node_fn = {k: _NodeFnWithSent(fn) for k, fn in update_node_fn.items()}
  else:
    wrapped_update_node_fn = {k: _NodeFnWithoutSent(fn) for k, fn in update_node_fn.items()}

  return GraphNetwork(
      update_edge_fn=wrapped_update_edge_fn,
      update_node_fn=wrapped_update_node_fn,
      update_global_fn=None,
      aggregate_edges_for_nodes_fn=aggregate_edges_for_nodes_fn)


def GraphMapFeatures(
    embed_edge_fn: Optional[Mapping[str, Callable[[TensorTree], TensorTree]]] = None,
    embed_node_fn: Optional[Mapping[str, Callable[[TensorTree], TensorTree]]] = None,
    embed_global_fn: Optional[Callable[[TensorTree], TensorTree]] = None,
) -> GraphMapFeaturesCallable:
  return GraphMapFeaturesCallable(
      embed_edge_fn=embed_edge_fn,
      embed_node_fn=embed_node_fn,
      embed_global_fn=embed_global_fn)


def _edge_update(
    graph: typed_graph.TypedGraph,
    edge_fn: Callable[[TensorTree, TensorTree, TensorTree, TensorTree], TensorTree],
    edge_set_key: typed_graph.EdgeSetKey,
) -> typed_graph.EdgeSet:
  sender_nodes = graph.nodes[edge_set_key.node_sets[0]]
  receiver_nodes = graph.nodes[edge_set_key.node_sets[1]]
  edge_set = graph.edges[edge_set_key]
  device_source = _first_tensor(edge_set.features)
  if device_source is None:
    device_source = _first_tensor(sender_nodes.features)
  if device_source is None:
    device_source = _first_tensor(receiver_nodes.features)
  device = device_source.device if device_source is not None else torch.device("cpu")

  senders = _as_tensor(edge_set.indices.senders, dtype=torch.long, device=device)
  receivers = _as_tensor(edge_set.indices.receivers, dtype=torch.long, device=device)

  sent_attributes = _tree_map(lambda n: n.index_select(0, senders), sender_nodes.features)
  received_attributes = _tree_map(lambda n: n.index_select(0, receivers), receiver_nodes.features)

  counts = _as_tensor(edge_set.n_edge, dtype=torch.long, device=device)
  global_features = _repeat_global(graph.context.features, counts)

  new_features = edge_fn(edge_set.features, sent_attributes, received_attributes, global_features)
  return edge_set._replace(features=new_features)


def _node_update(
    graph: typed_graph.TypedGraph,
    node_fn: NodeFn,
    node_set_key: str,
    aggregation_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
) -> typed_graph.NodeSet:
  node_set = graph.nodes[node_set_key]
  reference = _first_tensor(node_set.features)
  if reference is None:
    raise ValueError(f"Node set '{node_set_key}' does not contain tensor features.")
  sum_n_node = reference.shape[0]

  sent_features: dict[str, TensorTree] = {}
  for edge_set_key, edge_set in graph.edges.items():
    sender_node_set_key = edge_set_key.node_sets[0]
    if sender_node_set_key == node_set_key:
      senders = _as_tensor(edge_set.indices.senders, dtype=torch.long, device=reference.device)
      sent_features[edge_set_key.name] = _tree_map(
          lambda e: aggregation_fn(e, senders, sum_n_node), edge_set.features)

  received_features: dict[str, TensorTree] = {}
  for edge_set_key, edge_set in graph.edges.items():
    receiver_node_set_key = edge_set_key.node_sets[1]
    if receiver_node_set_key == node_set_key:
      receivers = _as_tensor(edge_set.indices.receivers, dtype=torch.long, device=reference.device)
      received_features[edge_set_key.name] = _tree_map(
          lambda e: aggregation_fn(e, receivers, sum_n_node), edge_set.features)

  counts = _as_tensor(node_set.n_node, dtype=torch.long, device=reference.device)
  global_features = _repeat_global(graph.context.features, counts)
  new_features = node_fn(node_set.features, sent_features, received_features, global_features)
  return node_set._replace(features=new_features)


def _global_update(
    graph: typed_graph.TypedGraph,
    global_fn: GlobalFn,
    edge_aggregation_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
    node_aggregation_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
) -> typed_graph.Context:
  reference = _first_tensor(graph.context.features)
  if reference is None:
    device = torch.device("cpu")
  else:
    device = reference.device

  n_graph = _as_tensor(graph.context.n_graph, dtype=torch.long, device=device)
  num_graph = n_graph.shape[0]
  graph_idx = torch.arange(num_graph, device=device)

  edge_features = {}
  for edge_set_key, edge_set in graph.edges.items():
    counts = _as_tensor(edge_set.n_edge, dtype=torch.long, device=device)
    edge_gr_idx = torch.repeat_interleave(graph_idx, counts, dim=0)
    edge_features[edge_set_key.name] = _tree_map(
        lambda e: edge_aggregation_fn(e, edge_gr_idx, num_graph), edge_set.features)

  node_features = {}
  for node_set_key, node_set in graph.nodes.items():
    counts = _as_tensor(node_set.n_node, dtype=torch.long, device=device)
    node_gr_idx = torch.repeat_interleave(graph_idx, counts, dim=0)
    node_features[node_set_key] = _tree_map(
        lambda n: node_aggregation_fn(n, node_gr_idx, num_graph), node_set.features)

  new_features = global_fn(node_features, edge_features, graph.context.features)
  return graph.context._replace(features=new_features)


AGGREGATORS = {
    "segment_sum": segment_sum,
    "segment_mean": segment_mean,
}
