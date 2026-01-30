"""PyTorch implementation of the DeepTypedGraphNet used in GraphCast."""

from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from graphcast import typed_graph
from graphcast import typed_graph_net


def _identity_graph(graph: typed_graph.TypedGraph) -> typed_graph.TypedGraph:
  return graph


class _AggregateAdapter:
  """Wrap aggregation so it remains picklable and dtype safe."""

  def __init__(self,
               base_fn: Callable[[torch.Tensor, torch.Tensor, int], torch.Tensor],
               use_f32: bool,
               normalization: Optional[float]):
    self._base_fn = base_fn
    self._use_f32 = use_f32
    self._normalization = normalization

  def __call__(self,
               data: torch.Tensor,
               segment_ids: torch.Tensor,
               num_segments: int) -> torch.Tensor:
    original_dtype = data.dtype
    work_data = data.float() if self._use_f32 else data
    aggregated = self._base_fn(work_data, segment_ids, num_segments)
    if self._normalization:
      aggregated = aggregated / self._normalization
    return aggregated.to(original_dtype)


class _EdgeBlockCallable:
  """Adapter that discards the unused global argument."""

  def __init__(self, module: nn.Module):
    self._module = module

  def __call__(self, edge_features, sender_features, receiver_features):
    return self._module(edge_features, sender_features, receiver_features, None)


class _NodeBlockCallableWithSent:
  """Adapter for node modules when sent messages are included."""

  def __init__(self, module: nn.Module):
    self._module = module

  def __call__(self, node_features, sent_features, received_features):
    return self._module(node_features, sent_features, received_features, None)


class _NodeBlockCallableNoSent:
  """Adapter for node modules when sent messages are skipped."""

  def __init__(self, module: nn.Module):
    self._module = module

  def __call__(self, node_features, received_features):
    return self._module(node_features, {}, received_features, None)


def _get_activation_module(name: str) -> nn.Module:
  name = name.lower()
  if name == "relu":
    return nn.ReLU()
  if name == "gelu":
    return nn.GELU()
  if name in {"silu", "swish"}:
    return nn.SiLU()
  if name == "tanh":
    return nn.Tanh()
  if name == "identity":
    return nn.Identity()
  raise ValueError(f"Unknown activation function: {name}")


class ConcatMLP(nn.Module):
  """MLP that operates on the last dimension, supporting lazy input size."""

  def __init__(self,
               hidden_size: int,
               num_hidden_layers: int,
               output_size: int,
               activation: str,
               use_layer_norm: bool):
    super().__init__()
    layers = []
    act_module = _get_activation_module(activation)

    if num_hidden_layers > 0:
      layers.append(nn.LazyLinear(hidden_size))
      if not isinstance(act_module, nn.Identity):
        layers.append(act_module)
      for _ in range(num_hidden_layers - 1):
        layers.append(nn.Linear(hidden_size, hidden_size))
        if not isinstance(act_module, nn.Identity):
          layers.append(act_module)
      layers.append(nn.Linear(hidden_size, output_size))
    else:
      layers.append(nn.LazyLinear(output_size))

    self.mlp = nn.Sequential(*layers)
    self.layer_norm = nn.LayerNorm(output_size) if use_layer_norm else None

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    original_shape = x.shape
    x = x.reshape(-1, original_shape[-1])
    x = self.mlp(x)
    if self.layer_norm is not None:
      x = self.layer_norm(x)
    new_shape = original_shape[:-1] + (x.shape[-1],)
    return x.reshape(new_shape)


class EdgeUpdateBlock(nn.Module):
  def __init__(self, mlp: ConcatMLP):
    super().__init__()
    self.mlp = mlp
    self.layer_norm = getattr(mlp, "layer_norm", None)

  def forward(self,
              edge_features: torch.Tensor,
              sender_features: Optional[torch.Tensor] = None,
              receiver_features: Optional[torch.Tensor] = None,
              global_features: Optional[torch.Tensor] = None) -> torch.Tensor:
    tensors = [edge_features]
    if isinstance(sender_features, torch.Tensor):
      tensors.append(sender_features)
    if isinstance(receiver_features, torch.Tensor):
      tensors.append(receiver_features)
    if isinstance(global_features, torch.Tensor):
      tensors.append(global_features)
    concat = torch.cat(tensors, dim=-1)
    return self.mlp(concat)


class NodeUpdateBlock(nn.Module):
  def __init__(self, mlp: ConcatMLP, include_sent: bool):
    super().__init__()
    self.mlp = mlp
    self.include_sent = include_sent
    self.layer_norm = getattr(mlp, "layer_norm", None)

  def forward(self,
              node_features: torch.Tensor,
              sent_features: Optional[Mapping[str, torch.Tensor]] = None,
              received_features: Optional[Mapping[str, torch.Tensor]] = None,
              global_features: Optional[torch.Tensor] = None) -> torch.Tensor:
    tensors = [node_features]
    if self.include_sent and sent_features is not None:
      for key in sorted(sent_features):
        tensor = sent_features[key]
        if isinstance(tensor, torch.Tensor):
          tensors.append(tensor)
    if received_features is not None:
      for key in sorted(received_features):
        tensor = received_features[key]
        if isinstance(tensor, torch.Tensor):
          tensors.append(tensor)
    if isinstance(global_features, torch.Tensor):
      tensors.append(global_features)
    concat = torch.cat(tensors, dim=-1)
    return self.mlp(concat)


class FeatureProjector(nn.Module):
  def __init__(self, mlp: ConcatMLP):
    super().__init__()
    self.mlp = mlp
    self.layer_norm = getattr(mlp, "layer_norm", None)

  def forward(self, features: torch.Tensor) -> torch.Tensor:
    return self.mlp(features)


class DeepTypedGraphNet(nn.Module):
  """Deep typed graph network implemented in PyTorch."""

  def __init__(self,
               *,
               node_latent_size: Mapping[str, int],
               edge_latent_size: Mapping[str, int],
               mlp_hidden_size: int,
               mlp_num_hidden_layers: int,
               num_message_passing_steps: int,
               num_processor_repetitions: int = 1,
               embed_nodes: bool = True,
               embed_edges: bool = True,
               node_output_size: Optional[Mapping[str, int]] = None,
               edge_output_size: Optional[Mapping[str, int]] = None,
               include_sent_messages_in_node_update: bool = False,
               use_layer_norm: bool = True,
               activation: str = "relu",
               f32_aggregation: bool = False,
               aggregate_edges_for_nodes_fn: str = "segment_sum",
               aggregate_normalization: Optional[float] = None,
               name: str | None = None):
    super().__init__()
    del name  # 与JAX版本兼容，无实际用?

    self._node_latent_size = node_latent_size
    self._edge_latent_size = edge_latent_size
    self._mlp_hidden_size = mlp_hidden_size
    self._mlp_num_hidden_layers = mlp_num_hidden_layers
    self._num_message_passing_steps = num_message_passing_steps
    self._num_processor_repetitions = num_processor_repetitions
    self._embed_nodes = embed_nodes
    self._embed_edges = embed_edges
    self._node_output_size = node_output_size
    self._edge_output_size = edge_output_size
    self._include_sent_messages_in_node_update = include_sent_messages_in_node_update
    self._use_layer_norm = use_layer_norm
    self._activation = activation
    self._f32_aggregation = f32_aggregation
    self._aggregate_edges_for_nodes_name = aggregate_edges_for_nodes_fn
    self._aggregate_normalization = aggregate_normalization
    self._initialized = False

    if aggregate_normalization is not None:
      assert aggregate_edges_for_nodes_fn == "segment_sum", (
          "aggregate_normalization currently only supports the 'segment_sum' aggregator")

    if aggregate_edges_for_nodes_fn not in typed_graph_net.AGGREGATORS:
      raise ValueError(
          f"Unknown aggregator: {aggregate_edges_for_nodes_fn}")
    self._aggregate_edges_for_nodes_fn = typed_graph_net.AGGREGATORS[
        aggregate_edges_for_nodes_fn]

  def forward(self, input_graph: typed_graph.TypedGraph) -> typed_graph.TypedGraph:
    if not self._initialized:
      self._build_networks(input_graph)
      self._initialized = True

    latent_graph_0 = self._embed(input_graph)
    latent_graph_m = self._process(latent_graph_0)
    return self._decode(latent_graph_m)

  def _build_networks(self, graph_template: typed_graph.TypedGraph) -> None:
    activation = self._activation

    # 编码?
    if self._embed_edges:
      self._edge_encoders = nn.ModuleDict({
          edge_set_key.name: FeatureProjector(ConcatMLP(
              self._mlp_hidden_size,
              self._mlp_num_hidden_layers,
              self._edge_latent_size[edge_set_key.name],
              activation,
              self._use_layer_norm))
          for edge_set_key in graph_template.edges.keys()
          if edge_set_key.name in self._edge_latent_size
      })
    else:
      self._edge_encoders = nn.ModuleDict()

    if self._embed_nodes:
      self._node_encoders = nn.ModuleDict({
          node_set_name: FeatureProjector(ConcatMLP(
              self._mlp_hidden_size,
              self._mlp_num_hidden_layers,
              self._node_latent_size[node_set_name],
              activation,
              self._use_layer_norm))
          for node_set_name in graph_template.nodes.keys()
          if node_set_name in self._node_latent_size
      })
    else:
      self._node_encoders = nn.ModuleDict()

    # 处理器（多轮消息传递）
    self._processor_edge_blocks = nn.ModuleList()
    self._processor_node_blocks = nn.ModuleList()
    for step in range(self._num_message_passing_steps):
      edge_blocks = nn.ModuleDict({
          edge_set_key.name: EdgeUpdateBlock(ConcatMLP(
              self._mlp_hidden_size,
              self._mlp_num_hidden_layers,
              self._edge_latent_size[edge_set_key.name],
              activation,
              self._use_layer_norm))
          for edge_set_key in graph_template.edges.keys()
          if edge_set_key.name in self._edge_latent_size
      })
      self._processor_edge_blocks.append(edge_blocks)

      node_blocks = nn.ModuleDict({
          node_set_name: NodeUpdateBlock(ConcatMLP(
              self._mlp_hidden_size,
              self._mlp_num_hidden_layers,
              self._node_latent_size[node_set_name],
              activation,
              self._use_layer_norm),
              include_sent=self._include_sent_messages_in_node_update)
          for node_set_name in graph_template.nodes.keys()
          if node_set_name in self._node_latent_size
      })
      self._processor_node_blocks.append(node_blocks)

    # 解码?
    self._edge_decoders = nn.ModuleDict()
    if self._edge_output_size:
      self._edge_decoders = nn.ModuleDict({
          edge_name: FeatureProjector(ConcatMLP(
              self._mlp_hidden_size,
              self._mlp_num_hidden_layers,
              self._edge_output_size[edge_name],
              activation,
              self._use_layer_norm))
          for edge_name in self._edge_output_size
      })

    self._node_decoders = nn.ModuleDict()
    if self._node_output_size:
      self._node_decoders = nn.ModuleDict({
          node_name: FeatureProjector(ConcatMLP(
              self._mlp_hidden_size,
              self._mlp_num_hidden_layers,
              self._node_output_size[node_name],
              activation,
              self._use_layer_norm))
          for node_name in self._node_output_size
      })

    # 构建GraphMapFeatures和InteractionNetwork包装
    if self._edge_encoders or self._node_encoders:
      embed_edge_fn = {
          name: module.forward for name, module in self._edge_encoders.items()
      } if self._edge_encoders else None
      embed_node_fn = {
          name: module.forward for name, module in self._node_encoders.items()
      } if self._node_encoders else None
    else:
      embed_edge_fn = None
      embed_node_fn = None
    self._embedder = typed_graph_net.GraphMapFeatures(
        embed_edge_fn=embed_edge_fn,
        embed_node_fn=embed_node_fn)

    aggregate_fn = _AggregateAdapter(
        self._aggregate_edges_for_nodes_fn,
        self._f32_aggregation,
        self._aggregate_normalization)

    self._processors = []
    for edge_blocks, node_blocks in zip(self._processor_edge_blocks, self._processor_node_blocks):
      update_edge_fn = {name: _EdgeBlockCallable(module) for name, module in edge_blocks.items()}

      if self._include_sent_messages_in_node_update:
        update_node_fn = {name: _NodeBlockCallableWithSent(module) for name, module in node_blocks.items()}
      else:
        update_node_fn = {name: _NodeBlockCallableNoSent(module) for name, module in node_blocks.items()}

      processor = typed_graph_net.InteractionNetwork(
          update_edge_fn=update_edge_fn,
          update_node_fn=update_node_fn,
          aggregate_edges_for_nodes_fn=aggregate_fn,
          include_sent_messages_in_node_update=self._include_sent_messages_in_node_update)
      self._processors.append(processor)

    if self._edge_decoders or self._node_decoders:
      decode_edge_fn = {
          name: module.forward for name, module in self._edge_decoders.items()
      } if self._edge_decoders else None
      decode_node_fn = {
          name: module.forward for name, module in self._node_decoders.items()
      } if self._node_decoders else None
      self._decoder = typed_graph_net.GraphMapFeatures(
          embed_edge_fn=decode_edge_fn,
          embed_node_fn=decode_node_fn)
    else:
      self._decoder = _identity_graph

  def _embed(self, input_graph: typed_graph.TypedGraph) -> typed_graph.TypedGraph:
    context_features = input_graph.context.features
    if isinstance(context_features, torch.Tensor) and context_features.numel() > 0:
      new_nodes = {}
      for node_name, node_set in input_graph.nodes.items():
        counts = torch.as_tensor(node_set.n_node, device=context_features.device, dtype=torch.long)
        repeated = torch.repeat_interleave(context_features, counts, dim=0)
        merged = torch.cat([node_set.features, repeated.to(node_set.features.dtype)], dim=-1)
        new_nodes[node_name] = node_set._replace(features=merged)
      input_graph = input_graph._replace(
          nodes=new_nodes,
          context=input_graph.context._replace(features=()))
    return self._embedder(input_graph)

  def _process(self, latent_graph_0: typed_graph.TypedGraph) -> typed_graph.TypedGraph:
    latent_graph = latent_graph_0
    for _ in range(self._num_processor_repetitions):
      for processor in self._processors:
        latent_graph = self._process_step(processor, latent_graph)
    return latent_graph

  def _process_step(self,
                    processor,
                    latent_graph_prev: typed_graph.TypedGraph) -> typed_graph.TypedGraph:
    latent_graph_new = processor(latent_graph_prev)

    nodes_with_residuals = {}
    for key, prev_set in latent_graph_prev.nodes.items():
      nodes_with_residuals[key] = prev_set._replace(
          features=prev_set.features + latent_graph_new.nodes[key].features)

    edges_with_residuals = {}
    for edge_key, prev_set in latent_graph_prev.edges.items():
      new_set = latent_graph_new.edges[edge_key]
      edges_with_residuals[edge_key] = prev_set._replace(
          features=prev_set.features + new_set.features)

    latent_graph_new = latent_graph_new._replace(nodes=nodes_with_residuals,
                                                 edges=edges_with_residuals)
    return latent_graph_new

  def _decode(self, latent_graph: typed_graph.TypedGraph) -> typed_graph.TypedGraph:
    return self._decoder(latent_graph)

  # ---------------------------------------------------------------------------
  # Parameter loading helpers (Haiku -> PyTorch).
  # ---------------------------------------------------------------------------

  def load_haiku_parameters(
      self,
      params: Mapping[str, np.ndarray],
      prefix: str,
  ) -> None:
    """Loads weights exported from the Haiku GraphCast implementation."""

    def _get(path: str) -> np.ndarray:
      key = f"{prefix}/~_networks_builder/{path}"
      if key not in params:
        raise KeyError(f"Missing parameter '{key}' in NPZ file.")
      return params[key]

    def _to_tensor(array: np.ndarray, reference: torch.Tensor) -> torch.Tensor:
      return torch.as_tensor(array, dtype=reference.dtype, device=reference.device)

    def _load_mlp(projector: FeatureProjector, scope: str) -> None:
      layers_module = projector.mlp
      if isinstance(layers_module, nn.Sequential):
        iterable_layers = layers_module
      elif hasattr(layers_module, "mlp") and isinstance(layers_module.mlp, nn.Sequential):
        iterable_layers = layers_module.mlp
      else:
        iterable_layers = layers_module
      mlp_layers: Sequence[nn.Module] = [
          layer for layer in iterable_layers if isinstance(layer, nn.Linear)
      ]
      for idx, linear in enumerate(mlp_layers):
        weight = _get(f"{scope}_mlp/~/linear_{idx}:w")
        bias = _get(f"{scope}_mlp/~/linear_{idx}:b")
        weight_tensor = _to_tensor(weight.T, linear.weight)
        bias_tensor = _to_tensor(bias, linear.bias)
        linear.weight.data.copy_(weight_tensor)
        linear.bias.data.copy_(bias_tensor)
      if projector.layer_norm is not None:
        try:
          scale = _get(f"{scope}_layer_norm:scale")
          offset = _get(f"{scope}_layer_norm:offset")
        except KeyError:
          pass
        else:
          projector.layer_norm.weight.data.copy_(
              _to_tensor(scale, projector.layer_norm.weight))
          projector.layer_norm.bias.data.copy_(
              _to_tensor(offset, projector.layer_norm.bias))

    for edge_name, module in self._edge_encoders.items():
      _load_mlp(module, f"encoder_edges_{edge_name}")
    for node_name, module in self._node_encoders.items():
      _load_mlp(module, f"encoder_nodes_{node_name}")

    for step, edge_modules in enumerate(self._processor_edge_blocks):
      for edge_name, module in edge_modules.items():
        _load_mlp(module, f"processor_edges_{step}_{edge_name}")
    for step, node_modules in enumerate(self._processor_node_blocks):
      for node_name, module in node_modules.items():
        _load_mlp(module, f"processor_nodes_{step}_{node_name}")

    for edge_name, module in self._edge_decoders.items():
      _load_mlp(module, f"decoder_edges_{edge_name}")
    for node_name, module in self._node_decoders.items():
      _load_mlp(module, f"decoder_nodes_{node_name}")
