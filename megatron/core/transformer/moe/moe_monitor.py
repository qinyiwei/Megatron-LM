# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""MoE Monitoring System for tracking expert utilization and routing behavior.

This module provides a three-level monitoring system:
- Level 0: Essential metrics with zero overhead (always computed by MoE)
- Level 1: Important derived metrics with light overhead
- Level 2: Raw data storage for post-hoc debugging (no heavy computation)
"""

import os
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from megatron.core import parallel_state

_FORWARD_METRICS_BUFFER: Dict[int, Dict[str, Any]] = {}


class MoELayerMonitor:
    """Monitor for a single MoE layer.
    
    Implements three levels of monitoring:
    - Level 0: Zero-overhead essentials (expert_token_counts)
    - Level 1: Light-weight derived statistics
    - Level 2: Raw data storage for post-hoc analysis
    
    Args:
        layer_id: Layer index
        num_experts: Total number of experts
        log_level_0_interval: Interval for Level 0 logging (None = disabled)
        log_level_1_interval: Interval for Level 1 logging (None = disabled)
        log_level_2_interval: Interval for Level 2 logging (None = disabled)
        is_level_2_layer: Whether this layer should record Level 2 data
        log_level_2_sample_tokens: Number of tokens to sample for Level 2
    """
    
    def __init__(
        self,
        layer_id: int,
        num_experts: int,
        log_level_0_interval: Optional[int],
        log_level_1_interval: Optional[int],
        log_level_2_interval: Optional[int],
        is_level_2_layer: bool = True,
        log_level_2_sample_tokens: int = 128,
    ):
        self.layer_id = layer_id
        self.num_experts = num_experts
        
        # Configuration
        self.log_level_0_interval = log_level_0_interval
        self.log_level_1_interval = log_level_1_interval
        self.log_level_2_interval = log_level_2_interval
        self.is_level_2_layer = is_level_2_layer
        self.log_level_2_sample_tokens = log_level_2_sample_tokens
        
        # Cached metrics
        self.level_0_metrics: Dict = {}
        self.level_1_metrics: Dict = {}
        self.level_2_raw_data: Dict = {}
    
    def is_enabled(self) -> bool:
        return (
            self.log_level_0_interval is not None
            or self.log_level_1_interval is not None
            or self.log_level_2_interval is not None
        )

    def should_log_level(self, level: int, step: int) -> bool:
        """Check if should log at given level and step.
        
        Args:
            level: Logging level (0, 1, or 2)
            step: Current training step
            
        Returns:
            True if should log at this step
        """
        if level == 0 and self.log_level_0_interval is not None:
            return step % self.log_level_0_interval == 0
        elif level == 1 and self.log_level_1_interval is not None:
            return step % self.log_level_1_interval == 0
        elif level == 2 and self.log_level_2_interval is not None:
            return self.is_level_2_layer and (step % self.log_level_2_interval == 0)
        return False
    
    def collect_forward_metrics(
        self,
        expert_token_counts: torch.Tensor,
        dropped_tokens: int,
        total_tokens: int,
        router_probs: Optional[Union[torch.Tensor, np.ndarray]] = None,
        routing_decisions: Optional[Union[torch.Tensor, np.ndarray]] = None,
        router_prob_stats: Optional[Dict[str, float]] = None,
        step: int = 0,
    ):
        """Collect metrics during forward pass.
        
        Args:
            expert_token_counts: [num_experts] tensor with token counts per expert
            dropped_tokens: Number of dropped tokens (if using capacity)
            total_tokens: Total number of tokens
            router_probs: [num_tokens, num_experts] router probabilities (optional)
            routing_decisions: [num_tokens, topk] expert indices (optional)
            step: Current training step
        """
        # === Level 0: Zero overhead essentials ===
        if self.should_log_level(0, step):
            counts = expert_token_counts.detach()
            counts_np = counts.cpu().numpy()
            mean_count = counts_np.mean() if counts_np.size > 0 else 0.0
            std_count = counts_np.std() if counts_np.size > 0 else 0.0
            max_count = float(counts_np.max()) if counts_np.size > 0 else 0.0
            mean_safe = mean_count + 1e-10
            max_violation = float((max_count - mean_count) / mean_safe) if counts_np.size > 0 else 0.0
            mad = float(np.abs(counts_np - mean_count).mean()) if counts_np.size > 0 else 0.0
            normalized_mad = float(mad / mean_safe) if counts_np.size > 0 else 0.0
            load_dist = counts / (counts.sum() + 1e-10)
            entropy = -(load_dist * torch.log(load_dist + 1e-10)).sum().item()
            max_entropy = float(np.log(self.num_experts)) if self.num_experts > 0 else 0.0
            normalized_entropy = entropy / (max_entropy + 1e-10)
            num_zero_experts = int((counts == 0).sum().item())
            drop_rate = float(dropped_tokens) / (total_tokens + 1e-10) if total_tokens > 0 else 0.0
            self.level_0_metrics = {
                'expert_token_counts_stats': {
                    # 'std': float(std_count),
                    'max_token_num': max_count,
                    'min_token_num': float(counts_np.min()) if counts_np.size > 0 else 0.0,
                    # 'cv': float(std_count / (mean_count + 1e-10)),
                    'normalized_entropy': float(normalized_entropy),
                    'num_zero_experts': num_zero_experts,
                    'max_violation': max_violation,
                    'normalized_mad': normalized_mad,
                },
                'drop_rate': drop_rate,
                'total_tokens': total_tokens,
            }
        
        # === Level 1: Important metrics with full expert arrays ===
        if self.should_log_level(1, step):
            self.level_1_metrics = {}
            if router_prob_stats is not None:
                self.level_1_metrics['router_prob_stats'] = router_prob_stats
        
        # === Level 2: Token-level sampling for deep analysis ===
        if self.should_log_level(2, step):
            # Sample router_probs for post-hoc analysis
            if router_probs is not None and self.log_level_2_sample_tokens > 0:
                if torch.is_tensor(router_probs):
                    num_tokens = router_probs.shape[0]
                    if num_tokens > self.log_level_2_sample_tokens:
                        indices = torch.randperm(num_tokens)[: self.log_level_2_sample_tokens]
                        sampled_probs = router_probs[indices].detach().cpu().numpy()
                    else:
                        sampled_probs = router_probs.detach().cpu().numpy()
                else:
                    router_probs_np = router_probs
                    num_tokens = router_probs_np.shape[0]
                    if num_tokens > self.log_level_2_sample_tokens:
                        sampled_probs = router_probs_np[: self.log_level_2_sample_tokens]
                    else:
                        sampled_probs = router_probs_np
                self.level_2_raw_data['router_probs_sample'] = sampled_probs
            
            # Sample routing decisions
            if routing_decisions is not None and self.log_level_2_sample_tokens > 0:
                if torch.is_tensor(routing_decisions):
                    num_tokens = routing_decisions.shape[0]
                    if num_tokens > self.log_level_2_sample_tokens:
                        indices = torch.randperm(num_tokens)[: self.log_level_2_sample_tokens]
                        sampled_decisions = routing_decisions[indices].detach().cpu().numpy()
                    else:
                        sampled_decisions = routing_decisions.detach().cpu().numpy()
                else:
                    routing_decisions_np = routing_decisions
                    num_tokens = routing_decisions_np.shape[0]
                    if num_tokens > self.log_level_2_sample_tokens:
                        sampled_decisions = routing_decisions_np[: self.log_level_2_sample_tokens]
                    else:
                        sampled_decisions = routing_decisions_np
                self.level_2_raw_data['routing_decisions_sample'] = sampled_decisions
    
    def collect_gradient_metrics(
        self,
        experts: nn.ModuleList,
        step: int,
    ):
        """Collect gradient metrics after backward pass.
        
        Args:
            experts: ModuleList of expert modules
            step: Current training step
        """
        # Level 1: Store expert gradient norms and statistics
        if self.should_log_level(1, step):
            expert_grad_norms = []
            for expert in experts:
                norm = 0.0
                for p in expert.parameters():
                    if p.grad is not None:
                        norm += p.grad.norm(2).item() ** 2
                expert_grad_norms.append(norm ** 0.5)
            
            expert_grad_norms = np.array(expert_grad_norms)
            
            # Store both full array and summary statistics
            if 'expert_grad_norms' not in self.level_1_metrics:
                self.level_1_metrics['expert_grad_norms'] = expert_grad_norms
                self.level_1_metrics['expert_grad_stats'] = {
                    'min': float(expert_grad_norms.min()),
                    'median': float(np.median(expert_grad_norms)),
                    'max': float(expert_grad_norms.max()),
                    'mean': float(expert_grad_norms.mean()),
                    'std': float(expert_grad_norms.std()),
                    'num_zero': int((expert_grad_norms < 1e-10).sum()),
                }
    
    def write_to_tensorboard(self, writer, step: int):
        """Write metrics to TensorBoard.
        
        Args:
            writer: TensorBoard SummaryWriter
            step: Current training step
        """
        if writer is None:
            return
        
        prefix = f'moe_layer_{self.layer_id}'
        
        # Level 0: Essential metrics
        if self.level_0_metrics:
            stats = self.level_0_metrics['expert_token_counts_stats']
            # writer.add_scalar(f'{prefix}_expert/std', stats['std'], step)
            writer.add_scalar(f'{prefix}_expert/max_token_num', stats['max_token_num'], step)
            writer.add_scalar(f'{prefix}_expert/min_token_num', stats['min_token_num'], step)
            # writer.add_scalar(f'{prefix}_expert/cv', stats['cv'], step)
            writer.add_scalar(
                f'{prefix}_expert/normalized_entropy', stats['normalized_entropy'], step
            )
            writer.add_scalar(f'{prefix}_expert/num_zero_experts', stats['num_zero_experts'], step)
            writer.add_scalar(f'{prefix}_expert/max_violation', stats['max_violation'], step)
            writer.add_scalar(
                f'{prefix}_expert/normalized_mad',
                stats['normalized_mad'],
                step,
            )
            writer.add_scalar(f'{prefix}_expert/drop_rate', self.level_0_metrics['drop_rate'], step)
            writer.add_scalar(f'{prefix}_expert/total_tokens', 
                                self.level_0_metrics['total_tokens'], step)

        # Level 1: Important metrics with full arrays
        if self.level_1_metrics:
            # Router probability stats
            if 'router_prob_stats' in self.level_1_metrics:
                for name, value in self.level_1_metrics['router_prob_stats'].items():
                    writer.add_scalar(f'{prefix}_prob/{name}', value, step)
            
            # Expert gradient norms
            if 'expert_grad_norms' in self.level_1_metrics:
                grad_norms = self.level_1_metrics['expert_grad_norms']
                # Histogram for distribution
                writer.add_histogram(
                    f'{prefix}/expert_grad_norm_distribution',
                    grad_norms,
                    step
                )
                
                # Summary statistics
                if 'expert_grad_stats' in self.level_1_metrics:
                    stats = self.level_1_metrics['expert_grad_stats']
                    writer.add_scalar(f'{prefix}/expert_grad_norm_median', 
                                    stats['median'], step)
                    writer.add_scalar(f'{prefix}/expert_grad_norm_max', 
                                    stats['max'], step)
                    writer.add_scalar(f'{prefix}/expert_grad_norms_zero_count', 
                                    stats['num_zero'], step)
                
                # Optionally log per-expert values (useful for <64 experts)
                if self.num_experts <= 64:
                    for expert_id, norm in enumerate(grad_norms):
                        writer.add_scalar(
                            f'{prefix}/expert_{expert_id}_grad_norm', 
                            float(norm), 
                            step
                        )
    
    def save_level_2_data(self, output_dir: str, step: int):
        """Save Level 2 raw data to file for post-hoc analysis.
        
        Args:
            output_dir: Directory to save data
            step: Current training step
        """
        if not self.level_2_raw_data:
            return
        
        try:
            import pickle
            rank = torch.distributed.get_rank()
            os.makedirs(output_dir, exist_ok=True)
            output_dir = os.path.join(output_dir, f"step_{step}")
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(
                output_dir,
                f"rank_{rank}_layer_{self.layer_id}.pkl",
            )
            with open(output_file, 'wb') as f:
                pickle.dump({
                    'step': step,
                    'layer_id': self.layer_id,
                    'data': self.level_2_raw_data,
                }, f)
            
            # Clear cache to save memory
            self.level_2_raw_data = {}
        except Exception as e:
            print(f"Warning: Failed to save Level 2 data for layer {self.layer_id}: {e}")


class GlobalMoEMonitor:
    """Global MoE monitoring coordinator.
    
    Manages monitors for all MoE layers and coordinates logging.
    
    Args:
        num_experts: Total number of experts
        log_level_0_interval: Interval for Level 0 logging
        log_level_1_interval: Interval for Level 1 logging
        log_level_2_interval: Interval for Level 2 logging
        log_level_2_layers: Set of layer indices to log at Level 2
        log_level_2_sample_tokens: Number of tokens to sample for Level 2
        log_level_2_output_dir: Directory for Level 2 data files
    """
    
    def __init__(
        self,
        num_experts: int,
        log_level_0_interval: Optional[int] = None,
        log_level_1_interval: Optional[int] = None,
        log_level_2_interval: Optional[int] = None,
        log_level_2_layers: Optional[Set[int]] = None,
        log_level_2_sample_tokens: int = 128,
        log_level_2_output_dir: Optional[str] = None,
    ):
        self.num_experts = num_experts
        self.log_level_0_interval = log_level_0_interval
        self.log_level_1_interval = log_level_1_interval
        self.log_level_2_interval = log_level_2_interval
        self.log_level_2_layers = log_level_2_layers if log_level_2_layers else set()
        self.log_level_2_sample_tokens = log_level_2_sample_tokens
        self.log_level_2_output_dir = log_level_2_output_dir
        
        # Dictionary of layer monitors
        self.layer_monitors: Dict[int, MoELayerMonitor] = {}
    
    def get_layer_monitor(self, layer_id: int) -> MoELayerMonitor:
        """Get or create monitor for a specific layer.
        
        Args:
            layer_id: Layer index
            
        Returns:
            MoELayerMonitor for the specified layer
        """
        if layer_id not in self.layer_monitors:
            is_level_2_layer = (
                layer_id in self.log_level_2_layers 
                if self.log_level_2_layers 
                else True  # Default: all layers
            )
            
            self.layer_monitors[layer_id] = MoELayerMonitor(
                layer_id=layer_id,
                num_experts=self.num_experts,
                log_level_0_interval=self.log_level_0_interval,
                log_level_1_interval=self.log_level_1_interval,
                log_level_2_interval=self.log_level_2_interval,
                is_level_2_layer=is_level_2_layer,
                log_level_2_sample_tokens=self.log_level_2_sample_tokens,
            )
        
        return self.layer_monitors[layer_id]
    
    def write_all_to_tensorboard(self, writer, step: int):
        """Write all layer metrics to TensorBoard.
        
        Args:
            writer: TensorBoard SummaryWriter
            step: Current training step
        """
        if writer is None:
            return
        
        for monitor in self.layer_monitors.values():
            monitor.write_to_tensorboard(writer, step)
    
    def save_all_level_2_data(self, step: int):
        """Save all Level 2 data to files.
        
        Args:
            step: Current training step
        """
        if self.log_level_2_output_dir is None:
            return
        
        for monitor in self.layer_monitors.values():
            monitor.save_level_2_data(self.log_level_2_output_dir, step)
    
    def is_enabled(self) -> bool:
        """Check if any level of monitoring is enabled.
        
        Returns:
            True if any monitoring level is active
        """
        return (
            self.log_level_0_interval is not None 
            or self.log_level_1_interval is not None 
            or self.log_level_2_interval is not None
        )


def _init_forward_metrics_entry(reference: torch.Tensor, iteration: int) -> Dict[str, Any]:
    device = reference.device
    zeros = torch.zeros_like(reference, dtype=torch.float32)
    return {
        "iteration": iteration,
        "pre_counts": torch.zeros_like(zeros),
        "kept_tokens": torch.zeros(1, device=device, dtype=torch.float32),
        "router_prob_stats": _init_router_prob_stats_tensors(device),
        "samples_probs": [],
        "samples_route": [],
    }


def _init_router_prob_stats_tensors(device: torch.device) -> Dict[str, torch.Tensor]:
    dtype = torch.float32
    def zeros():
        return torch.zeros(1, device=device, dtype=dtype)

    def inf(val: float):
        return torch.full((1,), val, device=device, dtype=dtype)

    return {
        "top1_sum": zeros(),
        "top1_sumsq": zeros(),
        "top1_min": inf(float("inf")),
        "top1_max": inf(float("-inf")),
        "top1_count": zeros(),
        "top2_sum": zeros(),
        "top2_count": zeros(),
        "top1_top2_ratio_sum": zeros(),
        "top1_top2_ratio_sumsq": zeros(),
        "top1_top2_ratio_min": inf(float("inf")),
        "top1_top2_ratio_max": inf(float("-inf")),
        "top1_top2_ratio_count": zeros(),
        "topk_prob_sum_sum": zeros(),
        "topk_prob_sum_count": zeros(),
        "entropy_sum": zeros(),
        "entropy_sumsq": zeros(),
        "entropy_min": inf(float("inf")),
        "entropy_max": inf(float("-inf")),
        "entropy_count": zeros(),
        "effective_sum": zeros(),
        "effective_count": zeros(),
        "prob_var_sum": zeros(),
        "prob_var_count": zeros(),
    }


def _accumulate_router_prob_stats(
    dest: Dict[str, torch.Tensor], src: Dict[str, float]
) -> None:
    if src is None:
        return
    device = next(iter(dest.values())).device
    for key in (
        "top1_sum",
        "top1_sumsq",
        "top1_count",
        "top2_sum",
        "top2_count",
        "top1_top2_ratio_sum",
        "top1_top2_ratio_sumsq",
        "top1_top2_ratio_count",
        "topk_prob_sum_sum",
        "topk_prob_sum_count",
        "entropy_sum",
        "entropy_sumsq",
        "entropy_count",
        "effective_sum",
        "effective_count",
        "prob_var_sum",
        "prob_var_count",
    ):
        dest[key] += torch.tensor([src.get(key, 0.0)], device=device, dtype=torch.float32)

    for key, reduce_fn in (
        ("top1_min", torch.minimum),
        ("top1_max", torch.maximum),
        ("top1_top2_ratio_min", torch.minimum),
        ("top1_top2_ratio_max", torch.maximum),
        ("entropy_min", torch.minimum),
        ("entropy_max", torch.maximum),
    ):
        value = src.get(key)
        if value is None:
            continue
        dest[key] = reduce_fn(
            dest[key], torch.tensor([value], device=device, dtype=torch.float32)
        )


def _reduce_router_prob_stats(stats: Dict[str, torch.Tensor], group):
    sum_keys = [
        "top1_sum",
        "top1_sumsq",
        "top1_count",
        "top2_sum",
        "top2_count",
        "top1_top2_ratio_sum",
        "top1_top2_ratio_sumsq",
        "top1_top2_ratio_count",
        "topk_prob_sum_sum",
        "topk_prob_sum_count",
        "entropy_sum",
        "entropy_sumsq",
        "entropy_count",
        "effective_sum",
        "effective_count",
        "prob_var_sum",
        "prob_var_count",
    ]
    for key in sum_keys:
        dist.all_reduce(stats[key], op=dist.ReduceOp.SUM, group=group)
    for key in ("top1_min", "top1_top2_ratio_min", "entropy_min"):
        dist.all_reduce(stats[key], op=dist.ReduceOp.MIN, group=group)
    for key in ("top1_max", "top1_top2_ratio_max", "entropy_max"):
        dist.all_reduce(stats[key], op=dist.ReduceOp.MAX, group=group)


def _finalize_router_prob_stats(stats: Dict[str, torch.Tensor]) -> Optional[Dict[str, float]]:
    result = {}

    def _finalize_mean_std(sum_key, sumsq_key, count_key, prefix):
        count = max(stats[count_key].item(), 0.0)
        if count <= 0:
            return
        mean = stats[sum_key].item() / count
        variance = max(stats[sumsq_key].item() / count - mean * mean, 0.0)
        result[f"{prefix}_mean"] = mean
        result[f"{prefix}_std"] = variance ** 0.5

    _finalize_mean_std("top1_sum", "top1_sumsq", "top1_count", "top1_prob")
    if stats["top1_count"].item() > 0:
        result["top1_prob_min"] = stats["top1_min"].item()
        result["top1_prob_max"] = stats["top1_max"].item()

    if stats["top1_top2_ratio_count"].item() > 0:
        _finalize_mean_std(
            "top1_top2_ratio_sum",
            "top1_top2_ratio_sumsq",
            "top1_top2_ratio_count",
            "top1_top2_ratio",
        )
        result["top1_top2_ratio_min"] = stats["top1_top2_ratio_min"].item()
        result["top1_top2_ratio_max"] = stats["top1_top2_ratio_max"].item()

    if stats["entropy_count"].item() > 0:
        _finalize_mean_std("entropy_sum", "entropy_sumsq", "entropy_count", "entropy")
        result["entropy_min"] = stats["entropy_min"].item()
        result["entropy_max"] = stats["entropy_max"].item()

    if stats["effective_count"].item() > 0:
        result["effective_experts_mean"] = (
            stats["effective_sum"].item() / stats["effective_count"].item()
        )

    if stats["prob_var_count"].item() > 0:
        result["prob_variance_mean"] = (
            stats["prob_var_sum"].item() / stats["prob_var_count"].item()
        )

    if stats["top2_count"].item() > 0:
        result["top2_prob_mean"] = stats["top2_sum"].item() / stats["top2_count"].item()

    if stats["topk_prob_sum_count"].item() > 0:
        result["topk_prob_sum_mean"] = (
            stats["topk_prob_sum_sum"].item() / stats["topk_prob_sum_count"].item()
        )

    return result or None


def record_moe_monitoring_forward_metrics(
    layer_id: int,
    iteration: int,
    pre_counts: torch.Tensor,
    kept_tokens: torch.Tensor,
    router_prob_stats: Optional[Dict[str, float]],
    sample_data: Optional[Dict[str, np.ndarray]] = None,
):
    entry = _FORWARD_METRICS_BUFFER.get(layer_id)
    if entry is None or entry["iteration"] != iteration:
        entry = _init_forward_metrics_entry(pre_counts, iteration)
        _FORWARD_METRICS_BUFFER[layer_id] = entry

    entry["pre_counts"] += pre_counts.detach().to(torch.float32)
    entry["kept_tokens"] += kept_tokens.detach().to(torch.float32)
    if router_prob_stats is not None:
        _accumulate_router_prob_stats(entry["router_prob_stats"], router_prob_stats)

    if sample_data is not None:
        if sample_data.get("router_probs") is not None:
            entry["samples_probs"].append(sample_data["router_probs"])
        if sample_data.get("routing_decisions") is not None:
            entry["samples_route"].append(sample_data["routing_decisions"])


def _reduce_forward_metrics_entry(entry: Dict[str, Any]):
    if not dist.is_initialized():
        return
    tp_dp_cp_group = parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True)
    if tp_dp_cp_group is None:
        return
    for key in ("pre_counts", "kept_tokens"):
        dist.all_reduce(entry[key], op=dist.ReduceOp.SUM, group=tp_dp_cp_group)
    _reduce_router_prob_stats(entry["router_prob_stats"], tp_dp_cp_group)


def _pop_forward_metrics_buffer() -> Dict[int, Dict[str, Any]]:
    global _FORWARD_METRICS_BUFFER
    buffer = _FORWARD_METRICS_BUFFER
    _FORWARD_METRICS_BUFFER = {}
    return buffer


def ingest_moe_monitoring_forward_metrics(step: int):
    if not _FORWARD_METRICS_BUFFER:
        return
    monitor = get_global_moe_monitor()
    buffer = _pop_forward_metrics_buffer()
    if monitor is None:
        return
    for layer_id, entry in buffer.items():
        _reduce_forward_metrics_entry(entry)
        layer_monitor = monitor.get_layer_monitor(layer_id)
        if layer_monitor is None:
            continue
        expert_counts = entry["pre_counts"]
        total_tokens = expert_counts.sum().item()
        expert_counts = expert_counts.detach().cpu()
        kept_tokens = entry["kept_tokens"].item()
        router_prob_stats = _finalize_router_prob_stats(entry["router_prob_stats"])
        router_probs = (
            np.concatenate(entry["samples_probs"], axis=0) if entry["samples_probs"] else None
        )
        routing_decisions = (
            np.concatenate(entry["samples_route"], axis=0) if entry["samples_route"] else None
        )
        layer_monitor.collect_forward_metrics(
            expert_token_counts=expert_counts,
            dropped_tokens=int(max(0.0, total_tokens - kept_tokens)),
            total_tokens=int(total_tokens),
            router_prob_stats=router_prob_stats,
            router_probs=router_probs,
            routing_decisions=routing_decisions,
            step=step,
        )


# Global monitor instance
_GLOBAL_MOE_MONITOR: Optional[GlobalMoEMonitor] = None


def get_global_moe_monitor() -> Optional[GlobalMoEMonitor]:
    """Get the global MoE monitor instance.
    
    Returns:
        GlobalMoEMonitor instance or None if not initialized
    """
    return _GLOBAL_MOE_MONITOR


def set_global_moe_monitor(monitor: GlobalMoEMonitor):
    """Set the global MoE monitor instance.
    
    Args:
        monitor: GlobalMoEMonitor instance to set
    """
    global _GLOBAL_MOE_MONITOR
    _GLOBAL_MOE_MONITOR = monitor


def initialize_moe_monitor_from_args(args) -> Optional[GlobalMoEMonitor]:
    """Initialize global MoE monitor from command line arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Initialized GlobalMoEMonitor or None if monitoring disabled
    """
    # Check if any monitoring is enabled
    if (args.moe_log_level_0_interval is None 
        and args.moe_log_level_1_interval is None 
        and args.moe_log_level_2_interval is None):
        return None
    
    # Parse Level 2 layers
    log_level_2_layers = None
    if args.moe_log_level_2_layers is not None:
        log_level_2_layers = set(map(int, args.moe_log_level_2_layers.split(',')))
    
    # Create monitor
    monitor = GlobalMoEMonitor(
        num_experts=args.num_experts,
        log_level_0_interval=args.moe_log_level_0_interval,
        log_level_1_interval=args.moe_log_level_1_interval,
        log_level_2_interval=args.moe_log_level_2_interval,
        log_level_2_layers=log_level_2_layers,
        log_level_2_sample_tokens=args.moe_log_level_2_sample_tokens,
        log_level_2_output_dir=args.moe_log_level_2_output_dir,
    )
    
    set_global_moe_monitor(monitor)
    return monitor

