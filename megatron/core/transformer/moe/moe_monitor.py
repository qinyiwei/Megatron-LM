# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""MoE monitoring utilities for tracking essential expert utilization metrics."""

from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist

from megatron.core import parallel_state

_FORWARD_METRICS_BUFFER: Dict[int, Dict[str, Any]] = {}


class MoELayerMonitor:
    """Monitor for a single MoE layer (Level 0 metrics only)."""
    
    def __init__(
        self,
        layer_id: int,
        num_experts: int,
        log_level_0_interval: Optional[int],
    ):
        self.layer_id = layer_id
        self.num_experts = num_experts
        
        # Configuration
        self.log_level_0_interval = log_level_0_interval
        
        # Cached metrics
        self.level_0_metrics: Dict = {}
    
    def is_enabled(self) -> bool:
        return (
            self.log_level_0_interval is not None
        )

    def should_log_level(self, step: int) -> bool:
        """Check if Level 0 metrics should be logged at the given step."""
        return (
            self.log_level_0_interval is not None
            and step % self.log_level_0_interval == 0
        )
    
    def collect_forward_metrics(
        self,
        expert_token_counts: torch.Tensor,
        dropped_tokens: int,
        total_tokens: int,
        step: int = 0,
    ):
        """Collect Level 0 metrics during the forward pass."""
        if self.should_log_level(step):
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
    
    def write_to_tensorboard(self, writer, step: int):
        """Write metrics to TensorBoard.
        
        Args:
            writer: TensorBoard SummaryWriter
            step: Current training step
        """
        if writer is None:
            return
        
        prefix = f'moe_layer_{self.layer_id}'
        
        if self.level_0_metrics:
            stats = self.level_0_metrics['expert_token_counts_stats']
            writer.add_scalar(f'{prefix}_expert/max_token_num', stats['max_token_num'], step)
            writer.add_scalar(f'{prefix}_expert/min_token_num', stats['min_token_num'], step)
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


class GlobalMoEMonitor:
    """Global MoE monitoring coordinator (Level 0 only)."""
    
    def __init__(
        self,
        num_experts: int,
        log_level_0_interval: Optional[int] = None,
    ):
        self.num_experts = num_experts
        self.log_level_0_interval = log_level_0_interval
        self.layer_monitors: Dict[int, MoELayerMonitor] = {}
    
    def get_layer_monitor(self, layer_id: int) -> MoELayerMonitor:
        """Get or create monitor for a specific layer.
        
        Args:
            layer_id: Layer index
            
        Returns:
            MoELayerMonitor for the specified layer
        """
        if layer_id not in self.layer_monitors:
            self.layer_monitors[layer_id] = MoELayerMonitor(
                layer_id=layer_id,
                num_experts=self.num_experts,
                log_level_0_interval=self.log_level_0_interval,
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
    
    def is_enabled(self) -> bool:
        """Check if monitoring is enabled."""
        return self.log_level_0_interval is not None


def _init_forward_metrics_entry(reference: torch.Tensor, iteration: int) -> Dict[str, Any]:
    device = reference.device
    zeros = torch.zeros_like(reference, dtype=torch.float32)
    return {
        "iteration": iteration,
        "pre_counts": torch.zeros_like(zeros),
        "kept_tokens": torch.zeros(1, device=device, dtype=torch.float32),
    }



def record_moe_monitoring_forward_metrics(
    layer_id: int,
    iteration: int,
    pre_counts: torch.Tensor,
    kept_tokens: torch.Tensor,
):
    entry = _FORWARD_METRICS_BUFFER.get(layer_id)
    if entry is None or entry["iteration"] != iteration:
        entry = _init_forward_metrics_entry(pre_counts, iteration)
        _FORWARD_METRICS_BUFFER[layer_id] = entry

    entry["pre_counts"] += pre_counts.detach().to(torch.float32)
    entry["kept_tokens"] += kept_tokens.detach().to(torch.float32)


def _reduce_forward_metrics_entry(entry: Dict[str, Any]):
    if not dist.is_initialized():
        return
    tp_dp_cp_group = parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True)
    if tp_dp_cp_group is None:
        return
    for key in ("pre_counts", "kept_tokens"):
        dist.all_reduce(entry[key], op=dist.ReduceOp.SUM, group=tp_dp_cp_group)


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
    if monitor is None or not monitor.is_enabled():
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
        layer_monitor.collect_forward_metrics(
            expert_token_counts=expert_counts,
            dropped_tokens=int(max(0.0, total_tokens - kept_tokens)),
            total_tokens=int(total_tokens),
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


def is_moe_monitoring_enabled() -> bool:
    """Return True if Level 0 MoE monitoring is active."""
    monitor = get_global_moe_monitor()
    return monitor is not None and monitor.is_enabled()


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
    # Check if monitoring is enabled
    if args.moe_log_level_0_interval is None:
        return None
    
    monitor = GlobalMoEMonitor(
        num_experts=args.num_experts,
        log_level_0_interval=args.moe_log_level_0_interval,
    )
    
    set_global_moe_monitor(monitor)
    return monitor

