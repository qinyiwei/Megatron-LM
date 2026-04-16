# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Pretrain and SFT GPT."""

# Capture the true program start time BEFORE any heavy imports.
import time
_PROGRAM_START_TIME = time.time()

import json

# Suppress warnings on all ranks but rank 0.
import os
import glob
import warnings
rank = int(os.environ.get('RANK', 0))
if rank != 0:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

from functools import partial
from typing import List, Optional, Tuple

import torch

from gpt_builders import gpt_builder
from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.utils import get_attr_wrapped_model, get_thd_batch_on_this_cp_rank, get_batch_on_this_hybrid_cp_rank, StragglerDetector
from megatron.core.tokenizers.text.utils.build_tokenizer import build_tokenizer
from megatron.training import (
    get_args,
    get_timers,
    get_tokenizer,
    inprocess_restart,
    pretrain,
    print_rank_0,
    set_startup_timestamps,
)
from megatron.training.datasets.sft_dataset import SFTDataset
from megatron.core.transformer.multi_token_prediction import mtp_on_this_rank, get_mtp_ranks
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.datasets.fim_dataset import GPTFIMDataset, GPTFIMDatasetConfig
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
    is_first_or_last_pipeline_stage,
)
from model_provider import model_provider

try:
    from megatron.post_training.arguments import add_modelopt_args
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

stimer = StragglerDetector()


def get_batch(data_iterator, vp_stage: Optional[int] = None):
    """Generate a batch."""
    args = get_args()
    config = core_transformer_config_from_args(args)
    # TODO: this is pretty hacky, find a better way
    if not is_first_or_last_pipeline_stage(vp_stage) and (
    (not mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage))):
        return None, None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(
        data_iterator,
        mtp_on_this_rank=mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage)
        )

    cu_seqlens = batch.pop('cu_seqlens', None)
    cu_seqlens_padded = batch.pop('cu_seqlens_padded', None)
    max_seqlen = batch.pop('max_seqlen', None)
    local_cp_size = batch.pop('local_cp_size', None)
    if local_cp_size is not None:
        local_cp_size = int(local_cp_size.item())

    if cu_seqlens is None and local_cp_size is None:
        # slice batch along sequence dimension for context parallelism
        batch = get_batch_on_this_cp_rank(batch)  # The implementation of this function is in MCore
        packed_seq_params = None
    elif local_cp_size is None:  # Packed THD format
        assert max_seqlen.dim() == 1
        batch, packed_seq_params = get_thd_batch_on_this_cp_rank(batch, cu_seqlens, cu_seqlens_padded, max_seqlen)
    else: # Hybrid CP format
        batch, packed_seq_params = get_batch_on_this_hybrid_cp_rank(batch, local_cp_size)
    
    return (*batch.values(), packed_seq_params)


# define spiky loss as a loss that's 10x the max loss observed
SPIKY_LOSS_FACTOR = 10


def loss_func(
    loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: Optional[GPTModel] = None
):
    """Loss function.

    Args:
        loss_mask (torch.Tensor): Used to mask out some portions of the loss
        output_tensor (torch.Tensor): The tensor with the losses
        model (GPTModel, optional): The model (can be wrapped)

    Returns:
        the loss scalar for this micro-batch
        the number of non-padded tokens in this microbatch
        a dict containing reporting metrics on the loss and number of tokens across
            the data parallel ranks
    """
    args = get_args()

    if has_nvidia_modelopt and getattr(args, 'modelopt_enabled', False):  # [ModelOpt]
        loss, num_tokens, report = loss_func_modelopt(loss_mask, output_tensor, model=model)
    else:
        losses = output_tensor.view(-1).float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses * loss_mask)

        num_tokens = loss_mask.sum().clone().detach().to(torch.int)
        report = {'lm loss': torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])}

    # Check individual rank losses are not NaN prior to DP all-reduce.
    rerun_state_machine = get_rerun_state_machine()
    if args.check_for_nan_in_loss_and_grad:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isnan,
            message="found NaN in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=torch.isinf,
            message="found Inf in local forward loss calculation",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=True,
        )
    # Check for spiky loss
    if args.check_for_spiky_loss:
        rerun_state_machine.validate_result(
            result=loss,
            rejection_func=partial(
                rerun_state_machine.is_unexpectedly_large,
                threshold=SPIKY_LOSS_FACTOR,
                context="loss",
            ),
            message="Spiky loss",
            tolerance=0.0,  # forward pass calculations are determinisic
            fatal=False,
        )

    return loss, num_tokens, report


def forward_step(data_iterator, model: GPTModel, return_schedule_plan: bool = False):
    """Forward training step.

    Args:
        data_iterator : Input data iterator
        model (GPTModel): The GPT Model
        return_schedule_plan (bool): Whether to return the schedule plan instead of the output tensor
    """
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator', log_level=2).start()
    global stimer
    with stimer(bdata=True):
        vp_stage = get_attr_wrapped_model(model, "vp_stage")
        tokens, labels, loss_mask, attention_mask, position_ids, packed_seq_params = get_batch(data_iterator, vp_stage)
    timers('batch-generator').stop()

    with stimer:
        if args.use_legacy_models:
            output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
        else:
            if return_schedule_plan:
                assert args.overlap_moe_expert_parallel_comm, \
                    "overlap_moe_expert_parallel_comm must be enabled to return the schedule plan"
                schedule_plan = model.build_schedule_plan(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )
                return schedule_plan, partial(loss_func, loss_mask, model=model)
            else:
                output_tensor = model(
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask, packed_seq_params=packed_seq_params
                )

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model)


def is_dataset_built_on_rank(vp_stage=None):
    args = get_args()
    config = core_transformer_config_from_args(args)
    return (
        is_first_or_last_pipeline_stage(vp_stage)
        or mtp_on_this_rank(config, ignore_virtual=False, vp_stage=vp_stage)
    ) and parallel_state.get_tensor_model_parallel_rank() == 0

# custom data load
def parse_dataset_config(config_file):
    """解析数据集配置，支持config file"""
    import json
    try:
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"配置文件不存在: {config_file}")
        
        print(f"📄 读取配置文件: {config_file}")
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        print(f"📊 使用JSON文件配置，包含 {len(config)} 个数据集")
        for i, cfg in enumerate(config):
            print(f"   [{i+1}] {cfg['path']} (权重: {cfg['weight']})")
        return config
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON文件格式错误: {e}")
    except Exception as e:
        raise ValueError(f"读取配置文件失败: {e}")


def generate_dataset_paths(config_file, pattern="_text_document"):
    """
    生成NeMo数据集路径列表，支持带权重配置
    
    Args:
        config_file: data config file
        pattern: 文件名模式，默认为"_text_document"
    
    Returns:
        list: Megatron格式的路径列表 ["weight1", "path1", "weight2", "path2", ...]
    """
    # 解析配置
    dataset_config = parse_dataset_config(config_file)
    
    # 检查所有目录是否存在
    print(f"📁 验证数据集目录...")
    for config in dataset_config:
        data_dir = config["path"]
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"数据目录不存在: {data_dir}")
        print(f"   ✅ {data_dir}")
    
    dataset_paths = []
    
    # 为每个目录生成文件级别的路径
    total_files = 0
    for config in dataset_config:
        data_dir = config["path"]
        folder_weight = config["weight"]
        
        print(f"\n📁 处理目录: {data_dir} (权重: {folder_weight})")
        
        # 搜索该目录下的.bin文件
        bin_files = glob.glob(os.path.join(data_dir, "**", "*_text_document.bin"), recursive=True)
        
        if not bin_files:
            print(f"   ⚠️  警告: 未找到任何.bin文件")
            continue
        
        valid_files = []
        file_sizes = []
        
        for bin_file in sorted(bin_files):
            # 检查文件大小
            bin_size = os.path.getsize(bin_file)
            if bin_size == 0:
                continue
                
            # 检查对应的.idx文件是否存在
            idx_file = bin_file.replace(".bin", ".idx")
            if not os.path.exists(idx_file) or os.path.getsize(idx_file) == 0:
                continue
            
            # 提取路径前缀
            prefix = bin_file.replace(f"{pattern}.bin", "")
            valid_files.append(prefix + pattern)
            file_sizes.append(bin_size)
        
        if not valid_files:
            print(f"   ❌ 该目录下没有有效的数据文件")
            continue
        
        print(f"   ✅ 找到 {len(valid_files)} 个有效文件")
        
        # 计算总大小
        total_size = sum(file_sizes)
        
        if total_size == 0:
            print(f"   ❌ 该目录下文件总大小为0")
            continue
        
        # 按文件大小分配权重
        for file_path, file_size in zip(valid_files, file_sizes):
            # 按文件大小比例分配权重
            file_weight = folder_weight * (file_size / total_size)
            dataset_paths.extend([str(file_weight), file_path])
        
        total_files += len(valid_files)
        
        # 显示文件大小统计
        total_size_mb = total_size / (1024 * 1024)
        print(f"   📊 总大小: {total_size_mb:.2f} MB")
    
    if not dataset_paths:
        raise FileNotFoundError(f"未找到有效的数据集文件")
    
    print(f"\n✅ 总计: {total_files} 个有效文件，生成 {len(dataset_paths)//2} 个带权重的数据集条目")
    
    # 验证权重总和
    weights = [float(dataset_paths[i]) for i in range(0, len(dataset_paths), 2)]
    weight_sum = sum(weights)
    print(f"📊 权重总和: {weight_sum:.6f}")
    
    return dataset_paths



def core_gpt_dataset_config_from_args(args):
    if args.legacy_tokenizer:
        tokenizer = get_tokenizer()
    else:
        tokenizer = build_tokenizer(args)

    args.data_path = generate_dataset_paths(args.data_path[0])

    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    sequences_per_dataset = None
    if args.per_dataset_sequences_path is not None:
        with open(args.per_dataset_sequences_path, "r") as f:
            sequences_per_dataset = json.load(f)

    data_args = {
        "random_seed": args.seed,
        "sequence_length": args.seq_length,
        "blend": blend,
        "blend_per_split": blend_per_split,
        "split": args.split,
        "multiple_validation_sets": args.multiple_validation_sets,
        "full_validation": args.full_validation,
        "num_dataset_builder_threads": args.num_dataset_builder_threads,
        "path_to_cache": args.data_cache_path,
        "mmap_bin_files": args.mmap_bin_files,
        "tokenizer": tokenizer,
        "reset_position_ids": args.reset_position_ids,
        "reset_attention_mask": args.reset_attention_mask,
        "eod_mask_loss": args.eod_mask_loss,
        "create_attention_mask": args.create_attention_mask_in_dataloader,
        "object_storage_cache_path": args.object_storage_cache_path,
        "mid_level_dataset_surplus": args.mid_level_dataset_surplus,
        "allow_ambiguous_pad_tokens": args.allow_ambiguous_pad_tokens,
        "fast_cache_load": args.dataloader_fast_cache_load,
        "sequences_per_dataset": sequences_per_dataset,
        "defer_npy_index_mmap": args.dataloader_defer_npy_index_mmap,
        "context_parallel_size": args.context_parallel_size,
        "data_parallel_size": args.data_parallel_size,
        "sequence_parallel_size": args.tensor_model_parallel_size*args.sequence_parallel,
        "hybrid_context_parallel": args.hybrid_context_parallel,
    }

    # add FIM args to the config
    if args.fim_data:
        extra_tokens = {
            "prefix": args.fim_prefix_token,
            "middle": args.fim_middle_token,
            "suffix": args.fim_suffix_token,
            "pad": args.fim_pad_token,
            "eod": args.fim_eod_token,
        }
        data_args.update(
            {
                "fim_rate": args.fim_rate,
                "fim_spm_rate": args.fim_spm_rate,
                "fim_extra_tokens": extra_tokens,
                "fim_split_sample": args.fim_split_sample,
                "fim_fragment_rate": args.fim_fragment_rate,
                "fim_no_prefix": args.fim_no_prefix,
            }
        )
        return GPTFIMDatasetConfig(**data_args)

    return GPTDatasetConfig(**data_args)


def train_valid_test_datasets_provider(train_val_test_num_samples, vp_stage=None):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args = get_args()

    config = core_gpt_dataset_config_from_args(args)

    if args.sft:
        dataset_type = SFTDataset
    else:
        if args.mock_data:
            dataset_type = MockGPTDataset
        elif args.fim_data:
            dataset_type = GPTFIMDataset
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    is_dataset_built = partial(is_dataset_built_on_rank, vp_stage=vp_stage)
    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, partial(is_dataset_built_on_rank, vp_stage=vp_stage), config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


def get_embedding_ranks(pp_ranks: List[int]):
    """Get the embedding ranks."""
    embedding_ranks = [pp_ranks[0]]
    if len(pp_ranks) > 1:
        args = get_args()
        if not args.untie_embeddings_and_output_weights:
            embedding_ranks.append(pp_ranks[-1])
        config = core_transformer_config_from_args(args)
        mtp_ranks = get_mtp_ranks(pp_ranks, config)
        embedding_ranks.extend(mtp_ranks)
    embedding_ranks = list(set(embedding_ranks))
    embedding_ranks = sorted(embedding_ranks)
    return embedding_ranks


if __name__ == "__main__":
    # Timestamp right after entering __main__ block (after all imports/library setup)
    _MAIN_ENTRY_TIME = time.time()

    # Register startup timestamps for timing report in pretrain()
    set_startup_timestamps(program_start=_PROGRAM_START_TIME, main_entry=_MAIN_ENTRY_TIME)

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        train_valid_test_datasets_provider,
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        store=store,
        get_embedding_ranks=get_embedding_ranks,
    )
