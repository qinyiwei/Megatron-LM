# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.

"""Pretrain and SFT GPT."""

import datetime
import os
import torch
import glob

from functools import partial
from typing import List, Optional, Tuple, Union
from megatron.core import parallel_state
from megatron.training import get_args
from megatron.training import inprocess_restart
from megatron.training import print_rank_0
from megatron.training import get_timers
from megatron.training import get_tokenizer
from megatron.core import mpu
from megatron.core.enums import ModelType
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig, MockGPTDataset
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import (
    get_gpt_decoder_block_spec,
    get_gpt_layer_local_spec,
    get_gpt_layer_with_transformer_engine_spec,
    get_gpt_mtp_block_spec,
)
from megatron.core.models.gpt.heterogeneous.heterogeneous_layer_specs import (
    get_gpt_heterogeneous_layer_spec,
)
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer.spec_utils import import_module
from megatron.core.utils import StragglerDetector
from megatron.training import get_args, get_timers, get_tokenizer, pretrain, print_rank_0
from megatron.training.arguments import core_transformer_config_from_args
from megatron.training.utils import (
    get_batch_on_this_cp_rank,
    get_batch_on_this_tp_rank,
    get_blend_and_blend_per_split,
)
from megatron.training.yaml_arguments import core_transformer_config_from_yaml
from megatron.training.datasets.sft_dataset import SFTDataset

import megatron.legacy.model  # isort: skip

# NOTE: Loading `megatron.legacy.model` earlier fails due to circular import

try:
    from megatron.post_training.arguments import add_modelopt_args, modelopt_args_enabled
    from megatron.post_training.loss_func import loss_func as loss_func_modelopt
    from megatron.post_training.model_provider import model_provider as model_provider_modelopt

    has_nvidia_modelopt = True
except ImportError:
    has_nvidia_modelopt = False

stimer = StragglerDetector()


def _get_transformer_layer_spec(use_te, config):
    """Get transformer layer specification based on configuration.
    
    Args:
        use_te (bool): Whether to use Transformer Engine
        args: Training arguments
        config: Model configuration
        
    Returns:
        transformer_layer_spec: The transformer layer specification
    """
    args = get_args()
    if use_te:
        return get_gpt_layer_with_transformer_engine_spec(
            args.num_experts,
            args.moe_grouped_gemm,
            args.qk_layernorm,
            args.multi_latent_attention,
            moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
            qk_l2_norm=args.qk_l2_norm,
            use_kitchen=config.use_kitchen,
        )
    else:
        return get_gpt_layer_local_spec(
            args.num_experts,
            args.moe_grouped_gemm,
            args.qk_layernorm,
            args.multi_latent_attention,
            moe_use_legacy_grouped_gemm=args.moe_use_legacy_grouped_gemm,
            normalization=args.normalization,
            use_kitchen=config.use_kitchen,
        )


def model_provider(
    pre_process=True, post_process=True, vp_stage: Optional[int] = None
) -> Union[GPTModel, megatron.legacy.model.GPTModel]:
    """Builds the model.

    If you set the use_legacy_models to True, it will return the legacy GPT model and if not the mcore GPT model.

    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.


    Returns:
        Union[GPTModel, megatron.legacy.model.GPTModel]: The returned model
    """
    args = get_args()

    if has_nvidia_modelopt and modelopt_args_enabled(args):  # [ModelOpt]
        return model_provider_modelopt(pre_process, post_process)

    use_te = args.transformer_impl == "transformer_engine"

    if args.record_memory_history:
        torch.cuda.memory._record_memory_history(
            True,
            # keep 100,000 alloc/free events from before the snapshot
            trace_alloc_max_entries=100000,
            # record stack information for the trace events
            trace_alloc_record_context=True,
        )

        def oom_observer(device, alloc, device_alloc, device_free):
            # snapshot right after an OOM happened
            print('saving allocated state during OOM')
            snapshot = torch.cuda.memory._snapshot()
            from pickle import dump

            dump(
                snapshot,
                open(f"oom_rank-{torch.distributed.get_rank()}_{args.memory_snapshot_path}", 'wb'),
            )

        torch._C._cuda_attach_out_of_memory_observer(oom_observer)

    print_rank_0('building GPT model ...')
    # Experimental loading arguments from yaml
    if args.yaml_cfg is not None:
        config = core_transformer_config_from_yaml(args, "language_model")
    else:
        config = core_transformer_config_from_args(args)

    if args.use_legacy_models:
        model = megatron.legacy.model.GPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process,
        )
    else:  # using core models
        if args.spec is not None:
            transformer_layer_spec = import_module(args.spec)
        else:
            if args.num_experts:
                # Define the decoder block spec
                transformer_layer_spec = get_gpt_decoder_block_spec(
                    config, use_transformer_engine=use_te, normalization=args.normalization, qk_l2_norm=args.qk_l2_norm, vp_stage=vp_stage
                )
            elif args.heterogeneous_layers_config_path is not None:
                transformer_layer_spec = get_gpt_heterogeneous_layer_spec(config, use_te)
            else:
                # Define the decoder layer spec
                transformer_layer_spec = _get_transformer_layer_spec(use_te, config)
        mtp_block_spec = None
        if args.mtp_num_layers is not None:
            if hasattr(transformer_layer_spec, 'layer_specs') and len(transformer_layer_spec.layer_specs) == 0:
                # Get the decoder layer spec explicitly if no decoder layer in the last stage,
                # Only happens with block spec (TransformerBlockSubmodules) when using MoE.
                transformer_layer_spec_for_mtp = _get_transformer_layer_spec(use_te, config)
            else:
                transformer_layer_spec_for_mtp = transformer_layer_spec
            mtp_block_spec = get_gpt_mtp_block_spec(
                config, transformer_layer_spec_for_mtp, use_transformer_engine=use_te, vp_stage=vp_stage
            )

        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            rope_scaling=args.use_rope_scaling,
            mtp_block_spec=mtp_block_spec,
            vp_stage=vp_stage,
        )

    return model


def get_batch(data_iterator):
    """Generate a batch."""

    # TODO: this is pretty hacky, find a better way
    if (not parallel_state.is_pipeline_first_stage(ignore_virtual=True)) and (
        not parallel_state.is_pipeline_last_stage(ignore_virtual=True)
    ):
        return None, None, None, None, None

    # get batches based on the TP rank you are on
    batch = get_batch_on_this_tp_rank(data_iterator)

    # slice batch along sequence dimension for context parallelism
    batch = get_batch_on_this_cp_rank(batch)

    return batch.values()


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

    if has_nvidia_modelopt and modelopt_args_enabled(args):  # [ModelOpt]
        return loss_func_modelopt(loss_mask, output_tensor, model=model)

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * loss_mask)

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

    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    reporting_loss = torch.cat([loss.clone().detach().view(1), num_tokens.view(1)])

    return (loss, num_tokens, {'lm loss': reporting_loss})


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
        tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
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
                    tokens, position_ids, attention_mask, labels=labels, loss_mask=loss_mask
                )

    # [ModelOpt]: model is needed to access ModelOpt distillation losses
    return output_tensor, partial(loss_func, loss_mask, model=model)


def is_dataset_built_on_rank():
    return (
        parallel_state.is_pipeline_first_stage(ignore_virtual=True)
        or parallel_state.is_pipeline_last_stage(ignore_virtual=True)
    ) and parallel_state.get_tensor_model_parallel_rank() == 0


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
    tokenizer = get_tokenizer()
    args.data_path = generate_dataset_paths(args.data_path[0])
    
    # Sometimes --data-path is too long, instead we parse it from a file.
    blend: Optional[Tuple[List[str], Optional[List[float]]]]
    blend_per_split: Optional[List[Optional[Tuple[List[str], Optional[List[float]]]]]]
    blend, blend_per_split = get_blend_and_blend_per_split(args)

    return GPTDatasetConfig(
        random_seed=args.seed,
        sequence_length=args.seq_length,
        blend=blend,
        blend_per_split=blend_per_split,
        split=args.split,
        multiple_validation_sets=args.multiple_validation_sets,
        full_validation=args.full_validation,
        num_dataset_builder_threads=args.num_dataset_builder_threads,
        path_to_cache=args.data_cache_path,
        mmap_bin_files=args.mmap_bin_files,
        tokenizer=tokenizer,
        reset_position_ids=args.reset_position_ids,
        reset_attention_mask=args.reset_attention_mask,
        eod_mask_loss=args.eod_mask_loss,
        create_attention_mask=args.create_attention_mask_in_dataloader,
        object_storage_cache_path=args.object_storage_cache_path,
        mid_level_dataset_surplus=args.mid_level_dataset_surplus,
    )


def train_valid_test_datasets_provider(train_val_test_num_samples):
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
        else:
            dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, is_dataset_built_on_rank, config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

    return train_ds, valid_ds, test_ds


if __name__ == "__main__":

    # Temporary for transition to core datasets
    train_valid_test_datasets_provider.is_distributed = True

    # Optionally enable inprocess restart on pretrain
    pretrain, store = inprocess_restart.maybe_wrap_for_inprocess_restart(pretrain)

    pretrain(
        train_valid_test_datasets_provider,
        model_provider,
        ModelType.encoder_or_decoder,
        forward_step,
        args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
        extra_args_provider=add_modelopt_args if has_nvidia_modelopt else None,
        store=store,
    )
