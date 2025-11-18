#!/usr/bin/env python3
import os
import sys

megatron_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, megatron_root)
from pretrain_gpt import generate_dataset_paths

import argparse
import torch
from datetime import timedelta
from typing import Union
import time
from megatron.core.datasets.gpt_dataset import GPTDataset, GPTDatasetConfig
import torch.distributed as dist
from megatron.training import print_rank_0
from megatron.training.utils import get_blend_and_blend_per_split
from typing import List, Optional, Tuple, Union
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core import parallel_state
from megatron.training.tokenizer import build_tokenizer

def parse_args():
    parser = argparse.ArgumentParser(description="NeMo Llama 3.2 预训练脚本")
    parser.add_argument(
        "--dataset_config_file",
        type=str,
        required=False,
        help="数据集配置JSON文件路径，包含folder和权重信息"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="保存路径"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234
    )
    parser.add_argument(
        "--seq_length",
        type=int,
        required=True,
        help="序列长度"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="99,1,0"
    )
    parser.add_argument(
        "--multiple_validation_sets",
        action="store_true",
    )
    parser.add_argument(
        "--full_validation",
        action="store_true",
    )
    parser.add_argument(
        "--mmap_bin_files",
        action="store_true",
    )
    parser.add_argument(
        "--reset_position_ids",
        action="store_true",
    )
    parser.add_argument(
        "--reset_attention_mask",
        action="store_true",
    )
    parser.add_argument(
        "--eod_mask_loss",
        action="store_true",
    )
    parser.add_argument(
        "--create_attention_mask_in_dataloader",
        action="store_true",
    )
    parser.add_argument(
        "--num_dataset_builder_threads",
        type=int,
        default=1
    )
    parser.add_argument(
        "--object_storage_cache_path",
        type=str,
        default=None
    )
    parser.add_argument(
        "--mid_level_dataset_surplus",
        type=float,
        default=0.005
    )
    parser.add_argument(
        "--train_iters",
        type=int,
        required=True,
        help="最大训练步数"
    )
    parser.add_argument(
        "--train_samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--train_data_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--valid_data_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--per_split_data_args_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--data_args_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--global_batch_size",
        type=int,
        required=True,
        help="全局批大小"
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        required=True,
        help="验证检查间隔"
    )
    parser.add_argument(
        "--eval_iters",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="实验名称"
    )
    
    parser.add_argument(
        "--mbs",
        type=int,
        required=True,
        help="微批大小"
    )
    parser.add_argument(
        "--tokenizer_model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default="HuggingFaceTokenizer",
    )
    parser.add_argument(
        "--make_vocab_size_divisible_by",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--tensor_model_parallel_size",
        type=int,
        required=True,
    )
    args = parser.parse_args()
    args.data_cache_path=f"{args.output_path}/{args.experiment_name}/dataset_cache"
    return args

def is_dataset_built_on_rank():
    return (
        parallel_state.is_pipeline_first_stage(ignore_virtual=True)
        or parallel_state.is_pipeline_last_stage(ignore_virtual=True)
    ) and parallel_state.get_tensor_model_parallel_rank() == 0
    
def is_built_on_rank():
    return True

def core_gpt_dataset_config_from_args(args):
    tokenizer = build_tokenizer(args)
    args.data_path = generate_dataset_paths(args.dataset_config_file)
    
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
    
def build_train_valid_test_datasets(args, rank, train_val_test_num_samples):
    """Build the train test and validation datasets.

    Args:
        train_val_test_num_samples : A list containing the number of samples in train test and validation.
    """
    args.rank = rank
    config = core_gpt_dataset_config_from_args(args)
    dataset_type = GPTDataset

    print_rank_0("> building train, validation, and test datasets for GPT ...")

    train_ds, valid_ds, test_ds = BlendedMegatronDatasetBuilder(
        dataset_type, train_val_test_num_samples, is_built_on_rank, config
    ).build()

    print_rank_0("> finished creating GPT datasets ...")

def get_train_valid_test_num_samples(args):
    """Train/valid/test num samples."""
    # Number of train/valid/test samples.
    if args.train_samples:
        train_samples = args.train_samples
    else:
        train_samples = args.train_iters * args.global_batch_size
    if args.full_validation:
        eval_samples = None
    else:
        eval_iters = (args.train_iters // args.eval_interval + 1) * args.eval_iters
        eval_samples = eval_iters * args.global_batch_size
    test_iters = args.eval_iters

    return (train_samples, eval_samples, test_iters * args.global_batch_size)


def main():
    args = parse_args()
    # 本地运行时设置默认环境变量
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '23456'
    
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
    else:
        rank = 0
    
    if rank != 0:
        print(f"Rank {rank}: Skipping dataset building (only rank 0 builds datasets)")
        return


    print_rank_0(f"Batch sizes: MBS={args.mbs}, GBS={args.global_batch_size}")
    print_rank_0("Building datasets using official Megatron function...")
    assert not dist.is_initialized(), "This function cannot be called inside an existing torch.distributed job."
    # The indices in Megatron are built on rank 0, so we set the world size to 1 here.
    timeout_delta = timedelta(hours=168)
    dist.init_process_group(
        world_size=1, 
        rank=0,
        device_id=torch.device('cuda:0'),
        timeout=timeout_delta
    )
    train_val_test_num_samples = get_train_valid_test_num_samples(args)
    print_rank_0(' > datasets target sizes (minimum size):')
    print_rank_0('    train:      {}'.format(train_val_test_num_samples[0]))
    print_rank_0('    validation: {}'.format(train_val_test_num_samples[1]))
    print_rank_0('    test:       {}'.format(train_val_test_num_samples[2]))
    
    start_time = time.time()
    rank = dist.get_rank()
    print("rank is:"+str(rank))
    build_train_valid_test_datasets(args, rank, train_val_test_num_samples)
    end_time = time.time()
    elapsed_minutes = (end_time - start_time) / 60
    print_rank_0("Dataset building completed!")
    print_rank_0(f"time cost (min):{elapsed_minutes}")

if __name__ == "__main__":
    main()