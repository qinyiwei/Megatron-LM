# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import hashlib
import json
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Union

import numpy
import torch

from megatron.core.datasets.blended_megatron_dataset_config import BlendedMegatronDatasetConfig
from megatron.core.datasets.indexed_dataset import IndexedDataset
from megatron.core.datasets.utils import Split

LowLevelDataset = Union[IndexedDataset, Iterable]


class MegatronDataset(ABC, torch.utils.data.Dataset):
    """The highest level wrapper class from which all dataset classes should inherit

    Args:
        dataset (LowLevelDataset): The dataset around which to build the MegatronDataset

        dataset_path (Optional[str]): The real path on disk to the dataset, for bookkeeping

        indices (numpy.ndarray): The set of the documents indices to expose

        num_samples (Optional[int]): The minimum number of samples to build from the indexed
            dataset. When None, build as many samples as correspond to one epoch.

        index_split (Split): The indices Split

        config (BlendedMegatronDatasetConfig): The config
    """

    def __init__(
        self,
        dataset: LowLevelDataset,
        dataset_path: Optional[str],
        indices: numpy.ndarray,
        num_samples: Optional[int],
        index_split: Split,
        config: BlendedMegatronDatasetConfig,
    ) -> None:
        self.dataset = dataset
        self.dataset_path = dataset_path
        self.indices = indices
        self.num_samples = num_samples
        self.index_split = index_split
        self.config = config

        self.unique_identifiers = OrderedDict()

        self.unique_identifiers["class"] = type(self).__name__
        self.unique_identifiers["dataset_path"] = self.dataset_path
        self.unique_identifiers["num_samples"] = self.num_samples
        self.unique_identifiers["index_split"] = self.index_split.name
        for attr in self._key_config_attributes():
            self.unique_identifiers[attr] = getattr(self.config, attr)

        self.unique_description = json.dumps(
            self.unique_identifiers, indent=4, default=lambda obj: obj.unique_identifiers
        )
        self.unique_description_hash = hashlib.md5(
            self.unique_description.encode("utf-8"), usedforsecurity=False
        ).hexdigest()
        #self.debug_unique_identifiers()
        
    def debug_unique_identifiers(self):
        """打印所有影响哈希值的信息，用于调试哈希不一致问题"""
        
        print("=" * 80)
        print("DEBUG: Unique Identifiers Information")
        print("=" * 80)
        
        # 1. 打印基本标识符
        print("\n[Basic Identifiers]")
        print(f"Class: {type(self).__name__}")
        print(f"Dataset Path: {self.dataset_path}")
        print(f"Num Samples: {self.num_samples}")
        print(f"Index Split: {self.index_split.name}")
        
        # 2. 打印配置属性
        print("\n[Config Attributes]")
        key_attrs = self._key_config_attributes()
        print(f"Key config attributes: {key_attrs}")
        for attr in key_attrs:
            value = getattr(self.config, attr)
            print(f"  {attr}: {value} (type: {type(value).__name__})")
        
        # 3. 打印完整的 unique_identifiers 字典
        print("\n[Complete Unique Identifiers Dict]")
        for key, value in self.unique_identifiers.items():
            if hasattr(value, 'unique_identifiers'):
                print(f"  {key}: {value} (has unique_identifiers attribute)")
                print(f"    -> {value.unique_identifiers}")
            else:
                print(f"  {key}: {value} (type: {type(value).__name__})")
        
        # 4. 打印 JSON 序列化前的字典（用于检查对象引用）
        print("\n[Dict Before JSON Serialization]")
        print(f"OrderedDict keys order: {list(self.unique_identifiers.keys())}")
        
        # 5. 打印 JSON 描述字符串（逐行）
        print("\n[JSON Description String]")
        print("--- START ---")
        print(self.unique_description)
        print("--- END ---")
        
        # 6. 打印 JSON 字符串的字节表示（前500字符）
        print("\n[JSON Bytes Representation (first 500 chars)]")
        json_bytes = self.unique_description.encode("utf-8")
        print(f"Total bytes length: {len(json_bytes)}")
        print(f"First 500 bytes: {json_bytes[:500]}")
        
        # 7. 打印最终哈希值
        print("\n[Final Hash]")
        print(f"MD5 Hash: {self.unique_description_hash}")
        
        # 8. 逐个字段重新计算哈希，检查哪个字段导致差异
        print("\n[Incremental Hash Calculation]")
        test_dict = OrderedDict()
        for key, value in self.unique_identifiers.items():
            test_dict[key] = value
            test_json = json.dumps(test_dict, indent=4, default=lambda obj: obj.unique_identifiers)
            test_hash = hashlib.md5(test_json.encode("utf-8"), usedforsecurity=False).hexdigest()
            print(f"  After adding '{key}': {test_hash[:16]}...")
        
        print("\n" + "=" * 80)


    @staticmethod
    def numel_low_level_dataset(low_level_dataset: LowLevelDataset) -> int:
        """Return the number of elements in the underlying low level dataset for the purpose of
        segregating the train/valid/test split indices

        It may be that the low level dataset can be split any number of ways, depending on the mid
        level dataset it supports, which is why we define the "number of elements" function
        separately from the __len__ function here in the mid level dataset class

        Args:
            low_level_dataset (LowLevelDataset): The underlying low level dataset

        Returns:
            int: The number of elements in the underlying low level dataset
        """
        raise NotImplementedError

    @staticmethod
    def build_low_level_dataset(
        dataset_path: str, config: BlendedMegatronDatasetConfig
    ) -> LowLevelDataset:
        """Build the low level dataset via a function to be called from within
        BlendedMegatronDatasetBuilder.build_generic_dataset

        It may be that the low level dataset spans any subset of train/valid/test splits, which is
        why we define a static "build" function separately from the constructor in the mid level
        dataset class

        Args:
            dataset_path (str): The real path on disk to the dataset

            config (BlendedMegatronDatasetConfig): The dataset config

        Returns:
            LowLevelDataset: The low level dataset
        """
        raise NotImplementedError

    @staticmethod
    def _key_config_attributes() -> List[str]:
        """Return all config attributes which contribute to uniquely identifying the dataset.

        These attributes will be used to build a uniquely identifying string and MD5 hash which
        will be used to cache/load dataset resources from run to run.

        Returns:
            List[str]: The key config attributes
        """
        return ["random_seed", "sequence_length", "split", "split_matrix", "tokenizer"]

    @abstractmethod
    def __len__(self) -> int:
        """Return the length of the dataset

        Returns:
            int: See abstract implementation
        """
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, numpy.ndarray]]:
        """Return from the dataset

        Args:
            idx (int): The index into the dataset

        Returns:
            Dict[str, Union[torch.Tensor, numpy.ndarray]]: See abstract implementation
        """
        pass
