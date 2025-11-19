import os
import json

def generate_weighted_json(folder_weights):
    """
    根据文件夹及其权重，按照子文件夹中的.bin文件总大小分配权重
    
    Args:
        folder_weights: 字典，格式为 {folder_path: weight}
    
    Returns:
        list: 包含路径和权重的字典列表
    """
    result = []
    
    for folder_path, folder_weight in folder_weights.items():
        if not os.path.exists(folder_path):
            print(f"警告: 路径不存在 {folder_path}")
            continue
            
        # 获取所有子文件夹（只统计文件夹，不统计文件）
        subfolders = [d for d in os.listdir(folder_path) 
                     if os.path.isdir(os.path.join(folder_path, d))]
        
        if not subfolders:
            print(f"警告: {folder_path} 下没有子文件夹")
            continue
        
        # 计算每个子文件夹的.bin文件总大小
        subfolder_file_sizes = {}
        total_size = 0
        
        for subfolder in subfolders:
            subfolder_path = os.path.join(folder_path, subfolder)
            # 计算所有.bin文件的总大小（字节）
            bin_size = sum([os.path.getsize(os.path.join(subfolder_path, f)) 
                             for f in os.listdir(subfolder_path) 
                             if os.path.isfile(os.path.join(subfolder_path, f)) and f.endswith('.bin')])
            subfolder_file_sizes[subfolder] = bin_size
            total_size += bin_size
        print(subfolder_file_sizes)
        # 按.bin文件总大小分配权重
        if total_size > 0:
            for subfolder, file_size in subfolder_file_sizes.items():
                subfolder_weight = folder_weight * (file_size / total_size)
                result.append({
                    "path": os.path.join(folder_path, subfolder),
                    "weight": subfolder_weight
                })
    
    return result

# 输入数据
# folder_weights = {
#     "/inspire/hdd/project/qproject-fundationmodel/liupengfei-24025/ttmi/Temp/Split/DATA_tokenized_qwen/book-final": 5,
#     "/inspire/hdd/project/qproject-fundationmodel/liupengfei-24025/ttmi/Temp/Split/DATA_tokenized_qwen/paper-final": 5,
# }
folder_weights = {
    "/inspire/hdd/project/qproject-fundationmodel/liupengfei-24025/ttmi/Temp/Split/DATA_tokenized_qwen/cc-final": 4280,
    "/inspire/hdd/project/qproject-fundationmodel/liupengfei-24025/ttmi/Temp/Split/DATA_tokenized_qwen/book-final": 251,
    "/inspire/hdd/project/qproject-fundationmodel/liupengfei-24025/ttmi/Temp/Split/DATA_tokenized_qwen/paper-final": 529,
    "/inspire/hdd/project/qproject-fundationmodel/liupengfei-24025/ttmi/Temp/Split/DATA_tokenized_qwen/code-final": 598,
    "/inspire/hdd/project/qproject-fundationmodel/liupengfei-24025/ttmi/Temp/Split/DATA_tokenized_qwen/math-final": 458,
}
output_path = "/inspire/ssd/project/qproject-fundationmodel/public/yiwei/megatron-workspace/source/Megatron-LM/scripts/qwen25_3b_from_scratch/dataset_config_stage_1_1.json"

# 生成结果
result = generate_weighted_json(folder_weights)
num = len(result)
print(f"total folders num: {num}")

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)