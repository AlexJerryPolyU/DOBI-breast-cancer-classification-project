"""
deal_mc_npy.py - Part of fNIR Base Model.
"""

import numpy as np
import pandas as pd
import os
from tqdm import tqdm  # 导入 tqdm 用于显示进度条
from config_data import parse_args  # 导入解析参数的脚本
from image_processing import ImageProcessor


def preprocess_and_save_npy(excel_path, sheet_names, data_folder, cache_folder, target_depth=22,
                            target_height=102, target_width=128, processor_args=None):
    # 创建 ImageProcessor 实例
    processor = ImageProcessor(**processor_args)

    # 加载存储图像名称和数据的 .npy 文件
    data_file_path = os.path.join(data_folder, 'datas.npy')

    try:
        # 加载数据字典
        data_data = np.load(data_file_path, allow_pickle=True).item()  # 加载字典
    except Exception as e:
        print(f"加载 .npy 文件时发生错误: {e}")
        return

    for sheet_name in sheet_names:
        # 从 Excel 文件加载当前 sheet 的数据
        df = pd.read_excel(excel_path, sheet_name=sheet_name)

        data = []
        labels = []
        names = []  # 用来存储文件名

        print(f"开始处理 {sheet_name} 数据...")

        # 遍历 DataFrame 中的每一行，加载对应的图像数据
        for idx in tqdm(range(len(df)), desc=f"Processing {sheet_name}", unit="file"):
            data_name = df.iloc[idx]['dcm_name']

            # 将 data_name 处理为与 names_data 中的 .mat 文件匹配
            mat_name = data_name.replace('.dcm', '.mat')  # 将 .dcm 转换为 .mat

            # 在 data_data 中查找当前文件名的图像数据
            try:
                # 通过 mat_name 查找对应的图像数据
                image = data_data[mat_name]  # 从 data_data 获取对应的图像数据
            except KeyError:
                print(f"警告: 文件 {data_name}（转换为 {mat_name}）在 datas.npy 中未找到，跳过此文件。")
                continue  # 如果找不到对应的文件名，跳过

            # 图像预处理
            resized_image = preprocess_image(image, data_name, sheet_name, processor, target_depth)

            # 将处理后的图像、标签和文件名添加到列表中
            data.append(resized_image)
            labels.append(df.iloc[idx]['tumor_nature'])
            names.append(data_name)  # 保存当前文件的名称

        # 转换为 numpy 数组
        # data = np.array(data, dtype=np.float32)
        data = np.array(data)
        labels = np.array(labels)
        names = np.array(names)  # 将文件名列表转换为 numpy 数组

        # 动态生成缓存文件夹名
        cache_folder_name = (f"{os.path.basename(cache_folder)}_interpolation_{processor.interpolation_method}_"
                             f"denoising_{processor.denoising_method}_normalization_{processor.normalization_method}_"
                             f"enhancement_{processor.enhancement_method}_edge_detection_{processor.edge_detection_method}")

        # 将新的缓存文件夹路径与原路径组合
        cache_folder_path = os.path.join(os.path.dirname(cache_folder), cache_folder_name)
        # 确保目标文件夹存在，如果不存在则创建它
        os.makedirs(cache_folder_path, exist_ok=True)

        # 保存为 .npy 文件
        np.save(f"{cache_folder_path}/{sheet_name}_data.npy", data)
        np.save(f"{cache_folder_path}/{sheet_name}_labels.npy", labels)
        np.save(f"{cache_folder_path}/{sheet_name}_names.npy", names)  # 保存文件名

        print(f"{sheet_name} 数据处理与保存完成。")


def preprocess_image(image, names, sheet_name, processor, target_depth):
    """
    处理单张图像的函数，包含重采样、去噪、标准化和增强。

    Args:
    - image: 输入图像
    - processor: 图像处理器实例
    - target_depth: 目标深度
    - single_layer: 是否为单层图像，默认为 False
    """
    # 使用优化后的 resize 方法调整尺寸
    resized_image = processor.resize_MC(image, target_depth, target_height=image.shape[0], target_width=image.shape[1])

    # 其他图像处理操作（例如去噪、标准化、增强等）
    resized_image = processor.denoise(resized_image)
    resized_image = processor.normalize(resized_image, names, sheet_name)
    resized_image = processor.enhance(resized_image)
    resized_image = processor.edge_detection(resized_image)

    return resized_image

def main():
    # 解析命令行参数
    args = parse_args()

    # 设置 processor_args，控制各种图像处理方法
    processor_args = {
        'interpolation_method': args.interpolation_method,
        'denoising_method': args.denoising_method,
        'normalization_method': args.normalization_method,
        'enhancement_method': args.enhancement_method
    }

    # 调用数据处理函数，并传递 single_layer 参数
    preprocess_and_save_npy(args.excel_path, args.sheet_names, args.data_folder, args.cache_folder,
                            target_depth=23, target_height=args.target_height,
                            target_width=args.target_width, processor_args=processor_args)

if __name__ == '__main__':
    main()
