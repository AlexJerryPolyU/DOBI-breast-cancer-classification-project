"""
data_npy.py - Part of fNIR Base Model.
"""

import pandas as pd
import scipy.io as sio
import os

from tqdm import tqdm  # 导入 tqdm 用于显示进度条
from .config_data import parse_args  # 导入解析参数的脚本
from .image_processing import ImageProcessor
import numpy as np
import matplotlib.pyplot as plt


def preprocess_and_save_npy_fNIR(excel_path, variable_name, data_folder, target_depth=23, single_layer=True, nor=2,image_aggregation_method='min'):
    """
    预处理图像数据并保存为 .npy 文件（针对单个默认 sheet）

    参数：
        excel_path (str): Excel 文件路径
        variable_name (str): .mat 文件中的变量名
        data_folder (str): 存储数据的文件夹路径
        target_depth (int): 目标深度
        single_layer (bool): 是否使用单层
    """
    # 从 Excel 文件加载默认的第一个 sheet 数据
    df = pd.read_excel(excel_path, sheet_name=0)  # 固定读取第一个 sheet
    processor = ImageProcessor(normalization_method=nor)
    data = []
    names = []  # 用来存储文件名

    print("开始处理数据...")

    # 遍历 DataFrame 中的每一行，加载所有的 .mat 文件
    for idx in tqdm(range(len(df)), desc="Processing", unit="file"):
        data_name = df.iloc[idx]['dcm_name']
        file_path = f"{data_folder}/{data_name.replace('.dcm', '.mat')}"

        try:
            # 加载 .mat 文件
            mat_data = sio.loadmat(file_path)
            if variable_name not in mat_data:
                print(f"警告: 变量 '{variable_name}' 在 {data_name} 的 .mat 文件中未找到，跳过此文件。")
                continue  # 如果变量未找到，跳过该文件

            # 提取图像数据
            image = mat_data[variable_name]
            # 使用 min 聚合方法（固定）
            if image_aggregation_method == 'min':
                image = np.min(image, axis=-1)
            elif image_aggregation_method == 'max':
                image = np.max(image, axis=-1)
            elif image_aggregation_method == 'median':
                image = np.median(image, axis=-1)
            elif image_aggregation_method == 'sum':
                image = np.sum(image, axis=-1)
            elif image_aggregation_method == 'std':
                image = np.std(image, axis=-1)
            elif image_aggregation_method == 'mean':
                image = np.mean(image, axis=-1)

            if variable_name == 'dNIR':
                zero_frame = np.zeros((102, 128, 1), dtype=np.float32)
                image = np.concatenate((zero_frame, image), axis=2)


            resized_image = processor.resize(image, target_depth, target_height=image.shape[0],
                                             target_width=image.shape[1], single_layer=single_layer)
            resized_image = processor.normalize(resized_image, data_name)


            # 将处理后的图像和文件名添加到列表中
            data.append(resized_image)
            names.append(data_name)

        except Exception as e:
            print(f"处理 {data_name} 时发生错误: {e}")

    # 转换为 numpy 数组
    data = np.array(data)
    names = np.array(names)

    # 保存为 .npy 文件
    np.save("npy/sanqi_datas.npy", data)
    np.save("npy/sanqi_names.npy", names)

    print("数据处理与保存完成。")



def preprocess_and_save_npy(excel_path, sheet_names, variable_name, data_folder, cache_folder, target_depth=23,
                            target_height=102, target_width=128, processor_args=None, single_layer=False,
                            image_aggregation_method='min'):
    """
    预处理图像数据并保存为 .npy 文件

    参数：
        excel_path (str): Excel 文件路径
        sheet_names (list): sheet 名称列表
        variable_name (str): .mat 文件中的变量名
        data_folder (str): 存储数据的文件夹路径
        cache_folder (str): 缓存文件夹路径
        target_depth (int): 目标深度
        target_height (int): 目标高度
        target_width (int): 目标宽度
        processor_args (dict): 图像处理相关的参数
        single_layer (bool): 是否使用单层
        image_aggregation_method (str): 控制图像聚合方法，选择 'min', 'max', 'median', 'sum', 'std'
    """
    # 创建 ImageProcessor 实例
    processor = ImageProcessor(**processor_args)

    # 创建缓存文件夹
    # os.makedirs(cache_folder, exist_ok=True)

    for sheet_name in sheet_names:
        # 从 Excel 文件加载当前 sheet 的数据
        df = pd.read_excel(excel_path, sheet_name=sheet_name)

        data = []
        labels = []
        names = []  # 用来存储文件名

        print(f"开始处理 {sheet_name} 数据...")

        # 遍历 DataFrame 中的每一行，加载所有的 .mat 文件
        for idx in tqdm(range(len(df)), desc=f"Processing {sheet_name}", unit="file"):
            data_name = df.iloc[idx]['dcm_name']
            file_path = f"{data_folder}/{data_name.replace('.dcm', '.mat')}"

            try:
                # 加载 .mat 文件
                mat_data = sio.loadmat(file_path)
                if variable_name not in mat_data:
                    print(f"警告: 变量 '{variable_name}' 在 {data_name} 的 .mat 文件中未找到，跳过此文件。")
                    continue  # 如果变量未找到，跳过该文件

                # 提取图像数据
                image = mat_data[variable_name]

                # (rfdNIR)
                # if variable_name == 'dNIR':
                #     height, width = image[0, 0].shape  # 假设每个数据的形状为 (102, 128)
                #     channels, depth = image.shape  # depth=23, channels=3
                #
                #     # # 初始化目标数组
                #     reshaped_data = np.zeros((height, width, depth, channels))
                #
                #     # # 填充数据
                #     for c in range(channels):
                #         for d in range(depth):
                #             reshaped_data[:, :, d, c] = image[c, d]
                #     image = reshaped_data

                # 根据指定的图像聚合方法选择相应的操作
                if image_aggregation_method == 'min':
                    image = np.min(image, axis=-1)
                elif image_aggregation_method == 'max':
                    image = np.max(image, axis=-1)
                elif image_aggregation_method == 'median':
                    image = np.median(image, axis=-1)
                elif image_aggregation_method == 'sum':
                    image = np.sum(image, axis=-1)
                elif image_aggregation_method == 'std':
                    image = np.std(image, axis=-1)
                elif image_aggregation_method == 'mean':
                    image = np.mean(image, axis=-1)
                # else:
                #     raise ValueError(f"不支持的图像聚合方法: {image_aggregation_method}")
                # print(f'归一化方法{image_aggregation_method}')
                # 图像预处理

                # (dNIR)
                # if variable_name == 'dNIR':
                #     zero_frame = np.zeros((102, 128, 1), dtype=np.float32)
                #     image = np.concatenate((zero_frame, image), axis=2)


                resized_image = preprocess_image(image, data_name, sheet_name, processor, target_depth, single_layer)
                # 将处理后的图像、标签和文件名添加到列表中
                data.append(resized_image)
                names.append(data_name)  # 保存当前文件的名称
                labels.append(df.iloc[idx]['tumor_nature'])

            except Exception as e:
                print(f"处理 {data_name} 时发生错误: {e}")

        # 转换为 numpy 数组
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


def preprocess_image(image, names, sheet_name, processor, target_depth, single_layer=False):
    """
    处理单张图像的函数，包含重采样、去噪、标准化和增强。

    Args:
    - image: 输入图像
    - processor: 图像处理器实例
    - target_depth: 目标深度
    - single_layer: 是否为单层图像，默认为 False
    """
    # 使用优化后的 resize 方法调整尺寸
    resized_image = processor.resize(image, target_depth, target_height=image.shape[0], target_width=image.shape[1], single_layer=single_layer)
    # 其他图像处理操作（例如去噪、标准化、增强等）
    resized_image = processor.denoise(resized_image)
    resized_image = processor.normalize(resized_image, names)
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
        'enhancement_method': args.enhancement_method,
        'edge_detection_method': args.edge_detection_method
    }

    # 调用数据处理函数，并传递 single_layer 参数
    preprocess_and_save_npy(args.excel_path, args.sheet_names, args.variable_name, args.data_folder, args.cache_folder,
                            target_depth=args.target_depth, target_height=args.target_height,
                            target_width=args.target_width,
                            processor_args=processor_args, single_layer=args.single_layer, image_aggregation_method=args.image_aggregation_method)


if __name__ == '__main__':
    main()
