"""
config_data.py - Part of fNIR Base Model.
"""

import argparse

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='Process .mat files and save as .npy files')

    # 添加参数
    # # 操作excel路径
        #                     help='Excel 文件路径（默认为 "data/excel/phase1exp.xlsx"）')
    parser.add_argument('--excel_path', type=str, default='data/excel/CVdsub_二期双10_20250212.xlsx',
                        help='Excel 文件路径（默认为 "data/excel/phase1exp.xlsx"）')
    # parser.add_argument('--excel_path', type=str, default='data/excel/phase1exp.xlsx',
    #                     help='Excel 文件路径（默认为 "data/excel/phase1exp.xlsx"）')
    # parser.add_argument('--sheet_names', type=str, nargs='+', default=["Sheet1"],
    #                     help='Excel 中的 sheet 名称（默认为 "train", "val", "test"）')

    parser.add_argument('--sheet_names', type=str, nargs='+', default=["train", "val", "test"],
                        help='Excel 中的 sheet 名称（默认为 "train", "val", "test"）')


    # 数据路径
    parser.add_argument('--single_layer', type=bool, default=False, help='重建层数（默认为 True：一层重建）')
    parser.add_argument('--variable_name', type=str, default='img4D', help='.mat 文件中的变量名')
    parser.add_argument('--data_folder', type=str,  default='data/source data2/fdDOT_DynamicMC_fix_ellips', help='数据文件夹路径')
    parser.add_argument('--cache_folder', type=str, default='data/npy2/fdDOT_DynamicMC_fix_ellips', help='缓存文件夹路径（默认为 "data"）')


    # 图像尺寸
    parser.add_argument('--target_depth', type=int, default=23, help='目标深度（默认为 23）')
    parser.add_argument('--target_height', type=int, default=102, help='目标高度（默认为 102）')
    parser.add_argument('--target_width', type=int, default=128, help='目标宽度（默认为 128）')


    # 预处理方法
    parser.add_argument('--image_aggregation_method', type=str, default=None, help='归一化方法（默认为 ：min）')
    parser.add_argument('--interpolation_method', type=int, default=0, help='插值方法（默认为 3：Cubic）')
    parser.add_argument('--denoising_method', type=int, default=0, help='去噪方法（默认为 1：高斯去噪）')
    parser.add_argument('--normalization_method', type=int, default=0, help='标准化方法（默认为 2：Z-score）')
    parser.add_argument('--enhancement_method', type=int, default=0, help='增强方法（默认为 1：CLAHE）')
    parser.add_argument('--edge_detection_method', type=int, default=0, help='边缘方法（默认为 0：不使用）')
    args = parser.parse_args()
    return args

