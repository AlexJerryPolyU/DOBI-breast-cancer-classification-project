"""
Configuration management for fNIR Base Model.

Defines all command-line arguments and default parameters.
"""

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN-LSTM model")
    #测试时使用路径 （无需修改）
    parser.add_argument('--excel_path', type=str, default="test_data/test.xlsx", help='Path to test data Excel file')
    parser.add_argument('--data_folder_root', type=str, default="test_data/11类数据/9类数据", help='Path to test data folder')





    # 需要修改的重要参数
    parser.add_argument("--model_module", type=str, default="lambda_net", help="Model module name")
    parser.add_argument("--model_class", type=str, default="CNNLSTMNet", help="Model class name")
    # 网络输入参数
    # parser.add_argument('--Recon_num_layers', type=int, default=1, choices=[1, 9], help="Number of layers in input image")
    parser.add_argument('--cnn_input_channels', type=int, default=3, choices=[1, 3],
                        help="1=original image single channel, 3=3x original image triple channel")

    # 在cnn_input_channels参数为1的基础上可以调试是否使用1转3的卷积
    parser.add_argument('--use_conv1to3', type=bool, default=False, choices=['True', 'False'],
                        help=" ")

    #Whether to use SE block，Required for lambda models
    parser.add_argument('--use_attention', type=bool, default=True, help='Whether to use attention mechanism')





    # 网络通用参数
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='Weight decay')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--hidden_size', type=int, default=64, help='Hidden size for LSTM')
    parser.add_argument('--num_lstm_layers', type=int, default=7, help='Number of LSTM layers')
    parser.add_argument('--output_size', type=int, default=1, help='Output size (1 for binary classification)')
    parser.add_argument('--input_size', type=int, default=128, help='Input size')


    # 数据集路径与npy文件名(data_path为npy路径) 训练使用
    parser.add_argument('--data_path', type=str, default='data/npy/fNIR_no_min_interpolation_0_denoising_0_normalization_2_enhancement_0_edge_detection_0', help='Root path to the dataset')
    parser.add_argument('--train_data', type=str, default='train_data.npy', help='Path to training data')
    parser.add_argument('--val_data', type=str, default='val_data.npy', help='Path to validation data')
    parser.add_argument('--test_data', type=str, default='test_data.npy', help='Path to test data')
    parser.add_argument('--train_labels', type=str, default='train_labels.npy', help='Path to training labels')
    parser.add_argument('--val_labels', type=str, default='val_labels.npy', help='Path to validation labels')
    parser.add_argument('--test_labels', type=str, default='test_labels.npy', help='Path to test labels')
    parser.add_argument('--train_names', type=str, default='train_names.npy', help='Path to training names')
    parser.add_argument('--val_names', type=str, default='val_names.npy', help='Path to validation names')
    parser.add_argument('--test_names', type=str, default='test_names.npy', help='Path to test names')

    # 结果保存路径
    parser.add_argument('--metrics_file', type=str, default='fNIR_no_min_i_0_d_0_n_2_e_0.xlsx',
                        help='Path to save training metrics')

    # 权重保存路径
    parser.add_argument('--weights_dir', type=str, default='weights', help='Directory to save model weights')

                
    return parser.parse_args()
