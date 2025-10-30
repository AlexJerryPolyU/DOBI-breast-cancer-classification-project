"""
Model evaluation and testing script.

Phase 1 Implementation Only.
"""

import importlib
import torch
import numpy as np
import pandas as pd
from mpmath.libmp import normalize
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, matthews_corrcoef
import json

from xlwt.ExcelFormulaLexer import name_pattern

from dobi_dataset import CustomDataset
import os
from torch.utils.data import DataLoader
from config import parse_args
from dataset.data_npy import preprocess_and_save_npy_fNIR


def load_model_configs(config_path="model_configs.json"):
    """Load JSON configuration file"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model config file not found: {config_path}")
    with open(config_path, 'r') as f:
        return json.load(f)


def get_model_params(args, model_class, config_dict):
    """Merge common and extra parameters"""
    # Use combination of module name and class name as key
    config_key = f"{args.model_module}.{model_class}"

    if config_key not in config_dict:
        print(f"Warning: No extra parameters defined for {config_key} in config file. Using only args.")
        extra_params = {}
    else:
        extra_params = config_dict[config_key]

    # Common parameters (from args)
    common_params = {
        "input_size": args.input_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_lstm_layers,
        "output_size": args.output_size,
        "use_conv1to3": args.use_conv1to3,
        "use_attention": args.use_attention,
        "cnn_input_channels": args.cnn_input_channels
    }

    # Merge parameters (extra_params override common_params)
    model_params = {**common_params, **extra_params}

    return model_params


def create_model(module_name, class_name, args, config_path="model_configs.json"):
    """Create model instance"""
    config_dict = load_model_configs(config_path)
    module = importlib.import_module(f"model.{module_name}")

    model_class = getattr(module, class_name)
    model_params = get_model_params(args, class_name, config_dict)
    return model_class(**model_params)

def compute_sen_spec(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0

    return sensitivity, specificity


def test_model(args, weights_path, dataset_paths, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    model = create_model(args.model_module, args.model_class, args).to(device)
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False), strict=False)

    # Weight validation
    if unexpected_keys:
        raise ValueError(f"unexpected_keys：{unexpected_keys}")
    model.to(device)

    model.eval()
    results = {}
    wrong_predictions = {}

    for dataset_type, paths in dataset_paths.items():
        test_dataset = CustomDataset(
            data_path=paths['data'],
            labels_path=paths['labels'],
            names_path=paths['names'],
            channels = args.cnn_input_channels
        )
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)

        all_test_predicted = np.array([])
        all_test_predicted_score = np.array([])
        all_test_labels = np.array([])
        all_names = []

        wrong_names = []
        wrong_labels = []
        wrong_predictions_list = []
        wrong_probs = []

        with torch.no_grad():
            for img, label, name in test_loader:
                test_img = img.to(device)
                test_targets = label.unsqueeze(1).float().to(device)
                test_outputs = model(test_img)
                probs = torch.sigmoid(test_outputs).squeeze().cpu().numpy()
                preds = (probs >= 0.5).astype(int)

                all_test_predicted_score = np.append(all_test_predicted_score, probs)
                all_test_predicted = np.append(all_test_predicted, preds)
                all_test_labels = np.append(all_test_labels, test_targets.cpu().numpy().squeeze())
                all_names.extend(name)

                # 记录预测错误的样本
                # for i in range(len(name)):
                #     print(name)
                #     if preds[i] != test_targets[i].cpu().numpy():
                #         wrong_names.append(name[i])
                #         wrong_labels.append(int(test_targets[i].cpu().numpy().squeeze()))
                #         wrong_predictions_list.append(preds[i])
                #         wrong_probs.append(probs[i])

        df = pd.DataFrame({
            'Image Name': all_names,
            'True Label': all_test_labels,
            'Predicted Label': all_test_predicted,
            'Predicted Probability': all_test_predicted_score
        })
        results[dataset_type] = df

        if wrong_names:
            wrong_df = pd.DataFrame({
                'Image Name': wrong_names,
                'True Label': wrong_labels,
                'Predicted Label': wrong_predictions_list,
                'Predicted Probability': wrong_probs
            })
            wrong_predictions[dataset_type] = wrong_df

        acc = accuracy_score(all_test_labels, all_test_predicted)
        sen, spe = compute_sen_spec(all_test_labels, all_test_predicted)
        fpr, tpr, _ = roc_curve(all_test_labels, all_test_predicted_score)
        auc_score = auc(fpr, tpr)
        mcc = matthews_corrcoef(all_test_labels, all_test_predicted)

        print(f"Dataset: {dataset_type}")
        print(f"Accuracy: {acc:.4f}")
        print(f"Sensitivity (Sen): {sen:.4f}")
        print(f"Specificity (Spe): {spe:.4f}")
        print(f"AUC: {auc_score:.4f}")
        print(f"MCC: {mcc:.4f}\n")

    return results, wrong_predictions


def save_results_to_excel(results, excel_output_path, wrong_predictions, wrong_excel_output_path):
    with pd.ExcelWriter(excel_output_path) as writer:
        for sheet_name, df in results.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"所有预测结果已保存到 {excel_output_path}")

    # with pd.ExcelWriter(wrong_excel_output_path) as writer:
    #     for sheet_name, df in wrong_predictions.items():
    #         df.to_excel(writer, sheet_name=sheet_name, index=False)
    # print(f"预测错误的样本已保存到 {wrong_excel_output_path}")


def test_model_88(args, weights_path, data_path, name_path, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    model = create_model(args.model_module, args.model_class, args).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False), strict=False)
    model.to(device)
    model.eval()

    # Create dataset without labels
    test_dataset = CustomDataset(
        data_path=data_path,
        names_path=name_path,
        channels=args.cnn_input_channels
    )
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    all_predicted = np.array([])
    all_predicted_score = np.array([])
    all_names = []

    with torch.no_grad():
        for img, name in test_loader:
            test_img = img.to(device)
            test_outputs = model(test_img)
            probs = torch.sigmoid(test_outputs).squeeze().cpu().numpy()
            preds = (probs >= 0.5).astype(int)

            all_predicted_score = np.append(all_predicted_score, probs)
            all_predicted = np.append(all_predicted, preds)
            all_names.extend(name)

    # Create results DataFrame without true labels
    results_df = pd.DataFrame({
        'Image Name': all_names,
        'Predicted Label': all_predicted,
        'Predicted Probability': all_predicted_score
    })

    return results_df

def save_results_to_excel_88(results_df, excel_output_path):
    results_df.to_excel(excel_output_path, index=False)
    print(f"预测结果已保存到 {excel_output_path}")

# 示例调用
if __name__ == '__main__':

    "————————————————————————————三期数据验证————————————————————————"
    # 对于新三期数据的测试
    # args = parse_args()
    # # 不需要修改
    # dataset_paths = 'npy/sanqi_datas.npy'
    # name_paths = 'npy/sanqi_names.npy'
    #
    # # 定义固定的文件路径和参数
        # data_folder = "data/dNIR" #对应Path to test data folder
    #
    #
    # # 修改数据处理
    # variable_name = "dNIR"
    # normalize = 0
    # aggregation_method = 'max'
    # # 权重修改
    # weights_path = "epoch/epoch_026.pth"  # 最优epoch
    # # 模型结构参数
    # args.use_conv1to3 = False
    # args.use_attention = False
    # args.model_module = "model"
    # args.model_class = "CNNLSTMNet"
    # args.cnn_input_channels = 3
    #
    # preprocess_and_save_npy_fNIR(
    #     excel_path=excel_path,
    #     variable_name=variable_name,
    #     data_folder=data_folder,
    #     nor=normalize,
    #     image_aggregation_method=aggregation_method
    #
    # )
    #
    # # 检查文件是否存在，如果不存在则生成数据
    # # if not (os.path.exists(dataset_paths) and os.path.exists(name_paths)):
    # #     print("数据文件不存在，正在生成数据...")
    # #     preprocess_and_save_npy_fNIR(
    # #         excel_path=excel_path,
    # #         variable_name=variable_name,
    # #         data_folder=data_folder,
    # #         nor=normalize,
    # #         image_aggregation_method=aggregation_method
    # #
    # #     )
    # # else:
    # #     print("数据文件已存在，跳过数据生成步骤。")
    #
    # # Get the config key and extract values from the JSON config
    # config_dict = load_model_configs("model_configs.json")
    # config_key = f"{args.model_module}.{args.model_class}"
    # extra_params = config_dict.get(config_key, {})  # Default to empty dict if key not found
    # config_values = "_".join(str(value) for value in extra_params.values())  # Join values with "_"
    #
    # # 执行测试和保存结果
    # weights_dir = os.path.join(
    #     "prob",
    #     f"{args.model_module}_{args.model_class}_conv1to3{args.use_conv1to3}_attention{args.use_attention}",  # 添加网络名，例如 "CNNLSTMNet"
    #     f"{variable_name}_{aggregation_method}_n{normalize}_channel{args.cnn_input_channels}_{config_values}",  # Use config values or "default" if empty
    #     f"bs_{args.batch_size}_lr_{args.learning_rate}_epoch_{args.num_epochs}_wd_{args.weight_decay}_hs_{args.hidden_size}_nl_{args.num_lstm_layers}"
    # )
    # if not os.path.exists(weights_dir):
    #     os.makedirs(weights_dir)
    # excel_out_path = os.path.join(weights_dir, 'predictions.xlsx')
        


    "————————————————————————————一期数据验证————————————————————————"
    # 测试一期fNIR是否正确
    args = parse_args()
    args.use_conv1to3 = False
    args.use_attention = False
    args.cnn_input_channels = 3
    args.model_module = "lambda_net"
    args.model_class = "CNNLSTMNet"
    weights_path = "epoch/epoch_023.pth"
    path = r'D:\fNIR_base_model\data\npy\fNIR_no_min_interpolation_0_denoising_0_normalization_2_enhancement_0_edge_detection_0'
    dataset_paths = {
        "train": {
            "data": f"{path}/train_data.npy",
            "labels": f"{path}/train_labels.npy",
            "names": f"{path}/train_names.npy"
        },
        "val": {
            "data": f"{path}/val_data.npy",
            "labels": f"{path}/val_labels.npy",
            "names": f"{path}/val_names.npy"
        },
        "test": {
            "data": f"{path}/test_data.npy",
            "labels": f"{path}/test_labels.npy",
            "names": f"{path}/test_names.npy"
        }
    }

    results, wrong_predictions = test_model(args, weights_path, dataset_paths)
    save_results_to_excel(results, "predictions.xlsx", wrong_predictions, "wrong_predictions.xlsx")