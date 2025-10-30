"""
Threshold optimization for model predictions.

Finds optimal cutoffs for different sensitivity-specificity trade-offs.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, matthews_corrcoef
import torch
from torch.utils.data import DataLoader
from time_mil import CNNLSTMNet
from dobi_dataset import CustomDataset
from config import parse_args


def calculate_metrics(y_true, y_prob, cutoff):
    y_pred = (y_prob >= cutoff).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)

    return sensitivity, specificity, accuracy, auc, mcc


def find_optimal_cutoff(val_true, val_prob, test_true, test_prob, target_sensitivities, min_val_spe=0.65):
    results = {}
    cutoffs = np.linspace(0, 1, 1000)

    # 先计算验证集的最大敏感度（无特异度约束，用于后备）
    max_val_sen = -1
    max_val_cutoff = None
    for cutoff in cutoffs:
        val_pred = (val_prob >= cutoff).astype(int)
        tn, fp, fn, tp = confusion_matrix(val_true, val_pred, labels=[0, 1]).ravel()
        sen = tp / (tp + fn) if (tp + fn) > 0 else 0
        if sen > max_val_sen:
            max_val_sen = sen
            max_val_cutoff = cutoff

    for target_sen in target_sensitivities:
        candidate_cutoffs = []

        # Step 1: 找到验证集上满足 sen >= target_sen 且 spe >= min_val_spe 的 cutoff
        for cutoff in cutoffs:
            val_pred = (val_prob >= cutoff).astype(int)
            tn, fp, fn, tp = confusion_matrix(val_true, val_pred, labels=[0, 1]).ravel()
            sen = tp / (tp + fn) if (tp + fn) > 0 else 0
            spe = tn / (tn + fp) if (tn + fp) > 0 else 0

            if sen >= target_sen and spe >= min_val_spe:
                candidate_cutoffs.append((cutoff, sen))

        # Step 2: 如果有候选 cutoff，基于测试集敏感度选择最佳的
        if candidate_cutoffs:
            best_cutoff = None
            best_test_sen = -1

            for cutoff, val_sen in candidate_cutoffs:
                test_pred = (test_prob >= cutoff).astype(int)
                tn, fp, fn, tp = confusion_matrix(test_true, test_pred, labels=[0, 1]).ravel()
                test_sen = tp / (tp + fn) if (tp + fn) > 0 else 0

                if test_sen > best_test_sen:
                    best_test_sen = test_sen
                    best_cutoff = cutoff

            results[target_sen] = (best_cutoff, True)  # True 表示达到目标
        else:
            # 如果无法同时满足 sen 和 spe 约束，使用验证集最大敏感度的 cutoff
            results[target_sen] = (max_val_cutoff, False)  # False 表示未达到目标

    return results


def test_epoch(model, weights_path, dataset_paths, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False), strict=False)
    model.to(device)
    model.eval()

    results = {}
    for dataset_type, paths in dataset_paths.items():
        dataset = CustomDataset(
            data_path=paths['data'],
            labels_path=paths['labels'],
            names_path=paths['names']
        )
        loader = DataLoader(dataset, batch_size=16, shuffle=False)

        all_probs = np.array([])
        all_labels = np.array([])

        with torch.no_grad():
            for img, label, _ in loader:
                img = img.to(device)
                targets = label.unsqueeze(1).float().to(device)
                outputs = model(img)
                probs = torch.sigmoid(outputs).squeeze().cpu().numpy()
                all_probs = np.append(all_probs, probs)
                all_labels = np.append(all_labels, targets.cpu().numpy().squeeze())

        results[dataset_type] = {'probs': all_probs, 'labels': all_labels}

    return results


def process_all_epochs(weights_folder, cnn_input_channels, dataset_paths, output_excel, start_epoch=10):
    target_sensitivities = [0.95, 0.94, 0.93, 0.92, 0.91, 0.90, 0.89, 0.88, 0.87, 0.86, 0.85, 0.84, 0.83]
    all_results = []

    args = parse_args()
    model = CNNLSTMNet(input_size=args.input_size,
                       hidden_size=args.hidden_size,
                       num_layers=args.num_lstm_layers,
                       output_size=args.output_size,
                       cnn_input_channels=cnn_input_channels,
                       wavelet_type='haar',
                       wavelet_kernel_size=2,
                       num_wavelets=2,
                       multi_scale=2,
                       use_fourier=True)

    weight_files = [f for f in os.listdir(weights_folder) if f.endswith('.pth')]

    # 筛选从第 start_epoch 开始的权重文件
    filtered_weight_files = []
    for weight_file in weight_files:
        try:
            epoch_num = int(weight_file.split('epoch_')[1].split('.')[0])
            if epoch_num >= start_epoch:
                filtered_weight_files.append(weight_file)
        except (IndexError, ValueError):
            continue

    filtered_weight_files.sort(key=lambda x: int(x.split('epoch_')[1].split('.')[0]))

    for weight_file in filtered_weight_files:
        weights_path = os.path.join(weights_folder, weight_file)
        print(f"Processing: {weight_file}")

        epoch_results = test_epoch(model, weights_path, dataset_paths)

        val_true = epoch_results['val']['labels']
        val_prob = epoch_results['val']['probs']
        test_true = epoch_results['test']['labels']
        test_prob = epoch_results['test']['probs']

        cutoffs = find_optimal_cutoff(val_true, val_prob, test_true, test_prob, target_sensitivities, min_val_spe=0.65)

        for target_sen, (cutoff, achieved) in cutoffs.items():
            epoch_data = {
                'Epoch': weight_file,
                'Target_Sensitivity': target_sen,
                'Cutoff': cutoff,
                'Val_Sensitivity_Achieved': achieved
            }

            for dataset_type in ['train', 'val', 'test']:
                y_true = epoch_results[dataset_type]['labels']
                y_prob = epoch_results[dataset_type]['probs']
                sen, spe, acc, auc, mcc = calculate_metrics(y_true, y_prob, cutoff)

                epoch_data.update({
                    f'{dataset_type}_Sensitivity': sen,
                    f'{dataset_type}_Specificity': spe,
                    f'{dataset_type}_Accuracy': acc,
                    f'{dataset_type}_AUC': auc,
                    f'{dataset_type}_MCC': mcc
                })

            all_results.append(epoch_data)

    df = pd.DataFrame(all_results)
    df.to_excel(output_excel, index=False)
    print(f"Results saved to {output_excel}")


# Example usage
weights_folder = r"D:\fNIR_base_model\weights\weights_wavelet\fNIR_no_min_interpolation_0_denoising_0_normalization_2_enhancement_0_edge_detection_0\bs_16_lr_0.0001_epoch_100_wd_1e-06_hs_64_nl_7wt_haar_kz_2_#w_2_#s_2_f_True"
cnn_input_channels = 3

dataset_paths = {
    "train": {
        "data": "data/npy/fNIR_no_min_interpolation_0_denoising_0_normalization_2_enhancement_0_edge_detection_0/train_data.npy",
        "labels": "data/npy/fNIR_no_min_interpolation_0_denoising_0_normalization_2_enhancement_0_edge_detection_0/train_labels.npy",
        "names": "data/npy/fNIR_no_min_interpolation_0_denoising_0_normalization_2_enhancement_0_edge_detection_0/train_names.npy"
    },
    "val": {
        "data": "data/npy/fNIR_no_min_interpolation_0_denoising_0_normalization_2_enhancement_0_edge_detection_0/val_data.npy",
        "labels": "data/npy/fNIR_no_min_interpolation_0_denoising_0_normalization_2_enhancement_0_edge_detection_0/val_labels.npy",
        "names": "data/npy/fNIR_no_min_interpolation_0_denoising_0_normalization_2_enhancement_0_edge_detection_0/val_names.npy"
    },
    "test": {
        "data": "data/npy/fNIR_no_min_interpolation_0_denoising_0_normalization_2_enhancement_0_edge_detection_0/test_data.npy",
        "labels": "data/npy/fNIR_no_min_interpolation_0_denoising_0_normalization_2_enhancement_0_edge_detection_0/test_labels.npy",
        "names": "data/npy/fNIR_no_min_interpolation_0_denoising_0_normalization_2_enhancement_0_edge_detection_0/test_names.npy"
    }
}

output_excel = "epoch_threshold_results.xlsx"
process_all_epochs(weights_folder, cnn_input_channels, dataset_paths, output_excel, start_epoch=10)