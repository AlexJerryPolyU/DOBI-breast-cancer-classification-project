import os
import torch
from torch.utils.data import DataLoader
from model.time_mil_new import CNNLSTMNet
from sklearn.metrics import roc_curve, auc, accuracy_score, matthews_corrcoef
import pandas as pd
from tqdm import tqdm
import numpy as np
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn
from config import parse_args
from dobi_dataset import CustomDataset
from unit.metrics import compute_sen_spec
import random
import itertools

# 默认随机种子
seed = 42


# 随机性控制
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

# 复现结果优先
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# Main training function
def train(args, hyperparams):
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    print("=======================【开始训练】===============================")

    # Update the paths dynamically based on arguments
    train_dataset = CustomDataset(
        data_path=os.path.join(args.data_path, args.train_data),
        labels_path=os.path.join(args.data_path, args.train_labels),
        names_path=os.path.join(args.data_path, args.train_names),
        channels=args.cnn_input_channels

    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    val_dataset = CustomDataset(
        data_path=os.path.join(args.data_path, args.val_data),
        labels_path=os.path.join(args.data_path, args.val_labels),
        names_path=os.path.join(args.data_path, args.val_names),
        channels=args.cnn_input_channels
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    test_dataset = CustomDataset(
        data_path=os.path.join(args.data_path, args.test_data),
        labels_path=os.path.join(args.data_path, args.test_labels),
        names_path=os.path.join(args.data_path, args.test_names),
        channels=args.cnn_input_channels
    )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)



    # 根据 num_layers 设定输入通道数
    # cnn_input_channels = args.Recon_num_layers  # 如果是1层或9层图像，设置相应的输入通道数

    print(hyperparams)
    # 初始化模型
    model = CNNLSTMNet(input_size=args.input_size,
                       hidden_size=args.hidden_size,
                       num_layers=args.num_lstm_layers,
                       output_size=args.output_size,
                       cnn_input_channels=args.cnn_input_channels,
                       **hyperparams
    )


    model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(1.9167 + 0.3).to(device))
    optimizer = optim.RAdam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=0.00001)

    # Directory for saving model checkpoints
    dataset_name = os.path.basename(args.data_path)
    # wavelet
    weights_dir = os.path.join(args.weights_dir, dataset_name,
                               f"bs_{args.batch_size}_lr_{args.learning_rate}_epoch_{args.num_epochs}_wd_{args.weight_decay}_hs_{args.hidden_size}_nl_{args.num_lstm_layers}"
                               f"wt_{hyperparams["wavelet_type"]}_kz_{hyperparams["wavelet_kernel_size"]}_#w_{hyperparams["num_wavelets"]}_#s_{hyperparams["multi_scale"]}_f_{hyperparams["use_fourier"]}_ls{hyperparams["num_recursive_layers"]}")

    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    # Initialize metrics storage
    metrics_data = []

    # Training Loop
    for epoch in range(args.num_epochs):
        model.train()
        all_train_predicted = np.array([])  # Predicted labels
        all_train_predicted_score = np.array([])  # Predicted scores
        all_train_labels = np.array([])  # True labels

        epoch_train_loss = 0.0
        epoch_val_loss = 0.0

        train_loader_tqdm = tqdm(train_loader, disable=False)
        # for idx, (img, label, name, roi) in enumerate(train_loader_tqdm):
        for idx, (img, label, name) in enumerate(train_loader_tqdm):
            train_img = img.to(device)
            # roi = roi.to(device)
            targets = label.unsqueeze(1).float().to(device)
            optimizer.zero_grad()
            # outputs = model(train_img, roi)
            outputs = model(train_img)
            train_loss = criterion(outputs, targets)
            epoch_train_loss += train_loss.item()
            train_loss.backward()

            optimizer.step()

            all_train_predicted_score = np.append(all_train_predicted_score,
                                                  torch.sigmoid(outputs).squeeze().cpu().detach().numpy())
            all_train_predicted = np.append(all_train_predicted,
                                            (torch.sigmoid(outputs) >= 0.5).squeeze().cpu().numpy().astype(int))
            all_train_labels = np.append(all_train_labels, targets.cpu().numpy().squeeze())

            # Clear unnecessary variables to avoid memory buildup
            del img, targets, outputs
            torch.cuda.empty_cache()

        scheduler.step()

        # Calculate metrics for training
        train_accuracy = accuracy_score(all_train_labels, all_train_predicted)
        train_sen, train_spe = compute_sen_spec(all_train_labels, all_train_predicted)
        fpr, tpr, _ = roc_curve(all_train_labels, all_train_predicted_score)
        train_auc = auc(fpr, tpr)
        train_mcc = matthews_corrcoef(all_train_labels, all_train_predicted)

        # Validation Loop
        model.eval()
        with torch.no_grad():
            all_val_predicted = np.array([])
            all_val_predicted_score = np.array([])
            all_val_labels = np.array([])

            val_loader_tqdm = tqdm(val_loader, disable=False)
            # for idx, (img, label, name, roi) in enumerate(val_loader_tqdm):
            for idx, (img, label, name) in enumerate(val_loader_tqdm):
                val_img = img.to(device)
                # roi = roi.to(device)
                val_targets = label.unsqueeze(1).float().to(device)
                # val_outputs = model(val_img, roi)
                val_outputs = model(val_img)
                val_loss = criterion(val_outputs, val_targets)
                epoch_val_loss += val_loss.item()

                all_val_predicted = np.append(all_val_predicted,
                                              (torch.sigmoid(val_outputs) >= 0.5).squeeze().cpu().numpy().astype(int))
                all_val_predicted_score = np.append(all_val_predicted_score,
                                                    torch.sigmoid(val_outputs).squeeze().cpu().numpy())
                all_val_labels = np.append(all_val_labels, val_targets.cpu().numpy().squeeze())

                # Clear unnecessary variables to avoid memory buildup
                del val_img, val_targets, val_outputs
                torch.cuda.empty_cache()

        # Calculate metrics for validation
        val_accuracy = accuracy_score(all_val_labels, all_val_predicted)
        val_sen, val_spe = compute_sen_spec(all_val_labels, all_val_predicted)
        fpr, tpr, _ = roc_curve(all_val_labels, all_val_predicted_score)
        val_auc = auc(fpr, tpr)
        val_mcc = matthews_corrcoef(all_val_labels, all_val_predicted)

        # 计算 train-val_mcc（train_mcc - val_mcc 的绝对值）
        train_val_mcc = abs(train_mcc - val_mcc)
        train_val_loss = abs((epoch_train_loss/ len(train_loader)) - (epoch_val_loss / len(val_loader)))

        # Test Loop
        # Test Loop
        all_test_predicted = np.array([])  # Predicted labels
        all_test_predicted_score = np.array([])  # Predicted scores
        all_test_labels = np.array([])  # True labels
        test_loss = 0.0  # Initialize test loss

        with torch.no_grad():
            # for idx, (img, label, name, roi) in enumerate(test_loader):
            for idx, (img, label, name) in enumerate(test_loader):
                test_img = img.to(device)
                test_targets = label.unsqueeze(1).float().to(device)
                # roi = roi.to(device)
                # test_outputs = model(test_img, roi)
                test_outputs = model(test_img)
                # Compute loss for this batch
                loss = criterion(test_outputs, test_targets)
                test_loss += loss.item()

                all_test_predicted_score = np.append(all_test_predicted_score,
                                                     torch.sigmoid(test_outputs).squeeze().cpu().numpy())
                all_test_predicted = np.append(all_test_predicted,
                                               (torch.sigmoid(test_outputs) >= 0.5).squeeze().cpu().numpy().astype(int))
                all_test_labels = np.append(all_test_labels, test_targets.cpu().numpy().squeeze())

                # Clear unnecessary variables to avoid memory buildup
                del test_img, test_targets, test_outputs
                torch.cuda.empty_cache()

        # Calculate metrics for test
        test_accuracy = accuracy_score(all_test_labels, all_test_predicted)
        test_sen, test_spe = compute_sen_spec(all_test_labels, all_test_predicted)
        fpr, tpr, _ = roc_curve(all_test_labels, all_test_predicted_score)
        test_auc = auc(fpr, tpr)
        test_mcc = matthews_corrcoef(all_test_labels, all_test_predicted)



        # Save model checkpoint
        torch.save(model.state_dict(), f'{weights_dir}/epoch_{epoch + 1:03}.pth')

        # Save metrics to Excel
        metrics_data.append({
            'Epoch': epoch + 1,
            'Train_Loss': epoch_train_loss / len(train_loader),
            'Val_Loss': epoch_val_loss / len(val_loader),
            'Test_Loss': test_loss / len(test_loader),  # Save test loss
            'Train_Accuracy': train_accuracy,
            'Train_Sensitivity': train_sen,
            'Train_Specificity': train_spe,
            'Train_AUC': train_auc,
            'Train_MCC': train_mcc,
            'Val_Accuracy': val_accuracy,
            'Val_Sensitivity': val_sen,
            'Val_Specificity': val_spe,
            'Val_AUC': val_auc,
            'Val_MCC': val_mcc,
            'Test_Accuracy': test_accuracy,
            'Test_Sensitivity': test_sen,
            'Test_Specificity': test_spe,
            'Test_AUC': test_auc,
            'Test_MCC': test_mcc,
            'Train-Val_MCC': train_val_mcc,
            'Train-Val_Loss':train_val_loss,
        })

        # 每个epoch输出一下结果
        print(f"Epoch [{epoch + 1}/{args.num_epochs}], "
              f"Train Loss: {epoch_train_loss / len(train_loader):.4f}, "
              f"Val Loss: {epoch_val_loss / len(val_loader):.4f}, "
              f"Test Loss: {test_loss / len(test_loader):.4f}")  # Output test loss
        print(f"Train Accuracy: {train_accuracy:.4f}, Train Sensitivity: {train_sen:.4f}, "
              f"Train Specificity: {train_spe:.4f}, Train AUC: {train_auc:.4f}, Train MCC: {train_mcc:.4f}")
        print(f"Val Accuracy: {val_accuracy:.4f}, Val Sensitivity: {val_sen:.4f}, "
              f"Val Specificity: {val_spe:.4f}, Val AUC: {val_auc:.4f}, Val MCC: {val_mcc:.4f}")
        print(f"Test Accuracy: {test_accuracy:.4f}, Test Sensitivity: {test_sen:.4f}, "
              f"Test Specificity: {test_spe:.4f}, Test AUC: {test_auc:.4f}, Test MCC: {test_mcc:.4f}")

        # 保存metrics到Excel文件
        metrics_file = os.path.join(weights_dir, args.metrics_file)  # Use the path from args
        df = pd.DataFrame(metrics_data)
        df.to_excel(metrics_file, index=False)

        # 在循环外定义（需要在 for epoch 循环外手动添加）
        output_file = os.path.join(weights_dir, '88_pro.xlsx')
        sheet_name = "Results"
        df_combined = pd.DataFrame()  # 初始化为空，用于累积所有 epoch 结果

        # 新一轮测试
        if not args.data_path_88:  # 检查 data_path_88 是否为空（None 或空字符串）
            print("参数 data_path_88 为空，跳过第二轮测试和保存。\n")
        else:
            # 第二测试集
            second_test_dataset = CustomDataset(
                data_path=os.path.join(args.data_path_88, args.second_test_data),
                labels_path=None,  # 没有标签
                names_path=os.path.join(args.data_path_88, args.second_test_names)
            )
            second_test_loader = DataLoader(second_test_dataset, batch_size=args.batch_size, shuffle=False)

            model.eval()
            second_test_predicted_score = []  # 存储当前轮次的概率值
            second_test_names = []  # 存储对应的文件名
            with torch.no_grad():
                second_test_loader_tqdm = tqdm(second_test_loader, desc=f'Second Test - Epoch {epoch + 1}',
                                               disable=False)
                for idx, (img, name) in enumerate(second_test_loader_tqdm):
                    second_test_img = img.to(device)
                    second_test_outputs = model(second_test_img)
                    probs = torch.sigmoid(second_test_outputs).squeeze().cpu().numpy()
                    second_test_predicted_score.extend(probs if probs.ndim > 0 else [probs])
                    second_test_names.extend(name)
                    del second_test_img, second_test_outputs
                    torch.cuda.empty_cache()

            # 创建当前 epoch 的 DataFrame
            df_new = pd.DataFrame({
                'dcm_name': second_test_names,
                f'prob_Epoch_{epoch + 1}': second_test_predicted_score
            })

            # 累积到 df_combined
            try:
                if epoch == 0:
                    # 第一个 epoch，初始化 df_combined
                    df_combined = df_new
                else:
                    # 后续 epoch，直接添加新列
                    df_combined[f'prob_Epoch_{epoch + 1}'] = df_new[f'prob_Epoch_{epoch + 1}']

                # 覆盖写入文件
                with pd.ExcelWriter(output_file, mode='w', engine='openpyxl') as writer:
                    df_combined.to_excel(writer, sheet_name=sheet_name, index=False)
                print(f"\nEpoch {epoch + 1} 的概率值已保存到: {output_file}")
            except Exception as e:
                print(f"保存失败: {e}")

         # 早停检查：如果 train_mcc > 0.85，则停止当前训练
        if train_mcc > 0.9:
             print(
                f"Train MCC reached {train_mcc:.4f} > 0.85 at epoch {epoch + 1}, stopping training for this parameter set.")
             break

def main():
    base_args = parse_args()
    base_args.num_epochs = 100  # 仅修改通用参数

    # 超参数组合
    wavelet_types = ['mexican_hat', 'haar', 'morlet', 'gabor']
    wavelet_kernel_sizes = [1, 2, 3]
    num_wavelets_list = [1, 2, 3]
    multi_scale_list = [1, 2, 3]
    use_fourier_list = [False]
    num_recursive_layers_list = [1, 2]

    for wavelet_type, kernel_size, num_wavelets, multi_scale, use_fourier, num_recursive_layers in itertools.product(
        wavelet_types, wavelet_kernel_sizes, num_wavelets_list, multi_scale_list, use_fourier_list, num_recursive_layers_list
    ):
        # ✅ **将所有超参数存入字典**
        hyperparams = {
            "wavelet_type": wavelet_type,
            "wavelet_kernel_size": kernel_size,
            "num_wavelets": num_wavelets,
            "multi_scale": multi_scale,
            "use_fourier": use_fourier,
            "num_recursive_layers": num_recursive_layers
        }

        print("\n==========================================")
        print(f" Starting new run with:")
        for key, value in hyperparams.items():
            print(f"   {key} = {value}")
        print("==========================================\n")

        train(base_args, hyperparams)


if __name__ == '__main__':
    main()
