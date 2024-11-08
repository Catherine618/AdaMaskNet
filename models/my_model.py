import numpy as np
from fvcore.nn import FlopCountAnalysis, parameter_count_table
from sklearn.manifold import TSNE
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from torch import nn
import seaborn as sns

from dataset.dataset_config import dataset_parameters
from models.baselines.ResCNN import ResCNN
from models.baselines.SimpleCNN import HAR_CNN
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, ConfusionMatrixDisplay, \
    confusion_matrix
import logging
import torch
import os
from matplotlib import pyplot as plt
from models.adaMaskNet.AdaMaskNet import AdaMaskNet

# 获取主脚本中配置的logger
logger = logging.getLogger('HARMODEL')



def save_checkpoint(model, optimizer, epoch, path):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(state, path)

def creat_model(dataset,
                num_classes,
                model_name,
                input_shape):

   if model_name == "AdaMaskNet":
        model = AdaMaskNet(n_class=num_classes,
                           input_shape=input_shape,
                           temporal_parameter_number_of_layer_list = dataset_parameters[dataset]["temporal"],
                           stride=dataset_parameters[dataset]["opt_stride"],
                           averagepool_size=dataset_parameters[dataset]["average_size"])

   elif model_name == "Simple_CNN":
        model = HAR_CNN(dataset, num_classes, input_shape, largeKernal=False)

   elif model_name == "LK_CNN":
        model = HAR_CNN(dataset, num_classes, input_shape, largeKernal=True)

   elif model_name == "Res_CNN":
        model = ResCNN(dataset, num_classes, input_shape, largeKernal=False)

   elif model_name == "Res_LK_CNN":
        model = ResCNN(dataset, num_classes, input_shape, largeKernal=True)

   return model


def adjust_learning_rate(learning_rate, optimizer, epoch):
    lr = learning_rate * (0.1 ** (epoch // 50))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    logger.info(f"current Learning_rate: {optimizer.param_groups[0]['lr']}")


def train_model(dataset,
                train_loader,
                test_loader,
                model_name,
                model,
                criterion,
                optimizer,
                num_epochs,
                learning_rate,
                output_dir,
                device,
                start_epoch,
                writer):
    model.train()

    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        adjust_learning_rate(learning_rate, optimizer, epoch)

        for inputs, labels in train_loader:
            if (model_name == "Simple_CNN"
                    or model_name == "LK_CNN"
                    or model_name == "Res_CNN"
                    or model_name == "Res_LK_CNN"):
                inputs = inputs.unsqueeze(1)

            elif model_name == "AdaMaskNet":
                inputs.requires_grad = False
                inputs = inputs.unsqueeze(1)
                labels = labels.view(-1)  # 将标签形状从 (batch_size, 1) 变为 (batch_size,)

            inputs = inputs.to(device)  # Move inputs to the same device as the model
            labels = labels.to(device)  # Move labels to the same device as the model

            optimizer.zero_grad()
            outputs, features = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_accuracy = correct / total

        writer.add_scalar('Train/Loss', epoch_loss, epoch)
        writer.add_scalar(f'Train/Accuracy', epoch_accuracy, epoch)

        logger.info(f'Epoch {epoch + 1}/{num_epochs}| Loss: {epoch_loss:.4f}| Accuracy: {epoch_accuracy:.4f}')

        # 保存模型权重
        checkpoint_path = os.path.join(output_dir, f'checkpoint_epoch_{epoch + 1}.pth')
        print('Saving checkpoint to', checkpoint_path)
        save_checkpoint(model, optimizer, epoch + 1, checkpoint_path)
        logger.info(f'Model checkpoint saved at {checkpoint_path}')

        evaluate_model(dataset, test_loader, model_name, model, device, writer, epoch)
        input_tensor = torch.randn(1, 1, inputs.shape[2], inputs.shape[3]).to(device)

# 定义一个函数来计算并添加比例到混淆矩阵中
def add_percentages_to_cm(cm):
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    return np.round(cm_normalized * 100, 2)


def evaluate_model(dataset,
                   test_loader,
                   model_name,
                   model,
                   device,
                   writer,
                   epoch=0):
    model.eval()

    correct = 0
    total = 0

    # 用于计算混淆矩阵
    all_predictions = []
    all_labels = []
    all_data = []

    # 获取所有数据的标签列表
    unique_labels = set()
    for data, labels in test_loader:
        all_data.append(data)
        unique_labels.update(labels.numpy())

    unique_labels = list(set(unique_labels))
    unique_labels.sort()

    with (torch.no_grad()):
        for inputs, labels in test_loader:
            if (model_name == "Simple_CNN"
                    or model_name == "LK_CNN"
                    or model_name == "Res_CNN"
                    or model_name == "Res_LK_CNN"):
                inputs = inputs.unsqueeze(1)

            elif model_name == "AdaMaskNet":
                inputs.requires_grad = False
                inputs = inputs.unsqueeze(1)
                labels = labels.view(-1)

            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs, features = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # 记录预测结果
            all_predictions.extend(predicted.view(-1).cpu().numpy())  # Collect predictions
            all_labels.extend(labels.view(-1).cpu().numpy())  # Collect actual labels

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # 计算混淆矩阵
    precision = precision_score(all_labels, all_predictions, average='weighted')
    recall = recall_score(all_labels, all_predictions, average='weighted')
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    accuracy = accuracy_score(all_labels, all_predictions)

    message = f'-----Test Precision:{precision:.4f}| Recall:{recall:.4f}| F1 Score:{f1:.4f}| Accuracy:{accuracy:.4f}-----'

    if logger.hasHandlers():
        logger.info(message)
    else:
        print(message)

    writer.add_scalar('Test/Precision', precision, epoch)
    writer.add_scalar('Test/Recall', recall, epoch)
    writer.add_scalar('Test/F1 Score', f1, epoch)
    writer.add_scalar('Test/Accuracy', accuracy, epoch)


