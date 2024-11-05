import logging
import os
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.utils.data as Data
from dataset.my_dataset import MyDataset

# 获取主脚本中配置的logger
logger = logging.getLogger('HARMODEL')

def load_dataset(dataset, batch_size):
    # 读取原数据
    x_train, y_train, x_test, y_test = read_raw_data(dataset)

    # 将数据转换为 PyTorch 张量
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    train_data = Data.TensorDataset(x_train_tensor, y_train_tensor)
    test_data = Data.TensorDataset(x_test_tensor, y_test_tensor)

    train_loader = Data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = Data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=0)

    # 合成train、test数据集
    # train_dataset = MyDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    # test_dataset = MyDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))

    # 合成dataloader
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=1, shuffle=True)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=1, shuffle=True)

    # 验证DataLoader是否正确
    for x_batch, y_batch in train_loader:
        logging.info(f'Batch X shape: {x_batch.shape}')
        logging.info(f'Batch y shape: {y_batch.shape}')
        break

    return train_loader, test_loader, (x_train.shape[1], x_train.shape[2])


def read_raw_data(dataset):
    if dataset == "PAMAP2":
        data_folder = r"dataset/PAMAP2"
    elif dataset == "OPPORTUNITY":
        data_folder = r"dataset/OPPORTUNITY"
    elif dataset == "UCI_HAR":
        data_folder = r"dataset/UCI_HAR"
    elif dataset == "UCI_HAR_2":
        data_folder = r"dataset/UCI_HAR_2"
    elif dataset == "UniMiB_SHAR":
        data_folder = r"dataset/UniMiB_SHAR"
    elif dataset == "USC_HAD":
        data_folder = r"dataset/USC_HAD"
    elif dataset == "WISDM":
        data_folder = r"dataset/WISDM"
    else:
        logging.error(f"Dataset {dataset} not supported.")
        return

    x_train = np.load(os.path.join(data_folder, 'x_train.npy'))
    y_train = np.load(os.path.join(data_folder, 'y_train.npy'))
    x_test = np.load(os.path.join(data_folder, 'x_test.npy'))
    y_test = np.load(os.path.join(data_folder, 'y_test.npy'))

    # reshape data
    x_train = x_train.transpose(0, 2, 1)
    x_test = x_test.transpose(0, 2, 1)

    # 输出加载的数据维度以验证
    logger.info(f'X_train shape: {x_train.shape}')
    logger.info(f'y_train shape: {y_train.shape}')
    logger.info(f'X_test shape: {x_test.shape}')
    logger.info(f'y_test shape: {y_test.shape}')

    return x_train, y_train, x_test, y_test