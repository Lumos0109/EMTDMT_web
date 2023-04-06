import numpy as np
import pandas as pd
import torch

def get_data_test(file_path,
             client_hello_length=256,
             time_step=20,):
    flow_features = pd.read_csv(file_path, header=None)
    print(flow_features.iloc[0, 0])
    print(flow_features.iloc[0, 1])
    print(flow_features.iloc[0, 2])
    return flow_features

def get_data(file_path,
             client_hello_length=256,
             time_step=20):

    flow_features = pd.read_csv(file_path, header=None)

    # 将数据补零得到可输入网络的数据集
    datas = {'ch': [], 'pls': [], 'pais': []}
    ch_padding = [0 for _ in range(client_hello_length)]
    pls_padding = [0 for _ in range(time_step)]
    pais_padding = [0 for _ in range(time_step)]

    input = (flow_features.iloc[0, 2].strip().split(',')[:-1] + ch_padding)[:len(ch_padding)]
    input = [int(c) for c in input]
    datas['ch'].append(input)

    input = (flow_features.iloc[0, 0].strip().split(',')[:-1] + pls_padding)[:len(pls_padding)]
    input = [int(c) for c in input]
    input = [(c if c < 1500 else 1500) for c in input]
    input = [(c if c > -1500 else -1500) for c in input]
    datas['pls'].append(input)

    input = (flow_features.iloc[0, 1].strip().split(',')[1:-1] + pais_padding)[:len(pais_padding)]
    input = [int(float(c) * 100) for c in input]
    input = [(c if c < 1500 else 1500) for c in input]
    input = [(c if c > -1500 else -1500) for c in input]
    datas['pais'].append(input)

    return datas


# 获取训练集、验证集、测试集的 DataLoader
def get_data_loader(datas, labels, normalized_target, list_target, test_set_retio=0.4, batch_size=256, eval_number=512):
    all_data_num = len(labels)
    train_data_num = int(all_data_num * (1 - test_set_retio))
    validate_data_num = (all_data_num - train_data_num) // 2
    permutation = np.random.permutation(all_data_num)
    print(all_data_num)
    print(train_data_num)
    print(validate_data_num)

    train_indexs = permutation[:train_data_num]
    validate_indexs = permutation[train_data_num:train_data_num + validate_data_num]
    test_indexs = permutation[train_data_num + validate_data_num:]

    train_dataset = TrafficDataset(datas, labels, train_indexs,  normalized_target, list_target)
    validate_dataset = TrafficDataset(datas, labels, validate_indexs, normalized_target, list_target)
    test_dataset = TrafficDataset(datas, labels, test_indexs, normalized_target, list_target)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=2)
    train_loader_evaluation = torch.utils.data.DataLoader(dataset=train_dataset,
                                                          batch_size=eval_number,
                                                          shuffle=True,
                                                          num_workers=2)
    validate_loader = torch.utils.data.DataLoader(dataset=validate_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=2)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=2)
    return train_loader, validate_loader, train_loader_evaluation, test_loader

# 用于将数据集转化成可以使用 DataLoader 迭代的 Dataset 类型
class TrafficDataset(torch.utils.data.Dataset):
    def __init__(self, datas, labels, data_indexs, normalized_target, list_target):
        self.ch_datas = torch.tensor(datas['ch'])[data_indexs, :]
        self.pls_datas = torch.tensor(datas['pls'])[data_indexs, :]
        self.pais_datas = torch.tensor(datas['pais'])[data_indexs, :]
        self.labels = torch.tensor(labels)[data_indexs]
        self.normalized_target = torch.tensor(normalized_target)[:, data_indexs]
        self.cipher_target = torch.tensor(list_target['cipher'])[data_indexs, :]
        self.ext_target = torch.tensor(list_target['ext'])[data_indexs, :]

        self.size = self.labels.shape[0]
        print("CustomDataset init:\nch_datas-%s pls_datas-%s pais_datas-%s"
              %(list(self.ch_datas.shape), list(self.pls_datas.shape), list(self.pais_datas.shape)))

    def __getitem__(self, index):
        return self.ch_datas[index, :], self.pls_datas[index, :], self.pais_datas[index, :], self.labels[index], self.normalized_target[:, index], self.cipher_target[index, :], self.ext_target[index, :]

    def __len__(self):
        return self.size
