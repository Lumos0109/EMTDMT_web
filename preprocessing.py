import numpy as np
import pandas as pd
import torch

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
