import os
import sys
import torch
import function
import preprocessing
from model import *
from scapy.all import *
import numpy as np

feature_path = 'features.csv'
MODEL_PATH = 'models/your_model.pt'
class_name = ['Normal', 'Bunitu', 'HTBot', 'Miuref', 'TrickBot', 'Dridex', 'Caphaw', 'Neris', 'Zbot']

def model_predict(datas, model):
	input1 = torch.tensor(datas['ch'])
	input2 = torch.tensor(datas['pls'])
	input3 = torch.tensor(datas['pais'])

	return model(input1, input2, input3)

def load_model(model):
    state_dict  = torch.load(MODEL_PATH)
    model.load_state_dict(state_dict)
    model.eval()
    print('Model loaded. Start serving...')

if __name__ == '__main__':
    model = Mymodel(143, 18)
    load_model(model)
    datas = preprocessing.get_data(feature_path)
    preds,_,_,_ = model_predict(datas, model)
    arr_preds = preds.detach().cpu().numpy()
    arr_preds = np.squeeze(arr_preds)
    class_num = arr_preds.argmax()
    max = np.max(arr_preds)
    pred_proba = np.around(max, decimals=3)
    pred_class = class_name[class_num]

    print(pred_class)
    print(pred_proba)
