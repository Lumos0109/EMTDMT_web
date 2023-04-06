import torch
import torch.nn as nn
from tqdm import tqdm
from preprocessing import get_data, get_data_loader
from model import *


class Trainer:
    def __init__(self):
        pass

    def prepare(self, file_path="../../data/Stratosphere_Lab/features/"):
        # 获得数据
        datas, labels, class_num, target, list_target, num_ciphers, num_exts = get_data(file_path)

        self.num_ciphers = num_ciphers
        self.num_exts = num_exts

        # 建立训练集、验证集、测试集
        train_loader, train_loader_evaluation, validate_loader, test_loader = get_data_loader(datas, labels, target, list_target)
        self.train_loader = train_loader
        self.train_loader_evaluation = train_loader_evaluation
        self.validate_loader = validate_loader
        self.test_loader = test_loader

    def train(self, epochs=50, batches_per_test=100, epochs_per_test=5, aux_num=24):

        device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
        model = Mymodel(self.num_ciphers, self.num_exts).to(device)
        optimizer = torch.optim.Adam(model.parameters())
        criterion = nn.CrossEntropyLoss()
        criterion_aux = nn.MSELoss()
        criterion_list = nn.BCEWithLogitsLoss()
        model.train()

        # 记录 loss 与 F1 变化
        print("记录 loss 与 F1 变化")
        record = {'loss': [], 'train_F1': [], 'dev_F1': []}

        for step in range(epochs):
            loop = tqdm(enumerate(self.train_loader), total=len(self.train_loader), leave=False)
            for i, data in loop:
                if (i + 1) % batches_per_test == 0:
                    print('EPOCHS: {} ,Loss: {:4f}'.format(step, float(loss_sum)))
                    record['loss'].append(float(loss_sum))
                    print("TRAIN:")
                    f1 = model.evaluate(self.train_loader_evaluation)
                    record['train_F1'].append(f1)
                    print("VALIDATE:")
                    f1 = model.evaluate(self.validate_loader)
                    record['dev_F1'].append(f1)
                    print("RECORD:")
                    for i in range(len(record['loss'])):
                        print('Loss: {:4f} train F1:{:4f} dev F1:{:4f}'.format(record['loss'][i], record['train_F1'][i], record['dev_F1'][i]))

                # training
                ch_datas, pls_datas, pais_datas, labels, target, cipher_target, ext_target = data
                target = target.t()
                ch_datas, pls_datas, pais_datas, labels, target, cipher_target, ext_target = ch_datas.to(device), pls_datas.to(device), pais_datas.to(device), labels.to(device), target.to(device), cipher_target.to(device), ext_target.to(device)

                prediction, prediction_aux, prediction_cipher, prediction_ext = model(ch_datas, pls_datas, pais_datas)

                loss_main = criterion(prediction, labels.long())
                loss_cipher = criterion_list(prediction_cipher, cipher_target.float())
                loss_ext = criterion_list(prediction_ext, ext_target.float())
                loss_sum = torch.sum(loss_main * 0.5 + loss_cipher * 0.05 + loss_ext * 0.05)
                for i in range(aux_num):
                    loss_aux = criterion_aux(torch.squeeze(prediction_aux[i]), target[i].float())
                    loss_sum = torch.sum(loss_sum + 0.3 / aux_num * loss_aux)

                optimizer.zero_grad()
                loss_sum.backward()
                optimizer.step()
                loop.set_description(f'Epoch {step}/{epochs}')
                loop.set_postfix(loss_sum=loss_sum.detach().cpu().numpy())

            if (step + 1) % epochs_per_test == 0:
                print("TEST:")
                model.evaluate(self.test_loader)
                torch.save(model.state_dict(), 'model.pt')
