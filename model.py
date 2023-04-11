import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout, max_len):
        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)
        return x

class Mymodel(nn.Module):
    def __init__(self,
                 num_ciphers,
                 num_exts,
                 time_step=20,
                 elements_number=260,
                 embedding_size=10,
                 length_number=3010,
                 eval_number=10000,
                 class_number=9,
                 aux_num = 24):

        super(Mymodel, self).__init__()

        self.time_step = time_step
        self.elements_number = elements_number
        self.embedding_size = embedding_size
        self.length_number = length_number
        self.eval_number = eval_number
        self.class_number = class_number
        self.aux_num = aux_num

        # 嵌入层
        self.pyload_embedding = nn.Embedding(elements_number, embedding_size)
        self.seq_embedding = nn.Embedding(length_number, embedding_size)

        # 位置编码
        self.pyload_position = PositionalEncoding(embedding_size, 0.1, elements_number * 2)
        self.seq_position = PositionalEncoding(embedding_size, 0.1, time_step)

        #ch编码层
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=5)
        self.encoder_ch = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # pls,pais编码层
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_size, nhead=5)
        self.encoder_seq = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.fc_ch = nn.Linear(256 * self.embedding_size, 400)
        self.fc_seq = nn.Linear(self.embedding_size * self.time_step * 2, 200)

        # 主任务全连接层
        self.fc_main = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(600, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, class_number),
        )

        # 辅助任务全连接层
        self.fc_aux = nn.ModuleList()
        for i in range(self.aux_num):
            fc = nn.Sequential(
                nn.ReLU(inplace=True),
                nn.Linear(600, 100),
                nn.ReLU(inplace=True),
                nn.Linear(100, 1)
            )
            self.fc_aux.append(fc)
        self.fc_cipher = nn.Linear(600, num_ciphers)
        self.fc_ext = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Linear(600, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_exts)
        )


    def forward(self, ch, pls, pais): # 三个输入：tls握手信息中client hello的部分有效载荷（ch）、数据包长序列（pls）、数据包到达时间间隔序列（pais）

        # 偏移，变正数
        pls = pls + (self.length_number // 2)
        pais = pais + (self.length_number // 2)

        # 嵌入层
        ch = self.pyload_embedding(ch.long() + 1)
        pls = self.seq_embedding(pls.long() + 1)
        pais = self.seq_embedding(pais.long() + 1)

        #ch
        input_block = self.pyload_position(ch)
        out1 = self.encoder_ch(input_block)
        out1 = out1.reshape(-1, 256 * self.embedding_size)
        out1 = self.fc_ch(out1)

        #pls,pais
        pls = self.seq_position(pls)
        pais = self.seq_position(pais)
        out2 = self.encoder_seq(torch.cat((pls, pais), dim=1))
        out2 = out2.reshape(-1, self.embedding_size * self.time_step * 2)
        out2 = self.fc_seq(out2)

        #main task
        out_main = self.fc_main(torch.cat((out1, out2), dim=1))

        #other tasks
        out_aux = []
        for i in range(self.aux_num):
            out_tmp = self.fc_aux[i](torch.cat((out1, out2), dim=1))
            out_aux.append(out_tmp)
        out_ciphers = self.fc_cipher(torch.cat((out1, out2), dim=1))
        out_exts = self.fc_ext(torch.cat((out1, out2), dim=1))

        return out_main, out_aux, out_ciphers, out_exts

    def evaluate(self, loader):
        correct = 0
        total = 0
        TP = [0.00001 for i in range(self.class_number)]
        FP = [0.00001 for i in range(self.class_number)]
        FN = [0.00001 for i in range(self.class_number)]
        F1s = []

        device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

        self.eval()
        with torch.no_grad():
            for ch_datas, pls_datas, pais_datas, labels, _, _, _ in loader:
                ch_datas, pls_datas, pais_datas, labels = ch_datas.to(
                    device), pls_datas.to(device), pais_datas.to(device), labels.to(
                    device)
                prediction,_,_,_= self(ch_datas, pls_datas, pais_datas)
                _, prediction = torch.max(prediction, 1)
                correct += (prediction == labels).sum()
                total += labels.size(0)
                for class_id in range(self.class_number):
                    TP[class_id] += ((prediction == class_id) & (labels == class_id)).sum()
                    FP[class_id] += ((prediction == class_id) & (labels != class_id)).sum()
                    FN[class_id] += ((prediction != class_id) & (labels == class_id)).sum()
        self.train()

        print('ACC : {} %'.format(100 * correct / total))
        print(len(loader))

        for class_id in range(self.class_number):
            precision = TP[class_id] / (TP[class_id] + FP[class_id])
            recall = TP[class_id] / (TP[class_id] + FN[class_id])
            F1 = 2 * (recall * precision) / (recall + precision)
            F1s.append(F1)
            print('NO.%d class: precision-%.4f recall-%.4f F1-%.4f TP-%.4f FP-%.4f FN-%.4f'
                  % (class_id, precision, recall, F1, TP[class_id], FP[class_id], FN[class_id]))

        f1 = sum(F1s) / self.class_number
        print('Macro-F1: {}'.format(f1))

        return f1
