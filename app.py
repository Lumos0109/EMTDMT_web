import os
import sys
import torch
import function
import preprocessing
import matplotlib.pyplot as plt
import base64
import numpy as np
from model import *
from scapy.all import *
from flask import Flask, request, render_template, jsonify
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

feature_path = 'features.csv'
MODEL_PATH = '../main/my_model.pth'
class_name = ['Normal', 'Bunitu', 'HTBot', 'Miuref', 'TrickBot', 'Dridex', 'Caphaw', 'Neris', 'Zbot']
num_ciphers = 18
num_exts = 143

def data_preprocess(pcap):
	if os.path.exists('features.csv'):
		os.remove('features.csv')
	tmp = function.get_features(pcap, feature_path)
	if tmp == -1:
		return None
	datas = preprocessing.get_data(feature_path)
	return datas

def model_predict(datas, model):
	input1 = torch.tensor(datas['ch'])
	input2 = torch.tensor(datas['pls'])
	input3 = torch.tensor(datas['pais'])

	return model(input1, input2, input3)

def picture(arr_2):
	# Min-Max 标准化
	min_val = min(arr_2)
	max_val = max(arr_2)
	arr_2 = [(pred - min_val) / (max_val - min_val) for pred in arr_2]
	total = np.sum(arr_2)
	arr_2 = arr_2 / total

	# 生成柱状图
	plt.figure(figsize=(6, 4), dpi=100)
	plt.bar(class_name, arr_2)
	plt.title('Classification Probabilities')
	plt.xlabel('Class')
	plt.ylabel('Probability')
	plt.ylim(0, 1)  # 设置纵坐标范围为0到1

	# 将图表转换为Base64编码的PNG图片格式
	buffer = io.BytesIO()
	plt.savefig(buffer, format='png')
	buffer.seek(0)
	image_png = buffer.getvalue()
	buffer.close()
	return base64.b64encode(image_png).decode('utf-8')


@app.route('/', methods=['GET'])
def index():
	# Main page
	return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
	if request.method == 'POST':
		temp_file = tempfile.NamedTemporaryFile(delete=False)  # 创建临时文件
		temp_file.write(request.get_data())  # 将请求的二进制流写入临时文件中
		temp_file.close()
		datas = data_preprocess(temp_file.name)
		if datas == None:
			return jsonify(result="InvalidFile!", probability=1)
		model = torch.load(MODEL_PATH, map_location="cuda:2").to("cpu")
		model.eval()
		print('Model loaded. Start serving...')
		preds, _, _, _ = model_predict(datas, model)
		arr_preds = preds.detach().cpu().numpy()
		arr_preds = np.squeeze(arr_preds)
		graphic = picture(arr_preds)
		print(arr_preds)
		class_num = arr_preds.argmax()
		max = np.max(arr_preds)
		pred_proba = np.around(max, decimals=3)
		pred_proba = float(pred_proba)
		pred_class = class_name[class_num]
		return jsonify(result=pred_class, probability=pred_proba, graphic=graphic)
	return None

	


if __name__ == '__main__':
	# app.run(port=5002, threaded=False)

	# Serve the app with gevent
	http_server = WSGIServer(('0.0.0.0', 40080), app)
	http_server.serve_forever()
