
#坑利用pytorch 保存模型乘pth預測效果非常差
  # 要加上 這行model.load_state_dict(torch.load(model_path ))
  #   model.eval()

import torch
from regression import Regression,feature_label,Data_procssing
import pandas as pd

device = torch.device("cpu")

Csv_path ='./car data.csv'
data=Data_procssing(Csv_path)
model = Regression()
model_path = './model.pth'
model.load_state_dict(torch.load(model_path ))
model.to(device)
# model.eval()  # 将模型设置为评估模式
# 預測間段===============================
model.eval()
features,labels=feature_label(data,device)
while(True):
    x = input("輸入資料集第幾筆資料：")

    if x == "q" or x== "Q":
        break
    x=int(x)
    # x=x-1
    features_test = features[x].to(device)
    prediction = model.forward(features_test)

    print(features_test)
    print(f"Predict Value： {prediction.item()} Correct Value： {labels[x].item()}")