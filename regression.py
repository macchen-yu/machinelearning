import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
def Data_procssing(Csv_path):
    #Read file
    data = pd.read_csv(Csv_path)
    #Preprocessing
    data.drop(columns= "Car_Name", inplace=True)
    data['Year'] = 2019 - data['Year']
    data['Kms_Driven'] = data['Kms_Driven'] / 10000
    fuel_type_mapping = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
    seller_type_mapping = {'Dealer': 0, 'Individual': 1}
    transmission_mapping = {'Manual': 0, 'Automatic': 1}
    data['Fuel_Type'] = data['Fuel_Type'].map(fuel_type_mapping)
    data['Seller_Type'] = data['Seller_Type'].map(seller_type_mapping)
    data['Transmission'] = data['Transmission'].map(transmission_mapping)
    return data
def DivideData(data,num):
    training_data=data.copy()
    val_data=pd.DataFrame()
    datalen  = int(len(data)-len(data)/num)
    for i in range(0, datalen , num):
        tempdata=training_data.iloc[i].copy()
        val_data=val_data._append(tempdata, ignore_index=True)
        training_data.drop(i, inplace=True)
    return training_data,val_data
def feature_label(data,device):
    features = data[['Year', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type']].values
    labels = data[['Selling_Price']].values
    features = torch.tensor(features, dtype=torch.float32).to(device)
    labels = torch.tensor(labels, dtype=torch.float32).to(device)
    return  features,labels
def train(num_epcoh,t_features,t_labels,model):
    model.train()
    output = model(t_features)
    loss = mse_l(output , t_labels)
    optimizer.zero_grad() #梯度歸零
    loss.backward()
    optimizer.step()
    # Save the trained model if needed
    torch.save(model.state_dict(), './model.pth')
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch+1}/{num_epcoh}], train_Loss: {loss.item():.4f}')
def val(num_epcoh,v_features,v_labels,model):
    model.eval()
    output = model(v_features)
    loss = mse_l(output, v_labels)
    if (epoch + 1) % 500 == 0:
        print(f'Epoch [{epoch + 1}/{num_epcoh}], Val_Loss: {loss.item():.4f}')


#Module
class Regression(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x
if __name__ == '__main__':
    # 如果有GPU 就用gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Csv_path='./car_data.csv'
    data=Data_procssing(Csv_path)
    #將資料分成 training_data,val_data num 是指每num個數量就把資料放進valdata裡面
    training_data,val_data = DivideData(data,12)

    #訓練feature labels
    t_features,t_labels=feature_label(training_data,device)
    v_features,v_labels=feature_label(val_data,device)

    #參數
    model = Regression().to(device)
    mse_l = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epcoh = 10000
    #訓練加上驗證
    for epoch in range(num_epcoh):
        train(num_epcoh,t_features,t_labels,model)
        val(num_epcoh, v_features, v_labels, model)
