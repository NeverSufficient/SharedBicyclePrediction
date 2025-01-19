import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import os


# 定义滑动窗口函数
def create_time_series_dataset(data, past_sequence_length, future_sequence_length):
    X, Y = [], []
    for i in range(len(data) - past_sequence_length - future_sequence_length):
        X.append(data[i:i + past_sequence_length, 1:10])
        Y.append(data[i + past_sequence_length:i + past_sequence_length + future_sequence_length, 10:12])
    X = np.array(X)
    Y = np.array(Y)
    return X, Y


# 数据预处理
def data_pre_process(batch_size, past_sequence_length, future_sequence_length):
    # 读取数据
    train_data_from_path = pd.read_csv('train_data.csv').values
    test_data_from_path = pd.read_csv('test_data.csv').values
    train_data = train_data_from_path[:, :]
    test_data = test_data_from_path[:, :]
    # 调用函数生成输入和输出
    X_train, Y_train = create_time_series_dataset(train_data, past_sequence_length, future_sequence_length)
    X_test, Y_test = create_time_series_dataset(test_data, past_sequence_length, future_sequence_length)
    # 转换为PyTorch张量
    X_train = torch.tensor(X_train, dtype=torch.float32)  # 输入形状：[样本数, past_sequence_length, num_features]
    Y_train = torch.tensor(Y_train, dtype=torch.float32)  # 输出形状：[样本数, future_sequence_length, num_features]
    X_test = torch.tensor(X_test, dtype=torch.float32)  # 输入形状：[样本数, past_sequence_length, num_features]
    Y_test = torch.tensor(Y_test, dtype=torch.float32)  # 输出形状：[样本数, future_sequence_length, num_features]
    # 使用DataLoader创建批量数据加载器
    train_dataset = TensorDataset(X_train, Y_train)
    test_dataset = TensorDataset(X_test, Y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# 定义LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # LSTM 层
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # 全连接层，用于预测未来时间点
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # LSTM 前向传播
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # 初始化隐藏状态
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  # 初始化记忆状态
        out, _ = self.lstm(x, (h0, c0))  # LSTM 输出 (batch_size, seq_length, hidden_size)
        out = self.fc(out[:, -1, :])     # 取最后一个时间步的输出
        return out


# 训练模型
def train_model(input_size, output_size, hidden_size, num_layers, lr, num_epochs, train_loader, device):
    # 定义LSTM模型
    model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)
    # model = torch.load('pretrained/LSTM.pt')
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    # 模型训练
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y[:, -1, :])  # 只预测未来时间序列最后一个时间点
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader):.8f}")
    torch.save(model, 'pretrained/LSTM.pt')   # torch.save()和torch.load()
    return model


# 评估模型
def evaluate(predictions, actuals):
    pre = (predictions[:, 0] + predictions[:, 1])
    act = (actuals[:, 0] + actuals[:, 1])
    mse = float(np.mean((pre - act)**2))
    mae = float(np.mean(np.abs(pre - act)))
    return mse, mae


if __name__ == '__main__':
    # 获取GPU资源
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # 数据预处理
    train_loader, test_loader = data_pre_process(batch_size=32, past_sequence_length=96, future_sequence_length=96)
    # 模型训练
    model = train_model(input_size=9, output_size=2, hidden_size=64, num_layers=1,
                        lr=0.001, num_epochs=300, train_loader=train_loader, device=device)
    '''
    # 加载模型
    model = torch.load('pretrained/LSTM.pt')
    '''
    # 模型预测
    model.eval()
    with torch.no_grad():
        predictions = []
        actuals = []
        for batch_X, batch_Y in test_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            outputs = model(batch_X)
            predictions.append(outputs)
            actuals.append(batch_Y[:, -1, :])
        predictions = torch.cat(predictions, dim=0).cpu().numpy().astype('int')
        actuals = torch.cat(actuals, dim=0).cpu().numpy().astype('int')
    print(predictions.shape)
    print(predictions)
    print(actuals.shape)
    print(actuals)
    # 性能评估  五次试验后使用np.std()求标准差
    # 短期预测96-96
    # MSE = []，mse = ，std =
    # MAE = []，mae = ，std =
    # 长期预测96-240
    # MSE = []，mse = ，std =
    # MAE = []，mae = ，std =
    mse, mae = evaluate(predictions, actuals)
    print("MSE为{:.4f}".format(mse))
    print("MAE为{:.4f}".format(mae))
    # 可视化
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    plt.figure(figsize=(10, 6))
    plt.plot((actuals[:, 0] + actuals[:, 1]), label="ground truth", color='blue')
    plt.plot((predictions[:, 0] + predictions[:, 1]), label="prediction", color='red')
    plt.title("LSTM Model Inference")
    plt.xlabel("Time")
    plt.ylabel("cnt")
    plt.legend()
    plt.savefig('images/LSTM.jpg')
    plt.show()
