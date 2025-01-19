import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
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


# 定义TCN模型
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        '''
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        '''
        self.net1 = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1)
        self.net2 = nn.Sequential(self.conv2, self.chomp2, self.relu2, self.dropout2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # out = self.net(x)
        out1 = self.net1(x)
        out2 = self.net2(out1)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out1 + out2 + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

class TCNPredictor(nn.Module):
    def __init__(self, input_dim, output_dim, num_channels, kernel_size, dropout):
        super(TCNPredictor, self).__init__()
        self.tcn = TemporalConvNet(input_dim, num_channels, kernel_size, dropout)
        self.linear = nn.Linear(num_channels[-1], output_dim)

    def forward(self, x):
        y1 = self.tcn(x.transpose(1, 2))
        return self.linear(y1[:, :, -1])



# 训练模型
def train_model(input_dim, output_dim, num_channels, kernel_size, dropout, future_sequence_length,
                lr, num_epochs, train_loader, device):
    # 定义TCN模型
    model = TCNPredictor(input_dim, output_dim, num_channels, kernel_size, dropout).to(device)
    # model = torch.load('pretrained/TCN.pt')
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
            outputs = model(batch_X).reshape(-1, future_sequence_length, output_dim // future_sequence_length)
            loss = criterion(outputs[:, -1, :], batch_Y[:, -1, :])  # 只预测未来时间序列最后一个时间点
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader):.8f}")
    torch.save(model, 'pretrained/MyModel.pt')   # torch.save()和torch.load()
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
    seq_len_future = 96
    output_dim = 2 * seq_len_future
    model = train_model(input_dim=9, output_dim=2*96, num_channels=[32, 64, 128], kernel_size=3, dropout=0.1, future_sequence_length=96,
                        lr=0.0001, num_epochs=100, train_loader=train_loader, device=device)
    '''
    # 加载模型
    model = torch.load('pretrained/MyModel.pt')
    '''
    # 模型预测
    model.eval()
    with torch.no_grad():
        predictions = []
        actuals = []
        for batch_X, batch_Y in test_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            outputs = model(batch_X)
            outputs = outputs.reshape(-1, seq_len_future, output_dim // seq_len_future)
            predictions.append(outputs[:, -1, :])
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
    plt.title("MyModel Inference")
    plt.xlabel("Time")
    plt.ylabel("cnt")
    plt.legend()
    plt.savefig('images/MyModel')
    plt.show()
