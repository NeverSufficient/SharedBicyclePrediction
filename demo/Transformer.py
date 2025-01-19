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


# 定义Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim, past_seq_len, dropout=0.1):
        super(TransformerModel, self).__init__()
        # 输入嵌入层（线性层）
        self.embedding = nn.Linear(input_dim, d_model)
        # 位置编码（可学习）
        self.positional_encoding = nn.Parameter(torch.zeros(1, past_seq_len, d_model))
        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # 输出层
        self.fc = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # 添加位置编码
        x = self.embedding(x) + self.positional_encoding
        # Transformer 编码器
        x = self.transformer_encoder(x)
        # 输出（仅取最后一层）
        out = self.fc(x)
        return out


# 训练模型
def train_model(input_dim, output_dim, d_model, nhead, num_layers, dim_feedforward, past_seq_len,
                lr, num_epochs, train_loader, device):
    # 定义Transformer模型
    model = TransformerModel(input_dim, d_model, nhead, num_layers, dim_feedforward, output_dim, past_seq_len).to(device)
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr)
    # 模型训练
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs[:, -1, :], batch_Y[:, -1, :])  # 只预测未来时间序列最后一个时间点
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader):.8f}")
    torch.save(model, 'pretrained/Transformer.pt')
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
    model = train_model(input_dim=9, output_dim=2, d_model=64, nhead=4, num_layers=2, dim_feedforward=256, past_seq_len=96, lr=0.0001, num_epochs=500, train_loader=train_loader, device=device)
    '''
    # 加载模型
    model = torch.load('pretrained/Transformer.pt')
    '''
    # 模型预测
    model.eval()
    with torch.no_grad():
        predictions = []
        actuals = []
        for batch_X, batch_Y in test_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            outputs = model(batch_X)
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
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("cnt")
    plt.title("Transformer Model Inference")
    plt.savefig('images/Transformer.jpg')
    plt.show()

