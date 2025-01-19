import numpy as np


# LSTM
# 短期预测
MSE1 = [35428.0010, 28337.0793, 22654.2739, 29721.6341, 18775.0447]
MAE1 = [131.3100, 116.0732, 93.2708, 125.2541, 91.0630]
MSE1_mean = np.mean(MSE1)
MSE1_std = np.std(MSE1)
MAE1_mean = np.mean(MAE1)
MAE1_std = np.std(MAE1)
# 长期预测
MSE2 = [28786.5197, 29370.2621, 29252.4918, 29161.3914, 30774.8662]
MAE2 = [112.5033, 116.8706, 119.0905, 119.0362, 119.5592]
MSE2_mean = np.mean(MSE2)
MSE2_std = np.std(MSE2)
MAE2_mean = np.mean(MAE2)
MAE2_std = np.std(MAE2)
print(f"LSTM模型：")
print(f"短期预测：MSE的均值为{MSE1_mean:.4f}，标准差为{MSE1_std:.4f}；MAE的均值为{MAE1_mean:.4f}，标准差为{MAE1_std:.4f}.")
print(f"长期预测：MSE的均值为{MSE2_mean:.4f}，标准差为{MSE2_std:.4f}；MAE的均值为{MAE2_mean:.4f}，标准差为{MAE2_std:.4f}.")

# Transformer
# 短期预测
MSE3 = [34868.6946, 34246.5691, 34663.3338, 34600.3140, 36504.3958]
MAE3 = [136.8613, 136.6291, 133.4538, 135.2266, 141.2637]
MSE3_mean = np.mean(MSE3)
MSE3_std = np.std(MSE3)
MAE3_mean = np.mean(MAE3)
MAE3_std = np.std(MAE3)
# 长期预测
MSE4 = [40300.8651, 43324.4967, 40357.8218, 38097.0132, 40503.4764]
MAE4 = [145.1371, 151.8936, 144.2933, 141.2928, 147.6047]
MSE4_mean = np.mean(MSE4)
MSE4_std = np.std(MSE4)
MAE4_mean = np.mean(MAE4)
MAE4_std = np.std(MAE4)
print(f"Transformer模型：")
print(f"短期预测：MSE的均值为{MSE3_mean:.4f}，标准差为{MSE3_std:.4f}；MAE的均值为{MAE3_mean:.4f}，标准差为{MAE3_std:.4f}.")
print(f"长期预测：MSE的均值为{MSE4_mean:.4f}，标准差为{MSE4_std:.4f}；MAE的均值为{MAE4_mean:.4f}，标准差为{MAE4_std:.4f}.")

# MyModel（TCN改）
# 短期预测
MSE5 = [22983.5823, 26139.4375, 23867.0894, 22887.2979, 22223.3882]
MAE5 = [101.7846, 123.9883, 116.8659, 111.7840, 109.1646]
MSE5_mean = np.mean(MSE5)
MSE5_std = np.std(MSE5)
MAE5_mean = np.mean(MAE5)
MAE5_std = np.std(MAE5)
# 长期预测
MSE6 = [32499.4287, 31665.0077, 32445.6502, 30488.7763, 30431.5482]
MAE6 = [125.7116, 130.2083, 127.9945, 135.3158, 131.7763]
MSE6_mean = np.mean(MSE6)
MSE6_std = np.std(MSE6)
MAE6_mean = np.mean(MAE6)
MAE6_std = np.std(MAE6)
print(f"MyModel模型：")
print(f"短期预测：MSE的均值为{MSE5_mean:.4f}，标准差为{MSE5_std:.4f}；MAE的均值为{MAE5_mean:.4f}，标准差为{MAE5_std:.4f}.")
print(f"长期预测：MSE的均值为{MSE6_mean:.4f}，标准差为{MSE6_std:.4f}；MAE的均值为{MAE6_mean:.4f}，标准差为{MAE6_std:.4f}.")
