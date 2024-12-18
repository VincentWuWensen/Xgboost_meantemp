# Import necessary libraries
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, YearLocator
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from xgboost import XGBClassifier

# Set random seeds
np.random.seed(42)

# Load and preprocess the dataset
filepath = 'D:\\pycharm\\pytorch_learn\\dataset\\archive\\DailyDelhiClimate\\DailyDelhiClimateTrain_pre.csv'
data = pd.read_csv(filepath)
data = data.sort_values('date')
data['date'] = pd.to_datetime(data['date'])

# Extract and scale predictors
predictors = ['humidity', 'wind_speed', 'meanpressure']
scaler = MinMaxScaler(feature_range=(-1, 1))
data[predictors] = scaler.fit_transform(data[predictors])

# Split the dataset into historical (meantemp > 0) and future (meantemp = 0) parts
historical_data = data.loc[data['meantemp'] > 1]
future_data = data.loc[data['meantemp'] <= 1]

# Scale the target variable (meantemp) for historical data
historical_data['meantemp'] = scaler.fit_transform(historical_data['meantemp'].values.reshape(-1, 1))
correlation_matrix = historical_data.corr()
print(correlation_matrix['meantemp'].sort_values(ascending=False))

# Prepare training and validation sets from historical data
train_ratio = 0.8
train_end = int(train_ratio * len(historical_data))

X_train = historical_data[predictors][:train_end]
X_val = historical_data[predictors][train_end:]
X_test = future_data[predictors]
y_train = historical_data['meantemp'][:train_end]
y_val = historical_data['meantemp'][train_end:]
y_test = future_data['meantemp']  # These values are placeholders (0)

xgb_model = XGBRegressor(max_depth=6,  # 每一棵树最大深度，默认6；
                         learning_rate=0.1,  # 学习率，每棵树的预测结果都要乘以这个学习率，默认0.3；
                         n_estimators=200,  # 使用多少棵树来拟合，也可以理解为多少次迭代。默认100；
                         objective='reg:squarederror',  # 此默认参数与 XGBClassifier不同
                         booster='gbtree',
                         # 有两种模型可以选择gbtree和gblinear。gbtree使用基于树的模型进行提升计算，gblinear使用线性模型进行提升计算。默认为gbtree
                         eval_metric='rmse', # 指定评估指标
                         n_jobs=-1,
                         gamma=0,  # 叶节点上进行进一步分裂所需的最小"损失减少"。默认0；
                         min_child_weight=1,  # 可以理解为叶子节点最小样本数，默认1；防止树分出来只有几个合格的
                         subsample=0.8,  # 训练集抽样比例，每次拟合一棵树之前，都会进行该抽样步骤。默认1，取值范围(0, 1]
                         colsample_bytree=0.8,  # 每次拟合一棵树之前，决定使用多少个特征，参数默认1，取值范围(0, 1]。
                         reg_alpha=0,  # 默认为0，控制模型复杂程度的权重值的 L1 正则项参数，参数值越大，模型越不容易过拟合。
                         reg_lambda=1,  # 默认为1，控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
                         random_state=42)  # 随机种子
xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=50)

# Evaluate the model
y_train_pred = xgb_model.predict(X_train)
y_val_pred = xgb_model.predict(X_val)
y_test_pred = xgb_model.predict(X_test)  # Predict future meantemp

# Calculate metrics for training and validation sets
train_rmse = mean_squared_error(y_train, y_train_pred, squared=False)
val_rmse = mean_squared_error(y_val, y_val_pred, squared=False)
train_r2 = r2_score(y_train, y_train_pred)
val_r2 = r2_score(y_val, y_val_pred)

print(f"Train RMSE: {train_rmse}, R²: {train_r2}")
print(f"Validation RMSE: {val_rmse}, R²: {val_r2}")

# Combine predictions for plotting
future_years = future_data['date']
pred_y = np.concatenate([y_train_pred, y_val_pred, y_test_pred])
true_y = np.concatenate([y_train, y_val, [None] * len(y_test_pred)])  # No true values for future data
years = pd.concat([historical_data['date'], future_years])

# Plotting results
plt.figure(figsize=(15, 9))
plt.title("XGBoost Predictions vs. True Values")
x0 = [i for i in range(len(true_y))]
plt.plot(x0, pred_y, marker="o", markersize=1, label="Predictions")
plt.plot(x0, true_y, marker="o", markersize=1, label="True Values")
plt.xlabel("Index")
plt.ylabel("Temperature (℃)")
plt.legend()
plt.show()

# Plotting results
plt.figure(figsize=(15, 9))
plt.title("XGBoost furture Predictions")
x1 = [i for i in range(len(y_test_pred))]
plt.plot(x1, y_test_pred, marker="o", markersize=1, label="Predictions")
plt.xlabel("Index")
plt.ylabel("Temperature (℃)")
plt.legend()
plt.show()

# Plotting results
plt.figure(figsize=(15, 9))
plt.title("XGBoost furture Predictions")
x2 = [i for i in range(len( y_val_pred))]
plt.plot(x2,  y_val_pred, marker="o", markersize=1, label="Predictions")
plt.plot(x2,  y_val, marker="o", markersize=1, label="Predictions")
plt.xlabel("Index")
plt.ylabel("Temperature (℃)")
plt.legend()
plt.show()

# Feature importance
feature_importances = xgb_model.feature_importances_
importance_df = pd.DataFrame({'Feature': predictors, 'Importance': feature_importances})
importance_df = importance_df.sort_values('Importance', ascending=False)

# Print feature importances
print("\nFeature Importances:")
print(importance_df)

# Plot feature importances
plt.figure(figsize=(12, 6))
sns.barplot(data=importance_df, x='Importance', y='Feature', palette="viridis")
plt.title('Feature Importances from XGBoost')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()