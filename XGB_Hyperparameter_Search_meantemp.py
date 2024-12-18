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
import numpy as np
import pandas as pd
import seaborn as sns
from io import StringIO
from IPython.display import Image
import pydotplus
from pprint import pprint
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score


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

# 超参数随机匹配择优（可能为局部最优）
n_estimators_range=[int(x) for x in np.linspace(start=50,stop=500,num=50)]
max_depth_range=[int(x) for x in np.linspace(10,200,num=50)]
min_child_weight_range=[int(x) for x in np.linspace(1,10,num=10)]
learning_rate_range=[0.1,0.01,0.001]
gamma_range=[0,0.1]
booster_range=['gblinear','gbtree']
subsample_range=[0.8,0.7]
colsample_bytree_range=[0.8,1]
reg_alpha_range=[int(x) for x in np.linspace(0,1,num=10)]
reg_lambda_range=[int(x) for x in np.linspace(0,1,num=10)]

xgb_hp_range={'n_estimators':n_estimators_range,
                        'max_depth':max_depth_range,
                        'min_child_weight':min_child_weight_range,
                        'learning_rate':learning_rate_range,
                        'gamma': gamma_range,
                        'booster': booster_range,
                        'subsample': subsample_range,
                        'colsample_bytree': colsample_bytree_range,
                        'reg_alpha': reg_alpha_range,
                        'reg_lambda': reg_lambda_range,
                        # 'bootstrap':bootstrap_range
                        }
pprint(xgb_hp_range)
xgb_model_test_base=XGBRegressor()
xgb_model = RandomizedSearchCV(estimator=xgb_model_test_base,
                                                   param_distributions=xgb_hp_range,
                                                   random_state=42
                                                   )
xgb_model.fit(X_train, y_train)
best_hp_now=xgb_model.best_params_
pprint(best_hp_now)

# 超参数遍历匹配择优
param_grid = {
    'n_estimators': [50, 77, 100],
    'min_child_weight': [1,2,3,4],
    'max_depth': [120, 141, 150, 200],
    'learning_rate':[0.1],
    'objective': ['reg:squarederror'],
    'gamma': [0],
    'subsample': [0.8],
    'colsample_bytree': [1],
    'reg_alpha': [0],
    'reg_lambda': [0],
    # 'bootstrap':bootstrap_range
}

grid_search = GridSearchCV( XGBRegressor(random_state=42),
                           param_grid, n_jobs=-1,cv=3, verbose=50, scoring='r2', return_train_score=True)
#scoring='neg_mean_squared_error'

# Print detailed results of hyperparameter tuning
grid_search.fit(X_train, y_train)
print("\nGrid Search Results:")
results = pd.DataFrame(grid_search.cv_results_)
print(results[['params', 'mean_test_score', 'mean_train_score']])
print("\nBest Parameters:", grid_search.best_params_)
best_hp_now_2=grid_search.best_params_
pprint(best_hp_now_2)

# Train the optimized model
xgb_model = grid_search.best_estimator_

xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=50)
scores = cross_val_score(xgb_model, X_train, y_train, cv=3, scoring='neg_mean_squared_error')
print("Cross-Validation RMSE:", (-scores.mean()) ** 0.5) #**0.5是开根号，从MSE--RMSE
#scores = cross_val_score(xgb_model, X_train, y_train, cv=3, scoring='r2')
#print("Cross-Validation RMSE:", (scores.mean())) #**0.5是开根号，从MSE--RMSE


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