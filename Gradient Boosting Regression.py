import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from xgboost import XGBRegressor 

import warnings
warnings.filterwarnings('ignore')

# 加载数据
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

print("训练集基本信息：")
print(train_data.info())
print("\n训练集描述统计：")
print(train_data.describe())

print("\n训练集缺失值情况：")
print(train_data.isnull().sum()[train_data.isnull().sum() > 0])

print("\n测试集缺失值情况：")
print(test_data.isnull().sum()[test_data.isnull().sum() > 0])

train_len = train_data.shape[0]
data = pd.concat([train_data, test_data], sort=False).reset_index(drop=True)

data.drop(['Id', 'Alley', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)


num_cols = data.select_dtypes(include=[np.number]).columns
num_imputer = SimpleImputer(strategy='median')
data[num_cols] = num_imputer.fit_transform(data[num_cols])


cat_cols = data.select_dtypes(include=['object']).columns
cat_imputer = SimpleImputer(strategy='most_frequent')
data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])


ordered_features = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                   'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond']

qual_mapping = {'Po': 1, 'Fa': 2, 'TA':3, 'Gd':4, 'Ex':5}

for feature in ordered_features:
    if feature in data.columns:
        data[feature] = data[feature].map(qual_mapping).fillna(0)

data = pd.get_dummies(data, drop_first=True)


if 'TotalBsmtSF' in data.columns and '1stFlrSF' in data.columns and '2ndFlrSF' in data.columns:
    data['TotalSF'] = data['TotalBsmtSF'] + data['1stFlrSF'] + data['2ndFlrSF']
else:

    data['TotalSF'] = 0
    if 'TotalBsmtSF' in data.columns:
        data['TotalSF'] += data['TotalBsmtSF']
    if '1stFlrSF' in data.columns:
        data['TotalSF'] += data['1stFlrSF']
    if '2ndFlrSF' in data.columns:
        data['TotalSF'] += data['2ndFlrSF']

if 'TotRmsAbvGrd' in data.columns and 'FullBath' in data.columns and 'HalfBath' in data.columns and 'BedroomAbvGr' in data.columns and 'KitchenAbvGr' in data.columns:
    data['TotalRooms'] = data['TotRmsAbvGrd'] + data['FullBath'] + data['HalfBath'] + data['BedroomAbvGr'] + data['KitchenAbvGr']
else:
    data['TotalRooms'] = 0
    if 'TotRmsAbvGrd' in data.columns:
        data['TotalRooms'] += data['TotRmsAbvGrd']
    if 'FullBath' in data.columns:
        data['TotalRooms'] += data['FullBath']
    if 'HalfBath' in data.columns:
        data['TotalRooms'] += data['HalfBath']
    if 'BedroomAbvGr' in data.columns:
        data['TotalRooms'] += data['BedroomAbvGr']
    if 'KitchenAbvGr' in data.columns:
        data['TotalRooms'] += data['KitchenAbvGr']


features_to_drop = ['TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'TotRmsAbvGrd', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr']
for feature in features_to_drop:
    if feature in data.columns:
        data.drop(feature, axis=1, inplace=True)


print("\n数据类型检查：")
print(data.dtypes.value_counts())


processed_train_data = data[:train_len]
if 'SalePrice' in processed_train_data.columns:
    processed_train_data_corr = processed_train_data.select_dtypes(include=[np.number])

    corr_matrix = processed_train_data_corr.drop('SalePrice', axis=1).corr()
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()
else:
    print("SalePrice not in data.columns")
    

X = data.drop(['SalePrice'], axis=1) if 'SalePrice' in data.columns else data
y = data['SalePrice'] if 'SalePrice' in data.columns else None

assert X.select_dtypes(include=['object']).empty, "X contains non-numeric data!"
if y is not None:
    assert y.dtype in [np.float64, np.int64], "y is not numeric!"

selector = SelectKBest(score_func=f_regression, k=30)
selector.fit(X, y)
selected_features = selector.get_support(indices=True)
selected_feature_names = X.columns[selected_features]
print("\n选择的特征数量：", len(selected_feature_names))
print("选择的特征名称：", selected_feature_names.tolist())


X = X[selected_feature_names]


scaler = StandardScaler()
X = scaler.fit_transform(X)

if 'SalePrice' in data.columns:
    X_train = X[:train_len]
    y_train = y.values
    X_test_final = X[train_len:]
else:
    X_train, X_test_final, y_train, y_test_final = train_test_split(X, y, test_size=0.2, random_state=42)


gbr = GradientBoostingRegressor(random_state=42)


param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'subsample': [0.8, 1.0]
}

grid_search = GridSearchCV(
    estimator=gbr,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring='neg_root_mean_squared_error'
)

grid_search.fit(X_train, y_train)

print("\n最佳参数：", grid_search.best_params_)
print("最佳得分（RMSE）：", -grid_search.best_score_)

best_gbr = grid_search.best_estimator_


cv = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(best_gbr, X_train, y_train, cv=cv, scoring='neg_root_mean_squared_error')
print("\n交叉验证平均RMSE：", -np.mean(cv_scores))

y_train_pred = best_gbr.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_mae = mean_absolute_error(y_train, y_train_pred)
train_r2 = r2_score(y_train, y_train_pred)
print(f"训练集 - RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}, R2: {train_r2:.4f}")

if 'SalePrice' in data.columns:
    y_pred = best_gbr.predict(X_train)
    test_rmse = np.sqrt(mean_squared_error(y_train, y_pred))
    test_mae = mean_absolute_error(y_train, y_pred)
    test_r2 = r2_score(y_train, y_pred)
    print(f"验证集 - RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}, R2: {test_r2:.4f}")
else:
    y_pred_final = best_gbr.predict(X_test_final)
    submission = pd.DataFrame({
        'Id': test_data['Id'],
        'SalePrice': y_pred_final
    })
    submission.to_csv('house_price_predictions.csv', index=False)
    print("预测结果已保存到 house_price_predictions.csv")


models = {
    'Random Forest': RandomForestRegressor(random_state=42),
    'XGBoost': XGBRegressor(random_state=42, objective='reg:squarederror')
}

params = {
    'Random Forest': {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'XGBoost': {
        'n_estimators': [100, 200],
        'learning_rate': [0.05, 0.1],
        'max_depth': [4, 6],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
}

# 模型训练
for name, model in models.items():
    print(f"\n训练模型：{name}")
    grid = GridSearchCV(
        estimator=model,
        param_grid=params[name],
        cv=5,
        n_jobs=-1,
        verbose=1,
        scoring='neg_root_mean_squared_error'
    )
    grid.fit(X_train, y_train)
    print(f"最佳参数：{grid.best_params_}")
    print(f"最佳RMSE：{-grid.best_score_:.2f}")
    cv_scores = cross_val_score(grid.best_estimator_, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
    print(f"交叉验证平均RMSE：{-np.mean(cv_scores):.2f}")


# 特征重要性可视化
importances = best_gbr.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12,8))
sns.barplot(x=importances[indices], y=np.array(selected_feature_names)[indices])
plt.title('梯度提升回归特征重要性')
plt.xlabel('重要性')
plt.ylabel('特征')
plt.show()

# 实际值 vs 预测值
plt.figure(figsize=(10,6))
plt.scatter(y_train, y_train_pred, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
plt.xlabel('实际值')
plt.ylabel('预测值')
plt.title('实际值 vs 预测值')
plt.show()

# 残差分布
residuals = y_train - y_train_pred
plt.figure(figsize=(10,6))
sns.histplot(residuals, kde=True)
plt.xlabel('残差')
plt.title('残差分布')
plt.show()
