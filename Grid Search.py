from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

# 设置随机种子
seed_value = 2023
np.random.seed(seed_value)

def grid_search(estimator, param_grid, X, y, cv):
    """
    网格搜索算法实现
    :param estimator: 估计器对象
    :param param_grid: 待搜索的参数空间，格式为dict
    :param X: 特征数据
    :param y: 目标数据
    :param cv: 交叉验证生成器对象
    :return: 最佳参数组合和对应的得分
    """
    # 将参数空间转换为列表
    param_lists = [param_grid[key] for key in param_grid]
    # 使用itertools.product()函数生成所有参数组合
    param_combinations = list(product(*param_lists))
    # 初始化最佳得分和最佳参数
    best_score = -np.inf
    best_params = None
    # 遍历所有参数组合并进行交叉验证
    for params in param_combinations:
        # 设置当前参数组合
        estimator.set_params(**dict(zip(param_grid.keys(), params)))
        # 计算当前参数组合的得分
        scores = []
        for train_index, test_index in cv.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            estimator.fit(X_train, y_train)
            scores.append(estimator.score(X_test, y_test))
        # 计算当前参数组合的平均得分
        mean_score = np.mean(scores)
        # 如果当前参数组合的得分更好，则更新最佳得分和最佳参数
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    # 返回最佳参数组合和对应的得分
    return best_params, best_score

# 导入数据
X, y = load_iris(return_X_y=True)
X = X[:, :2]

# 划分训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=seed_value)

# 定义评估器
model = DecisionTreeClassifier()

# 定义搜索参数列表
params = {
            
    'max_depth': [3, 5, 7],
    'criterion': ['gini', 'entropy'],
    'min_samples_split': [2, 3, 4]
}

# 生成器对象
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 网格搜索
grid_search(model, params, X, y, cv)

# 使用最优参数定义模型
model = DecisionTreeClassifier(max_depth=3, criterion='gini', min_samples_split=2)
model.fit(X_train, y_train)

# 结果
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

score_train = model.score(X_train, y_train)
score_test = model.score(X_test, y_test)

print('训练集Accuracy: ', score_train)
print('测试集Accuracy: ', score_test)

# 可视化决策边界
x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))
Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)
plt.contourf(xx1, xx2, Z, cmap=plt.cm.Spectral)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
plt.savefig('a.png', dpi=720)
plt.show()