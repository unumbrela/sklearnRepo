import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 设置随机种子
seed_value = 2023
np.random.seed(seed_value)

class NaiveBayes:
    def __init__(self):
        pass
    
    # 模型训练
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.mean = {
            }
        self.var = {
            }
        self.prior = {
            }
        # 计算每个类别的先验概率
        for c in self.classes:
            self.prior[c] = np.sum(self.y == c) / len(self.y)
        # 计算每个类别的均值和方差
        for c in self.classes:
            X_c = self.X[self.y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0)
            
    # 模型预测
    def predict(self, X):
        posterior = []
        for c in self.classes:
            prior = np.log(self.prior[c])
            likelihood = np.sum(np.log(self.prob_density(X, c)), axis=1)
            posterior_c = prior + likelihood
            posterior.append(posterior_c)
        # 返回具有最大后验概率的类别
        return self.classes[np.argmax(posterior, axis=0)]
    
    # 计算给定类别和特征下的概率密度
    def prob_density(self, X, c):
        mean = self.mean[c]
        var = self.var[c]
        numerator = np.exp(-(X - mean)**2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
    
    # 精度
    def score(self, y_pred, y):
        accuracy = (y_pred == y).sum() / len(y)
        return accuracy

# 导入数据
X, y = load_iris(return_X_y=True)
X = X[:, :2]

# 划分训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=seed_value)

# 训练贝叶斯模型
model = NaiveBayes()
model.fit(X_train, y_train)

# 结果
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

score_train = model.score(y_train_pred, y_train)
score_test = model.score(y_test_pred, y_test)

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