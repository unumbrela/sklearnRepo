import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 设置随机种子
seed_value = 2023
np.random.seed(seed_value)

# 定义KNN模型
class KNN:
    def __init__(self, k=3):
        self.k = k
        
    # 模型训练
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        
    # 模型预测
    def predict(self, X):
        y_pred = []
        for sample in X:
            distances = []
            for i in range(len(self.X_train)):
                distance = np.sqrt(np.sum((sample - self.X_train[i])**2))
                distances.append((distance, self.y_train[i]))
            distances.sort()
            neighbors = distances[:self.k]
            classes = {
            }
            for neighbor in neighbors:
                if neighbor[1] in classes:
                    classes[neighbor[1]] += 1
                else:
                    classes[neighbor[1]] = 1
            y_pred.append(max(classes, key=classes.get))
        return y_pred

    # 计算准确率
    def score(self, y_pred, y):
        accuracy = (y_pred == y).sum() / len(y)
        return accuracy

# 导入数据
X, y = load_iris(return_X_y=True)
X = X[:, :2]

# 划分训练集、测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=seed_value)

# 训练K近邻模型
model = KNN(k=10)
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