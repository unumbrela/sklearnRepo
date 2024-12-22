import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置随机种子
seed_value = 2023
np.random.seed(seed_value)

class KMeans:
    def __init__(self, k=3, max_iter=100, tol=0.0001):
        self.k = k  # 聚类的数量
        self.max_iter = max_iter  # 最大迭代次数
        self.tol = tol  # 收敛容忍度

    def fit(self, X):
        # 随机初始化聚类中心
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False), :]

        # 迭代聚类过程
        for i in range(self.max_iter):
            # 计算每个点到聚类中心的距离
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))

            # 分配每个点到距离最近的聚类中心
            labels = distances.argmin(axis=0)

            # 计算新的聚类中心
            new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(self.k)])

            # 计算收敛误差
            delta = np.abs(self.centroids - new_centroids).max()

            # 如果收敛误差小于容忍度，退出迭代
            if delta < self.tol:
                break

            # 更新聚类中心
            self.centroids = new_centroids

    def predict(self, X):
        # 计算每个点到聚类中心的距离
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))

        # 分配每个点到距离最近的聚类中心
        labels = distances.argmin(axis=0)

        return labels

# 导入数据
data = pd.read_csv("./cluster_500-10_7.csv", encoding="gbk")
X = data.iloc[:, 1:-2].values
y = data.iloc[:, -1].values.flatten()

# 训练KMeans模型
cluster = 7
model = KMeans(k=cluster)
model.fit(X)

y_pred = model.predict(X)

# 聚类结果可视化
# 绘制颜色
color = ['red', 'green', 'orange', 'dimgray', 'gold', 'khaki', 'lime']

# 对每个类别样本进行绘制散点图
for i in range(cluster):
    # X[y == i][:, 0]的意思就是先获取等于i类别的索引，然后取出对应的数据，然后取出第一列用于x轴
    plt.scatter(X[y == i][:, 0],
                X[y == i][:, 1],
                c=color[i])
    
plt.savefig('a.png', dpi=720)
plt.show()

# 对每个类别样本进行绘制散点图
for i in range(cluster):
    plt.scatter(X[y_pred == i][:, 0],
                X[y_pred == i][:, 1],
                c=color[i])
    
plt.savefig('b.png', dpi=720)
plt.show()