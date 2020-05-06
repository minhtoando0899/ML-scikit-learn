"""
K-means Clustering
Phân cụm k-mean
"""
# Các sơ đồ hiển thị trước hết những gì thuật toán K-mean sẽ mang lại bằng cách
# sử dụng ba cụm. Sau đó, nó cho thấy tác động của việc khởi tạo xấu là gì đối
# với quá trình phân loại: Bằng cách đặt n_init thành chỉ 1 (mặc định là 10), số
# lần thuật toán sẽ được chạy với các hạt trung tâm khác nhau sẽ giảm. Cốt truyện
# tiếp theo hiển thị những gì sử dụng tám cụm sẽ cung cấp và cuối cùng là sự thật
# nền tảng.

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
# Mặc dù việc nhập sau không được sử dụng trực tiếp, nhưng nó được yêu cầu
# để chiếu 3D hoạt động.
from mpl_toolkits.mplot3d import Axes3D

from sklearn.cluster import KMeans
from sklearn import datasets

np.random.seed(5)

iris = datasets.load_iris()
X = iris.data
y = iris.target

estimators = [('k_mean_iris_8', KMeans(n_clusters=8)),
              ('k_mean_iris_3', KMeans(n_clusters=3)),
              ('k_mean_iris_bad_init', KMeans(n_clusters=3, n_init=1,
                                              init='random'))]
fignum = 1
titles = ['8 clusters', '3 clusters', '3 clusters, bad initialization']
for name, est in estimators:
    fig = plt.figure(fignum, figsize=(4, 3))
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    est.fit(X)
    labels = est.labels_

    ax.scatter(X[:, 3], X[:, 0], X[:, 2],
               c=labels.astype(np.float), edgecolor='k')

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    ax.set_title(titles[fignum - 1])
    ax.dist = 12
    fignum = fignum + 1

# phác họa xác thật
fig = plt.figure(fignum, figsize=(4, 3))
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

for name, label in [('Setosa', 0),
                    ('Versicolour', 1),
                    ('Virginica', 2)]:
    ax.text3D(X[y == label, 3].mean(),
              X[y == label, 0].mean(),
              X[y == label, 2].mean() + 2, name,
              horizontalalignment='center',
              bbox=dict(alpha=.2, edgecolor='w', facecolor='w'))
# Sắp xếp lại các nhãn để có màu tương xứng với kết quả của phân cụm
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y, edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])
ax.set_xlabel('Petal width')
ax.set_ylabel('Sepal length')
ax.set_zlabel('Petal length')
ax.set_title("Ground Truth")
ax.dist = 12

fig.show()
