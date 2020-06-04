"""
Inductive Clustering
(Phân loại quy nạp)
"""
# Phân cụm có thể tốn kém, đặc biệt là khi tập dữ liệu của chúng tôi chứa hàng triệu
# điểm dữ liệu. Nhiều thuật toán phân cụm không phải là quy nạp và do đó không thể
# áp dụng trực tiếp vào các mẫu dữ liệu mới mà không tính toán lại phân cụm, có thể
# không thể điều chỉnh được. Thay vào đó, chúng ta có thể sử dụng phân cụm để sau đó
# tìm hiểu một mô hình quy nạp với trình phân loại, có một số lợi ích:
#
# nó cho phép các cụm chia tỷ lệ và áp dụng cho dữ liệu mới
#
# Không giống như lắp lại các cụm vào các mẫu mới, nó đảm bảo quy trình ghi nhãn
# nhất quán theo thời gian
#
# nó cho phép chúng ta sử dụng các khả năng suy luận của trình phân loại để mô tả
# hoặc giải thích các cụm
#
# Ví dụ này minh họa một triển khai chung của một công cụ ước tính meta mở rộng
# phân cụm bằng cách tạo ra một bộ phân loại từ các nhãn cụm.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, clone
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.metaestimators import if_delegate_has_method

N_SAMPLES = 5000
RANDOM_STATE = 42


class InductiveClusterer(BaseEstimator):
    def __init__(self, clusterer, classifier):
        self.clusterer = clusterer
        self.classifier = classifier

    def fit(self, X, y=None):
        self.clusterer_ = clone(self.clusterer)
        self.classifier_ = clone(self.classifier)
        y = self.clusterer_.fit_predict(X, y)
        self.classifier_.fit(X,y)
        return self

    @if_delegate_has_method(delegate='classifier_')
    def predict(self, X):
        return self.classifier_.predict(X)

    @if_delegate_has_method(delegate='classifier_')
    def dicision_function(self, X):
        return self.classifier_.decision_function(X)


def plot_scatter(X, color, alpha=0.5):
    return plt.scatter(X[:, 0],
                       X[:, 1],
                       c=color,
                       alpha=alpha,
                       edgecolors='k')


# tạo một số dữ liệu huấn luyện từ phân cụm
X, y = make_blobs(n_samples=N_SAMPLES,
                  cluster_std=[1.0, 1.0, 0.5],
                  centers=[(-5, -5), (0, 0), (5, 5)],
                  random_state=RANDOM_STATE)

# Huấn luyện một thuật toán phân loại trên dữ liệu huấn luyện và lấy các nhãn phân loại
clusterer = AgglomerativeClustering(n_clusters=3)
cluster_labels = clusterer.fit_predict(X)

plt.figure(figsize=(12, 4))

plt.subplot(131)
plot_scatter(X, cluster_labels)
plt.title("Ward Linkage")

# Tạo các mẫu mới và phác họa chúng dọc theo với bộ dữ liệu nguyên bản
X_new, y_new = make_blobs(n_samples=10,
                          centers=[(-7, -1), (-2, 4), (3, 6)],
                          random_state=RANDOM_STATE)

plt.subplot(132)
plot_scatter(X, cluster_labels)
plot_scatter(X_new, 'black', 1)
plt.title("Unknown instances")

# Khai báo mô hình học quy nạp sẽ được sử dụng để dự đoán phân loại thành viên cho
# các trường hợp không xác định.
classifier = RandomForestClassifier(random_state=RANDOM_STATE)
inductive_learner = InductiveClusterer(clusterer, classifier).fit(X)

probable_clusters = inductive_learner.predict(X_new)

plt.subplot(133)
plot_scatter(X, cluster_labels)
plot_scatter(X_new, probable_clusters)

# phác họa vùng quyết định
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

Z = inductive_learner.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.title("Classify unknown instance")

plt.show()
