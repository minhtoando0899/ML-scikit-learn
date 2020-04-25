"""
================================================================================
Plot Hierarchical Clustering Dendrogram (Phác họa sơ đồ cụm phân cấp Dendrogram)
================================================================================

Ví dụ này phác họa sơ đồ dendrogram tương ứng của một cụm phân cấp bằng cách
sử dụng AgglomerativeClustering và phương pháp dendrogram có sẵn trong scipy.
"""

import numpy as np

from matplotlib import pyplot as plt
from scipy.cluster.hierarchy import dendrogram
from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering


def plot_dendrogram(model, **kwargs):
    # Tạo ma trận liên kết và sau đó phác họa sơ đồ dendrogram.

    # Tạo số lượng mẫu dưới mỗi nút.
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # nút cửa
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # phác họa sơ đồ dendrogram tương ứng
    dendrogram(linkage_matrix, **kwargs)


iris = load_iris()
X = iris.data

# thiết lập distance_threshould=0 đảm bảo chúng tôi tính toán toàn bộ tree.
model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)

model = model.fit(X)
plt.title('Hierarchilcal Clustering Dendrogram')
# phác họa sơ đồ 3 cấp độ cao nhất của dendrogram
plot_dendrogram(model, truncate_mode='level', p=3)
plt.xlabel("Number of points in  node (or index of point if no parenthesis).")
plt.show()
