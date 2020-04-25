"""
=============================================================
A demo of the mean-shift clustering algorithm
(một bản demo của thuật toán phân cụm mean-shift)
=============================================================
"""
print(__doc__)

import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets import make_blobs

# ##########################################################################
# Tạo dữ liệu mẫu
centers = [[1, 1], [-1, -1], [1, -1]]
X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)

# ##########################################################################
# Tính toán phân cụm với Meanshift

# Băng thông sau có thể được tự động phát hiện bằng cách sử dụng
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_cluster_ = len(labels_unique)

print("Number of estimated clusters : %d" % n_cluster_)

# ###############################################################################
# plot result
import matplotlib.pyplot as plt
from itertools import cycle

plt.figure(1)
plt.clf()

colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_cluster_), colors):
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
plt.title('Estimated number of clusters: %d' % n_cluster_)
plt.show()
