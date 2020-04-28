"""
===================================================
Agglomerative clustering with and without structure
(Phân cụm kết tụ có và không có cấu trúc).
===================================================
Ví dụ này cho thấy hiệu quả của việc áp đặt biểu đồ
kết nối để nắm bắt cấu trúc cục bộ trong dữ liệu.
Biểu đồ chỉ đơn giản là biểu đồ của 20 hàng xóm gần nhất.
"""
# Hai hậu quả của việc áp đặt một kết nối có thể được nhìn thấy.
# Phân cụm đầu tiên với một ma trận kết nối nhanh hơn nhiều.
#
# Thứ hai, khi sử dụng ma trận kết nối, liên kết đơn, trung bình
# và hoàn chỉnh không ổn định và có xu hướng tạo ra một số cụm
# tăng trưởng rất nhanh. Thật vậy, liên kết trung bình và hoàn chỉnh
# chống lại hành vi xâm nhập này bằng cách xem xét tất cả khoảng
# cách giữa hai cụm khi hợp nhất chúng (trong khi liên kết đơn sẽ
# phóng đại hành vi bằng cách chỉ xem xét khoảng cách ngắn nhất
# giữa các cụm). Biểu đồ kết nối phá vỡ cơ chế này để liên kết
# trung bình và hoàn chỉnh, làm cho chúng giống với liên kết đơn
# dễ vỡ hơn. Hiệu ứng này rõ rệt hơn đối với các biểu đồ rất thưa thớt
# (thử giảm số lượng hàng xóm trong kneighbor_graph) và với liên kết
# hoàn chỉnh. Đặc biệt, có một số lượng rất nhỏ các lân cận trong
# biểu đồ, áp đặt một hình học gần với liên kết đơn, được biết đến là
# có sự mất ổn định màu sắc này.

import time
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

# Tạo dữ liệu mẫu
n_samples = 1500
np.random.seed(0)
t = 1.5 * np.pi * (1 + 3 * np.random.rand(1, n_samples))
x = t * np.cos(t)
y = t * np.sin(t)


X = np.concatenate((x, y))
X += .7 * np.random.randn(2, n_samples)
X = X.T

# Tạo một biểu đồ chụp kết nối cục bộ. Số lượng lớn hơn của hàng xóm
# sẽ cung cấp các cụm đồng nhất hơn cho chi phí tính toán
# thời gian. Một số lượng lớn hàng xóm phân phối đồng đều hơn
# kích thước cụm, nhưng có thể không áp đặt cấu trúc đa tạp cục bộ của
# dữ liệu
knn_graph = kneighbors_graph(X, 30, include_self=False)

for connectivity in (None, knn_graph):
    for n_clusters in (30, 3):
        plt.figure(figsize=(10, 4))
        for index, linkage in enumerate(('average',
                                         'complete',
                                         'ward',
                                         'single')):
            plt.subplot(1, 4, index + 1)
            model = AgglomerativeClustering(linkage=linkage,
                                            connectivity=connectivity,
                                            n_clusters=n_clusters)
            t0 = time.time()
            model.fit(X)
            elapsed_time = time.time() - t0
            plt.scatter(X[:, 0], X[:, 1], c=model.labels_,
                        cmap=plt.cm.nipy_spectral)
            plt.title('linkage=%s\n(time %.2fs)' % (linkage, elapsed_time),
                      fontdict=dict(verticalalignment='top'))
            plt.axis('equal')
            plt.axis('off')

            plt.subplots_adjust(bottom=0, top=.89, wspace=0,
                                left=0, right=1)
            plt.suptitle('n_cluster=%i, connectivity=%r' %
                         (n_clusters, connectivity is not None), size=17)

plt.show()
