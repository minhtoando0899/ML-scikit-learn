"""
Agglomerative clustering with different metrics
(phân cụm kết tụ với các số liệu khác nhau)

Thể hiện tác dụng của các số liệu khác nhau trên phân cụm phân cấp.
"""

# Ví dụ được thiết kế để hiển thị hiệu quả của việc lựa chọn các số liệu khác nhau.
# Nó được áp dụng cho dạng sóng, có thể được xem như là vectơ chiều cao. Thật vậy,
# sự khác biệt giữa các số liệu thường rõ rệt hơn ở chiều cao (đặc biệt đối với
# euclid và cityblock).
#
# Chúng tôi tạo dữ liệu từ ba nhóm dạng sóng. Hai trong số các dạng sóng (dạng sóng
# 1 và dạng sóng 2) tỷ lệ thuận với nhau. Khoảng cách cosin là bất biến đối với tỷ lệ
# của dữ liệu, do đó, nó không thể phân biệt hai dạng sóng này. Do đó, ngay cả khi
# không có nhiễu, việc phân cụm sử dụng khoảng cách này sẽ không tách rời dạng sóng
# 1 và 2.
#
# Chúng tôi thêm nhiễu quan sát cho các dạng sóng này. Chúng tôi tạo ra tiếng ồn rất thưa
# thớt: chỉ có 6% số điểm thời gian có chứa tiếng ồn. Kết quả là, chỉ tiêu l1 của tiếng ồn
# này (tức là khoảng cách city cityblock) nhỏ hơn nhiều so với định mức L2 (khoảng cách
# của e eididean). Điều này có thể được nhìn thấy trên các ma trận khoảng cách giữa các lớp:
# các giá trị trên đường chéo, đặc trưng cho sự lan truyền của lớp, lớn hơn nhiều đối với
# khoảng cách Euclide so với khoảng cách cityblock.
#
# Khi chúng tôi áp dụng phân cụm cho dữ liệu, chúng tôi thấy rằng phân cụm phản ánh
# những gì trong ma trận khoảng cách. Thật vậy, đối với khoảng cách Euclide, các lớp
# bị tách biệt do nhiễu và do đó, cụm không tách rời các dạng sóng. Đối với khoảng cách
# cityblock, sự phân tách là tốt và các lớp dạng sóng được phục hồi. Cuối cùng, khoảng
# cách cosin không tách rời ở tất cả các dạng sóng 1 và 2, do đó, cụm này đặt chúng vào
# cùng một cụm.

import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances

np.random.seed(0)

# Tạo dữ liệu dang sóng
n_features = 2000
t = np.pi * np.linspace(0, 1, n_features)


def sqr(X):
    return np.sign(np.cos(X))


X = list()
y = list()
for i, (phi, a) in enumerate([(.5, .15), (.5, .6), (.3, .2)]):
    for _ in range(30):
        phase_noise = .01 * np.random.normal()
        amplitude_noise = .04 * np.random.normal()
        additional_noise = 1 - 2 * np.random.rand(n_features)
        # làm cho tiếng ồn dải rác
        additional_noise[np.abs(additional_noise) < .997] = 0

        X.append(12 * ((a + amplitude_noise)
                       * (sqr(6 * (t + phi + phase_noise)))
                       + additional_noise))
        y.append(i)

X = np.array(X)
y = np.array(y)

n_clusters = 3

labels = ('Waveform 1', 'Waveform 2', 'Waveform 3')

# phác đồ nhãn ground-truth
plt.figure()
plt.axes([0, 0, 1, 1])
for l, c, n in zip(range(n_clusters), 'rgb',
                   labels):
    lines = plt.plot(X[y == l].T, c=c, alpha=.5)
    lines[0].set_label(n)

plt.legend(loc='best')

plt.axis('tight')
plt.axis('off')
plt.suptitle("Ground truth", size=20)

# phác họa khoảng cách
for index, metric in enumerate(["cosine", "euclidean", "cityblock"]):
    avg_dist = np.zeros((n_clusters, n_clusters))
    plt.figure(figsize=(5, 4.5))
    for i in range(n_clusters):
        for j in range(n_clusters):
            avg_dist[i, j] = pairwise_distances(X[y == i], X[y == j],
                                                metric=metric).mean()
    avg_dist /= avg_dist.max()
    for i in range(n_clusters):
        for j in range(n_clusters):
            plt.text(i, j, '%5.3f' % avg_dist[i, j],
                     verticalalignment='center',
                     horizontalalignment='center')

    plt.imshow(avg_dist, interpolation='nearest', cmap=plt.cm.gnuplot2,
               vmin=0)
    plt.xticks(range(n_clusters), labels, rotation=45)
    plt.yticks(range(n_clusters), labels)
    plt.colorbar()
    plt.suptitle("Interclass %s distance" % metric, size=18)
    plt.tight_layout()

# phác họa kết quả phân cụm
for index, metric in enumerate(["cosine", "euclidean", "cityblock"]):
    model = AgglomerativeClustering(n_clusters=n_clusters,
                                    linkage="average", affinity=metric)
    model.fit(X)
    plt.figure()
    plt.axes([0, 0, 1, 1])
    for l, c in zip(np.arange(model.n_clusters), 'rgbk'):
        plt.plot(X[model.labels_ == l].T, c=c, alpha=.5)
    plt.axis('tight')
    plt.axis('off')
    plt.suptitle("AgglomerativeClustering(affinity=%s)" % metric, size=20)

plt.show()
