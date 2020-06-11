"""
Adjustment for chance in clustering performance evaluation
(Điều chỉnh cơ hội trong đánh giá hiệu suất phân cụm)
"""
# Các sơ đồ sau đây cho thấy tác động của số lượng cụm và số lượng mẫu trên các số
# liệu đánh giá hiệu suất phân cụm khác nhau.

# Các biện pháp không được điều chỉnh như Biện pháp V cho thấy sự phụ thuộc giữa số
# lượng cụm và số lượng mẫu: V-Đo trung bình của ghi nhãn ngẫu nhiên tăng đáng kể
# khi số lượng cụm gần với tổng số mẫu được sử dụng để tính toán các biện pháp.

# Được điều chỉnh để đo lường cơ hội như ARI ​​hiển thị một số biến thể ngẫu nhiên
# tập trung quanh điểm trung bình 0,0 cho bất kỳ số lượng mẫu và cụm.

# Do đó, chỉ các biện pháp được điều chỉnh mới có thể được sử dụng một cách an toàn
# như một chỉ số đồng thuận để đánh giá tính ổn định trung bình của các thuật toán
# phân cụm cho giá trị k cho trên các mẫu con chồng chéo khác nhau của tập dữ liệu.

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn import metrics


def uniform_labelings_scores(score_func, n_samples, n_clusters_range,
                             fixed_n_classes=None, n_runs=5, seed=42):
    """Tính toán điểm cho 2 nhãn phân cụm đồng nhất ngẫu nhiên.

    Cả hai nhã ngẫu nhiên có cùng số cụm cho mỗi giá trị, giá
    trị có thể có trong ``n_clustftimewe``.

    Khi các lớp cố định không phải là không có nhã đầu tiên
    được coi là mặt đất phân công lớp thật với số lượng cố định
    của các lớp.
    """

    random_lables = np.random.RandomState(seed).randint
    scores = np.zeros((len(n_clusters_range), n_runs))

    if fixed_n_classes is not None:
        labels_a = random_lables(low=0, high=fixed_n_classes, size=n_samples)

    for i, k in enumerate(n_clusters_range):
        for j in range(n_runs):
            if fixed_n_classes is None:
                labels_a = random_lables(low=0, high=k, size=n_samples)
            labels_b = random_lables(low=0, high=k, size=n_samples)
            scores[i, j] = score_func(labels_a, labels_b)
    return scores


def ami_score(U, V):
    return metrics.adjusted_mutual_info_score(U, V)


score_funcs = [
    metrics.adjusted_rand_score,
    metrics.v_measure_score,
    ami_score,
    metrics.mutual_info_score,
]

# 2 phân cụm ngẫu nhiên độc lập với số cụm bằng nhau

n_samples = 100
n_clusters_range = np.linspace(2, n_samples, 10).astype(np.int)

plt.figure(1)

plots = []
names = []
for score_func in score_funcs:
    print("Computing %s for %d values of n_clusters and n_samples=%d"
          % (score_func.__name__, len(n_clusters_range), n_samples))

    t0 = time()
    scores = uniform_labelings_scores(score_func, n_samples, n_clusters_range)
    print("done in %0.3fs" % (time() - t0))
    plots.append(plt.errorbar(
        n_clusters_range, np.median(scores, axis=1), scores.std(axis=1))[0])
    names.append(score_func.__name__)

plt.title("Clustering measures for 2 random uniform labelings\n"
          "with equal number of clusters")
plt.xlabel('Number of clusters (Number of samples is fixed to %d)' % n_samples)
plt.ylabel('Score value')
plt.legend(plots, names)
plt.ylim(bottom=-0.05, top=1.05)

# Ghi nhãn ngẫu nhiên với n_cluster khác nhau so với nhãn lớp mặt đất
# với số cụm nhất định

n_samples = 1000
n_clusters_range = np.linspace(2, 100, 10).astype(np.int)
n_classes = 10

plt.figure(2)

plots = []
names = []
for score_func in score_funcs:
    print("Computing %s for %d values of n_clusters and n_samples=%d"
           % (score_func.__name__, len(n_clusters_range), n_samples))

    t0 = time()
    scores = uniform_labelings_scores(score_func, n_samples, n_clusters_range,
                                      fixed_n_classes=n_classes)
    print("done in %0.3fs" % (time() - t0))
    plots.append(plt.errorbar(
        n_clusters_range, scores.mean(axis=1), scores.std(axis=1))[0])
    names.append(score_func.__name__)

plt.title("Clustering measures for random uniform labeling\n"
          "against reference assignment with %d classes" % n_classes)
plt.xlabel('Number of clusters (Number of samples is fixed to %d)' % n_samples)
plt.ylabel('Score value')
plt.ylim(bottom=-0.05, top=1.05)
plt.legend(plots, names)
plt.show()
