"""Empirical evaluation of the impact of k-means initialization
(Đánh giá thực nghiệm về tác động của khởi tạo k-mean)
"""
# Đánh giá khả năng của các chiến lược khởi tạo k-mean để làm cho thuật toán hội tụ
# mạnh mẽ khi được đo bằng độ lệch chuẩn tương đối của quán tính của cụm (nghĩa là
# tổng khoảng cách bình phương đến trung tâm cụm gần nhất).
#
# Biểu đồ đầu tiên cho thấy quán tính tốt nhất đạt được cho mỗi kết hợp của mô hình
# (KMeans hoặc MiniBatchKMeans) và phương thức init (init = "Random" hoặc
# init = "kmeans ++") để tăng giá trị của tham số n_init kiểm soát số lượng khởi tạo.
#
# Biểu đồ thứ hai thể hiện một lần chạy duy nhất của công cụ ước tính MiniBatchKMeans
# bằng cách sử dụng init = "ngẫu nhiên" và n_init = 1. Chạy này dẫn đến một sự hội
# tụ xấu (tối ưu cục bộ) với các trung tâm ước tính bị mắc kẹt giữa các cụm sự thật mặt đất.
#
# Tập dữ liệu được sử dụng để đánh giá là một lưới 2D của các cụm Gaussian đẳng
# hướng cách đều nhau.

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.utils import shuffle
from sklearn.utils import check_random_state
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans

random_state = np.random.RandomState(0)

# Số lần chạy (với bộ dữ liệu được tạo ngẫu nhiên) cho mỗi chiến lược
# để có thế tính toán được độ lệnh chuẩn.
n_runs = 5

# các mô hình k-means có thể thực hiện một số thao tác ngẫn nhiên để có
# thể giao dịch thời gian CPU cho sự hội tụ mạnh mẽ
n_init_range = np.array([1, 5, 10, 15, 20])

# Thông số tạo các tập dữ liệu
n_samples_per_center = 100
grid_size = 3
scale = 0.1
n_clusters = grid_size ** 2


def make_data(random_state, n_samples_per_center, grid_size, scale):
    random_state = check_random_state(random_state)
    centers = np.array([[i, j]
                        for i in range(grid_size)
                        for j in range(grid_size)])
    n_clusters_true, n_features = centers.shape

    noise = random_state.normal(
        scale=scale, size=(n_samples_per_center, centers.shape[1]))

    X = np.concatenate([c + noise for c in centers])
    y = np.concatenate([[i] * n_samples_per_center
                        for i in range(n_clusters_true)])
    return shuffle(X, y, random_state=random_state)


# phần 1: đánh giá định lượng của các phương pháp init khác nhau

plt.figure()
plots = []
legends = []

cases = [
    (KMeans, 'k-means++', {}),
    (KMeans, 'random', {}),
    (MiniBatchKMeans, 'k-means++', {'max_no_improvement': 3}),
    (MiniBatchKMeans, 'random', {'max_no_improvement': 3, 'init_size': 500}),
]

for factory, init, params in cases:
    print("Evaluation of %s with %s init" % (factory.__name__, init))
    inertia = np.empty((len(n_init_range), n_runs))

    for run_id in range(n_runs):
        X, y = make_data(run_id, n_samples_per_center, grid_size, scale)
        for i, n_init in enumerate(n_init_range):
            km = factory(n_clusters=n_clusters, init=init, random_state=run_id,
                         n_init=n_init, **params).fit(X)
            inertia[i, run_id] = km.inertia_
    p = plt.errorbar(n_init_range, inertia.mean(axis=1), inertia.std(axis=1))
    plots.append(p[0])
    legends.append("%s with %s init" % (factory.__name__, init))

plt.xlabel('n_init')
plt.ylabel('inertia')
plt.legend(plots, legends)
plt.title("Mean inertia for various k-means init across %d runs" % n_runs)

# Phần 2: kiểm tra trực quan định tính về sự hội tụ

X, y = make_data(random_state, n_samples_per_center, grid_size, scale)
km = MiniBatchKMeans(n_clusters=n_clusters, init='random', n_init=1,
                    random_state=random_state).fit(X)

plt.figure()
for k in range(n_clusters):
    my_members = km.labels_ == k
    color = cm.nipy_spectral(float(k) / n_clusters, 1)
    plt.plot(X[my_members, 0], X[my_members, 1], 'o', marker='.', c=color)
    cluster_center = km.cluster_centers_[k]
    plt.plot(cluster_center[0], cluster_center[1], 'o',
             markerfacecolor=color, markeredgecolor='k', markersize=6)
    plt.title("Example cluster allocation with a single random init\n"
              "with MinibatchKMeans")

plt.show()