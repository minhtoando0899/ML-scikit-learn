"""
Compare BIRCH and MiniBatchKMeans
(So sánh BIRCH và MinibatchkMeans)

Ví dụ này so sánh thời gian của Birch (có và không có bước phân cụm toàn cầu)
và MiniBatchKMeans trên một tập dữ liệu tổng hợp có 100.000 mẫu và 2 tính năng
được tạo bằng make_blobs.

Nếu n_cluster được đặt thành Không, dữ liệu sẽ giảm từ 100.000 mẫu xuống còn
158 cụm. Điều này có thể được xem như một bước tiền xử lý trước bước phân cụm
(toàn cầu) cuối cùng làm giảm thêm 158 cụm này xuống còn 100 cụm.
"""

print(__doc__)

from itertools import cycle
from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from sklearn.cluster import Birch, MiniBatchKMeans
from sklearn.datasets import make_blobs

# Tạo trung tâm cho các đốm màu để nó tạo thành một lưới 10 x 10.
xx = np.linspace(-22, 22, 10)
yy = np.linspace(-22, 22, 10)
xx, yy = np.meshgrid(xx, yy)
n_centers = np.hstack((np.ravel(xx)[:, np.newaxis],
                       np.ravel(yy)[:, np.newaxis]))

# Tạo các đốm màu để so sánh giữa MiniBatchKMeans và Birch
X, y = make_blobs(n_samples=100000, centers=n_centers, random_state=0)

# Sử dụng tất cả các màu mà matplotlib cung cấp theo mặc định
colors_ = cycle(colors.cnames.keys())

fig = plt.figure(figsize=(12, 4))
fig.subplots_adjust(left=0.04, right=0.98, bottom=0.1, top=0.9)

# tính toán phân cụm với Birch và không có bước phân cụm cuối cùng và phác họa.
birch_models = [Birch(threshold=1.7, n_clusters=None),
                Birch(threshold=1.7, n_clusters=100)]
final_step = ['without global clustering', 'with global clustering']

for ind, (birch_models, info) in enumerate(zip(birch_models, final_step)):
    t = time()
    birch_models.fit(X)
    time_ = time() - t
    print("Birch %s as the final step took %0.2f second" % (
        info, (time() - t)))

    # phác họa kết quả
    labels = birch_models.labels_
    centroids = birch_models.subcluster_centers_
    n_clusters = np.unique(labels).size
    print("n_clusters : %d" % n_clusters)

    ax = fig.add_subplot(1, 3, ind + 1)
    for this_centroid, k, col in zip(centroids, range(n_clusters), colors_):
        mask = labels == k
        ax.scatter(X[mask, 0], X[mask, 1],
                   c='w', edgecolor=col, marker='.', alpha=0.5)
        if birch_models.n_clusters is None:
            ax.scatter(this_centroid[0], this_centroid[1], marker='+',
                       c='k', s=25)
    ax.set_ylim([-25, 25])
    ax.set_xlim([-25, 25])
    ax.set_autoscaley_on(False)
    ax.set_title('Birch %s' % info)

# Tính toán phân cụm với MiniBatchKMeans.
mbk = MiniBatchKMeans(init='k-means++', n_clusters=100, batch_size=100,
                      n_init=10, max_no_improvement=10, verbose=0,
                      random_state=0)
t0 = time()
mbk.fit(X)
t_mini_batch = time() - t0
print("Time taken to run MiniBatchKMeans %0.2f seconds" % t_mini_batch)
mbk_means_labels_unique = np.unique(mbk.labels_)

ax = fig.add_subplot(1, 3, 3)
for this_centroid, k, col in zip(mbk.cluster_centers_,
                                 range(n_clusters), colors_):
    mask = mbk.labels_ == k
    ax.scatter(X[mask, 0], X[mask, 1], marker='.',
               c='w', edgecolors=col, alpha=0.5)
    ax.scatter(this_centroid[0], this_centroid[1], marker='+',
               c='k', s=25)

ax.set_xlim([-25, 25])
ax.set_ylim([-25, 25])
ax.set_title("MiniBatchKMeans")
ax.set_autoscaley_on(False)
plt.show()
