"""
Various Agglomerative Clustering on a 2D embedding of digits
Phân cụm kết tụ khác nhau trên 1 bản in 2D của các số
"""

# Một minh họa về các tùy chọn liên kết khác nhau để phân cụm kết tụ
# trên một nhúng 2D của bộ dữ liệu chữ số.
#
# Mục tiêu của ví dụ này là chỉ ra bằng trực giác cách thức các số liệu
# ứng xử và không tìm thấy các cụm tốt cho các chữ số. Đây là lý do tại
# sao ví dụ hoạt động trên nhúng 2D.
#
# Điều mà ví dụ này cho chúng ta thấy là hành vi "Giàu giàu ngày càng giàu
# hơn" của cụm phân cụm có xu hướng tạo ra các kích thước cụm không đồng đều.
# Hành vi này được phát âm cho chiến lược liên kết trung bình, kết thúc
# bằng một vài cụm đơn, trong trường hợp liên kết đơn, chúng ta có một cụm
# trung tâm duy nhất với tất cả các cụm khác được rút ra từ các điểm nhiễu
# xung quanh rìa.

print(__doc__)
from time import time

import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt

from sklearn import manifold, datasets

X, y = datasets.load_digits(return_X_y=True)
n_samples, n_features = X.shape

np.random.seed(0)


def nudge_images(X, y):
    # Có một tập dữ liệu lớn cho thấy rõ hơn về hành vi của
    # các phương pháp, nhưng chúng ta nhân kích thước của tập dữ liệu chỉ với 2,
    # như chi phí của các phương phác phân cụm phân cụm phân cấp mạnh mẽ
    # siêu tuyến tính trong n_samples
    shift = lambda x: ndimage.shift(x.reshape((8, 8)),
                                    .3 * np.random.normal(size=2),
                                    mode='constant',
                                    ).ravel()
    X = np.concatenate([X, np.apply_along_axis(shift, 1, X)])
    Y = np.concatenate([y, y], axis=0)
    return X, Y


X, y = nudge_images(X, y)


# -------------------------------------------------------------------
# Trực quan hóa phân cụm
def plot_clustering(X_red, labels, title=None):
    x_min, x_max = np.min(X_red, axis=0), np.max(X_red, axis=0)
    X_red = (X_red - x_min) / (x_max - x_min)

    plt.figure(figsize=(6, 4))
    for i in range(X_red.shape[0]):
        plt.text(X_red[i, 0], X_red[i, 1], str(y[i]),
                 color=plt.cm.nipy_spectral(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# ----------------------------------------------------------------------
# Nhúng 2D của tập dữ liệu các số
print("Computing embedding")
X_red = manifold.SpectralEmbedding(n_components=2).fit_transform(X)
print("Done.")

from sklearn.cluster import AgglomerativeClustering

for linkage in ('ward', 'average', 'complete', 'single'):
    clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)
    t0 = time()
    clustering.fit(X_red)
    print("%s :\t%.2fs" % (linkage, time() - t0))

    plot_clustering(X_red, clustering.labels_, "%s linkage" % linkage)

plt.show()
