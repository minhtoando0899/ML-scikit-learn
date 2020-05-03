"""
Segmenting the picture of greek coins in regions
(Phân chia từng ảnh của tiền Hy lạp trong khu vực)
"""
# Ví dụ này sử dụng phân cụm Spectral trên biểu đồ được tạo từ sự khác biệt giữa voxel-voxel
# trên một hình ảnh để chia hình ảnh này thành nhiều vùng đồng nhất một phần.

# Quy trình này (phân cụm phổ trên một hình ảnh) là một giải pháp gần đúng hiệu quả
# để tìm các biểu đồ cắt chuẩn hóa.

# Có hai tùy chọn để gán nhãn:

# với phân cụm phổ ‘kmeans sẽ phân cụm các mẫu trong không gian nhúng bằng thuật toán kmeans

# trong khi đó, rời rạc sẽ lặp đi lặp lại tìm kiếm không gian phân vùng gần nhất với không gian nhúng.

print(__doc__)

import time

import numpy as np
from distutils.version import LooseVersion
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import skimage
from skimage.data import coins
from skimage.transform import rescale

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering

#  Chúng được giới thiệu trong skimage-0.14
if LooseVersion(skimage.__version__) >= '0.14':
    rescale_params = {'anti_aliasing': False, 'multichannel': False}
else:
    rescale_params = {}

# Tải các đồng tiền như 1 dãy numpy
orig_coins = coins()

# Thay đổi kích thước của nó thành 20% kích thước ban đầu để tăng tốc độ chương trình
# Áp dụng bộ lọc Gaussian để làm mịn trước khi thu nhỏ
# Giảm hiện vật răng cưa
smoothened_coins = gaussian_filter(orig_coins, sigma=2)
rescaled_coins = rescale(smoothened_coins, 0.2, mode="reflect",
                         **rescale_params)

# Thêm các hình ảnh vào trong đồ thị với giá trị của gradient trên các cạnh
graph = image.img_to_graph(rescaled_coins)

# lấy 1 hàm decreasing của gradient: số mũ
# beta càng nhỏ, phân đoạn càng độc lập phân khúc của hình ảnh thực tế.
# cho beta =1, phân khúc gần với 1 voronoi
beta = 10
eps = 1e-6
graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps

# Ứng dụng phân cụm spectral ( bước này đi nhanh hơn nếu bạn có cái đặt pyamg)
N_REGIONS = 25

# #####################################################################
# Hình dung các khu vực kết quả
# ---------------------------

for assign_label in ('kmeans', 'discretize'):
    t0 = time.time()
    labels = spectral_clustering(graph, n_clusters=N_REGIONS,
                                 assign_labels=assign_label, random_state=42)
    t1 = time.time()
    labels = labels.reshape(rescaled_coins.shape)

    plt.figure(figsize=(5, 5))
    plt.imshow(rescaled_coins, cmap=plt.cm.gray)
    for l in range(N_REGIONS):
        plt.contour(labels == l,
                    colors=[plt.cm.nipy_spectral(l / float(N_REGIONS))])
    plt.xticks(())
    plt.yticks(())
    title = 'Spectral clustering: %s, %.2fs' % (assign_label, (t1 - t0))
    print(title)
    plt.title(title)
plt.show()
