"""
A demo of structured Ward hierarchical clustering on an image of coins
(1 bản demo của phân cụm thứ bậc được xậy dựng trên ảnh của các đồng xu)

Tính toán phân đoạn hình ảnh 2D với phân cụm phân cấp Ward.
Việc phân cụm bị hạn chế về mặt không gian để mỗi khu vực
được phân chia thành một phần.
"""

print(__doc__)

import time as time

import numpy as np
from distutils.version import LooseVersion
from scipy.ndimage.filters import gaussian_filter

import matplotlib.pyplot as plt

import skimage
from skimage.data import coins
from skimage.transform import rescale

from sklearn.feature_extraction.image import grid_to_graph
from sklearn.cluster import AgglomerativeClustering

# chúng được giới thiệu trong skimage-0.14
if LooseVersion(skimage.__version__) >= '0.14':
    rescale_params = {'anti_aliasing': False, 'multichannel': False}
else:
    rescale_params = {}

# #######################################################################
# tạo dữ liệu
orig_coins = coins()

# Thay đổi kích thước của nó thành 20% của kích thước ban đầu để tăng tốc độ xử lý
# ứng dụng bộ lọc Gaussian để làm mịn trước khi giảm tỷ lệ làm giảm các tạo phẩm răng cưa
smoothened_coins = gaussian_filter(orig_coins, sigma=2)
rescaled_coins = rescale(smoothened_coins, 0.2, mode="reflect",
                      **rescale_params)

X = np.reshape(rescaled_coins, (-1, 1))

# ###############################################################################
# Định nghĩa cấu trúc A của dữ liệu. Điểm ảnh kết nối với kế cận của chúng.
connectivity = grid_to_graph(*rescaled_coins.shape)

# ##############################################################################
# tính toán phân cụm
print("Compute structured  hierarchical clustering...")
st = time.time()
n_clusters = 27  # số của khu vực
ward = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward',
                               connectivity=connectivity)
ward.fit(X)
label = np.reshape(ward.labels_, rescaled_coins.shape)
print("Elapsed time: ", time.time() - st)
print("Number of pixels: ", label.size)
print("Number of clusters: ", np.unique(label).size)

# #############################################################################
# phác họa kết quả trên hình ảnh
plt.figure(figsize=(5, 5))
plt.imshow(rescaled_coins, cmap=plt.cm.gray)
for l in range(n_clusters):
    plt.contour(label == l,
                colors=[plt.cm.nipy_spectral(l / float(n_clusters)), ])
plt.xticks(())
plt.yticks(())
plt.show()
