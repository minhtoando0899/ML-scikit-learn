"""
Hierarchical clustering: structured vs unstructured ward
(phân cụm phân cấp: khu vực có cấu trúc và không có cấu trúc)
"""
# Ví dụ xây dựng một tập dữ liệu cuộn swiss và chạy phân cụm theo thứ bậc trên vị trí
# của chúng.

# Để biết thêm thông tin, xem Phân cụm phân cấp.

# Trong bước đầu tiên, việc phân cụm phân cấp được thực hiện mà không bị ràng buộc
# về kết nối trên cấu trúc và chỉ dựa trên khoảng cách, trong khi ở bước thứ hai, việc
# phân cụm được giới hạn trong biểu đồ k-Nearest Neighbors: nó phân cụm theo phân cấp
# có cấu trúc trước.

# Một số cụm được học mà không có ràng buộc kết nối không có quan hệ cấu trúc của cuộn
# swiss và mở rộng qua các nếp gấp khác nhau của đa tạp. Ngược lại, khi đối lập các
# ràng buộc kết nối, các cụm tạo thành một phân chia tốt đẹp của cuộn swiss.

print(__doc__)

import time as time
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_swiss_roll

# #####################################################################################
# tạo dữ liệu (dữ liệu cuộn swiss)
n_samples = 1500
noise = 0.05
X, _ = make_swiss_roll(n_samples, noise)
# làm cho n mỏng hơn
X[:, 1] *= .5

# ######################################################################################
# tính toán phân cụm
print("Compute unstructured hierarchical clustering...")
st = time.time()
ward = AgglomerativeClustering(n_clusters=6, linkage='ward').fit(X)
elapsed_time = time.time() - st
label = ward.labels_
print("Elapsed time: %.2fs" % elapsed_time)
print("Number of points: %i" % label.size)

# ######################################################################################
# phác họa kết quả
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -80)
for l in np.unique(label):
    ax.scatter(X[label == l, 0], X[label == l, 1], X[label == l, 2],
               color=plt.cm.jet(np.float(l) / np.max(label + 1)),
               s=20, edgecolors='k')
plt.title('Without connectivity constraints (time %.2fs)' % elapsed_time)

# #####################################################################################
# Định nghĩa cấu trúc A của dữ liệu. ở đây là 10 nearest neighbors
from sklearn.neighbors import kneighbors_graph
connectivity = kneighbors_graph(X, n_neighbors=10, include_self=False)

# ####################################################################################
# tính toán phân cụm
print("Compute structured hierarchical clustering...")
st = time.time()
ward = AgglomerativeClustering(n_clusters=6, connectivity=connectivity,
                               linkage='ward').fit(X)
elapsed_time = time.time() - st
label = ward.labels_
print("Elapsed time: %.2fs" % elapsed_time)
print("Number of points: %i" % label.size)

# #######################################################################################
# phác họa kết quả
fig = plt.figure()
ax = p3.Axes3D(fig)
ax.view_init(7, -80)
for l in np.unique(label):
    ax.scatter(X[label == l, 0], X[label == l, 1], X[label == l, 2],
               color=plt.cm.jet(float(l) / np.max(label + 1)),
               s=20, edgecolors='k')
plt.title('With connectivity constraints (time %.2fs)' % elapsed_time)

plt.show()
