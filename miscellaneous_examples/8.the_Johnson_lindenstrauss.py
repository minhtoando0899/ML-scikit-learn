"""
=========================================================================
Johnson-Lindenstrauss bị ràng buộc để nhúng với các phép chiếu ngẫu nhiên
=========================================================================
"""
# Bổ đề Johnson-Lindenstrauss nói rằng bất kỳ tập dữ liệu chiều cao nào cũng
# có thể được chiếu ngẫu nhiên vào không gian Euclide chiều thấp hơn trong khi
# kiểm soát biến dạng trong khoảng cách theo cặp.

print(__doc__)

import sys
from time import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from distutils.version import LooseVersion
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from sklearn.random_projection import SparseRandomProjection
from sklearn.datasets import fetch_20newsgroups_vectorized
from sklearn.datasets import load_digits
from sklearn.metrics.pairwise import euclidean_distances

# `normed` is being deprecated in favor of `density` in histograms
if LooseVersion(matplotlib.__version__) >= '2.1':
    density_param = {'density': True}
else:
    density_param = {'normed': True}

####################################################################
# Giới hạn lý thuyết
# ------------------------------------------------------------------

# Biến thể được giới thiệu bởi một phép chiếu ngẫu nhiên p được khẳng định bởi
# thực tế là p đang xác định nhúng eps với xác suất tốt như được định nghĩa bởi:
# (1 - eps) \|u - v\|^2 < \|p(u) - p(v)\|^2 < (1 + eps) \|u - v\|^2

# Trong đó u và v là bất kỳ hàng nào được lấy từ bộ dữ liệu hình dạng
# [n_samples, n_features] và p là hình chiếu của ma trận Gaussian N (0, 1)
# ngẫu nhiên có hình dạng [n_components, n_features] (hoặc ma trận
# Achlioptas thưa thớt).
#
# Số lượng thành phần tối thiểu để đảm bảo nhúng eps được đưa ra bởi:
# n\_components >= 4 log(n\_samples) / (eps^2 / 2 - eps^3 / 3)

# Biểu đồ đầu tiên cho thấy với số lượng mẫu n_samples ngày càng tăng,
# số lượng kích thước tối thiểu n_components đã tăng logarit để
# đảm bảo nhúng eps.

# phạm vi biến dạng được chấp nhận
eps_range = np.linspace(0.1, 0.99, 5)
colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(eps_range)))

# phạm vi số lượng mẫu (quan sát) để nhúng
n_samples_range = np.logspace(1, 9, 9)

plt.figure()
for eps, color in zip(eps_range, colors):
    min_n_components = johnson_lindenstrauss_min_dim(n_samples_range, eps=eps)
    plt.loglog(n_samples_range, min_n_components, color=color)

plt.legend(["eps = %0.1f" % eps for eps in eps_range], loc="lower right")
plt.xlabel("number of observations to eps-embed")
plt.ylabel("Minimum number of dimensions")
plt.title("Johnson-Linderstrauss bounds\nn_samples vs n_components")
plt.show()

# Biểu đồ thứ hai cho thấy rằng sự gia tăng của méo méo có thể chấp nhận
# cho phép giảm đáng kể số lượng kích thước tối thiểu cho một số
# lượng mẫu n_samples đã cho

# phạm vi biến dạng được chấp nhận
eps_range = np.linspace(0.01, 0.99, 100)

# phạm vi số lượng mẫu (quan sát) để nhúng
n_samples_range = np.logspace(2, 6, 5)
colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(n_samples_range)))

plt.figure()
for n_samples, color in zip(n_samples_range, colors):
    min_n_components = johnson_lindenstrauss_min_dim(n_samples, eps=eps_range)
    plt.semilogy(eps_range, min_n_components, color=color)

plt.legend(["n_samples = %d" % n for n in n_samples_range], loc="upper right")
plt.xlabel("Distortion eps")
plt.ylabel("Minimum number of dimensions")
plt.title("Johnson-Lindenstrauss bounds: \nn_components vs eps")
plt.show()

# ##############################################################################
# Xác nhận thực nghiệm
# --------------------
# Chúng tôi xác nhận các giới hạn trên trên tập dữ liệu văn bản 20 nhóm tin (tần số từ TF-IDF)
# hoặc trên tập dữ liệu chữ số:

# đối với 20 tập tin nhóm dữ liệu, khoảng 500 tài liệu với tổng số 100 nghìn tính năng được
# dự kiến ​​sử dụng ma trận ngẫu nhiên thưa thớt đến các không gian euclide nhỏ hơn với các
# giá trị khác nhau cho số lượng kích thước mục tiêu n_components.

# đối với tập dữ liệu chữ số, một số dữ liệu pixel độ xám 8 x 8 cho 500 hình ảnh chữ số viết tay
# được chiếu ngẫu nhiên vào khoảng trắng cho số lượng kích thước lớn hơn n_components.

# Bộ dữ liệu mặc định là bộ dữ liệu 20 nhóm tin. Để chạy ví dụ trên tập dữ liệu chữ số, chuyển đối
# số dòng lệnh --use-chữ số-tập dữ liệu cho tập lệnh này.

if '__use-digits-dataset' in sys.argv:
    data = load_digits().data[:500]
else:
    data = fetch_20newsgroups_vectorized().data[:500]

# Đối với mỗi giá trị của n_components, chúng tôi vẽ:

# Phân phối 2D của các cặp mẫu với khoảng cách theo cặp trong không gian ban đầu và được chiếu
# theo trục x và y tương ứng.

# Biểu đồ 1D của tỷ lệ các khoảng cách đó (dự kiến / bản gốc).

n_samples, n_features = data.shape
print("Embedding %d samples with dim %d using various random projections"
      % (n_samples, n_features))

n_components_range = np.array([300, 1000, 10000])
dists = euclidean_distances(data, squared=True).ravel()

# chỉ chọn các cặp mẫu không giống nhau
nonzero = dists != 0
dists = dists[nonzero]

for n_components in n_components_range:
    t0 = time()
    rp = SparseRandomProjection(n_components=n_components)
    projected_data = rp.fit_transform(data)
    print("Projected %d samples from %d to %d in %0.3fs"
          % (n_samples, n_features, n_components, time() - t0))
    if hasattr(rp, 'components_'):
        n_bytes = rp.components_.data.nbytes
        n_bytes += rp.components_.indices.nbytes
        print("Random matric with size: %0.3fMB" % (n_bytes / 1e6))

    projected_dists = euclidean_distances(
        projected_data, squared=True).ravel()[nonzero]

    plt.figure()
    min_dist = min(projected_dists.min(), dists.min())
    max_dist = max(projected_dists.max(), dists.max())
    plt.hexbin(dists, projected_dists, gridsize=100, cmap=plt.cm.PuBu,
               extent=[min_dist, max_dist, min_dist, max_dist])
    plt.xlabel("Pairwise squared distances in original space")
    plt.ylabel("Pairwise squared distances in projected space")
    plt.title("Pairwise distances distribution for n_components = %d" %
              n_components)
    cb = plt.colorbar()
    cb.set_label('Sample pairs counts')

    rates = projected_dists / dists
    print("Mean distances rate: %0.2f (%0.2f)"
          % (np.mean(rates), np.std(rates)))

    plt.figure()
    plt.hist(rates, bins=50, range=(0., 2.), edgecolor='k', **density_param)
    plt.xlabel("Squared distances rate: (0.,2.), edgecolor='k', **density_param")
    plt.ylabel("Squared distances rate: projected / original")
    plt.title("Histogram of pairwise distance rates for n_components=%d" %
              n_components)

    # TODO: tính giá trị mong đợi của eps và thêm chúng vào cốt truyện trước

    # dưới dạng đường / vùng dọc

plt.show()


# Chúng ta có thể thấy rằng đối với các giá trị thấp của n_components, phân phối rộng với nhiều cặp
# bị biến dạng và phân phối bị lệch (do giới hạn cứng của tỷ lệ 0 ở bên trái vì khoảng cách luôn luôn dương)
# trong khi đối với các giá trị lớn hơn của n_components thì độ méo được kiểm soát và các khoảng cách được bảo
# quản tốt bởi phép chiếu ngẫu nhiên

# ##############################################################################
# Nhận xét
# --------

# Theo bổ đề JL, chiếu 500 mẫu mà không bị biến dạng quá nhiều sẽ cần ít nhất vài nghìn kích thước,
# bất kể số lượng tính năng của bộ dữ liệu gốc.

# Do đó, việc sử dụng các phép chiếu ngẫu nhiên trên tập dữ liệu chữ số chỉ có 64 tính năng trong
# không gian đầu vào không có ý nghĩa: nó không cho phép giảm kích thước trong trường hợp này.

# Mặt khác, trong hai mươi nhóm tin tức, chiều có thể giảm từ 56436 xuống còn 10000 trong khi vẫn
# giữ khoảng cách cặp đôi một cách hợp lý.