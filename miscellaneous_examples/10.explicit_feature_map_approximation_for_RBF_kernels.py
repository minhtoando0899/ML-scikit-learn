"""
===================================================
Tính gần đúng bản đồ tính toán cho các hạt nhân RBF
===================================================
"""

# Một ví dụ minh họa sự gần đúng của bản đồ tính năng của hạt nhân RBF.
#
# Nó cho thấy cách sử dụng RBFSampler và Nystroem để xấp xỉ bản đồ tính năng của hạt nhân
# RBF để phân loại với một SVM trên tập dữ liệu chữ số. Kết quả sử dụng một SVM tuyến tính
# trong không gian ban đầu, một SVM tuyến tính sử dụng ánh xạ gần đúng và sử dụng một SVM
# được nhân được so sánh. Thời gian và độ chính xác cho các mức lấy mẫu Monte Carlo khác
# nhau (trong trường hợp RBFSampler, sử dụng các tính năng Fourier ngẫu nhiên) và các tập
# hợp con có kích thước khác nhau của tập huấn luyện (đối với Nystroem) cho ánh xạ gần đúng
# được hiển thị.
#
# Xin lưu ý rằng bộ dữ liệu ở đây không đủ lớn để hiển thị các lợi ích của xấp xỉ kernel,
# vì SVM chính xác vẫn còn khá nhanh.
#
# Lấy mẫu nhiều kích thước rõ ràng dẫn đến kết quả phân loại tốt hơn, nhưng có chi phí
# lớn hơn. Điều này có nghĩa là có sự đánh đổi giữa thời gian chạy và độ chính xác, được
# đưa ra bởi tham số n_components. Lưu ý rằng việc giải quyết SVM tuyến tính và cả SVM
# hạt nhân gần đúng có thể được tăng tốc đáng kể bằng cách sử dụng độ dốc dốc ngẫu nhiên
# thông qua sklearn.linear_model.SGDClassifier. Điều này là không dễ dàng đối với trường
# hợp của SVM được nhân.

# ##################################################################################
# Nhập gói và dữ liệu Python, tải dữ liệu
# ------------------------------------------------

print(__doc__)

# Nhập Python khoa học tiêu chuẩn
import matplotlib.pyplot as plt
import numpy as np
from time import time

# Nhập bộ dữ liệu, phân loại và số liệu hiệu suất
from sklearn import datasets, svm, pipeline
from sklearn.kernel_approximation import (RBFSampler,
                                          Nystroem)
from sklearn.decomposition import PCA

# Tập dữ liệu chữ số
digits = datasets.load_digits(n_class=9)

# ###################################################################################
# Phác đồ thời gian và độ chính xác
# -------------------------

# Để áp dụng trình phân loại trên dữ liệu này, chúng ta cần làm phẳng hình ảnh, để biến
# dữ liệu theo ma trận (mẫu, tính năng):

n_samples = len(digits.data)
data = digits.data / 16.
data -= data.mean(axis=0)

# Chúng tôi tìm hiểu các chữ số ở nửa đầu của các chữ số
data_train, targets_train = (data[:n_samples // 2],
                             digits.target[:n_samples // 2])

# Bây giờ dự đoán giá trị của chữ số trên nửa thứ hai:
data_test, targets_test = (data[n_samples // 2:],
                           digits.target[n_samples // 2:])
# data_test = scaler.transform(data_test)

# Tạo trình phân loại: trình phân loại vectơ hỗ trợ
kernel_svm = svm.SVC(gamma=.2)
linear_svm = svm.LinearSVC()

# tạo đường ống từ xấp xỉ kernel
# và Svm tuyến tính
feature_map_fourier = RBFSampler(gamma=.2, random_state=1)
feature_map_nystroem = Nystroem(gamma=.2, random_state=1)
fourier_approx_svm = pipeline.Pipeline([("feature_map", feature_map_fourier),
                                        ("svm", svm.LinearSVC())])
nystroem_approx_svm = pipeline.Pipeline([("feature_map", feature_map_nystroem),
                                         ("svm", svm.LinearSVC())])

# phù hợp và dự đoán bằng cách sử dụng svm tuyến tính và kernel:

kernel_svm_time = time()
kernel_svm.fit(data_train, targets_train)
kernel_svm_score = kernel_svm.score(data_test, targets_test)
kernel_svm_time = time() - kernel_svm_time

linear_svm_time = time()
linear_svm.fit(data_train, targets_train)
linear_svm_score = linear_svm.score(data_test, targets_test)
linear_svm_time = time() - linear_svm_time

sample_sizes = 30 * np.arange(1, 10)
fourier_scores = []
nystroem_scores = []
fourier_times = []
nystroem_times = []

for D in sample_sizes:
    fourier_approx_svm.set_params(feature_map__n_components=D)
    nystroem_approx_svm.set_params(feature_map__n_components=D)
    start = time()
    nystroem_approx_svm.fit(data_train, targets_train)
    nystroem_times.append(time() - start)

    start = time()
    fourier_approx_svm.fit(data_train, targets_train)
    fourier_times.append(time() - start)

    fourier_score = fourier_approx_svm.score(data_test, targets_test)
    nystroem_score = nystroem_approx_svm.score(data_test, targets_test)
    nystroem_scores.append(nystroem_score)
    fourier_scores.append(fourier_score)

# vẽ kết quả:
plt.figure(figsize=(16, 4))
accuracy = plt.subplot(121)
# Trục thứ hai y cho thời gian
timescale = plt.subplot(122)

accuracy.plot(sample_sizes, nystroem_scores, label="Nystroem approx. kernel")
timescale.plot(sample_sizes, nystroem_times, '--',
               label='Nystroem approx. kernel')

accuracy.plot(sample_sizes, fourier_scores, label='Fourier approxy. kernel')
timescale.plot(sample_sizes, fourier_times, '--',
               label='Fourier approxy.kernel')

# đường ngang cho hạt nhân rbf và tuyến tính chính xác:
accuracy.plot([sample_sizes[0], sample_sizes[-1]],
              [linear_svm_score, linear_svm_score], label="linear svm")
timescale.plot([sample_sizes[0], sample_sizes[-1]],
               [linear_svm_time, linear_svm_time], '--', label='linear svm')

accuracy.plot([sample_sizes[0], sample_sizes[-1]],
              [kernel_svm_score, kernel_svm_score], label="rbf svm")
timescale.plot([sample_sizes[0], sample_sizes[-1]],
               [kernel_svm_time, kernel_svm_time], '--', label='rbf svm')

# đường thẳng đứng cho tập dữ liệu thứ nguyên = 64
accuracy.plot([64, 64], [0.7, 1], label="n_features")

# legends and labels
accuracy.set_title("Classification accuracy")
timescale.set_title("Training times")
accuracy.set_xlim(sample_sizes[0], sample_sizes[-1])
accuracy.set_xticks(())
accuracy.set_ylim(np.min(fourier_scores), 1)
timescale.set_xlabel("Sampling steps = transformed feature dimension")
accuracy.set_ylabel("Classification accuracy")
timescale.set_ylabel("Training time in seconds")
accuracy.legend(loc='best')
timescale.legend(loc='best')
plt.tight_layout()
plt.show()

# ###################################################################################
# Bề mặt quyết định của RBF Kernel SVM và tuyến tính SVM
# --------------------------------------------------

# Biểu đồ thứ hai trực quan hóa các bề mặt quyết định của hạt nhân RBF SVM và SVM tuyến tính
# với các bản đồ hạt nhân gần đúng. Biểu đồ cho thấy các bề mặt quyết định của các phân loại
# được chiếu lên hai thành phần chính đầu tiên của dữ liệu. Hình dung này nên được thực hiện
# với một hạt muối vì nó chỉ là một lát cắt thú vị thông qua bề mặt quyết định trong 64 chiều.
# Đặc biệt lưu ý rằng một datapoint (được biểu thị dưới dạng một dấu chấm) không nhất thiết
# phải được phân loại vào khu vực mà nó nằm, vì nó sẽ không nằm trên mặt phẳng mà hai thành
# phần chính đầu tiên trải dài. Việc sử dụng RBFSampler và Nystroem được mô tả chi tiết trong
# xấp xỉ hạt nhân.

# trực quan hóa bề mặt quyết định, chiếu xuống đầu tiên
# hai thành phần chính của bộ dữ liệu
pca = PCA(n_components=8).fit(data_train)

X = pca.transform(data_train)

# Tạo lưới dọc theo hai thành phần chính đầu tiên
multiples = np.arange(-2, 2, 0.1)
# bước dọc theo thành phần đầu tiên
first = multiples[:, np.newaxis] * pca.components_[0, :]
# steps along second component
second = multiples[:, np.newaxis] * pca.components_[1, :]
# phối hợp
grid = first[np.newaxis, :, :] + second[:, np.newaxis, :]
flat_grid = grid.reshape(-1, data.shape[1])

# tiêu đề cho các lô
titles = ['SVC with rbf kernel',
         'SVC (linear kernel)\n with Fourier rbf feature map\n'
         'n_components=100',
         'SVC (linear kernel)\n with Nystroem rbf feature map\n'
         'n_components=100']

plt.figure(figsize=(18, 7.5))
plt.rcParams.update({'font.size': 14})
# dự đoán và cốt truyện
for i, clf in enumerate((kernel_svm, nystroem_approx_svm,
                         fourier_approx_svm)):
    # Vẽ ranh giới quyết định. Vì vậy, chúng tôi sẽ chỉ định một màu cho mỗi
    # điểm trong danh sách [x_min, x_max]x[y_min, y_max].
    plt.subplot(1, 3, i + 1)
    Z = clf.predict(flat_grid)

    # Đặt kết quả vào một ô màu
    Z = Z.reshape(grid.shape[:-1])
    plt.contourf(multiples, multiples, Z, cmap=plt.cm.Paired)
    plt.axis('off')

    # Âm mưu cũng là điểm đào tạo
    plt.scatter(X[:, 0], X[:, 1], c=targets_train, cmap=plt.cm.Paired,
                edgecolors=(0, 0, 0))
    plt.title(titles[i])
plt.tight_layout()
plt.show()
