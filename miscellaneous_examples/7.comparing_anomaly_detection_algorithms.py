"""
========================================================================================
So sánh các thuật toán phát hiện dị thường để phát hiện ngoại lệ trên bộ dữ liệu đồ chơi
========================================================================================
Ví dụ này cho thấy các đặc điểm của các thuật toán phát hiện dị thường khác nhau trên
các bộ dữ liệu 2D. Các bộ dữ liệu chứa một hoặc hai chế độ (vùng có mật độ cao) để
minh họa khả năng của thuật toán đối phó với dữ liệu đa phương thức.
"""
# Đối với mỗi tập dữ liệu, 15% mẫu được tạo ra dưới dạng nhiễu đồng nhất ngẫu nhiên.
# Tỷ lệ này là giá trị được cung cấp cho tham số nu của OneClassSVM và tham số ô nhiễm
# của các thuật toán phát hiện ngoại lệ khác. Ranh giới quyết định giữa các ngoại lệ
# và ngoại lệ được hiển thị màu đen ngoại trừ Yếu tố ngoại lệ cục bộ (LOF) vì nó không
# có phương pháp dự đoán nào được áp dụng trên dữ liệu mới khi được sử dụng để phát hiện
# ngoại lệ.
#
# Sklearn.svm.OneClassSVM được biết là nhạy cảm với các ngoại lệ và do đó không hoạt động tốt
# để phát hiện ngoại lệ. Công cụ ước tính này phù hợp nhất để phát hiện tính mới khi tập huấn
# luyện không bị ô nhiễm bởi các ngoại lệ. Điều đó nói rằng, phát hiện ngoại lệ ở chiều cao
# hoặc không có bất kỳ giả định nào về phân phối dữ liệu bên trong là rất khó khăn và một SVM
# một lớp có thể cho kết quả hữu ích trong các tình huống này tùy thuộc vào giá trị của siêu
# đường kính của nó.
#
# sklearn.covariance.EllipticEnvel giả sử dữ liệu là Gaussian và học một hình elip. Do đó, nó
# xuống cấp khi dữ liệu không phải là không chính thống. Tuy nhiên, lưu ý rằng công cụ ước tính
# này là mạnh mẽ đối với các ngoại lệ.
#
# sklearn.ensemble.IsolationForest và sklearn.neighbor.LocalOutlierFactor dường như hoạt động khá
# tốt đối với các tập dữ liệu đa phương thức. Ưu điểm của sklearn.neighbor.LocalOutlierFactor so
# với các công cụ ước tính khác được hiển thị cho tập dữ liệu thứ ba, trong đó hai chế độ có mật
# độ khác nhau. Ưu điểm này được giải thích bởi khía cạnh cục bộ của LOF, có nghĩa là nó chỉ so sánh
# điểm bất thường của một mẫu với điểm của các nước láng giềng.
#
# Cuối cùng, đối với tập dữ liệu cuối cùng, thật khó để nói rằng một mẫu bất thường hơn một mẫu
# khác vì chúng được phân phối đồng đều trong một hypercube. Ngoại trừ sklearn.svm.OneClassSVM
# mặc trang phục một chút, tất cả các công cụ ước tính đều đưa ra các giải pháp hợp lý cho tình
# huống này. Trong trường hợp như vậy, sẽ là khôn ngoan nếu nhìn kỹ hơn vào điểm bất thường của
# các mẫu vì một người ước lượng tốt sẽ chỉ định điểm tương tự cho tất cả các mẫu.
#
# Mặc dù các ví dụ này cung cấp một số trực giác về các thuật toán, trực giác này có thể không
# áp dụng cho dữ liệu chiều rất cao.
#
# Cuối cùng, lưu ý rằng các tham số của các mô hình đã ở đây được lựa chọn cẩn thận nhưng trong
# thực tế, chúng cần phải được điều chỉnh. Trong trường hợp không có dữ liệu được dán nhãn, vấn đề
# hoàn toàn không được giám sát nên việc lựa chọn mô hình có thể là một thách thức.
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.datasets import make_moons, make_blobs
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

print(__doc__)

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

# Cài đặt ví dụ
n_samples = 300
outliers_fraction = 0.15
n_outliers = int(outliers_fraction * n_samples)
n_inliers = n_samples - n_outliers

# định nghĩa các phương pháp phát hiện ngoại lệ / dị thường được so sánh
anomaly_algorithms = [
    ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
    ("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf",
                                      gamma=0.1)),
    ("Isolation Forest", IsolationForest(contamination=outliers_fraction,
                                         random_state=42)),
    ("Local Outlier Factor", LocalOutlierFactor(
        n_neighbors=35, contamination=outliers_fraction))]

# Định nghĩa bộ dữ liệu
blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
datasets = [
    make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5,
               **blobs_params)[0],
    make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[0.5, 0.5],
               **blobs_params)[0],
    make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[1.5, .3],
               **blobs_params)[0],
    4. * (make_moons(n_samples=n_samples, noise=.05, random_state=0)[0] -
          np.array([0.5, 0.25])),
    14. * (np.random.RandomState(42).rand(n_samples, 2) - 0.5)]

# So sánh các phân loại nhất định trong các cài đặt đã cho
xx, yy = np.meshgrid(np.linspace(-7, 7, 150),
                     np.linspace(-7, 7, 150))

plt.figure(figsize=(len(anomaly_algorithms) * 2 + 3, 12.5))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)

plot_num = 1
rng = np.random.RandomState(42)

for i_dataset, X in enumerate(datasets):
    # Thêm các ngoại lệ
    X = np.concatenate([X, rng.uniform(low=-6, high=6,
                                       size=(n_outliers, 2))], axis=0)

    for name, algorithm in anomaly_algorithms:
        t0 = time.time()
        algorithm.fit(X)
        t1 = time.time()
        plt.subplot(len(datasets), len(anomaly_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(name, size=18)

        # phù hợp với dữ liệu và thẻ ngoại lệ
        if name == "Local Outlier Factor":
            y_pred = algorithm.fit_predict(X)
        else:
            y_pred = algorithm.fit(X).predict(X)

        # vẽ các đường mức và điểm
        if name != "Local Outlier Factor":  # LOF does not implement predict
            Z = algorithm.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black')

        colors = np.array(['#377eb8', '#ff7f00'])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_pred + 1) // 2])

        plt.xlim(-7, 7)
        plt.ylim(-7, 7)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1

plt.show()
