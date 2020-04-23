"""
=============================================
Comparison of kernel ridge regression and SVR
=============================================
"""
# Cả hồi quy sườn hạt nhân (KRR) và SVR đều học một hàm phi tuyến tính bằng cách
# sử dụng thủ thuật kernel, tức là, chúng học một hàm tuyến tính trong không gian
# do hạt nhân tương ứng tạo ra tương ứng với hàm phi tuyến tính trong không gian
# ban đầu. Chúng khác nhau ở các hàm mất (sườn so với mất không nhạy cảm với epsilon).
# Trái ngược với SVR, việc lắp KRR có thể được thực hiện ở dạng đóng và thường
# nhanh hơn đối với các bộ dữ liệu cỡ trung bình. Mặt khác, mô hình đã học là không
# thưa thớt và do đó chậm hơn so với SVR tại thời điểm dự đoán.
#
# Ví dụ này minh họa cả hai phương pháp trên một tập dữ liệu nhân tạo, bao gồm hàm
# mục tiêu hình sin và nhiễu mạnh được thêm vào mỗi datapoint thứ năm. Hình đầu tiên
# so sánh mô hình đã học của KRR và SVR khi cả độ phức tạp / chính quy hóa và băng
# thông của hạt nhân RBF được tối ưu hóa bằng cách sử dụng tìm kiếm dạng lưới. Các
# chức năng đã học rất giống nhau; tuy nhiên, KRR phù hợp là khoảng. nhanh hơn bảy
# lần so với lắp SVR (cả hai đều có lưới tìm kiếm). Tuy nhiên, dự đoán 100000 giá
# trị mục tiêu nhanh hơn nhiều lần so với SVR vì nó đã học được một mô hình thưa thớt
# chỉ sử dụng khoảng. 1/3 trong số 100 datapoint đào tạo như các vectơ hỗ trợ.
#
# Hình tiếp theo so sánh thời gian để phù hợp và dự đoán KRR và SVR cho các kích cỡ
# khác nhau của tập huấn luyện. Lắp KRR nhanh hơn SVR cho các bộ huấn luyện cỡ trung bình
# (dưới 1000 mẫu); tuy nhiên, đối với tập huấn luyện lớn hơn, thang điểm SVR tốt hơn.
# Về thời gian dự đoán, SVR nhanh hơn KRR cho tất cả các kích cỡ của tập huấn luyện vì
# giải pháp thưa thớt đã học. Lưu ý rằng mức độ thưa thớt và do đó thời gian dự đoán
# phụ thuộc vào các tham số epsilon và C của SVR.


import time

import numpy as np

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve
from sklearn.kernel_ridge import KernelRidge
import matplotlib.pyplot as plt

rng = np.random.RandomState(0)

#################################################################
# Tạo dữ liệu mẫu
X = 5 * rng.rand(10000, 1)
y = np.sin(X).ravel()

# Thêm tiếng ồn cho mục tiêu
y[::5] += 3 * (0.5 - rng.rand(X.shape[0] // 5))

X_plot = np.linspace(0, 5, 100000)[:, None]

##################################################################
# Mô hình hồi quy phù hợp
train_size = 100
svr = GridSearchCV(SVR(kernel='rbf', gamma=0.1),
                   param_grid={"C": [1e0, 1e1, 1e2, 1e3],
                               "gamma": np.logspace(-2, 2, 5)})
kr = GridSearchCV(KernelRidge(kernel='rbf', gamma=0.1),
                  param_grid={"alpha": [1e0, 0.1, 1e-2, 1e-3],
                              "gamma": np.logspace(-2, 2, 5)})
t0 = time.time()
svr.fit(X[:train_size], y[:train_size])
svr_fit = time.time() - t0
print("SVR complexity and bandwidth selected and model fitted in %.3f s"
      % svr_fit)

t0 = time.time()
kr.fit(X[:train_size], y[:train_size])
kr_fit = time.time() - t0
print("KRR complexity and bandwidth selected and model fitted in %.3f s"
      % kr_fit)

sv_ratio = svr.best_estimator_.support_.shape[0] / train_size
print("support vector ratio: %.3f" % sv_ratio)

t0 = time.time()
y_svr = svr.predict(X_plot)
svr_predict = time.time() - t0
print("SVR prediction for %d inputs in %.3f s"
      % (X_plot.shape[0], svr_predict))

t0 = time.time()
y_kr = kr.predict(X_plot)
kr_predict = time.time() - t0
print("KRR prediction for %d inputs %.3f s"
      % (X_plot.shape[0], kr_predict))

#######################################################################
# Nhìn vào kết quả
sv_ind = svr.best_estimator_.support_
plt.scatter(X[sv_ind], y[sv_ind], c='r', s=50, label='SVR support vectors',
            zorder=2, edgecolors=(0, 0, 0))
plt.scatter(X[:100], y[:100], c='k', label='data', zorder=1,
            edgecolors=(0, 0, 0))
plt.plot(X_plot, y_svr, c='r',
         label='SVR(fit:%.3fs, predict: %.3fs)' % (svr_fit, svr_predict))
plt.plot(X_plot, y_kr, c='g',
         label='KRR (fit: %.3fs, predict: %.3fs)' % (kr_fit, kr_predict))
plt.xlabel('data')
plt.ylabel('target')
plt.title('SVR versus kernel Ridge')
plt.legend()

# Trực quan hóa thời gian đào tạo và dự đoán
plt.figure()

# Tạo dữ liệu mẫu
X = 5 * rng.rand(10000, 1)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(X.shape[0] // 5))
sizes = np.logspace(1, 4, 7).astype(np.int)
for name, estimator in {"KRR": KernelRidge(kernel='rbf', alpha=0.1,
                                           gamma=10),
                        "SVR": SVR(kernel='rbf', C=1e1, gamma=10)}.items():
    train_time = []
    test_time = []
    for train_test_size in sizes:
        t0 = time.time()
        estimator.fit(X[:train_test_size], y[:train_test_size])
        train_time.append(time.time() - t0)

        t0 = time.time()
        estimator.predict(X_plot[:1000])
        test_time.append(time.time() - t0)

    plt.plot(sizes, train_time, 'o-', color="r" if name == "SVR" else "g",
             label="%s (train)" % name)
    plt.plot(sizes, test_time, 'o--', color='r' if name == "SVR" else "g",
             label="%s (test)" % name)

plt.xscale("log")
plt.yscale("log")
plt.xlabel("Train size")
plt.ylabel("Time (second)")
plt.title("Execution Time")
plt.legend(loc="best")

# Trực quan hóa các đường cong học tập
plt.figure()

svr = SVR(kernel='rbf', C=1e1, gamma=0.1)
kr = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.1)
train_sizes, train_scores_svr, test_scores_svr = \
    learning_curve(svr, X[:100], y[:100], train_sizes=np.linspace(0.1, 1, 10),
                   scoring="neg_mean_squared_error", cv=10)
train_sizes_abs, train_scores_kr, test_scores_kr = \
    learning_curve(kr, X[:100], y[:100], train_sizes=np.linspace(0.1, 1, 10),
                   scoring="neg_mean_squared_error", cv=10)
plt.plot(train_sizes, -test_scores_svr.mean(1), 'o-', color="r",
         label="SVR")
plt.plot(train_sizes, -test_scores_kr.mean(1), 'o-', color="g",
         label="KRR")
plt.xlabel("Train size")
plt.ylabel("Mean Squared Error")
plt.title('Learning cuves')
plt.legend(loc="best")

plt.show()
