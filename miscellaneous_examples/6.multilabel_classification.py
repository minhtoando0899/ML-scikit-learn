"""
=================
Phân loại đa nhãn
=================
Ví dụ này mô phỏng một bài toán phân loại tài liệu đa nhãn.
Bộ dữ liệu được tạo ngẫu nhiên dựa trên quy trình sau:

chọn số lượng nhãn: n ~ Poisson (n_labels)

n lần, chọn một lớp c: c ~ Multinomial (theta)

chọn độ dài tài liệu: k ~ Poisson (length)

k lần, chọn một từ: w ~ Multinomial (theta_c)

"""
# Trong quy trình trên, lấy mẫu từ chối được sử dụng để đảm bảo rằng
# n lớn hơn 2 và độ dài tài liệu không bao giờ bằng không. Tương tự
# như vậy, chúng tôi từ chối các lớp đã được chọn. Các tài liệu được
# gán cho cả hai lớp được vẽ bao quanh bởi hai vòng tròn màu.
#
# Việc phân loại được thực hiện bằng cách chiếu đến hai thành phần chính
# đầu tiên được PCA và CCA tìm thấy cho mục đích trực quan hóa, tiếp theo
# là sử dụng siêu dữ liệu sklearn.multiclass.OneVsRestClassifier sử dụng
# hai SVC với hạt nhân tuyến tính để tìm hiểu mô hình phân biệt đối xử
# cho mỗi lớp. Lưu ý rằng PCA được sử dụng để thực hiện giảm kích thước
# không giám sát, trong khi CCA được sử dụng để thực hiện giám sát.
#
# Lưu ý: trong cốt truyện, các mẫu không gắn nhãn, không có nghĩa là
# chúng tôi không biết nhãn (như trong học tập bán giám sát) nhưng
# các mẫu đơn giản là không có nhãn.
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_multilabel_classification
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA


def plot_hyperplane(clf, min_x, max_x, linestyle, label):
    # lấy siêu phẳng tách ra
    w = clf.coef_[0]
    a = -w[0] / w[1]
    xx = np.linspace(min_x - 5, max_x + 5)  # đảm bảo dòng này đủ dài
    yy = a * xx - (clf.intercept_[0] / w[1])
    plt.plot(xx, yy, linestyle, label=label)


def plot_subfigure(X, y, subplot, title, transform):
    if transform == "pca":
        X = PCA(n_components=2).fit_transform(X)
    elif transform == "cca":
        X = CCA(n_components=2).fit(X, Y).transform(X)
    else:
        raise ValueError
    min_x = np.min(X[:, 0])
    max_x = np.max(X[:, 0])

    min_y = np.min(X[:, 1])
    max_y = np.max(X[:, 1])

    classif = OneVsRestClassifier(SVC(kernel='linear'))
    classif.fit(X, Y)

    plt.subplot(2, 2, subplot)
    plt.title(title)

    zero_class = np.where(Y[:, 0])
    one_class = np.where(Y[:, 1])
    plt.scatter(X[:, 0], X[:, 1], s=40, c='gray', edgecolors=(0, 0, 0))
    plt.scatter(X[zero_class, 0], X[zero_class, 1], s=160, edgecolors='b',
                facecolor='none', linewidths=2, label='class 1')
    plt.scatter(X[one_class, 0], X[one_class, 1], s=80, edgecolors='orange',
                facecolor='none', linewidths=2, label='class 2')

    plot_hyperplane(classif.estimators_[0], min_x, max_x, 'k--',
                    'Boundary\nfor class 1')
    plot_hyperplane(classif.estimators_[1], min_x, max_x, 'k-.',
                    'Boundary\nfor class 2')

    plt.xticks(())
    plt.yticks(())

    plt.xlim(min_x - .5 * max_x, max_x + .5 * max_x)
    plt.ylim(min_y - .5 * max_y, max_y + .5 * max_y)
    if subplot == 2:
        plt.xlabel('First principal component')
        plt.ylabel('Second principal component')
        plt.legend(loc="upper left")


plt.figure(figsize=(8, 6))
X, Y = make_multilabel_classification(n_classes=2, n_labels=1,
                                      allow_unlabeled=True,
                                      random_state=1)

plot_subfigure(X, Y, 1, "with unlabeled sample +CCA", "cca")
plot_subfigure(X, Y, 2, "with unlabeled sample +PCA", "pca")

X, Y = make_multilabel_classification(n_classes=2, n_labels=1,
                                      allow_unlabeled=False,
                                      random_state=1)
plot_subfigure(X, Y, 3, "without unlabeled sample +CCA", "cca")
plot_subfigure(X, Y, 4, "without unlabeled sample + PCA", "pca")

plt.subplots_adjust(.04, .02, .97, .94, .09, .2)
plt.show()
