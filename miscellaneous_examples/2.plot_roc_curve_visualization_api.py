'''
====================================
Đường cong ROB với API trực quan hóa
====================================
Scikit-learn định nghĩa 1 API đơn giản để tạo trực quan hóa cho
machine learning. Các tính năng chính của API này là cho phép điều
chỉnh phác họa đồ thị và hình ảnh nhanh chóng mà không cần tính toán lại.
Trong ví dụ này, chúng tôi sẽ trình bày cách sử dụng API trực
quan hóa bằng cách so sánh các đường cong ROC.
'''
print(__doc__)

###################################################################
# Tải Dữ liệu và huấn luyện 1 SVC
# -------------------------
# Đầu tiên, chúng tôi tải tập dữ liệu wine và chuyển đổi nó thành một bài
# toán phân loại nhị phân. Sau đó, chúng tôi đào tạo một trình phân loại
# vector hỗ trợ trên một tập dữ liệu đào tạo.

import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_roc_curve
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

X, y = load_wine(return_X_y=True)
y = y == 2

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
svc = SVC(random_state=42)
svc.fit(X_train, y_train)

###########################################################################
# Phác họa đồ thị đường cong ROC
# -----------------------
# Tiếp theo, chúng tôi vẽ đường cong ROC bằng một lệnh gọi đến sklearn.metrics.plot_roc_curve.
# Đối tượng svc_disp được trả về cho phép chúng ta tiếp tục sử dụng đường cong ROC đã được
# tính toán cho SVC trong các phác đồ tương lai.

scv_disp = plot_roc_curve(svc, X_test, y_test)
plt.show()

###################################################################################
# Huấn luyện 1 Random Forest và phác họa đồ thị đường cong ROC
# ---------------------------------------------------
# Chúng tôi đào tạo một trình phân loại Random Forest và tạo ra một phác họa so sánh
# nó với đường cong SVC ROC. Lưu ý cách svc_disp sử dụng biểu đồ để phác họa đường cong
# SVC ROC mà không tính toán lại các giá trị của chính đường cong roc. Hơn nữa, chúng ta
# chuyển alpha = 0.8 cho các hàm phạc họa để điều chỉnh giá trị alpha của các đường cong.


rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(X_train, y_train)
ax = plt.gca()
rfc_disp = plot_roc_curve(rfc, X_test, y_test, ax=ax, alpha=0.8)
scv_disp.plot(ax=ax, alpha=0.8)
plt.show()