"""
===================================================
Phác họa đồ thị nâng cao với sự phụ thuộc một phần
===================================================
Hàm plot_partial_dependence trả về một đối tượng PartialDependenceDisplay
có thể được sử dụng để phác họa đồ thị mà không cần tính toán lại sự phụ thuộc
một phần. Trong ví dụ này, chúng tôi trình bày cách phác họa đồ thị phụ thuộc
một phần và cách nhanh chóng tùy chỉnh phác họa bằng API trực quan hóa.
"""
print(__doc__)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.inspection import plot_partial_dependence

# ######################################################################################
# Mô hình đào tạo trên bộ dữ liệu giá nhà ở boston
# -------------------------------------------------
# Đầu tiên, chúng tôi đào tạo một decision tree và một
# multi-layer perceptron trên bộ dữ liệu giá nhà ở boston.

boston = load_boston()
X = pd.DataFrame(boston.data, columns=boston.feature_names)
y = boston.target

tree = DecisionTreeRegressor()
mlp = make_pipeline(StandardScaler(),
                    MLPRegressor(hidden_layer_sizes=(100, 100),
                                 tol=1e-2, max_iter=500, random_state=0))
tree.fit(X, y)
mlp.fit(X, y)

# ######################################################################################
# Phác họa đồ thị phụ thuộc một phần cho 2 tính năng
# --------------------------------------------
# Chúng tôi phác họa các đường cong phụ thuộc một phần cho các tính năng của LSTAT, và của RM RM
# cho decision tree. Với hai tính năng, plot_partial_dependence dự kiến sẽ vẽ hai đường cong.
# Ở đây, chức năng vẽ đồ thị đặt một lưới gồm hai ô bằng cách sử dụng khoảng trắng được xác định bởi ax.

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_title("Decision Tree")
tree_disp = plot_partial_dependence(tree, X, ["LSTAT", "RM"], ax=ax)

# Các đường cong phụ thuộc một phần có thể được vẽ cho perceptionron nhiều lớp. Trong trường hợp này,
# line_kw được chuyển đến plot_partial_dependence để thay đổi màu của đường cong.

fig, ax = plt.subplots(figsize=(12, 6))
ax.set_title("multi-layer Perceptron")
mlp_disp = plot_partial_dependence(mlp, X, ["LSTAT", "RM"], ax=ax,
                                   line_kw={"c": "red"})

# ######################################################################################
# Phác họa đồ thị phụ thuộc một phần của hai mô hình với nhau
# ------------------------------------------------------
# Các đối tượng Tree_disp và mlp_disp PartialDependenceDisplay chứa tất cả các thông tin được tính toán
# cần thiết để tạo lại các đường cong phụ thuộc một phần. Điều này có nghĩa là chúng ta có thể dễ dàng
# tạo các ô bổ sung mà không cần tính toán lại các đường cong.

# Một cách để vẽ các đường cong là đặt chúng trong cùng một hình, với các đường cong của mỗi mô hình
# trên mỗi hàng. Đầu tiên, chúng ta tạo một hình với hai trục trong hai hàng và một cột. Hai trục được
# truyền cho các hàm cốt truyện của tree_disp và mlp_disp. Các trục đã cho sẽ được sử dụng bởi hàm phác
# họa để vẽ sự phụ thuộc một phần. Biểu đồ kết quả đặt các đường cong phụ thuộc một phần của decision tree
# vào hàng đầu tiên của multi-layer Perceptron ở hàng thứ hai.

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
tree_disp.plot(ax=ax1)
ax1.set_title("Decision Tree")
mlp_disp.plot(ax=ax2, line_kw={"c": "red"})
ax2.set_title("Multi-layer Perceptron")

# Một cách khác để so sánh các đường cong là vẽ chúng lên nhau. Ở đây, chúng ta tạo một hình với một hàng
# và hai cột. Các trục được truyền vào hàm vẽ dưới dạng một danh sách, nó sẽ vẽ các đường cong phụ thuộc
# một phần của mỗi mô hình trên cùng một trục. Độ dài của danh sách trục phải bằng số lượng ô được vẽ.

# Đặt hình ảnh này làm hình thu nhỏ cho thư viện nhân sư
# sphinx_gallery_thumbnail_number = 4
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
tree_disp.plot(ax=[ax1, ax2], line_kw={"label": "multi-layer Perceptron",
                                       "c": "red"})
ax1.legend
ax2.legend

# tree_disp.axes_ là một mảng chứa numpy các trục được sử dụng để vẽ các ô phụ thuộc một phần.
# Điều này có thể được truyền cho mlp_disp để có cùng ảnh hưởng của việc vẽ các ô trên đầu nhau.
# Hơn nữa, mlp_disp.figure_ lưu trữ hình, cho phép thay đổi kích thước hình sau khi gọi biểu đồ.
# Trong trường hợp này tree_disp.axes_ có hai thứ nguyên, do đó, cốt truyện sẽ chỉ hiển thị nhãn
# y và dấu tick y ở bên trái hầu hết các ô.

tree_disp.plot(line_kw={"label": "Decision Tree"})
mlp_disp.plot(line_kw={"label": "Multi layer Perceptron", "c": "red"},
              ax=tree_disp.axes_)
tree_disp.figure_.set_size_inches(10, 6)
tree_disp.axes_[0, 0].legend()
tree_disp.axes_[0, 1].legend()
plt.show()

# ######################################################################################
# Phác họa phụ thuộc một phần cho một tính năng
# -------------------------------------------
# Ở đây, chúng tôi phác họa các đường cong phụ thuộc một phần cho một tính năng duy nhất,
# “LSTAT”, trên cùng một trục. Trong trường hợp này, tree_disp.axes_ được truyền vào hàm
# phác họa thứ hai.

tree_disp = plot_partial_dependence(tree, X, ["LSTAT"])
mlp_disp = plot_partial_dependence(mlp, X, ["LSTAT"],
                                   ax=tree_disp.axes_, line_kw={"c": "red"})
