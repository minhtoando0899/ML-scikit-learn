"""
===================
Hồi quy Đẳng hướng
===================
Một minh họa về hồi quy đẳng hướng trên dữ liệu được tạo.
Hồi quy đẳng hướng tìm thấy một xấp xỉ không giảm của hàm
trong khi giảm thiểu sai số bình phương trung bình trên
dữ liệu huấn luyện. Lợi ích của một mô hình như vậy là nó
không giả định bất kỳ hình thức nào cho hàm mục tiêu như
tuyến tính. Để so sánh một hồi quy tuyến tính cũng được trình bày.
"""
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.linear_model import LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import check_random_state

n = 100
x = np.arange(n)
rs = check_random_state(0)
y = rs.randint(-50, 50, size=(n,)) + 50. * np.log1p(np.arange(n))

# ###############################################################################
# Các mô hình Fit IsotonicRegression và linearRegression
ir = IsotonicRegression()
y_ = ir.fit_transform(x, y)
lr = LinearRegression()
lr.fit(x[:, np.newaxis], y)  # x cần là 2d cho linearRegression

# ###############################################################################
# kết quả phác họa

segments = [[[i, y[i]], [i, y_[i]]] for i in range(n)]
lc = LineCollection(segments, zorder=0)
lc.set_array(np.ones(len(y)))
lc.set_linewidth(np.full(n, 0.5))

fig = plt.figure()
plt.plot(x, y, 'r.', markersize=12)
plt.plot(x, y_, 'b.-', markersize=12)
plt.plot(x, lr.predict(x[:, np.newaxis]), 'b-')
plt.gca().add_collection(lc)
plt.legend(('data', 'Isotonic Fit', 'Linear Fit'), loc='lower right')
plt.title('Isotonic regression')
plt.show()