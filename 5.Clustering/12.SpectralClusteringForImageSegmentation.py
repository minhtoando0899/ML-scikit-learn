"""
Spectral clustering for image segmentation
(phân cụm phổ cho phân đoạn hình ảnh)

Trong ví dụ này, một hình ảnh với các vòng tròn
được kết nối được tạo ra và phân cụm phổ được sử
dụng để phân tách các vòng tròn.
"""
# Trong các cài đặt này, cách tiếp cận phân cụm phổ giải quyết vấn đề được gọi là
# 'cắt đồ thị chuẩn hóa': hình ảnh được xem như một biểu đồ của các voxels được
# kết nối và thuật toán phân cụm phổ sẽ chọn cách cắt đồ thị xác định các vùng trong
# khi giảm thiểu tỷ lệ của độ dốc dọc cắt, và khối lượng của khu vực.

# Vì thuật toán cố gắng cân bằng âm lượng (tức là cân bằng các kích thước vùng), nếu
# chúng ta lấy các vòng tròn với các kích thước khác nhau, việc phân đoạn sẽ thất bại.

# Ngoài ra, vì không có thông tin hữu ích về cường độ của hình ảnh, hoặc độ dốc của nó,
# chúng tôi chọn thực hiện phân cụm phổ trên biểu đồ chỉ được thông báo yếu bởi độ dốc.
# Điều này gần với việc thực hiện phân vùng Voronoi của biểu đồ.

# Ngoài ra, chúng tôi sử dụng mặt nạ của các đối tượng để hạn chế biểu đồ cho phác thảo
# của các đối tượng. Trong ví dụ này, chúng tôi quan tâm đến việc tách các đối tượng này
# với đối tượng khác, chứ không phải từ nền.

print(__doc__)

import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction import image
from sklearn.cluster import spectral_clustering

l = 100
x, y = np.indices((l, l))

center1 = (28, 24)
center2 = (40, 50)
center3 = (67, 58)
center4 = (24, 70)

radius1, radius2, radius3, radius4 = 16, 14, 15, 14

circle1 = (x - center1[0]) ** 2 + (y - center1[1]) ** 2 < radius1 ** 2
circle2 = (x - center2[0]) ** 2 + (y - center2[1]) ** 2 < radius2 ** 2
circle3 = (x - center3[0]) ** 2 + (y - center3[1]) ** 2 < radius2 ** 2
circle4 = (x - center4[0]) ** 2 + (y - center4[1]) ** 2 < radius2 ** 2

# ############################################################################
# 4 vòng tròn
img = circle1 + circle2 + circle3 + circle4

# Chúng tôi sử dụng mặt nạ giới hạn ở tiền cảnh: vấn đề chúng tôi quan tâm ở đây
# là không tách các đối tượng khỏi nền, nhưng tách chúng ra khỏi cái kia.
mask = img.astype(bool)

img = img.astype(float)
img += 1 + 0.2 * np.random.randn(*img.shape)

# Chèn hình ảnh vào trong 1 đồ thị với giá trị của gradient trên cạnh
graph = image.img_to_graph(img, mask=mask)

# Thực hiện chức năng giảm độ dốc: chúng tôi lấy nó một cách yếu
# phụ thuộc từ độ dốc, phân đoạn gần với voronoi
graph.data = np.exp(-graph.data / graph.data.std())

# Buộc người giải phải là arpack, vì amg là số không ổn định trong ví dụ này.
labels = spectral_clustering(graph, n_clusters=4, eigen_solver='arpack')
label_im = np.full(mask.shape, -1.)
label_im[mask] = labels

plt.matshow(img)
plt.matshow(label_im)

# ##############################################################################
# 2 vòng tròn
img = circle1 + circle2
mask = img.astype(bool)
img = img.astype(float)

img += 1 + 0.2 * np.random.randn(*img.shape)

graph = image.img_to_graph(img, mask=mask)
graph.data = np.exp(-graph.data / graph.data.std())

labels = spectral_clustering(graph, n_clusters=2, eigen_solver='arpack')
label_im = np.full(mask.shape, -1.)
label_im[mask] = labels

plt.matshow(img)
plt.matshow(label_im)

plt.show()
