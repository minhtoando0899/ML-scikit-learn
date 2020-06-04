"""
Color Quantization using K-Means
(định lượng màu sử dụng K-Means)
"""
# Thực hiện lượng tử hóa Vector (VQ) pixel-wise của hình ảnh cung điện mùa hè
# (Trung Quốc), giảm số màu sắc cần thiết để hiển thị hình ảnh tử 96,615 xuống
# còn 64, trong khi vẫn giữ được chất lượng ảnh.

# Trong ví dụ này, các pixel được biểu diễn trong không gian 3D và K-Means
# được sử dụng để tìm 64 cụm màu. Trong tài liệu xử lý ảnh, bảng mã thu được
# từ K-mean (trung tâm cụm) được gọi là bảng màu. Sử dụng một byte đơn, có thể
# xử lý tối đa 256 màu, trong khi mã hóa RGB yêu cầu 3 byte cho mỗi pixel. Định
# dạng tệp GIF, ví dụ, sử dụng bảng màu như vậy.

# Để so sánh, một hình ảnh được lượng tử hóa bằng cách sử dụng một cuốn sách mã
# ngẫu nhiên (màu sắc được chọn ngẫu nhiên) cũng được hiển thị.

print(__doc__)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from time import time

n_colors = 64

# tải ảnh cung điện mùa hè
china = load_sample_image("china.jpg")

# chuyển đổi thành các số thực thay vì mã hóa 8 bit mặc định.
# Chia cho 255 là rất quan trọng để plt.imshow hoạt động tốt
# trên dữ liệu nổi( cần phải nằm trong phạm vi [0-1])
china = np.array(china, dtype=np.float64) / 225

# Tải ảnh và chuyển đổi thành 1 mảng numpy 2D
w, h, d = original_shape = tuple(china.shape)
assert d == 3
image_array = np.reshape(china, (w * h, d))

print("Fitting model on a small sub-sample of the data")
t0 = time()
image_array_sample = shuffle(image_array, random_state=0)[:1000]
kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(image_array_sample)
print("done in %0.3fs." % (time() - t0))

# lấy các nhãn cho tất cả các điểm
print("Predicting color indices on the full image (k-means)")
t0 = time()
labels = kmeans.predict(image_array)
print("done in %0.3fs." % (time() - t0))

codebook_random = shuffle(image_array, random_state=0)[:n_colors]
print("Predicting color indices on the full image (random)")
t0 = time()
labels_random = pairwise_distances_argmin(codebook_random,
                                          image_array,
                                          axis=0)
print("done in %0.3fs." % (time() - t0))


def recreate_image(codebook, labels, w, h):
    """Recreate the (compressed) image from the code book & labels"""
    d = codebook.shape[1]
    image = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = codebook[labels[label_idx]]
            label_idx += 1
    return image


# Hiển thị các kết quả, cung với hình ảnh gốc.
plt.figure(1)
plt.clf()
plt.axis('off')
plt.title('Original image (96,615 colors)')
plt.imshow(china)

plt.figure(2)
plt.clf()
plt.axis('off')
plt.title('Quantized image (64 colors, K-Means)')
plt.imshow(recreate_image(kmeans.cluster_centers_, labels, w, h))

plt.figure(3)
plt.clf()
plt.axis('off')
plt.title('Quantized image (64 colors, Random)')
plt.imshow(recreate_image(codebook_random, labels_random, w, h))
plt.show()