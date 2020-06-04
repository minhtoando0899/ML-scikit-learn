"""
=================================================
Online learning of a dictionary of parts of faces
Học trực tuyến từ điển các bộ phận của khuôn mặt
=================================================
Ví dụ này sử dụng một bộ dữ liệu lớn về khuôn mặt
để tìm hiểu một tập hợp các bản vá hình ảnh 20 x 20
tạo thành khuôn mặt.
"""
# Từ quan điểm lập trình, thật thú vị bởi vì nó cho thấy cách sử dụng API
# trực tuyến của scikit-learn để xử lý một tập dữ liệu rất lớn theo từng khối.
# Cách chúng tôi tiến hành là chúng tôi tải một hình ảnh tại một thời điểm và
# trích xuất ngẫu nhiên 50 bản vá từ hình ảnh này. Khi chúng tôi đã tích lũy
# được 500 bản vá này (sử dụng 10 hình ảnh), chúng tôi sẽ chạy phương thức
# part_fit của đối tượng KMeans trực tuyến, MiniBatchKMeans.

# Cài đặt dài dòng trên MiniBatchKMeans cho phép chúng tôi thấy rằng một số cụm
# được gán lại trong các cuộc gọi liên tiếp để phù hợp một phần. Điều này là do
# số lượng các bản vá mà chúng đại diện đã trở nên quá ít, và tốt hơn là chọn một
# cụm mới ngẫu nhiên.

print(__doc__)

import time

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.image import extract_patches_2d

faces = datasets.fetch_olivetti_faces()

# ##############################################################################
# Tìm hiểu từ điển của các hình ảnh

print('Learning the dictionary... ')
rng = np.random.RandomState(0)
kmeans = MiniBatchKMeans(n_clusters=81, random_state=rng, verbose=True)
patch_size = (20, 20)

buffer = []
t0 = time.time()

# Phần học trực tuyến: xoay vòng dữ liệu 6 lần.
index = 0
for _ in range(6):
    for img in faces.images:
        data = extract_patches_2d(img, patch_size, max_patches=50,
                                  random_state=rng)
        data = np.reshape(data, (len(data), -1))
        buffer.append(data)
        index += 1
        if index % 10 == 0:
            data = np.concatenate(buffer, axis=0)
            data -= np.mean(data, axis=0)
            data /= np.std(data, axis=0)
            kmeans.partial_fit(data)
            buffer = []
        if index % 100 == 0:
            print('Partial fit of %4i out of %i'
                  % (index, 6 * len(faces.images)))

dt = time.time() - t0
print('done in %.2fs.' % dt)

# ####################################################################
# Phác họa sơ đồ kết quả
plt.figure(figsize=(4.2, 4))
for i, patch in enumerate(kmeans.cluster_centers_):
    plt.subplot(9, 9, i + 1)
    plt.imshow(patch.reshape(patch_size), cmap=plt.cm.gray,
               interpolation='nearest')
    plt.xticks(())
    plt.yticks(())

plt.suptitle('Patches of face\nTrain time %.1fs on %d patches' %
             (dt, 8 * len(faces.images)), fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

plt.show()