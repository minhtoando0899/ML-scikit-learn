"""
=================================
Biểu diễn ước lượng vắn tắt
=================================
Ví dụ này minh họa việc sử dụng tham số toàn cầu the print_changed_only
Đặt print_changed_only thành True sẽ thay thế biểu diễn của các công cụ ước
tính để chỉ hiển thị các tham số đã được đặt thành giá trị không mặc định.
Điều này có thể được sử dụng để có đại diện vắn tắt hơn.
"""
print(__doc__)

from sklearn.linear_model import LogisticRegression
from sklearn import set_config


lr = LogisticRegression(penalty='l1')
print('Default representation:')
print(lr)
# LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#                    intercept_scaling=1, l1_ratio=None, max_iter=100,
#                    multi_class='auto', n_jobs=None, penalty='l1',
#                    random_state=None, solver='warn', tol=0.0001, verbose=0,
#                    warm_start=False)

set_config(print_changed_only=True)
print('\nWith changed_only option:')
print(lr)
# LogisticRegression(penalty='l1')
