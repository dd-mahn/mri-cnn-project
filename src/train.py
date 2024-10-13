import tensorflow as tf
import os
import numpy as np
from model import build_resnet

# Lấy thư mục hiện tại của script
current_dir = os.path.dirname(__file__)

# Định nghĩa đường dẫn đến thư mục dữ liệu đã xử lý
data_dir = os.path.join(current_dir, '..', 'data', 'processed')
history_dir = os.path.join(current_dir, '..', 'history')

# Tải dữ liệu đã được tiền xử lý
X_train = np.load(os.path.join(data_dir, 'X_train.npy'), allow_pickle=True)
X_val = np.load(os.path.join(data_dir, 'X_val.npy'), allow_pickle=True)
y_train = np.load(os.path.join(data_dir, 'y_train.npy'), allow_pickle=True)
y_val = np.load(os.path.join(data_dir, 'y_val.npy'), allow_pickle=True)

# Chuyển đổi dữ liệu sang kiểu float32 để tối ưu hóa hiệu suất
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')

# In ra thông tin về kích thước và nội dung của y_train để kiểm tra
print(f"Kích thước của y_train: {y_train.shape}")
print(f"Nội dung của y_train: {y_train}")

# Chuyển đổi nhãn chuỗi thành chỉ số số nguyên
label_to_index = {label: index for index, label in enumerate(np.unique(y_train))}
y_train = np.array([label_to_index[label] for label in y_train])
y_val = np.array([label_to_index[label] for label in y_val])

# In ra thông tin về kích thước và nội dung của X_train và X_val
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_val: {X_val.shape}")

# In ra thông tin về số lượng nhãn duy nhất
print(f"Number of unique labels: {len(np.unique(y_train))}")

# Xây dựng mô hình với kích thước đầu vào được cập nhật
model = build_resnet((224, 224, 3), num_classes=len(label_to_index))

# Biên dịch mô hình
model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

# Thiết lập các callback cho quá trình huấn luyện
callbacks = [
    # Dừng sớm nếu mô hình không cải thiện sau 10 epoch
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),
    # Lưu mô hình tốt nhất dựa trên val_loss
    tf.keras.callbacks.ModelCheckpoint('model_best.keras', save_best_only=True)
]

# Huấn luyện mô hình
history = model.fit(X_train, y_train, 
                    epochs=100,  # Số epoch tối đa
                    batch_size=32,  # Kích thước batch
                    validation_data=(X_val, y_val),  # Dữ liệu validation
                    callbacks=callbacks)  # Sử dụng callbacks đã định nghĩa

# Lưu mô hình cuối cùng
model.save('mri_classification_model.h5')

# In ra thông tin về quá trình huấn luyện
print("Quá trình huấn luyện đã hoàn tất.")
print(f"Độ chính xác trên tập huấn luyện: {history.history['accuracy'][-1]:.4f}")
print(f"Độ chính xác trên tập validation: {history.history['val_accuracy'][-1]:.4f}")

# Lưu lại lịch sử huấn luyện
np.save(os.path.join(history_dir, 'training_history.npy'), history.history)

