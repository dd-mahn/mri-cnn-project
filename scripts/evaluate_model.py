project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

import tensorflow as tf
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns
from src.data_preprocessing import load_test_data
from sklearn.metrics import precision_score, recall_score

# Định nghĩa đường dẫn đến thư mục dữ liệu kiểm tra
current_dir = os.path.dirname(__file__)
test_dir = os.path.join(current_dir, '..', 'data', 'testing')

# Tải dữ liệu kiểm tra
X_test, y_test = load_test_data(test_dir)

# Tải mô hình
model_path = os.path.join(project_root, 'model', 'model_best.keras')
model = tf.keras.models.load_model(model_path)

# Đánh giá mô hình
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Độ chính xác trên tập kiểm tra: {test_acc}")

# Dự đoán và tính toán precision và recall
predictions = model.predict(X_test)
y_pred = np.argmax(predictions, axis=1)
y_true = y_test

test_precision = precision_score(y_true, y_pred, average='weighted')
test_recall = recall_score(y_true, y_pred, average='weighted')

print(f"Độ chính xác (precision) trên tập kiểm tra: {test_precision}")
print(f"Độ nhạy (recall) trên tập kiểm tra: {test_recall}")

# In một số dự đoán (tùy chọn)
print(f"Một số dự đoán: {predictions[:5]}")

# Tải lịch sử huấn luyện
history = np.load('training_history.npy', allow_pickle=True).item()

# Vẽ quá trình huấn luyện
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history['loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['accuracy'], label='Training Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig('training_process.png')
plt.close()

# Tính toán các chỉ số bổ sung
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='weighted')
auc_roc = roc_auc_score(y_true, predictions, multi_class='ovr', average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Precision: {test_precision}")
print(f"Recall: {test_recall}")
print(f"F1-score: {f1}")
print(f"AUC-ROC: {auc_roc}")

# Tạo ma trận nhầm lẫn
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('confusion_matrix.png')
plt.close()

# Lưu kết quả vào tệp
with open('evaluation_results.txt', 'w') as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Precision: {test_precision}\n")
    f.write(f"Recall: {test_recall}\n")
    f.write(f"F1-score: {f1}\n")
    f.write(f"AUC-ROC: {auc_roc}\n")
