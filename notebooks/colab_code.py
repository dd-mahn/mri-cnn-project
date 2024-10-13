import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from google.colab import drive

# Gắn kết Google Drive
drive.mount('/content/drive')

# Thiết lập các đường dẫn
base_path = '/content/drive/My Drive/mri-cnn-project'
data_dir = os.path.join(base_path, 'data', 'raw')
processed_dir = os.path.join(base_path, 'data', 'processed')
model_dir = os.path.join(base_path, 'model')
results_dir = os.path.join(base_path, 'results')

# Tạo các thư mục nếu chúng chưa tồn tại
for dir_path in [processed_dir, model_dir, results_dir]:
    os.makedirs(dir_path, exist_ok=True)

# Hàm tiền xử lý dữ liệu
def load_and_preprocess_image(file_path, target_size=(224, 224)):
    # Tải hình ảnh và chuyển đổi sang mảng numpy
    img = tf.keras.preprocessing.image.load_img(file_path, color_mode='grayscale', target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # Chuẩn hóa giá trị pixel
    img_array = img_array / 255.0
    # Chuyển đổi hình ảnh grayscale thành 3 kênh
    img_array = np.stack((img_array,) * 3, axis=-1)
    return img_array

# Hàm áp dụng các bộ lọc tùy chỉnh cho hình ảnh
def apply_filters(image):
    # Tính giá trị trung bình của hình ảnh
    mean = np.mean(image)
    # Điều chỉnh độ tương phản
    image = (image - mean) * 2 + mean
    # Cắt giá trị để duy trì phạm vi hợp lệ
    image = np.clip(image, 0, 1)
    # Điều chỉnh độ sáng ngẫu nhiên
    image = image + np.random.uniform(-0.2, 0.2)
    image = np.clip(image, 0, 1)
    return image

# Hàm chuẩn bị tập dữ liệu
def prepare_dataset(data_dirs, target_size=(224, 224)):
    images = []
    labels = []
    # Duyệt qua từng thư mục dữ liệu
    for data_dir in data_dirs:
        for sub_dir in os.listdir(data_dir):
            sub_dir_path = os.path.join(data_dir, sub_dir)
            if not os.path.isdir(sub_dir_path):
                continue  # Bỏ qua nếu không phải là thư mục
            label = sub_dir  # Sử dụng tên thư mục con làm nhãn
            print(f"Xử lý thư mục: {sub_dir_path}")
            for img_name in os.listdir(sub_dir_path):
                img_path = os.path.join(sub_dir_path, img_name)
                if not os.path.isfile(img_path):
                    continue  # Bỏ qua nếu không phải là tệp
                try:
                    # Tải và tiền xử lý hình ảnh
                    img_array = load_and_preprocess_image(img_path, target_size)
                    # Áp dụng bộ lọc tùy chỉnh
                    img_array = apply_filters(img_array)
                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f"Lỗi khi xử lý {img_path}: {e}")
    
    if not images:
        print("Không tìm thấy hình ảnh. Vui lòng kiểm tra đường dẫn thư mục và nội dung.")
        return None, None, None, None, None, None
    
    # Chuyển đổi danh sách hình ảnh và nhãn thành mảng numpy
    X = np.array(images)
    y = np.array(labels)
    # Chia dữ liệu thành tập huấn luyện và tập tạm thời
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    # Chia tập tạm thời thành tập xác thực và tập kiểm tra
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

# Hàm lưu dữ liệu đã xử lý vào thư mục đầu ra
def save_processed_data(output_dir, X_train, X_val, X_test, y_train, y_val, y_test):
    # Lưu từng tập dữ liệu vào tệp .npy
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

# Hàm xây dựng mô hình
def build_resnet(input_shape, num_classes):
    # Tải mô hình ResNet50 với trọng số từ ImageNet, không bao gồm phần đầu ra
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    
    # Đóng băng các lớp đầu tiên của mô hình cơ sở
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    
    # Thêm các lớp tùy chỉnh vào mô hình
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)  # Lớp pooling toàn cục
    x = layers.Dense(1024, activation='relu')(x)  # Lớp fully connected với 1024 đơn vị và hàm kích hoạt ReLU
    outputs = layers.Dense(num_classes, activation='softmax')(x)  # Lớp đầu ra với số lượng đơn vị bằng số lớp và hàm softmax
    
    # Tạo mô hình hoàn chỉnh
    model = models.Model(inputs=base_model.input, outputs=outputs)
    
    return model

# Hàm huấn luyện mô hình
def train_mri_classification_model(X_train, y_train, X_val, y_val):
    # Chuyển đổi nhãn chuỗi thành chỉ số số nguyên
    label_to_index = {label: index for index, label in enumerate(np.unique(y_train))}
    y_train = np.array([label_to_index[label] for label in y_train])
    y_val = np.array([label_to_index[label] for label in y_val])

    # Xây dựng mô hình ResNet với kích thước đầu vào và số lớp
    model = build_resnet((224, 224, 3), num_classes=len(label_to_index))
    # Biên dịch mô hình với optimizer Adam và hàm loss sparse_categorical_crossentropy
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Thiết lập các callback cho quá trình huấn luyện
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10),  # Dừng sớm nếu không cải thiện sau 10 epoch
        tf.keras.callbacks.ModelCheckpoint(os.path.join(model_dir, 'model_best.h5'), save_best_only=True)  # Lưu mô hình tốt nhất
    ]

    # Huấn luyện mô hình
    history = model.fit(X_train, y_train, 
                        epochs=100,  # Số epoch tối đa
                        batch_size=32,  # Kích thước batch
                        validation_data=(X_val, y_val),  # Dữ liệu validation
                        callbacks=callbacks)  # Sử dụng callbacks đã định nghĩa

    # Lưu mô hình cuối cùng
    model.save(os.path.join(model_dir, 'mri_classification_model.h5'))
    # Lưu lại lịch sử huấn luyện
    np.save(os.path.join(results_dir, 'training_history.npy'), history.history)

    return model, history, label_to_index

# Hàm đánh giá mô hình
def evaluate_model(model, X_test, y_test, label_to_index):
    # Chuyển đổi nhãn chuỗi thành chỉ số số nguyên
    y_test_encoded = np.array([label_to_index[label] for label in y_test])
    # Đánh giá mô hình trên tập kiểm tra
    test_loss, test_acc = model.evaluate(X_test, y_test_encoded)
    print(f"Độ chính xác trên tập kiểm tra: {test_acc}")

    # Dự đoán nhãn cho tập kiểm tra
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)

    # Tính toán các chỉ số đánh giá
    accuracy = accuracy_score(y_test_encoded, y_pred)
    precision = precision_score(y_test_encoded, y_pred, average='weighted')
    recall = recall_score(y_test_encoded, y_pred, average='weighted')
    f1 = f1_score(y_test_encoded, y_pred, average='weighted')
    auc_roc = roc_auc_score(y_test_encoded, predictions, multi_class='ovr', average='weighted')

    print(f"Độ chính xác: {accuracy}")
    print(f"Độ chính xác (precision): {precision}")
    print(f"Độ nhạy (recall): {recall}")
    print(f"F1-score: {f1}")
    print(f"AUC-ROC: {auc_roc}")

    # Tạo ma trận nhầm lẫn
    cm = confusion_matrix(y_test_encoded, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Ma trận nhầm lẫn')
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.savefig(os.path.join(results_dir, 'confusion_matrix.png'))
    plt.close()

    # Lưu kết quả đánh giá vào tệp
    with open(os.path.join(results_dir, 'evaluation_results.txt'), 'w') as f:
        f.write(f"Độ chính xác: {accuracy}\n")
        f.write(f"Độ chính xác (precision): {precision}\n")
        f.write(f"Độ nhạy (recall): {recall}\n")
        f.write(f"F1-score: {f1}\n")
        f.write(f"AUC-ROC: {auc_roc}\n")

    return accuracy, precision, recall, f1, auc_roc

# Hàm so sánh kết quả
def compare_results():
    # Đọc kết quả đánh giá từ tệp
    with open(os.path.join(results_dir, 'evaluation_results.txt'), 'r') as f:
        lines = f.readlines()
        your_results = {line.split(':')[0].strip(): float(line.split(':')[1].strip()) for line in lines}

    # Kết quả của các nghiên cứu trước đó
    previous_studies = [
        {"name": "Study A", "accuracy": 0.85, "precision": 0.86, "recall": 0.84, "f1_score": 0.85, "auc_roc": 0.92},
        {"name": "Study B", "accuracy": 0.88, "precision": 0.89, "recall": 0.87, "f1_score": 0.88, "auc_roc": 0.94},
    ]

    # Thêm kết quả của bạn vào danh sách
    previous_studies.append({"name": "Your Study", **your_results})
    df = pd.DataFrame(previous_studies)

    # Vẽ biểu đồ so sánh
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
    plt.figure(figsize=(12, 6))

    for i, metric in enumerate(metrics):
        plt.subplot(2, 3, i+1)
        plt.bar(df['name'], df[metric])
        plt.title(metric.capitalize())
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'comparison_with_previous_studies.png'))
    plt.close()

    # In bảng so sánh
    print(df.to_string(index=False))

# Khối thực thi chính
if __name__ == "__main__":
    # Chuẩn bị tập dữ liệu
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset([data_dir])
    if X_train is not None:
        # Lưu dữ liệu đã xử lý
        save_processed_data(processed_dir, X_train, X_val, X_test, y_train, y_val, y_test)
        print("Hoàn thành tiền xử lý dữ liệu.")

        # Huấn luyện mô hình
        model, history, label_to_index = train_mri_classification_model(X_train, y_train, X_val, y_val)
        print("Hoàn thành huấn luyện mô hình.")

        # Đánh giá mô hình
        evaluate_model(model, X_test, y_test, label_to_index)
        print("Hoàn thành đánh giá mô hình.")

        # So sánh kết quả
        compare_results()
        print("Hoàn thành so sánh kết quả.")
    else:
        print("Chuẩn bị dữ liệu thất bại. Vui lòng kiểm tra thư mục dữ liệu và thử lại.")