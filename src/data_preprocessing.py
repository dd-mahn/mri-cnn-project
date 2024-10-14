import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

def load_and_preprocess_image(file_path, target_size=(224, 224)):
    """
    Tải và tiền xử lý hình ảnh.

    Tham số:
    - file_path (str): Đường dẫn đến tệp hình ảnh.
    - target_size (tuple): Kích thước mục tiêu để thay đổi kích thước hình ảnh.

    Trả về:
    - img_array (numpy array): Mảng numpy của hình ảnh đã được tiền xử lý.
    """
    # Tải hình ảnh
    img = Image.open(file_path).convert('L')  # Chuyển đổi sang ảnh xám
    
    # Thay đổi kích thước
    img = img.resize(target_size)
    
    # Chuyển đổi sang mảng numpy và chuẩn hóa
    img_array = np.array(img) / 255.0
    
    # Chuyển đổi ảnh xám sang RGB bằng cách xếp chồng kênh đơn
    img_array = np.stack((img_array,) * 3, axis=-1)
    
    return img_array

def apply_filters(image):
    """
    Áp dụng các bộ lọc tùy chỉnh cho hình ảnh.

    Tham số:
    - image (numpy array): Mảng numpy của hình ảnh.

    Trả về:
    - image (numpy array): Mảng numpy của hình ảnh đã được áp dụng bộ lọc.
    """
    # Điều chỉnh độ tương phản bằng NumPy
    mean = np.mean(image)
    image = (image - mean) * 2 + mean  # Điều chỉnh độ tương phản đơn giản
    # Cắt giá trị để duy trì phạm vi hợp lệ
    image = np.clip(image, 0, 1)
    # Điều chỉnh độ sáng ngẫu nhiên
    image = image + np.random.uniform(-0.2, 0.2)
    image = np.clip(image, 0, 1)
    return image

def prepare_dataset(data_dirs, target_size=(224, 224)):
    """
    Chuẩn bị tập dữ liệu từ thư mục hình ảnh.

    Tham số:
    - data_dirs (list of str): Danh sách đường dẫn đến thư mục chứa hình ảnh.
    - target_size (tuple): Kích thước mục tiêu để thay đổi kích thước hình ảnh.

    Trả về:
    - X_train, X_val, X_test (numpy arrays): Các tập dữ liệu huấn luyện, xác thực và kiểm tra.
    - y_train, y_val, y_test (numpy arrays): Các nhãn tương ứng cho các tập dữ liệu.
    """
    images = []
    labels = []
    
    # Xử lý từng hình ảnh trong thư mục
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
                    img_array = load_and_preprocess_image(img_path, target_size)
                    img_array = apply_filters(img_array)
                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f"Lỗi khi xử lý {img_path}: {e}")
    if not images:
        print("Không tìm thấy hình ảnh. Vui lòng kiểm tra đường dẫn thư mục và nội dung.")
        return None, None, None, None
    
    # Điều chỉnh tỷ lệ chia dữ liệu
    X_train, X_temp, y_train, y_temp = train_test_split(np.array(images), np.array(labels), test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def load_test_data(test_dir, target_size=(224, 224)):
    """
    Tải dữ liệu kiểm tra từ thư mục hình ảnh.

    Tham số:
    - test_dir (str): Đường dẫn đến thư mục chứa hình ảnh kiểm tra.
    - target_size (tuple): Kích thước mục tiêu để thay đổi kích thước hình ảnh.

    Trả về:
    - test_images (numpy array): Mảng numpy của hình ảnh kiểm tra.
    - test_labels (numpy array): Mảng numpy của nhãn kiểm tra.
    """
    test_images = []
    test_labels = []
    for class_name in os.listdir(test_dir):
        class_dir = os.path.join(test_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for img_name in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_name)
            img_array = load_and_preprocess_image(img_path, target_size)
            test_images.append(img_array)
            test_labels.append(class_name)
    
    # Chuyển đổi nhãn sang định dạng số
    unique_labels = list(set(test_labels))
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    test_labels = [label_to_index[label] for label in test_labels]
    
    return np.array(test_images), np.array(test_labels)

def save_processed_data(output_dir, X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Lưu dữ liệu đã xử lý vào thư mục đầu ra.

    Tham số:
    - output_dir (str): Đường dẫn đến thư mục đầu ra.
    - X_train, X_val, X_test (numpy arrays): Các tập dữ liệu huấn luyện, xác thực và kiểm tra.
    - y_train, y_val, y_test (numpy arrays): Các nhãn tương ứng cho các tập dữ liệu.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)

def extract_image_attributes(image):
    """
    Extract attributes from the image.
    """
    attributes = {
        'mean_intensity': np.mean(image),
        'std_intensity': np.std(image),
        'min_intensity': np.min(image),
        'max_intensity': np.max(image),
        'shape': image.shape
    }
    return attributes

# Sử dụng
if __name__ == "__main__":
    # Lấy thư mục hiện tại của script
    current_dir = os.path.dirname(__file__)

    # Định nghĩa đường dẫn tương đối đến thư mục dữ liệu
    data_dir = os.path.join(current_dir, '..', 'data', "raw")

    # Chuyển đổi sang đường dẫn tuyệt đối (tùy chọn, nhưng có thể hữu ích để rõ ràng)
    data_dir = os.path.abspath(data_dir)
    
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_dataset([data_dir])
    if X_train is not None:
        print(f"Hình dạng tập huấn luyện: {X_train.shape}")
        print(f"Hình dạng tập xác thực: {X_val.shape}")
        print(f"Hình dạng tập kiểm tra: {X_test.shape}")
    
    # Định nghĩa đường dẫn đến thư mục đầu ra
    output_dir = os.path.join(current_dir, '..', 'data', 'processed')
    output_dir = os.path.abspath(output_dir)
    
    if X_train is not None:
        save_processed_data(output_dir, X_train, X_val, X_test, y_train, y_val, y_test)
