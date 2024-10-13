import pandas as pd
import matplotlib.pyplot as plt
import os

# Định nghĩa đường dẫn đến thư mục lịch sử
current_dir = os.path.dirname(__file__)
history_dir = os.path.join(current_dir, '..', 'history')

# Tải kết quả của bạn từ tệp trong thư mục lịch sử
with open(os.path.join(history_dir, 'evaluation_results.txt'), 'r') as f:
    lines = f.readlines()
    your_results = {line.split(':')[0].strip(): float(line.split(':')[1].strip()) for line in lines}

# Kết quả của các nghiên cứu trước đó (dữ liệu ví dụ, thay thế bằng dữ liệu thực tế)
previous_studies = [
    {"name": "Study A", "accuracy": 0.85, "precision": 0.86, "recall": 0.84, "f1_score": 0.85, "auc_roc": 0.92},
    {"name": "Study B", "accuracy": 0.88, "precision": 0.89, "recall": 0.87, "f1_score": 0.88, "auc_roc": 0.94},
    # Thêm nhiều nghiên cứu hơn nếu cần
]

# Thêm kết quả của bạn vào danh sách
previous_studies.append({"name": "Your Study", **your_results})

# Tạo DataFrame từ danh sách các nghiên cứu
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
plt.savefig('comparison_with_previous_studies.png')
plt.close()

# In bảng so sánh
print(df.to_string(index=False))

