import pandas as pd
import matplotlib.pyplot as plt
import os

# Định nghĩa đường dẫn đến thư mục lịch sử
current_dir = os.path.dirname(__file__)
history_dir = os.path.join(current_dir, '..', 'history')

# Tải kết quả từ tệp trong thư mục lịch sử
evaluation_results_path = os.path.join(history_dir, 'evaluation_results.txt')
with open(evaluation_results_path, 'r') as f:
    lines = f.readlines()
    results = {line.split(':')[0].strip(): float(line.split(':')[1].strip()) for line in lines}

# Kết quả của các nghiên cứu trước đó
previous_studies = []

# Thêm kết quả của bạn vào danh sách
previous_studies.append({"name": "Your Study", **results})

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
comparison_plot_path = os.path.join(history_dir, 'comparison_with_previous_studies.png')
plt.savefig(comparison_plot_path)
plt.close()

# In bảng so sánh
print("Bảng so sánh kết quả:")
print(df.to_string(index=False))

