import pandas as pd
import matplotlib.pyplot as plt
import os

# CSV 파일 경로
csv_file_path = 'depth_data/2025-06-06_18-03-12/depth_data.csv'

# CSV 파일 읽기
try:
    # Assuming 'MinDepth' and 'MaxDepth' columns are added
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print(f"Error: The file {csv_file_path} was not found.")
    exit()

# 데이터에 'MinDepth'와 'MaxDepth' 컬럼이 있는지 확인
if 'MinDepth' not in df.columns or 'MaxDepth' not in df.columns:
    print("Error: 'MinDepth' or 'MaxDepth' columns not found in the CSV file. Please ensure these columns are present.")
    exit()

# 그래프 생성
plt.figure(figsize=(12, 6))
plt.plot(df['Frame'], df['AverageDepth'], label='Average Depth', marker='o', linestyle='-', markersize=4, zorder=2)

# 각 프레임의 Min-Max 범위 표시
plt.fill_between(df['Frame'], df['MinDepth'], df['MaxDepth'], color='blue', alpha=0.2, label='Min-Max Range')

# 이상치 데이터 포인트 표시
outliers = df[df['IsOutlier'] == True]
plt.scatter(outliers['Frame'], outliers['AverageDepth'], color='red', label='Outlier', zorder=5)

# 그래프 제목 및 축 라벨 설정
plt.title('Average Depth over Frames')
plt.xlabel('Frame')
plt.ylabel('Average Depth')
plt.grid(True)
plt.legend()

# 그래프 저장 경로 설정 및 저장
output_dir = './analysis_output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

output_plot_path = os.path.join(output_dir, 'depth_plot_no_outlier.png')
plt.savefig(output_plot_path)

print(f"Plot saved to {output_plot_path}")
