import pandas as pd
import matplotlib.pyplot as plt

# 첫 번째 데이터 파일 경로 (outlier 없음)
file_path_no_outliers = 'depth_data/2025-06-06_18-03-12/depth_data.csv'
# 두 번째 데이터 파일 경로 (outlier 존재)
file_path_with_outliers = 'depth_data/2025-06-06_17-53-28/depth_data.csv'

# CSV 파일 읽기
try:
    df_no_outliers = pd.read_csv(file_path_no_outliers)
    df_with_outliers = pd.read_csv(file_path_with_outliers)

    # 그래프 그리기
    plt.figure(figsize=(10, 5))

    # Outlier 없는 데이터 플롯
    plt.plot(df_no_outliers['Frame'], df_no_outliers['AverageDepth'], label='Non Outliers - Average Depth', linestyle='-', markersize=4, zorder=2)
    # Outlier 없는 데이터의 Min-Max 범위 표시
    plt.fill_between(df_no_outliers['Frame'], df_no_outliers['MinDepth'], df_no_outliers['MaxDepth'], color='blue', alpha=0.1, label='Non Outliers - Min-Max Range')

    # Outlier 있는 데이터 플롯
    plt.plot(df_with_outliers['Frame'], df_with_outliers['AverageDepth'], label='Outliers - Average Depth', linestyle='-', markersize=4, zorder=2)
    # Outlier 있는 데이터의 Min-Max 범위 표시
    plt.fill_between(df_with_outliers['Frame'], df_with_outliers['MinDepth'], df_with_outliers['MaxDepth'], color='orange', alpha=0.1, label='Outliers - Min-Max Range')

    # Outlier 있는 데이터에서 이상치 포인트 표시
    outliers_with_data = df_with_outliers[df_with_outliers['IsOutlier'] == True]
    plt.scatter(outliers_with_data['Frame'], outliers_with_data['AverageDepth'], color='red', label='Outlier (Outliers data)', zorder=5)

    # 그래프 제목 및 축 라벨 설정
    plt.title('data compare')
    plt.xlabel('frame')
    plt.ylabel('depth')
    plt.legend()
    plt.grid(True)

    # 그래프 표시
    plt.savefig('depth_data/depth_data_compare.png')
    plt.show()

except FileNotFoundError as e:
    print(f"파일을 찾을 수 없습니다: {e}")
except KeyError as e:
    print(f"필요한 컬럼이 파일에 없습니다: {e}")
except Exception as e:
    print(f"오류가 발생했습니다: {e}")
