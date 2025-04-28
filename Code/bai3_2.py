import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Hàm chuyển đổi tuổi
def convert_age(age_str):
    if isinstance(age_str, str) and '-' in age_str:
        try:
            year, days = map(int, age_str.split('-'))
            return round((year + days / 365), 2)
        except:
            return None
    return None

# Đọc dữ liệu
file_path = r'E:\ket_qua_bai_tap_lon\Code\results.csv'
df = pd.read_csv(file_path)

# Chuyển đổi tuổi
df['Age'] = df['Age'].apply(convert_age)

# Thay thế 'N/a' bằng 0
df.replace('N/a', 0, inplace=True)

# Lấy danh sách cột dữ liệu cần xử lý
headers = list(df.columns)[4:]

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
headers_t = [header + '_t' for header in headers]
df[headers_t] = scaler.fit_transform(df[headers])

# PCA giảm chiều
pca = PCA(n_components=2)
x_pca = pca.fit_transform(df[headers_t])
df_pca = pd.DataFrame(data=x_pca, columns=['PC1', 'PC2'])

n_clu=[7,8,9,10,11,12]
# KMeans clustering
for n in n_clu:
    kmeans = KMeans(n_clusters=n)
    clusters = kmeans.fit_predict(df_pca)
    df_pca['Cluster'] = clusters

    # Vẽ biểu đồ scatter
    plt.figure(figsize=(8, 6))
    plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['Cluster'])
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('KMeans Clustering with PCA,n_clusters={}'.format(n))
    plt.colorbar(label='Cụm')
    plt.grid(True)
    plt.show()
