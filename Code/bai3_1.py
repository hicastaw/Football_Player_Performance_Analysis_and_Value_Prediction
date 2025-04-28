import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def convert_age(age_str):
    if isinstance(age_str, str) and '-' in age_str:
        try:
            year, days = map(int, age_str.split('-'))
            return round((year + days / 365),2)
        except:
            return None
    return None
def sil():
    sil_score = []
    for k in range(2, 75):
        kmeans = KMeans(n_clusters=k, random_state=0)
        labels = kmeans.fit_predict(df[headers_t])
        score = silhouette_score(df[headers_t], labels)
        sil_score.append(score)
    return sil_score
def optimize_kmean(data, k_range):
    means=[]
    inertia = []
    for k in range(2,k_range):
        kmeans = KMeans(n_clusters=k, random_state=0)
        kmeans.fit(data)
        means.append(k)
        inertia.append(kmeans.inertia_)
    return inertia

file_path = r'E:\ket_qua_bai_tap_lon\Code\results.csv'
df = pd.read_csv(file_path)
df['Age'] = df['Age'].apply(convert_age)
df.replace('N/a', 0, inplace=True)

headers=list(df.columns)[4:]
headers_t=[header +'_t' for header in headers]

scaler=StandardScaler()
df[headers_t]=scaler.fit_transform(df[headers])

inertia=optimize_kmean(df[headers_t], 75)
sil_score = sil()

fig, ax = plt.subplots(1,2,figsize=(14, 5))

ax[0].plot(range(2, 75), sil_score, marker='o')
ax[0].set_title('Silhouette Score')
ax[0].set_xlabel('Number of clusters')
ax[0].set_ylabel('Silhouette Score')
ax[0].axvline(x=8, color='red', linestyle='--')

ax[1].plot(range(2, 75), inertia, marker='o')
ax[1].set_title('Elbow Method')
ax[1].set_xlabel('Number of clusters')
ax[1].set_ylabel('Elbow')
ax[1].axvline(x=8, color='red', linestyle='--')

plt.suptitle('KMeans Clustering Analysis')
plt.tight_layout()

plt.show()



