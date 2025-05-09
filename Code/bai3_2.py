import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Function to convert age to decimal format (e.g., "25-150" -> 25.41)
def convert_age(age_str):
    if isinstance(age_str, str) and '-' in age_str:
        try:
            year, days = map(int, age_str.split('-'))  # Split the string into year and days, then convert to integers
            return round((year + days / 365), 2)  # Convert to decimal format
        except:
            return None  # Return None if there is an error
    return None  # Return None if format is incorrect

# Read data from CSV file
file_path = r'Code\\results.csv'
df = pd.read_csv(file_path)

# Apply age conversion function to the 'Age' column
df['Age'] = df['Age'].apply(convert_age)

# Replace 'N/a' values with 0 for consistency in the dataset
df.replace('N/a', 0, inplace=True)

# Get the list of columns (from the 5th column onwards) for processing
headers = list(df.columns)[4:]

# Standardize the data using StandardScaler (important for clustering)
scaler = StandardScaler()
headers_t = [header + '_t' for header in headers]
df[headers_t] = scaler.fit_transform(df[headers])  # Apply transformation

# Apply PCA for dimensionality reduction (reduce to 2 components for visualization)
pca = PCA(n_components=2)
x_pca = pca.fit_transform(df[headers_t])  # Apply PCA to the standardized data
df_pca = pd.DataFrame(data=x_pca, columns=['PC1', 'PC2'])  # Create a DataFrame with the PCA results

# List of possible cluster counts to try
n_clu = [7, 8, 9, 10, 11, 12]

# Loop through different cluster sizes to apply KMeans clustering
for n in n_clu:
    kmeans = KMeans(n_clusters=n)  # Initialize KMeans with the current number of clusters
    clusters = kmeans.fit_predict(df_pca)  # Perform clustering on the PCA-transformed data
    df_pca['Cluster'] = clusters  # Assign the cluster labels to the DataFrame

    # Plot the clustering result in a 2D scatter plot
    plt.figure(figsize=(8, 6))
    plt.scatter(df_pca['PC1'], df_pca['PC2'], c=df_pca['Cluster'])  # Scatter plot with color by cluster
    plt.xlabel('PC1')  # Label for the first principal component
    plt.ylabel('PC2')  # Label for the second principal component
    plt.title(f'KMeans Clustering with PCA, n_clusters={n}')  # Title for the plot
    plt.colorbar(label='Cluster')  # Add colorbar to indicate cluster labels
    plt.grid(True)  # Show grid on the plot
    plt.show()  # Display the plot
