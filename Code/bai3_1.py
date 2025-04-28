import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Function to convert age from string format (e.g., "25-150") into a decimal number
def convert_age(age_str):
    if isinstance(age_str, str) and '-' in age_str:
        try:
            year, days = map(int, age_str.split('-'))  # Split the string by "-" into year and days, then convert to integers
            return round((year + days / 365), 2)  # Convert the years and days into a decimal year format
        except:
            return None  # Return None if there's an error in conversion
    return None  # Return None if the string format is not valid

# Function to calculate silhouette scores for different cluster sizes
def sil():
    sil_score = []
    for k in range(2, 75):  # Loop through a range of cluster numbers (from 2 to 74)
        kmeans = KMeans(n_clusters=k, random_state=0)  # Initialize KMeans with k clusters
        labels = kmeans.fit_predict(df[headers_t])  # Fit the model and predict cluster labels
        score = silhouette_score(df[headers_t], labels)  # Calculate the silhouette score for the clustering
        sil_score.append(score)  # Append the score for the current k
    return sil_score  # Return the list of silhouette scores

# Function to calculate inertia for different cluster sizes (for elbow method)
def optimize_kmean(data, k_range):
    means = []
    inertia = []
    for k in range(2, k_range):  # Loop through a range of cluster numbers (from 2 to k_range)
        kmeans = KMeans(n_clusters=k, random_state=0)  # Initialize KMeans with k clusters
        kmeans.fit(data)  # Fit the KMeans model
        means.append(k)  # Append the current k value
        inertia.append(kmeans.inertia_)  # Append the inertia (sum of squared distances of samples to their centroids)
    return inertia  # Return the list of inertia values

# Load the dataset from the specified file path
file_path = r'E:\ket_qua_bai_tap_lon\Code\results.csv'
df = pd.read_csv(file_path)

# Apply the age conversion function to the 'Age' column
df['Age'] = df['Age'].apply(convert_age)

# Replace 'N/a' values with 0 in the dataset for consistency
df.replace('N/a', 0, inplace=True)

# List the columns to be used for clustering (starting from the 4th column onward)
headers = list(df.columns)[4:]
# Create new column names for the standardized data (adding '_t' suffix)
headers_t = [header + '_t' for header in headers]

# Standardize the data using StandardScaler (for clustering)
scaler = StandardScaler()
df[headers_t] = scaler.fit_transform(df[headers])

# Get the inertia values using the elbow method (for optimal number of clusters)
inertia = optimize_kmean(df[headers_t], 75)

# Get the silhouette scores for different cluster sizes
sil_score = sil()

# Create a figure with two subplots for visualizing results
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Plot silhouette scores
ax[0].plot(range(2, 75), sil_score, marker='o')  # Plot silhouette scores for k = 2 to 74
ax[0].set_title('Silhouette Score')  # Set the title of the plot
ax[0].set_xlabel('Number of clusters')  # Label for the x-axis
ax[0].set_ylabel('Silhouette Score')  # Label for the y-axis
ax[0].axvline(x=8, color='red', linestyle='--')  # Add a vertical line at k=8 (chosen for analysis)

# Plot inertia (Elbow Method)
ax[1].plot(range(2, 75), inertia, marker='o')  # Plot inertia values for k = 2 to 74
ax[1].set_title('Elbow Method')  # Set the title of the plot
ax[1].set_xlabel('Number of clusters')  # Label for the x-axis
ax[1].set_ylabel('Elbow')  # Label for the y-axis
ax[1].axvline(x=8, color='red', linestyle='--')  # Add a vertical line at k=8 (suggested by the elbow method)

# Set the overall title for the plots
plt.suptitle('KMeans Clustering Analysis')

# Adjust the layout to ensure no overlapping of plots
plt.tight_layout()

# Display the plots
plt.show()
