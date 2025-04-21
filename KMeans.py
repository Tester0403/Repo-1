# 1. Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 2. Load the dataset
df = pd.read_csv("Mall_Customers.csv")

# Display basic info
print(df.head())
print("\nSummary:\n", df.describe())

# 3. Preprocess and normalize 'Age' and 'Annual Income'
X = df[['Age', 'Annual Income (k$)']]  # You can add 'Spending Score' too
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Apply KMeans clustering with k = 3
kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# 5. Visualize the clusters
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='Age', y='Annual Income (k$)', hue='Cluster', palette='Set1')
plt.title("Customer Segments (K=3)")
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")
plt.legend()
plt.show()

# 6. Analyze characteristics of each cluster
cluster_analysis = df.groupby('Cluster')[['Age', 'Annual Income (k$)']].mean()
print("\nCluster Characteristics:\n", cluster_analysis)

# Optional: Show how many customers in each group
print("\nCustomer count per cluster:\n", df['Cluster'].value_counts())
