import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset (replace 'file_path' with the actual file path of the Excel file)
file_path = 'Online_Retail.xlsx'
data = pd.read_excel(file_path, usecols=["InvoiceNo", "CustomerID", "Quantity", "UnitPrice", "InvoiceDate"])

# Drop rows with missing CustomerID values as clustering is based on customer ID
data = data.dropna(subset=['CustomerID'])

# Calculate the 'TotalSpent' for each transaction
data['TotalSpent'] = data['Quantity'] * data['UnitPrice']

# Aggregate the data per CustomerID to create clustering features
customer_data = data.groupby('CustomerID').agg({
    'TotalSpent': 'sum',                # Total spending per customer
    'InvoiceNo': 'nunique'               # Frequency of purchases (unique invoices per customer)
}).rename(columns={'InvoiceNo': 'PurchaseFrequency'})

# Standardize the data to ensure fair clustering
scaler = StandardScaler()
customer_data_scaled = scaler.fit_transform(customer_data)

# Apply K-means clustering with 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
customer_data['Cluster'] = kmeans.fit_predict(customer_data_scaled)

# Display the cluster means to understand the segments
cluster_summary = customer_data.groupby('Cluster').mean()
print(cluster_summary)
import matplotlib.pyplot as plt

# Scatter plot for visualizing clusters
plt.figure(figsize=(10, 6))

# Plot each cluster with a different color
for cluster in customer_data['Cluster'].unique():
    cluster_data = customer_data[customer_data['Cluster'] == cluster]
    plt.scatter(
        cluster_data['PurchaseFrequency'],
        cluster_data['TotalSpent'],
        label=f'Cluster {cluster}'
    )

# Labeling the plot
plt.xlabel('Purchase Frequency')
plt.ylabel('Total Spent')
plt.title('Customer Segments based on Purchase Frequency and Total Spent')
plt.legend()
plt.grid(True)
plt.show()
