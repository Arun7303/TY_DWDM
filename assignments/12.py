import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt

# Load and preprocess data (replace 'file_path' with the actual file path)
file_path = 'Online_Retail.xlsx'
data = pd.read_excel(file_path, usecols=["InvoiceNo", "CustomerID", "Quantity", "UnitPrice", "InvoiceDate"])
data = data.dropna(subset=['CustomerID'])

# Calculate 'TotalSpent' for each transaction
data['TotalSpent'] = data['Quantity'] * data['UnitPrice']

# Aggregate per CustomerID to create features for clustering
customer_data = data.groupby('CustomerID').agg({
    'TotalSpent': 'sum',                  
    'InvoiceNo': 'nunique'                 
}).rename(columns={'InvoiceNo': 'PurchaseFrequency'})

# Standardize the features
scaler = StandardScaler()
customer_data_scaled = scaler.fit_transform(customer_data)

# Apply Hierarchical Clustering
linked = linkage(customer_data_scaled, method='ward')  # 'ward' minimizes variance within clusters

# Plot the dendrogram
plt.figure(figsize=(15, 10))
dendrogram(linked, orientation='top', distance_sort='ascending', show_leaf_counts=False)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Customer Index')
plt.ylabel('Distance')
plt.show()
