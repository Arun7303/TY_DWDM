import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load and preprocess data (replace 'file_path' with the actual path of the Excel file)
file_path = 'Online_Retail.xlsx'
data = pd.read_excel(file_path, usecols=["InvoiceNo", "CustomerID", "Quantity", "UnitPrice", "InvoiceDate"])
data = data.dropna(subset=['CustomerID'])

# Calculate 'TotalSpent' for each transaction
data['TotalSpent'] = data['Quantity'] * data['UnitPrice']

# Aggregate per CustomerID to create features for classification
customer_data = data.groupby('CustomerID').agg({
    'TotalSpent': 'sum',                  
    'InvoiceNo': 'nunique'                 
}).rename(columns={'InvoiceNo': 'PurchaseFrequency'})

# Create classification target based on spending behavior
# We'll define 'TotalSpent' classes: low, medium, high spenders
customer_data['SpenderCategory'] = pd.qcut(customer_data['TotalSpent'], q=3, labels=['Low', 'Medium', 'High'])

# Define features and target
X = customer_data[['TotalSpent', 'PurchaseFrequency']]
y = customer_data['SpenderCategory']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialize and train the Naive Bayes classifier
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Predict on the test set
y_pred = nb_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Assuming y_test and y_pred are defined from the Naive Bayes code
conf_matrix = confusion_matrix(y_test, y_pred, labels=['Low', 'Medium', 'High'])

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low', 'Medium', 'High'], 
            yticklabels=['Low', 'Medium', 'High'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Naive Bayes Classification")
plt.show()
