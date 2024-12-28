# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# Step 1: Load the dataset from a string (since you provided it directly)
data = """Outlook,Temperature,Humidity,Windy,Play
Sunny,Hot,High,False,No
Sunny,Hot,High,True,No
Overcast,Hot,High,False,Yes
Rain,Mild,High,False,Yes
Rain,Cool,Normal,False,Yes
Rain,Cool,Normal,True,No
Overcast,Cool,Normal,True,Yes
Sunny,Mild,High,False,No
Sunny,Cool,Normal,False,Yes
Rain,Mild,Normal,False,Yes
Sunny,Mild,Normal,True,Yes
Overcast,Mild,High,True,Yes
Overcast,Hot,Normal,False,Yes
Rain,Mild,High,True,No"""

# Read the data into a pandas DataFrame
from io import StringIO
df = pd.read_csv(StringIO(data))

# Print the column names to confirm the DataFrame structure
print("Column names in the dataset:")
print(df.columns)

# Step 2: Preprocess the data
# Convert categorical variables to numerical using one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Define features and target variable
target_column_name = 'Play_Yes'  # Use the encoded target column name
X = df.drop(target_column_name, axis=1)  # Features
y = df[target_column_name]                 # Target variable

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Create and train the Decision Tree model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Make predictions
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

# Step 8: Visualize the Decision Tree (optional)
plt.figure(figsize=(12, 8))
plot_tree(model, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
plt.title('Decision Tree Visualization')
plt.show()
