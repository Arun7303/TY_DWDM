
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score


file_path = 'reuters.csv' 
data = pd.read_csv(file_path)

# Step 2: Define the target variable (optional, based on thresholds)
# Example: Consider a post viral if it has more than 500 likes, 100 shares, and 50 comments
data['Viral'] = ((data['likes'] > 200) & (data['shares'] > 50) & (data['comments'] > 20)).astype(int)

# Step 3: Preprocess Text Data (option 1: exclude text column)
X = data[['likes', 'comments', 'shares']]  # features without text
y = data['Viral']  # target variable


# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Initialize and train the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = dt_classifier.predict(X_test)

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Print the accuracy and classification report
print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_rep)

# Optional: To visualize the tree, use the following code
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Create the figure with larger size for better readability
fig, ax = plt.subplots(figsize=(25,15))

# Plot the tree with adjusted parameters for better spacing and appearance
plot_tree(dt_classifier, 
          feature_names=X.columns, 
          class_names=['Non-Viral', 'Viral'], 
          filled=True, 
          proportion=False,  # Display class proportion at each node
          rounded=True,      # Round the corners of the boxes
          precision=2,       # Display only two decimal points
          fontsize=10,       # Adjust the font size for better readability
          ax=ax)

# Adjust layout to add some space around the tree
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)

# Show the plot
plt.show()
