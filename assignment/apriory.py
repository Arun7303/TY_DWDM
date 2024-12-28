
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Sample dataset simulating social media hashtags usage
data = {'#food': [1, 1, 1, 0, 1],
        '#travel': [1, 1, 0, 1, 1],
        '#fashion': [0, 1, 1, 1, 1],
        '#fitness': [1, 0, 0, 1, 1]}

# Creating a DataFrame
df = pd.DataFrame(data)

# Convert the DataFrame to boolean values
df = df.astype(bool)

# Applying the Apriori Algorithm for hashtag association
min_support = 0.5
frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

# Generate num_itemsets dictionary for older versions of mlxtend
num_itemsets = {frozenset(itemset): itemset_count 
                for itemset, itemset_count in zip(frequent_itemsets['itemsets'], frequent_itemsets['support'] * len(df))}

# Generate association rules based on the frequent itemsets
min_confidence = 0.7
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence, num_itemsets=num_itemsets)

# Display the results
print("Frequent Hashtag Itemsets:\n", frequent_itemsets)
print("\nAssociation Rules:\n", rules)
