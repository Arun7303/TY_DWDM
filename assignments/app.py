import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth

# Loading the Data
data = pd.read_excel('Online_Retail.xlsx')
data.head()

# Exploring the columns of the data
data.columns

# Exploring the different regions of transactions
data.Country.unique()

# Stripping extra spaces in the description
data['Description'] = data['Description'].str.strip()

# Dropping the rows without any invoice number
data.dropna(axis=0, subset=['InvoiceNo'], inplace=True)
data['InvoiceNo'] = data['InvoiceNo'].astype('str')

# Dropping all transactions which were done on credit
data = data[~data['InvoiceNo'].str.contains('C')]

# Transactions done in France
basket_France = (data[data['Country'] == "France"]
                 .groupby(['InvoiceNo', 'Description'])['Quantity']
                 .sum().unstack().reset_index().fillna(0)
                 .set_index('InvoiceNo'))

# Transactions done in the United Kingdom
basket_UK = (data[data['Country'] == "United Kingdom"]
             .groupby(['InvoiceNo', 'Description'])['Quantity']
             .sum().unstack().reset_index().fillna(0)
             .set_index('InvoiceNo'))

# Transactions done in Portugal
basket_Por = (data[data['Country'] == "Portugal"]
              .groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().reset_index().fillna(0)
              .set_index('InvoiceNo'))

# Transactions done in Sweden
basket_Sweden = (data[data['Country'] == "Sweden"]
                 .groupby(['InvoiceNo', 'Description'])['Quantity']
                 .sum().unstack().reset_index().fillna(0)
                 .set_index('InvoiceNo'))

# Defining the hot encoding function to make the data suitable for the concerned libraries
def hot_encode(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1

# Encoding the datasets with map (since applymap is deprecated)
basket_France = basket_France.map(hot_encode)
basket_UK = basket_UK.map(hot_encode)
basket_Por = basket_Por.map(hot_encode)
basket_Sweden = basket_Sweden.map(hot_encode)

# --------------------
# Building models with FP-Growth instead of Apriori
# --------------------

# France
frq_items = fpgrowth(basket_France, min_support=0.05, use_colnames=True)
rules = association_rules(frq_items, metric="lift", min_threshold=1)
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
print("France Rules")
print(rules.head())

# UK (using higher min_support to avoid memory issues)
frq_items = fpgrowth(basket_UK, min_support=0.02, use_colnames=True)
rules = association_rules(frq_items, metric="lift", min_threshold=1)
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
print("UK Rules")
print(rules.head())

# Portugal
frq_items = fpgrowth(basket_Por, min_support=0.05, use_colnames=True)
rules = association_rules(frq_items, metric="lift", min_threshold=1)
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
print("Portugal Rules")
print(rules.head())

# Sweden
frq_items = fpgrowth(basket_Sweden, min_support=0.05, use_colnames=True)
rules = association_rules(frq_items, metric="lift", min_threshold=1)
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])
print("Sweden Rules")
print(rules.head())
