# 1. Import libraries
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
import matplotlib.pyplot as plt

# 2. Load and preprocess the dataset
# Assuming the dataset is in this format: one transaction per line, items separated by commas
with open('groceries.csv', 'r') as f:
    transactions = [line.strip().split(',') for line in f.readlines()]

# 3. Convert to one-hot encoded DataFrame
te = TransactionEncoder()
te_array = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_array, columns=te.columns_)

# 4. Apply Apriori with min_support = 0.01
frequent_itemsets = apriori(df, min_support=0.01, use_colnames=True)

# 5. Generate association rules with min_confidence = 0.3
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.3)

# 6. Sort rules by lift and display top 5
sorted_rules = rules.sort_values(by='lift', ascending=False)
print("\nTop 5 Rules by Lift:\n", sorted_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head())

# 7. Visualize top 10 items by frequency
item_freq = df.sum().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
item_freq.plot(kind='bar', color='orange')
plt.title('Top 10 Most Frequent Items')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.show()

# 8. Analyze a strong rule
strong_rule = sorted_rules.iloc[0]
print("\nStrong Rule Example:")
print(f"If a person buys {list(strong_rule['antecedents'])}, they are likely to also buy {list(strong_rule['consequents'])}")
print(f"Support: {strong_rule['support']:.2f}, Confidence: {strong_rule['confidence']:.2f}, Lift: {strong_rule['lift']:.2f}")
