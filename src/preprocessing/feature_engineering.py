import pandas as pd
df = pd.read_csv("data/synthetic_telco_churn.csv")

df['AvgPurchaseValue'] = df['TotalSpend'] / df['PurchaseCount']
df['Recency'] = (df['LastPurchaseDate'] - df['SignUpDate']).dt.days
df.to_csv("data/features/engineered_features.csv", index=False)
