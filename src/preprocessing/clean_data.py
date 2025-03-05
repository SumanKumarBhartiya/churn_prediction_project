import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv("data/synthetic_telco_churn.csv")

df.fillna(df.median(), inplace=True)
encoder = LabelEncoder()
df['SubscriptionType'] = encoder.fit_transform(df['SubscriptionType'])

scaler = StandardScaler()
df[['Age', 'MonthlySpend']] = scaler.fit_transform(df[['Age', 'MonthlySpend']])

df.to_csv("data/processed/cleaned_data.csv", index=False)
