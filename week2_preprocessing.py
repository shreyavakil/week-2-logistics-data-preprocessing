import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# 1. Load the logistics dataset
df = pd.read_csv("logistics_sample_data.csv")

# 2. Display basic information
print("Dataset shape:", df.shape)
print(df.head())
print(df.info())

# 3. Check missing values
print("\nMissing values:")
print(df.isnull().sum())

# 4. Remove duplicate records
df = df.drop_duplicates()

# 5. Convert date columns
df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
df["Shipping_Date"] = pd.to_datetime(df["Shipping_Date"], errors="coerce")

# 6. Convert numerical columns
for col in ["Sales", "Quantity", "Profit"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 7. Handle missing numerical values
for col in ["Sales", "Quantity", "Profit"]:
    df[col] = df[col].fillna(df[col].median())

# 8. Handle missing categorical values
df["Shipping_Mode"] = df["Shipping_Mode"].fillna("Unknown")

# 9. Create delivery-time feature
df["Delivery_Days"] = (
    df["Shipping_Date"] - df["Order_Date"]
).dt.days

# 10. Detect and handle impossible negative delivery times
df.loc[df["Delivery_Days"] < 0, "Delivery_Days"] = pd.NA

# 11. Normalize numerical variables
scaler = MinMaxScaler()
scale_columns = ["Sales", "Quantity", "Profit"]

df[scale_columns] = scaler.fit_transform(df[scale_columns])

# 12. Save cleaned dataset
df.to_csv("cleaned_logistics_data.csv", index=False)

print("\nPreprocessing completed successfully.")
print("Cleaned dataset shape:", df.shape)
