
import pandas as pd
import numpy as np

# ==========================================
# 1. LOAD DATA
# ==========================================

df = pd.read_csv("sales_data_500_rows.csv")

print("\n--- DATASET PREVIEW ---")
print(df.head())

print("\nDataset shape:", df.shape)
print("\nColumn names:", df.columns.tolist())

print("\n--- DATA TYPES AND NON-NULL COUNTS ---")
df.info()

print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())


# ==========================================
# 2. BASIC SALES ANALYSIS
# ==========================================

print("\n--- LAST 10 ROWS ---")
print(df.tail(10))

print("\n--- SELECTED COLUMNS ---")
print(df[["Product", "Quantity", "Revenue"]])

total_revenue = df["Revenue"].sum()
print("\nTotal Revenue:", total_revenue)

top_products = (
    df.groupby("Product")["Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(5)
)

print("\n--- TOP 5 PRODUCTS BY REVENUE ---")
print(top_products)

online_orders = (df["Sales_Channel"] == "Online").sum()
print("\nNumber of Online Orders:", online_orders)


# ==========================================
# 3. INITIAL DATA VALIDATION
# ==========================================

df["Order_Date"] = pd.to_datetime(
    df["Order_Date"], errors="coerce"
)

print("\n--- DATA TYPES ---")
print(df.dtypes)

print("\nMissing values after date conversion:")
print(df.isnull().sum())

print(
    "\nDuplicate order IDs:",
    df["Order_ID"].duplicated().sum()
)

expected_revenue = df["Quantity"] * df["Unit_Price"]
expected_cost = df["Quantity"] * df["Unit_Cost"]
expected_profit = expected_revenue - expected_cost

print(
    "\nIncorrect revenue:",
    (~np.isclose(df["Revenue"], expected_revenue)).sum()
)

print(
    "Incorrect total cost:",
    (~np.isclose(df["Total_Cost"], expected_cost)).sum()
)

print(
    "Incorrect profit:",
    (~np.isclose(df["Profit"], expected_profit)).sum()
)


# ==========================================
# 4. CLEAN THE DATA
# ==========================================

clean_df = df.copy()

# Convert numeric columns safely
numeric_columns = [
    "Quantity", "Unit_Price", "Unit_Cost"
]

for col in numeric_columns:
    clean_df[col] = pd.to_numeric(
        clean_df[col], errors="coerce"
    )

# Remove duplicate order IDs
clean_df = clean_df.drop_duplicates(
    subset=["Order_ID"],
    keep="first"
)

# Required fields
required_columns = [
    "Order_ID", "Order_Date", "Product",
    "Category", "Region", "Sales_Channel",
    "Quantity", "Unit_Price", "Unit_Cost",
    "Customer_ID"
]

# Remove records missing required information
clean_df = clean_df.dropna(
    subset=required_columns
).copy()

# Keep only valid quantities, prices, and costs
valid_rows = (
    (clean_df["Quantity"] > 0)
    & (clean_df["Quantity"] % 1 == 0)
    & (clean_df["Unit_Price"] > 0)
    & (clean_df["Unit_Cost"] >= 0)
)

clean_df = clean_df.loc[valid_rows].copy()

# Convert quantity to integer after validation
clean_df["Quantity"] = clean_df["Quantity"].astype("int64")

# Recalculate financial metrics
clean_df["Revenue"] = (
    clean_df["Quantity"] * clean_df["Unit_Price"]
).round(2)

clean_df["Total_Cost"] = (
    clean_df["Quantity"] * clean_df["Unit_Cost"]
).round(2)

clean_df["Profit"] = (
    clean_df["Revenue"] - clean_df["Total_Cost"]
).round(2)

clean_df["Profit_Margin_Pct"] = (
    clean_df["Profit"] / clean_df["Revenue"] * 100
).round(2)


# ==========================================
# 5. VERIFY CLEANED DATA
# ==========================================

print("\n--- CLEANING SUMMARY ---")
print("Original rows:", len(df))
print("Cleaned rows:", len(clean_df))
print("Rows removed:", len(df) - len(clean_df))

print(
    "Remaining missing values:",
    clean_df.isnull().sum().sum()
)

print(
    "Remaining duplicate order IDs:",
    clean_df["Order_ID"].duplicated().sum()
)

print(
    "Incorrect revenue calculations:",
    (~np.isclose(
        clean_df["Revenue"],
        clean_df["Quantity"] * clean_df["Unit_Price"]
    )).sum()
)


# ==========================================
# 6. EXPORT CLEANED DATA
# ==========================================

clean_df.to_csv(
    "sales_data_cleaned.csv",
    index=False
)

print("\nCleaned dataset exported successfully!")
print("File: sales_data_cleaned.csv")
