import pandas as pd

file_path = "data/household_power_consumption.txt"

df = pd.read_csv(
    file_path,
    sep=";",
    nrows=1000,
    na_values=["?"],
    low_memory=False
)

print("\nFIRST 10 ROWS")
print(df.head(10).to_string(index=False))

print("\nDATASET SHAPE")
print(df.shape)

print("\nCOLUMN NAMES")
print(df.columns.tolist())

print("\nDATA TYPES")
print(df.dtypes)

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nDUPLICATE ROWS")
print(df.duplicated().sum())

print("\nSTATISTICAL SUMMARY")
print(df.describe().round(2).to_string())