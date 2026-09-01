import pandas as pd

cols = ["date", "time", "open", "high", "low", "close", "volume"]
df = pd.read_csv("ibm_1min.csv", names=cols)
df["ts"] = pd.to_datetime(df["date"] + " " + df["time"], format="%m/%d/%Y %H:%M")

print(df[["ts", "close"]].head())
print(df["ts"].dtype)