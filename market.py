import pandas as pd
from speech import sentences, results   # your Phase 1 output

# 1. load price data
cols = ["date", "time", "open", "high", "low", "close", "volume"]
df = pd.read_csv("ibm_1min.csv", names=cols)
df["ts"] = pd.to_datetime(df["date"] + " " + df["time"], format="%m/%d/%Y %H:%M")

# 2. define the function
def market_move(df, spoken_time, window_minutes=15):
    before = df[df["ts"] >= spoken_time].iloc[0]
    after_time = spoken_time + pd.Timedelta(minutes=window_minutes)
    after = df[df["ts"] >= after_time].iloc[0]
    return (after["close"] - before["close"]) / before["close"]

# 3. the join
start = pd.Timestamp("2026-06-03 10:00")

rows = []
for i, (sent, score) in enumerate(zip(sentences, results)):
    spoken = start + pd.Timedelta(seconds=i * 5)   # each sentence 5s later
    try:
        move = market_move(df, spoken, window_minutes=15)
    except IndexError:
        continue        # ran past end of data — skip
    rows.append({
        "sentence": sent,
        "signed": score["signed"],
        "move": move
    })

paired = pd.DataFrame(rows)
print(paired.shape)
print(paired.head())
print(paired["signed"].corr(paired["move"]))