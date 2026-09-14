import pandas as pd

def fetch_series(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url, parse_dates=["observation_date"], index_col="observation_date")
    numbers = pd.to_numeric(df[series_id], errors="coerce")
    return numbers.dropna()

def zscore(series):
    return (series - series.mean()) / series.std()

# pull all four
traders   = fetch_series("NFCI")
deficit   = fetch_series("FYFSGDA188S")
fedfunds  = fetch_series("FEDFUNDS")
cpi       = fetch_series("CPIAUCSL")
corpdebt  = fetch_series("NCBDBIQ027S")

# resample each to quarterly, taking the last value in each quarter
receipts = fetch_series("FGRECPT").resample("QE").last()
outlays  = fetch_series("FGEXPND").resample("QE").last()
gdp = fetch_series("GDP").resample("QE").last()

deficit_q  = (outlays - receipts) /gdp * 100
traders_q  = traders.resample("QE").last()
fedfunds_q = fedfunds.resample("QE").last()
cpi_q      = cpi.resample("QE").last()
corpdebt_q = corpdebt.resample("QE").last()

mktcap = fetch_series("NCBEILQ027S").resample("QE").last()   # corp equities held
valuation = mktcap / gdp
valuation_score = zscore(valuation) 

# print(traders_q.tail())
# print(corpdebt_q.tail())

# real rate = nominal - year-over-year inflation
inflation_yoy = cpi_q.pct_change(4) * 100      # 4 quarters = 1 year
real_rate = fedfunds_q - inflation_yoy
# print(real_rate.tail())


# each score: higher = more irresponsible
traders_score  = zscore(-traders_q)                    # flip: low NFCI = risky
treasury_score = zscore(deficit_q)                     # high deficit = risky
fed_score      = zscore(-real_rate)                    # flip: low real rate = risky
company_score  = zscore(corpdebt_q / gdp)              # debt/GDP, high = risky

# Republican presidents by term
rep_periods = [
    ("1969-01-20","1977-01-20"),  # Nixon/Ford
    ("1981-01-20","1993-01-20"),  # Reagan/Bush
    ("2001-01-20","2009-01-20"),  # Bush
    ("2017-01-20","2021-01-20"),  # Trump
    ("2025-01-20","2029-01-20"),  # Trump again
]

# combine into one indicator
scores = pd.DataFrame({
    "traders": traders_score,
    "treasury": treasury_score,
    "fed": fed_score,
    "company": company_score,
    "valuation": valuation_score,
}).dropna()

prez = pd.Series(0, index=scores.index)
for start, end in rep_periods:
    prez[(prez.index >= start) & (prez.index < end)] = 1

scores["prez"] = prez


actor_cols = ["traders", "treasury", "fed", "company", "valuation", "prez"]
scores["indicator"] = scores[actor_cols].mean(axis=1)
# print(scores.tail(8))

## backtest
import yfinance as yf
# crash onset quarters (peak, just before the fall)
crashes = ["2000-03-31", "2007-09-30", "2020-03-31", "2022-03-31"]

for c in crashes:
    val = scores["indicator"].asof(c)     # indicator value at/before that date
    print(c, "indicator:", round(val, 2))

print("\nhistorical median:", round(scores["indicator"].median(), 2))
print("historical 90th pct:", round(scores["indicator"].quantile(0.9), 2))

high = scores["indicator"] > 0.89        # above 90th percentile

print(scores.index[high].tolist())
print(scores["prez"].value_counts())   # should show counts of 0s and 1s
print(scores[["valuation","prez"]].tail())
print(scores[["traders","treasury","fed","company","valuation","prez"]].tail(3))
