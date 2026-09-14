import pandas as pd

def fetch_series(series_id):
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    df = pd.read_csv(url, parse_dates=["observation_date"], index_col="observation_date")
    numbers = pd.to_numeric(df[series_id], errors="coerce")
    return numbers.dropna()

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


# print(traders_q.tail())
# print(corpdebt_q.tail())

# real rate = nominal - year-over-year inflation
inflation_yoy = cpi_q.pct_change(4) * 100      # 4 quarters = 1 year
real_rate = fedfunds_q - inflation_yoy
# print(real_rate.tail())

def zscore(series):
    return (series - series.mean()) / series.std()

# each score: higher = more irresponsible
traders_score  = zscore(-traders_q)                    # flip: low NFCI = risky
treasury_score = zscore(deficit_q)                     # high deficit = risky
fed_score      = zscore(-real_rate)                    # flip: low real rate = risky
company_score  = zscore(corpdebt_q / gdp)              # debt/GDP, high = risky

# combine into one indicator
scores = pd.DataFrame({
    "traders": traders_score,
    "treasury": treasury_score,
    "fed": fed_score,
    "company": company_score,
}).dropna()

scores["indicator"] = scores.mean(axis=1)
# print(scores.tail(8))

## backtest
import yfinance as yf
sp = yf.download("^GSPC", start="1990-01-01")["Close"].resample("QE").last()
sp = sp.squeeze()
peak = sp.cummax()
drawdown = (sp - peak) / peak * 100

print(drawdown.min())
print(drawdown.idxmin())

bt = pd.DataFrame({
    "indicator": scores["indicator"],
    "drawdown": drawdown,
    }).dropna()

# was the indicator high before drawdowns?
print(bt.tail())
print(bt["indicator"].corr(bt["drawdown"]))
