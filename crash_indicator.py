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
traders_q  = traders.resample("QE").last()
deficit_q  = deficit.resample("QE").last()
fedfunds_q = fedfunds.resample("QE").last()
cpi_q      = cpi.resample("QE").last()
corpdebt_q = corpdebt.resample("QE").last()

print(traders_q.tail())
print(corpdebt_q.tail())

# real rate = nominal - year-over-year inflation
inflation_yoy = cpi_q.pct_change(4) * 100      # 4 quarters = 1 year
real_rate = fedfunds_q - inflation_yoy
print(real_rate.tail())

def zscore(series):
    return (series - series.mean()) / series.std()