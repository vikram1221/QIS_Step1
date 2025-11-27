import pandas as pd
import numpy as np
import yfinance as yf
import time

# CLEAN + VALIDATE TICKERS

def clean_tickers_for_yahoo(tickers):
    cleaned = []
    for t in tickers:
        t = t.replace(".", "-")  # BRK.B → BRK-B
        cleaned.append(t)
    return cleaned


def filter_valid_tickers(tickers, start="2020-01-01", end="2024-01-01"):
    """
    Validate each ticker by checking if Yahoo returns price data.
    """
    valid = []
    print("\nChecking which tickers have real price history...\n")
    for t in tickers:
        try:
            df = yf.download(t, start=start, end=end, progress=False)
            if not df.empty:
                valid.append(t)
        except:
            pass
    print("\nValid tickers:", valid)
    return valid


# ROBUST PRICE DOWNLOAD (FIXED VERSION)

def get_prices(tickers, start, end):

    print("\n=== DOWNLOADING PRICE DATA ===\n")

    df = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker"
    )

    # If bulk returned nothing:
    if df.empty:
        print("Bulk download failed. Trying individually...\n")
        working = []
        for t in tickers:
            sub = yf.download(t, start=start, end=end, progress=False)
            if not sub.empty:
                working.append(t)

        print("Working tickers:", working)
        if not working:
            raise ValueError("No tickers have valid data.")

        df = yf.download(
            tickers=working,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="ticker"
        )

    # ------------------------------------------------------------------
    # PRINT COLUMN STRUCTURE SO WE SEE WHAT YAHOO RETURNED
    # ------------------------------------------------------------------
    print("\nYahoo returned columns:")
    print(df.columns)

    # ------------------------------------------------------------------
    # CASE A: MULTIINDEX → MANY TICKERS
    # ------------------------------------------------------------------
    if isinstance(df.columns, pd.MultiIndex):

        # Preferred: Adj Close
        if "Adj Close" in df.columns.get_level_values(1):
            adj = df.xs("Adj Close", axis=1, level=1)

        # Fallback 1: Close
        elif "Close" in df.columns.get_level_values(1):
            adj = df.xs("Close", axis=1, level=1)

        # Fallback 2: Only one column exists (very unusual)
        else:
            first_level = df.columns.get_level_values(1)
            unique_cols = list(set(first_level))
            raise ValueError(f"MultiIndex returned but no usable price column. Found: {unique_cols}")

    # ------------------------------------------------------------------
    # CASE B: SINGLE TICKER → NOT MULTIINDEX
    # ------------------------------------------------------------------
    else:
        cols = df.columns

        # Preferred: Adj Close
        if "Adj Close" in cols:
            adj = df[["Adj Close"]]
            adj.columns = [tickers[0]]

        # Fallback 1: Close
        elif "Close" in cols:
            adj = df[["Close"]]
            adj.columns = [tickers[0]]

        else:
            raise ValueError(f"Single ticker returned but no Adj Close or Close. Columns: {cols}")

    # ------------------------------------------------------------------
    # Convert to long format
    # ------------------------------------------------------------------
    long_df = adj.stack().reset_index()
    long_df.columns = ["date", "ticker", "adj_close"]

    print("\nDownloaded", len(long_df), "rows of price data.")
    print(long_df.head())

    return long_df

# FUNDAMENTALS + META DATA

def get_meta_data(tickers):
    rows = []
    for symbol in tickers:
        try:
            t = yf.Ticker(symbol)
            info = t.info

            sector = info.get("sector")
            market_cap = info.get("marketCap")
            price = info.get("currentPrice")

            if price is None:
                hist = t.history(period="1d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])

            rows.append({
                "ticker": symbol,
                "sector": sector,
                "market_cap": market_cap,
                "price": price
            })

            time.sleep(0.15)

        except Exception as e:
            print("Meta fail:", symbol, e)

    return rows


def get_fundamentals(tickers, as_of_date):
    rows = []
    for symbol in tickers:
        try:
            t = yf.Ticker(symbol)
            info = t.info
            pe = info.get("trailingPE")
            if pe is None:
                continue

            rows.append({
                "date": as_of_date,
                "ticker": symbol,
                "pe": pe
            })

            time.sleep(0.15)

        except Exception as e:
            print(f"Fundamental fail for {symbol}: {e}")

    return pd.DataFrame(rows)

def fill_monthly_fundamentals(fund_df, start, end):
    months = pd.date_range(start, end, freq="M")
    out = []
    for ticker in fund_df["ticker"].unique():
        pe = float(fund_df[fund_df["ticker"] == ticker]["pe"].iloc[0])
        for m in months:
            out.append({"date": m, "ticker": ticker, "pe": pe})
    return pd.DataFrame(out)

#                      SECTOR UNIVERSE

class SectorUniverse:
    def __init__(self, sector_name, stock_meta_data, min_market_cap=None, min_price=None):
        self.__sector_name = sector_name
        self.__stock_meta_data = stock_meta_data
        self.__min_market_cap = min_market_cap
        self.__min_price = min_price

        self.__tickers = self.__filter()

    def __filter(self):
        out = []
        for s in self.__stock_meta_data:

            if s.get("sector") != self.__sector_name:
                continue

            if self.__min_market_cap is not None:
                if s.get("market_cap") is None or s.get("market_cap") < self.__min_market_cap:
                    continue

            if self.__min_price is not None:
                if s.get("price") is None or s.get("price") < self.__min_price:
                    continue

            out.append(s["ticker"])

        return out

    def get_tickers(self):
        return list(self.__tickers)

    def __repr__(self):
        return f"SectorUniverse(sector={self.__sector_name}, n={len(self.__tickers)})"


# DATASTORE

class DataStore:
    def __init__(self, prices_df):
        self.__prices = prices_df.copy()
        self.__prices["date"] = pd.to_datetime(self.__prices["date"])
        self.__prices = self.__prices.sort_values(["ticker", "date"])

    def get_price_panel(self, tickers=None, start_date=None, end_date=None):
        df = self.__prices
        if tickers is not None:
            df = df[df["ticker"].isin(tickers)]
        if start_date is not None:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date is not None:
            df = df[df["date"] <= pd.to_datetime(end_date)]
        return df.copy()

    def __repr__(self):
        if self.__prices.empty:
            return "DataStore(n_tickers=0)"
        uni = self.__prices["ticker"].unique()
        start = self.__prices["date"].min().date()
        end = self.__prices["date"].max().date()
        return f"DataStore(n_tickers={len(uni)}, start={start}, end={end})"


# FACTOR MODEL + VALUE FACTOR

class FactorModel:
    def __init__(self, name, fundamentals_df, universe):
        self._name = name
        self._fundamentals = fundamentals_df.copy()
        self._universe = universe

        self._fundamentals["date"] = pd.to_datetime(self._fundamentals["date"])

    def __repr__(self):
        return f"FactorModel(name={self._name})"


class ValueFactorModel(FactorModel):
    def __init__(self, fundamentals_df, universe, metric_column):
        super().__init__("Value", fundamentals_df, universe)
        self.metric = metric_column

    def compute_scores(self, start_date=None, end_date=None):
        df = self._fundamentals
        allowed = self._universe.get_tickers()

        df = df[df["ticker"].isin(allowed)]
        if start_date:
            df = df[df["date"] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df["date"] <= pd.to_datetime(end_date)]

        df = df[["date", "ticker", self.metric]].rename(columns={self.metric: "value_score"})

        return df.sort_values(["date", "value_score"])


# RANKER + STRATEGY

class Ranker:
    def __init__(self, factor_model, top_quartile=0.25):
        self.model = factor_model
        self.top_q = top_quartile

    def compute_ranks(self, start_date=None, end_date=None):
        df = self.model.compute_scores(start_date, end_date)
        df["rank"] = df.groupby("date")["value_score"].rank(ascending=True)
        max_rank = df.groupby("date")["rank"].transform("max")
        df["pct_rank"] = df["rank"] / max_rank
        return df

    def generate_signals(self, start_date=None, end_date=None):
        df = self.compute_ranks(start_date, end_date)
        df["signal"] = (df["pct_rank"] <= self.top_q).astype(int)
        return df


class LongOnlyValueStrategy:
    def __init__(self, name, ranker, data_store):
        self.name = name
        self.ranker = ranker
        self.ds = data_store

    def get_signals(self, start_date=None, end_date=None):
        return self.ranker.generate_signals(start_date, end_date)

    def get_rebalance_dates(self, start_date=None, end_date=None):
        sig = self.get_signals(start_date, end_date)
        rebal = sig["date"].dt.to_period("M").dt.to_timestamp("M")
        rebal = sorted(set(rebal))
        return rebal
    
    def get_weights_on_date(self, date):
        sig = self.get_signals(date, date)
        sig = sig.drop_duplicates(subset=["date", "ticker"])
        sig = sig[(sig["date"] == pd.to_datetime(date)) & (sig["signal"] == 1)]

        if sig.empty:
            return pd.DataFrame(columns=["ticker", "weight"])

        w = 1 / len(sig)
        wdf = sig[["ticker"]].copy()
        wdf["weight"] = w
        return wdf


# BACKTEST

class Backtest:
    def __init__(self, strategy, data_store, start, end):
        self.strategy = strategy
        self.ds = data_store
        self.start = start
        self.end = end

    def run(self):

        rebal = self.strategy.get_rebalance_dates(self.start, self.end)
        rebal = sorted(set(rebal))     # ← KEEP UNIQUE MONTH-ENDS


        if len(rebal) < 2:
            raise ValueError("Not enough rebalance periods.")

        prices = self.ds.get_price_panel(start_date=self.start, end_date=self.end)
        results = []

        for i in range(len(rebal)-1):

            start = pd.to_datetime(rebal[i])
            end = pd.to_datetime(rebal[i+1])

            weights = self.strategy.get_weights_on_date(start)

            if weights.empty:
                results.append({"date": end, "port_ret": 0})
                continue

            tickers = list(weights["ticker"])
            weights = weights.set_index("ticker")["weight"]

            # prices for tickers only
            panel = prices[prices["ticker"].isin(tickers)]

            # LAST AVAILABLE PRICE BEFORE/ON START DATE
            p0 = (
                panel[panel["date"] <= start]
                .sort_values(["ticker", "date"])
                .groupby("ticker")
                .tail(1)
            )

            # LAST AVAILABLE PRICE BEFORE/ON END DATE
            p1 = (
                panel[panel["date"] <= end]
                .sort_values(["ticker", "date"])
                .groupby("ticker")
                .tail(1)
            )

            if p0.empty or p1.empty:
                results.append({"date": end, "port_ret": 0})
                continue

            if p0.empty or p1.empty:
                print("EMPTY PRICES FOR PERIOD:", start, "->", end)
                results.append({"date": end, "port_ret": 0})
                continue

            merge = p0.merge(
                p1,
                on="ticker",
                suffixes=("_start", "_end")
            )

            merge = merge.merge(weights, on="ticker")

            merge["ret"] = merge["adj_close_end"] / merge["adj_close_start"] -1
            port_ret = float((merge["ret"] * merge["weight"]).sum())
            results.append({"date": end, "port_ret": port_ret})

        result_df = pd.DataFrame(results).sort_values("date")
        result_df["cum_ret"] = (1 + result_df["port_ret"]).cumprod() - 1
        return result_df


# PERFORMANCE REPORT

class PerformanceReport:
    def __init__(self, df, periods=12):
        self.df = df
        self.periods = periods

    def compute(self):
        df = self.df
        total = df["cum_ret"].iloc[-1]

        n = len(df)
        years = n / self.periods
        cagr = (1 + total)**(1/years) - 1

        vol = df["port_ret"].std() * np.sqrt(self.periods)
        sharpe = cagr / vol if vol > 0 else np.nan

        equity = (1 + df["port_ret"]).cumprod()
        dd = (equity / equity.cummax() - 1).min()

        return total, cagr, vol, sharpe, dd

    def print(self):
        t, c, v, s, d = self.compute()
        print(f"\nTotal Return: {t*100:.2f}%")
        print(f"CAGR:         {c*100:.2f}%")
        print(f"Volatility:   {v*100:.2f}%")
        print(f"Sharpe:       {s:.2f}")
        print(f"Max Drawdown: {d*100:.2f}%")


# MAIN PIPELINE

print("\n=== STEP 1: READ TICKERS ===")
tickers_df = pd.read_csv(
    r"C:\Users\vikra\OneDrive\Desktop\Python Trading Programs\QIS_Step_1\sp_500_stocks.csv"
)
tickers_df = tickers_df.rename(columns={"Ticker": "ticker"})

raw = tickers_df["ticker"].dropna().unique().tolist()
clean = clean_tickers_for_yahoo(raw)

print("Raw tickers:", len(raw))
print("Clean tickers:", len(clean))

print("\n=== STEP 2: FILTER VALID TICKERS ===")
valid = filter_valid_tickers(clean)

print("\n=== STEP 3: META DATA ===")
meta = get_meta_data(valid)
print("Example meta row:", meta[0])

print("\n=== STEP 4: BUILD SECTOR UNIVERSE ===")
universe = SectorUniverse(
    sector_name="Financial Services",
    stock_meta_data=meta,
    min_market_cap=1_000_000_000,
    min_price=5.0
)
univ_tickers = universe.get_tickers()
print(universe)
print("Universe tickers:", univ_tickers)

print("\n=== STEP 5: DOWNLOAD PRICES ===")
prices = get_prices(
    tickers=univ_tickers,
    start="2020-01-01",
    end="2024-01-01"
)
store = DataStore(prices)
print(store)

print("\n=== STEP 6: FUNDAMENTALS ===")

raw_fund = get_fundamentals(univ_tickers, "2024-01-01")
fund = fill_monthly_fundamentals(raw_fund, "2020-01-01", "2024-01-01")
print(fund.head())

print("\n=== STEP 7: FACTOR MODEL + RANKER ===")
value_model = ValueFactorModel(fund, universe, "pe")
ranker = Ranker(value_model, top_quartile=0.25)

strategy = LongOnlyValueStrategy("Value Strategy", ranker, store)
print(strategy.name)

print("\n=== STEP 8: SIGNALS TABLE ===")
signals = strategy.get_signals()
print(signals)

print("\n=== STEP 9: RUN BACKTEST ===")
bt = Backtest(strategy, store, "2020-01-01", "2024-01-01")
results = bt.run()
print(results)

print("\n=== STEP 10: PERFORMANCE REPORT ===")
report = PerformanceReport(results)
report.print()

print("\n=== DEBUG: SIGNALS SUMMARY ===")
print(signals.head(20))
print(signals["signal"].value_counts())


print("\n=== STEP 11: EXPORT TO EXCEL ===")

output_path = r"C:\Users\vikra\OneDrive\Desktop\Python Trading Programs\QIS_Step_1\value_strategy_output.xlsx"

with pd.ExcelWriter(output_path, engine="xlsxwriter") as writer:

    # ---- SHEET 1: Backtest Results ----
    results.to_excel(writer, sheet_name="Backtest Results", index=False)

    # ---- SHEET 2: Performance Summary ----
    total, cagr, vol, sharpe, dd = report.compute()
    perf_df = pd.DataFrame({
        "Metric": ["Total Return", "CAGR", "Volatility", "Sharpe Ratio", "Max Drawdown"],
        "Value": [total, cagr, vol, sharpe, dd]
    })
    perf_df.to_excel(writer, sheet_name="Performance Summary", index=False)

    # ---- SHEET 3: Signals Table ----
    signals.to_excel(writer, sheet_name="Signals", index=False)
