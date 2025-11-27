# QIS Step 1 — Data Pipeline & Value Strategy


## This step focuses on building a fully reproducibledata pipeline for quantitative investment research. The output of this workflow becomes the foundation for all future signal construction, factor models, and backtesting.

### 1. Load & Clean Ticker Inputs
- Read raw tickers from CSV or a user-provided list.
- Standardize formats (e.g., convert BRK.B → BRK-B for Yahoo Finance).
- Remove duplicates and sanitize symbols.

### 2. Validate Tickers Using Real Price History
- Query each ticker individually from Yahoo Finance.
- Reject any ticker that returns:
- Empty DataFrames,
- Missing or partial histories,
- Non-trading assets or delisted tickers.
- Produce a clean, validated list for all downstream steps.

### 3. Robust Price Downloader (OHLCV Engine)
- Download Open, High, Low, Close, Adj Close, Volume using retry logic.
- Automatically handle: API failures, Incomplete downloads, Tickers that stall or rate-limit.
- Store results in tidy DataFrames with clear naming conventions.

### 4. Build Synchronized Multi-Asset Panels
- Align all tickers on the same trading calendar.
- Forward-fill or drop based on validation rules.
- Produce wide-format parquet panels for each field:
          open_panel.parquet
          high_panel.parquet
          low_panel.parquet
          close_panel.parquet
          adj_close_panel.parquet
          volume_panel.parquet

### 5. Create Debug & Transparency Tools
Utilities include:
- Panel shape viewer
- Date-range inspector
- Missing-data diagnostics
- Ticker-level history preview
- Logging layer for all download attempts
These tools ensure the dataset remains trustworthy and easy to validate.

### 6. Compute Value Scores & Signals
- Calculate value metrics (e.g., Earnings Yield) from fundamentals.
- Rank stocks cross-sectionally by percentile.
- Assign long/short signals based on top and bottom quantiles.
- Save output in tidy, long-format DataFrames.

### 7. Backtest the Strategy
- Merge price returns with signals.
- Compute:  
      Daily portfolio returns
      Cumulative returns
      Equal-weight long/short performance
      Export clean results for review.

### 8. Export Final Research Outputs
- The pipeline exports all results into an Excel workbook:
- Sheets include:
      Signals (3,000+ rows)
      Backtest Results (daily & cumulative returns)
      Performance Summary (Sharpe, Vol, CAGR, MDD)
      All reports are formatted for non-technical users.
