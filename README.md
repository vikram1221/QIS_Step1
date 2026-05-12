# Quantitative Investment Strategies Step 1 — Data Pipeline & Value Strategy

## This step focuses on building a fully reproducible data pipeline for quantitative investment research. The output of this programs becomes the foundation for all future signal construction, portfolios, and backtesting.


### 1. Load & Clean Ticker Inputs
- Reading tickers from the CSV which contains all SP500 stocks, clean listing formats and standardize. 

### 2. Validate Tickers Using Real Price History
- Querying each ticker using the yahoo finance api. Remove of all tickers which present incomplete or strange to create a clean final list of tickers. 

### 3. Robust Price Downloader (OHLCV Engine)
- Downloading OHLCV data using yf api, and store all of the data in neat dataframes with clear naming conventions. 

### 4. Build Synchronized Multi-Asset Panels
- Aligning tickers on the same calender date to produce cross-sections. If data is missing for a stock at a particular date, it is not filled. The Open, High, Low, Close and Volume are saved as in different parquet files to compress size. 

### 5. Create Debug & Transparency Tools
- Building in utilities for viewing panel shape size, date-range inspector, missing-data diagnostics and ticker-level history preview. These built tools make it easier to obtain information about the dataset. 

### 6. Compute Value Scores & Signals
- Calculating value metrics (e.g., Earnings Yield) from fundamentals
- Ranking stocks cross-sectionally by percentile
- Assigning long/short signals based on top and bottom quantiles
- Saving output in tidy, long-format DataFrames

### 7. Backtest the Strategy
- Merging price returns with signals.
- Computing:  
      Daily portfolio returns
      Cumulative returns
      Equal-weight long/short performance
      Export clean results for review

### 8. Export Final Research Outputs
- The pipeline exports all results into an Excel workbook: Individual day signals (3000+ rows), backtest summary (cumulative returns), performance summary (Sharpe, Volume, CAGR, Max Drawdown)
