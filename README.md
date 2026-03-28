# SavingsSimulationAndPrediction

Simulation of financial asset evolution and investment strategies, combined with a predictive model for monthly S&P 500 trends using Random Forest.

---

## Project Overview

**SavingsSimulationAndPrediction** is a end-to-end quantitative finance project that covers:

- Financial data collection and preprocessing via Python
- Asset evolution and savings simulations in R
- Interactive web applications for investment visualization
- Predictive modeling (Random Forest) for monthly S&P 500 classification (up/down)

---

## Project Structure
```
SavingsSimulationAndPrediction/
│
├── DataCs/                    # CSV files for financial assets & S&P 500 descriptive variables
│
├── PyData/                    # Python scripts for data acquisition
│   ├── yfinanceDATA.py        # Fetches monthly OHLCV data for multiple assets (yfinance)
│   └── RecupDataSNP500.py     # Fetches S&P 500 dataset with macro descriptive variables
│
├── HtmlWS/                    # Interactive web simulators
│   ├── SimulationImmob.html   # Real estate investment simulator
│   └── PortfolioSimulator.html # Multi-asset portfolio simulator
│
├── Epargne.Rmd                # R Markdown — asset evolution & savings visualization
├── ModelMultiple.Rmd          # R Markdown — S&P 500 predictive modeling
└── README.md
```

---

## Web Applications

### Real Estate Investment Simulator (`SimulationImmob.html`)
- Inputs: property price, down payment, loan rate, inflation, loan duration
- Outputs: nominal value, real value (inflation-adjusted), net equity, invested capital
- Visualizations: year-by-year evolution charts

### Multi-Asset Portfolio Simulator (`PortfolioSimulator.html`)
- 75+ assets across 7 categories: ETFs, individual stocks, bonds, commodities, energy, crypto, savings
- Inputs: monthly contribution, initial capital, duration, inflation, tax wrapper (CTO, PEA, AV, PER, ISA, 401k...)
- Multi-currency support with real-time conversion (EUR, USD, CHF, GBP, JPY, and more)
- Outputs: gross value, net value (after tax), real value, per-asset breakdown

---

## R Scripts

### `Epargne.Rmd`
- Simulates and compares the evolution of multiple asset classes over time
- Visualizes savings strategies (cash, real estate, financial assets)

### `ModelMultiple.Rmd`
Three predictive approaches tested for monthly S&P 500 trend classification:

| Model | Result |
|---|---|
| Multiple Regression | Underperformed (R² ≈ -2%) |
| Logistic Regression | Underperformed (AUC ≈ 0.49) |
| **Random Forest** | **Best performance** |

The Random Forest model predicts whether the S&P 500 will rise or fall each month using lagged macro variables (VIX, momentum, spreads, Fed rate, CPI, unemployment, etc.).

---

## Data Description

- **`DataCs/`** — CSV files for SNP500, CAC40, Gold, BTC and macro variables
- **`PyData/yfinanceDATA.py`** — Monthly OHLCV data from first available date to today for each asset
- **`PyData/RecupDataSNP500.py`** — Extended S&P 500 dataset including macro and sentiment variables via FRED API

---

## Required Libraries

### Python
```
yfinance
pandas
numpy
fredapi
```

### R
```r
install.packages(c(
  "ggplot2", "factoextra", "FactoMineR", "corrplot",
  "caret", "dplyr", "tidyverse", "scales", "reshape2",
  "ROSE", "randomForest", "pROC", "zoo"
))
```

---

## Usage

### 1. Data Preparation
```bash
cd PyData/
python yfinanceDATA.py       # Fetch asset price data
python RecupDataSNP500.py    # Fetch S&P 500 macro dataset
```

### 2. Savings Simulation (R)
Open `Epargne.Rmd` in RStudio and knit to visualize asset evolution.

### 3. Predictive Modeling (R)
Open `ModelMultiple.Rmd` in RStudio and knit to reproduce the Random Forest model.

### 4. Interactive Visualization
Open `HtmlWS/SimulationImmob.html` or `HtmlWS/PortfolioSimulator.html` directly in any browser — no server required.

---

## Notes

- Ensure all CSV files in `DataCs/` are generated before running R scripts
- The Random Forest model uses a chronological 80/20 train/test split (no data leakage)
- All macro variables are lagged by 1 month to avoid look-ahead bias
- Web applications are fully standalone HTML files with no dependencies to install
