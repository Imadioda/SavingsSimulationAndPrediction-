SavingsSimulationAndPrediction
Simulation of financial asset evolution and investment strategies, combined with a predictive model for monthly S&P 500 trends using: Multiple Regression, Logistic Regression, Random Forest and Gradient Boosting (XGBoost).

Project Overview
SavingsSimulationAndPrediction is an end-to-end quantitative finance project that covers:

Financial data collection and preprocessing via Python
Asset evolution and savings simulations in R
Interactive web applications for investment visualization
Predictive modeling for monthly S&P 500 classification (up/down)


Project Structure
SavingsSimulationAndPrediction/
│
├── DataCsv/                        # CSV files for financial assets & S&P 500 descriptive variables
│
├── PyData/                         # Python scripts for data acquisition
│   ├── yfinanceDATA.py             # Fetches monthly OHLCV data for multiple assets (yfinance)
│   └── RecupDataSNP500.py          # Fetches S&P 500 dataset with macro descriptive variables
│
├── HtmlWS/                         # Interactive web simulators
│   ├── SimulationImmob.html        # Real estate investment simulator
│   └── PortfolioSimulator.html     # Multi-asset portfolio simulator
│
├── Epargne/                        # Asset evolution & savings simulation
│   ├── Epargne.Rmd                 # R Markdown — savings visualization
│   └── Epargne.R                   # R Script — savings simulation
│
├── RegressionModels/               # Regression-based predictive models
│   ├── ModelMultipleReg.Rmd        # R Markdown — multiple & logistic regression
│   └── RegressionModels.R          # R Script — multiple regression + logistic (simple & optimised)
│
├── RFandXGB/                       # Tree-based predictive models
│   ├── ForXgbModel.Rmd             # R Markdown — Random Forest & XGBoost (R)
│   ├── RFandXGB.R                  # R Script — Random Forest & XGBoost (R)
│   └── snp500_modeles_predictifs.py # Python — Random Forest & XGBoost (sklearn / xgboost)
│
└── README.md

Web Applications
Real Estate Investment Simulator (SimulationImmob.html)

Inputs: property price, down payment, loan rate, inflation, loan duration
Outputs: nominal value, real value (inflation-adjusted), net equity, invested capital
Visualizations: year-by-year evolution charts

Multi-Asset Portfolio Simulator (PortfolioSimulator.html)

75+ assets across 7 categories: ETFs, individual stocks, bonds, commodities, energy, crypto, savings
Inputs: monthly contribution, initial capital, duration, inflation, tax wrapper (CTO, PEA, AV, PER, ISA, 401k...)
Multi-currency support with real-time conversion (EUR, USD, CHF, GBP, JPY, and more)
Outputs: gross value, net value (after tax), real value, per-asset breakdown


Predictive Modeling — S&P 500 Monthly Trend Classification
Problem Statement
The core question of this project is: can macroeconomic and market variables observed at month t predict whether the S&P 500 will rise or fall at month t+1?
All models use a chronological 80/20 train/test split and all features are lagged by 1 month to avoid look-ahead bias.

Model Results
Multiple Linear Regression (RegressionModels/)
Predicts the continuous monthly return, converted to a binary signal (positive = up, negative = down).
MetricScoreR² (test)≈ -2%RMSE6.51
Conclusion: The model fails to generalise — a negative R² means it performs worse than simply predicting the mean. Linear regression is not suited for a classification problem with noisy financial data.

Logistic Regression — Simple (RegressionModels/)
Direct binary classification (up/down) trained on lagged macro variables.
MetricScoreAccuracy69.6%AUC0.478Sensitivity95.0%Specificity6.25%
Conclusion: The model almost always predicts "up", which inflates accuracy given the market rises ~65% of months. The near-zero specificity and AUC below 0.5 confirm it has no real discriminatory power.

Logistic Regression — Optimised (RegressionModels/)
Adds class weighting and optimal threshold selection via ROC curve.
MetricScoreAccuracy57.1%AUC0.472Sensitivity60.0%Specificity50.0%
Conclusion: Rebalancing improved specificity significantly (6% → 50%), meaning the model now identifies down months more reliably. However the AUC remains below 0.5, indicating the signal is still very weak overall.

Random Forest (RFandXGB/)
Ensemble of 500 decision trees trained on 30+ macro, momentum, rate and sentiment features.
MetricScoreOOB Accuracy62.4%OOB Error37.6%AUC0.460Accuracy54.5%Sensitivity58.5%Specificity47.1%F1 Score0.628
Conclusion: The Random Forest shows the most balanced profile between sensitivity and specificity. However, the AUC below 0.5 indicates the model does not reliably discriminate between up and down months on the test set. The signal extracted from macro variables alone remains insufficient.

Gradient Boosting — XGBoost (RFandXGB/)
Optimised via randomised search (60 combinations) on a time-series cross-validation.
Best hyperparameters found:

n_estimators: 202 — max_depth: 6 — learning_rate: 0.079
colsample_bytree: 0.651 — gamma: 0.261 — subsample: 0.726 — min_child_weight: 5

MetricScoreAUC0.483Accuracy62.6%Sensitivity93.9%Specificity2.94%F1 Score0.767
Conclusion: XGBoost suffers from the same bias as the simple logistic regression — it predicts "up" almost systematically (specificity of 2.94%). The high accuracy is misleading and purely driven by the class imbalance (~65% up months). The near-zero specificity makes this model unusable as a real trading signal.

Overall Comparison
ModelAUCAccuracySensitivitySpecificityF1Multiple Regression—————Logistic Simple0.47869.6%95.0%6.25%—Logistic Optimised0.47257.1%60.0%50.0%—Random Forest0.46054.5%58.5%47.1%0.628XGBoost0.48362.6%93.9%2.94%0.767
Next Month Signal (January 2026 data)
ModelP(Up)SignalRandom Forest63.4%LONG ↑XGBoost67.2%LONG ↑

Key Takeaway
All models struggle to beat a random baseline (AUC ≈ 0.5). This is consistent with the Efficient Market Hypothesis — monthly S&P 500 direction is extremely difficult to predict from public macroeconomic data alone. The best result in terms of balance between sensitivity and specificity is the Optimised Logistic Regression, while the Random Forest offers the most stable profile across all metrics.

Data Description

DataCsv/ — CSV files for SNP500, CAC40, Gold, BTC and macro variables
PyData/yfinanceDATA.py — Monthly OHLCV data from first available date to today for each asset
PyData/RecupDataSNP500.py — Extended S&P 500 dataset including macro and sentiment variables via FRED API


Required Libraries
Python
yfinance
pandas
numpy
scikit-learn
xgboost
matplotlib
scipy
fredapi
R
rinstall.packages(c(
  "ggplot2", "factoextra", "FactoMineR", "corrplot",
  "caret", "dplyr", "tidyverse", "scales", "reshape2",
  "ROSE", "randomForest", "ranger", "xgboost",
  "pROC", "zoo", "doParallel"
))

Usage
1. Data Preparation
bashcd PyData/
python yfinanceDATA.py         # Fetch asset price data
python RecupDataSNP500.py      # Fetch S&P 500 macro dataset
2. Savings Simulation (R)
Open Epargne/Epargne.Rmd in RStudio and knit to visualize asset evolution.
3. Regression Models (R)
Open RegressionModels/ModelMultipleReg.Rmd in RStudio and knit to run multiple and logistic regression models.
4. Random Forest & XGBoost (R)
Open RFandXGB/ForXgbModel.Rmd in RStudio and knit to run tree-based models.
5. Random Forest & XGBoost (Python)
bashcd RFandXGB/
python snp500_modeles_predictifs.py
6. Interactive Visualization
Open HtmlWS/SimulationImmob.html or HtmlWS/PortfolioSimulator.html directly in any browser — no server required.

Notes

Ensure all CSV files in DataCsv/ are generated before running R scripts
All models use a chronological 80/20 train/test split to prevent data leakage
All macro variables are lagged by 1 month to avoid look-ahead bias
Web applications are fully standalone HTML files with no dependencies to install
