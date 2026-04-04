# =============================================================================
# MODÈLES PRÉDICTIFS S&P 500 — RANDOM FOREST vs GRADIENT BOOSTING
# Conversion Python du script R original
# =============================================================================

# =============================================================================
# 0. IMPORTS & CONFIGURATION
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score,
    roc_curve, accuracy_score, f1_score
)
from sklearn.inspection import permutation_importance

import xgboost as xgb
from xgboost import XGBClassifier

from scipy.stats import randint, uniform

warnings.filterwarnings("ignore")
np.random.seed(42)


# =============================================================================
# 1. CHARGEMENT & NETTOYAGE DES DONNÉES
# =============================================================================

raw = pd.read_csv(
    "snp500_enrichi.csv",
    sep=";",
    decimal=",",
    na_values=["", "NA", "N/A", "NaN"]
)

print(f"Dimensions brutes : {raw.shape[0]} x {raw.shape[1]}")

# Parsing de la date
raw["Date"] = pd.to_datetime(raw["Date"], dayfirst=True)

# Tri chronologique
df = raw.sort_values("Date").reset_index(drop=True)


# =============================================================================
# 2. INGÉNIERIE DE LA VARIABLE CIBLE
# Cible : est-ce que le mois SUIVANT sera haussier (1) ou baissier (0) ?
# =============================================================================

df["Target_Rendement"] = df["Rendement_Mensuel_Pct"].shift(-1)
df["Target"] = (df["Target_Rendement"] > 0).astype(int)

print("\nDistribution de la cible :")
print(df["Target"].value_counts(dropna=False))
print(f"  0 = Baisse | 1 = Hausse")


# =============================================================================
# 3. SÉLECTION DES FEATURES & TRAITEMENT DES NA
# =============================================================================

feature_cols = [
    # Momentum / Prix
    "Momentum_12_1_Mois", "Momentum_6_Mois", "Momentum_3_Mois", "Momentum_1_Mois",
    # Taux
    "Fed_Taux_Directeur", "Fed_Taux_Variation",
    "Taux_10ans", "Taux_2ans", "Taux_3mois",
    "Spread_10ans_2ans", "Spread_10ans_3mois", "Spread_Calcule_10_2",
    "Taux_Hypothecaire_30ans",
    # Macroéconomie
    "CPI_Variation_Pct", "Taux_Chomage", "Chomage_Variation",
    "Production_Indus_Pct", "Ventes_Detail_Pct",
    # Crédit & Liquidité
    "Credit_Spread_IG", "Credit_Spread_HY", "TED_Spread",
    "M2_Variation_Pct",
    # Risque & Volatilité
    "VIX_Niveau", "VIX_Variation", "Volatilite_Realisee_Ann",
    "Variance_Risk_Premium",
    # Sentiment
    "Sentiment_Michigan", "Sentiment_Michigan_Var",
    # Matières premières & FX
    "Petrole_WTI_Pct", "EURUSD_Pct",
]

# Filtrer les lignes avec cible disponible
model_df = df.dropna(subset=["Target"]).copy()
model_df = model_df[["Date"] + feature_cols + ["Target", "Target_Rendement"]].copy()

# Imputation par la médiane (robuste, évite le look-ahead bias)
for col in feature_cols:
    median_val = model_df[col].median()
    model_df[col] = model_df[col].fillna(median_val)

print(f"\nLignes après nettoyage : {len(model_df)}")
print(f"NA restants : {model_df.isnull().sum().sum()}")


# =============================================================================
# 4. SPLIT TRAIN / TEST (validation temporelle — pas de data leakage)
# 80% train (chronologique), 20% test
# =============================================================================

split_idx = int(np.floor(0.80 * len(model_df)))
train_df  = model_df.iloc[:split_idx].copy()
test_df   = model_df.iloc[split_idx:].copy()

X_train = train_df[feature_cols].values
y_train = train_df["Target"].values
X_test  = test_df[feature_cols].values
y_test  = test_df["Target"].values

print(f"\nPériode d'entraînement : {train_df['Date'].min().date()} → {train_df['Date'].max().date()}")
print(f"Période de test        : {test_df['Date'].min().date()} → {test_df['Date'].max().date()}")
print(f"Train : {len(train_df)} obs | Test : {len(test_df)} obs")


# =============================================================================
# 5. VALIDATION CROISÉE TEMPORELLE (Time Series Split)
# Équivalent du trainControl(method="timeslice") de caret
# =============================================================================

from sklearn.model_selection import TimeSeriesSplit

ts_cv = TimeSeriesSplit(
    n_splits=10,         # ~10 fenêtres glissantes
    gap=0,
    max_train_size=None  # fenêtre expansive (expanding window)
)


# =============================================================================
# 6. RANDOM FOREST
# Équivalent de ranger avec grid search sur ROC
# =============================================================================

print("\n========== RANDOM FOREST ==========")

from sklearn.model_selection import RandomizedSearchCV

rf_param_grid = {
    "n_estimators": [200, 500],
    "max_features": [3, 7, 13],           # mtry
    "min_samples_leaf": [5, 15],          # min.node.size
    "criterion": ["gini", "entropy"],     # splitrule (gini / extratrees approx)
}

rf_base = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    oob_score=True
)

rf_search = RandomizedSearchCV(
    estimator=rf_base,
    param_distributions=rf_param_grid,
    n_iter=12,
    scoring="roc_auc",
    cv=ts_cv,
    random_state=42,
    n_jobs=-1,
    verbose=0,
    refit=True
)

rf_search.fit(X_train, y_train)
rf_model = rf_search.best_estimator_

print("Meilleurs hyperparamètres RF :")
print(rf_search.best_params_)
print(f"\nOOB Score  : {rf_model.oob_score_:.4f}")
print(f"OOB Error  : {1 - rf_model.oob_score_:.4f}")

# Prédictions RF
rf_pred_prob  = rf_model.predict_proba(X_test)[:, 1]
rf_pred_class = (rf_pred_prob >= 0.65).astype(int)   # seuil 0.65 comme dans R

rf_auc = roc_auc_score(y_test, rf_pred_prob)
rf_acc = accuracy_score(y_test, rf_pred_class)
rf_cm  = confusion_matrix(y_test, rf_pred_class)
rf_f1  = f1_score(y_test, rf_pred_class)

# Sensibilité et Spécificité manuelles
tn, fp, fn, tp = rf_cm.ravel()
rf_sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
rf_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print("\n--- Matrice de confusion RF ---")
print(f"           Prédit Baisse  Prédit Hausse")
print(f"Réel Baisse      {tn:>5}          {fp:>5}")
print(f"Réel Hausse      {fn:>5}          {tp:>5}")
print(f"\nAUC         : {rf_auc:.4f}")
print(f"Accuracy    : {rf_acc:.4f}")
print(f"Sensitivity : {rf_sensitivity:.4f}")
print(f"Specificity : {rf_specificity:.4f}")
print(f"F1 Score    : {rf_f1:.4f}")


# =============================================================================
# 7. GRADIENT BOOSTING OPTIMISÉ (XGBoost)
# =============================================================================

print("\n========== GRADIENT BOOSTING (XGBoost) ==========")

xgb_param_dist = {
    "n_estimators":      randint(100, 301),      # nrounds
    "max_depth":         randint(3, 7),           # max_depth
    "learning_rate":     uniform(0.01, 0.09),     # eta
    "gamma":             uniform(0, 0.5),         # gamma
    "colsample_bytree":  uniform(0.6, 0.4),      # colsample_bytree
    "min_child_weight":  randint(1, 11),         # min_child_weight
    "subsample":         uniform(0.7, 0.3),      # subsample
}

xgb_base = XGBClassifier(
    objective="binary:logistic",
    eval_metric="auc",
    use_label_encoder=False,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
)

xgb_search = RandomizedSearchCV(
    estimator=xgb_base,
    param_distributions=xgb_param_dist,
    n_iter=60,
    scoring="roc_auc",
    cv=ts_cv,
    random_state=42,
    n_jobs=-1,
    verbose=0,
    refit=True
)

xgb_search.fit(X_train, y_train)
xgb_model = xgb_search.best_estimator_

print("Meilleurs hyperparamètres XGBoost :")
print(xgb_search.best_params_)

# Prédictions XGBoost
xgb_pred_prob  = xgb_model.predict_proba(X_test)[:, 1]
xgb_pred_class = xgb_model.predict(X_test)

xgb_auc = roc_auc_score(y_test, xgb_pred_prob)
xgb_acc = accuracy_score(y_test, xgb_pred_class)
xgb_cm  = confusion_matrix(y_test, xgb_pred_class)
xgb_f1  = f1_score(y_test, xgb_pred_class)

tn2, fp2, fn2, tp2 = xgb_cm.ravel()
xgb_sensitivity = tp2 / (tp2 + fn2) if (tp2 + fn2) > 0 else 0
xgb_specificity = tn2 / (tn2 + fp2) if (tn2 + fp2) > 0 else 0

print("\n--- Matrice de confusion XGBoost ---")
print(f"           Prédit Baisse  Prédit Hausse")
print(f"Réel Baisse      {tn2:>5}          {fp2:>5}")
print(f"Réel Hausse      {fn2:>5}          {tp2:>5}")
print(f"\nAUC         : {xgb_auc:.4f}")
print(f"Accuracy    : {xgb_acc:.4f}")
print(f"Sensitivity : {xgb_sensitivity:.4f}")
print(f"Specificity : {xgb_specificity:.4f}")
print(f"F1 Score    : {xgb_f1:.4f}")


# =============================================================================
# 8. TABLEAU COMPARATIF DES PERFORMANCES
# =============================================================================

print("\n========== COMPARAISON DES MODÈLES ==========")

results = pd.DataFrame({
    "Modèle":      ["Random Forest", "Gradient Boosting (XGBoost)"],
    "AUC":         [round(rf_auc, 4),         round(xgb_auc, 4)],
    "Accuracy":    [round(rf_acc, 4),          round(xgb_acc, 4)],
    "Sensitivity": [round(rf_sensitivity, 4),  round(xgb_sensitivity, 4)],
    "Specificity": [round(rf_specificity, 4),  round(xgb_specificity, 4)],
    "F1":          [round(rf_f1, 4),           round(xgb_f1, 4)],
})
print(results.to_string(index=False))


# =============================================================================
# 9. IMPORTANCE DES VARIABLES
# =============================================================================

# RF — importance Gini
rf_imp_vals = rf_model.feature_importances_
rf_imp = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": rf_imp_vals
}).sort_values("Importance", ascending=False).head(15)

# XGBoost — importance Gain
xgb_imp_vals = xgb_model.feature_importances_
xgb_imp = pd.DataFrame({
    "Feature": feature_cols,
    "Importance": xgb_imp_vals
}).sort_values("Importance", ascending=False).head(15)

# Courbes ROC
rf_fpr,  rf_tpr,  _ = roc_curve(y_test, rf_pred_prob)
xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_pred_prob)

# ── Graphiques ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Importance des variables", fontsize=13, fontweight="bold")

# RF importance
axes[0].barh(rf_imp["Feature"][::-1], rf_imp["Importance"][::-1],
             color="#2E86AB", alpha=0.85)
axes[0].set_title("Random Forest (Gini)", fontsize=11)
axes[0].set_xlabel("Importance")
axes[0].tick_params(axis="y", labelsize=9)

# XGBoost importance
axes[1].barh(xgb_imp["Feature"][::-1], xgb_imp["Importance"][::-1],
             color="#E84855", alpha=0.85)
axes[1].set_title("XGBoost (Gain)", fontsize=11)
axes[1].set_xlabel("Importance")
axes[1].tick_params(axis="y", labelsize=9)

plt.tight_layout()
plt.savefig("snp500_importance.pdf", bbox_inches="tight")
plt.close()
print("\nGraphique d'importance sauvegardé dans 'snp500_importance.pdf'")

# ── Courbes ROC ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(rf_fpr,  rf_tpr,  color="#2E86AB", linewidth=1.5,
        label=f"RF (AUC={rf_auc:.3f})")
ax.plot(xgb_fpr, xgb_tpr, color="#E84855", linewidth=1.5,
        label=f"XGBoost (AUC={xgb_auc:.3f})")
ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=0.8)
ax.set_xlabel("Taux de Faux Positifs (1 - Spécificité)")
ax.set_ylabel("Taux de Vrais Positifs (Sensibilité)")
ax.set_title("Courbes ROC — Comparaison des modèles")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("snp500_resultats.pdf", bbox_inches="tight")
plt.close()
print("Courbes ROC sauvegardées dans 'snp500_resultats.pdf'")


# =============================================================================
# 10. BACKTESTING : SIMULATION D'UNE STRATÉGIE LONG/CASH
# =============================================================================

backtest = test_df[["Date", "Target_Rendement"]].copy()
backtest["Rendement_Reel"] = backtest["Target_Rendement"] / 100
backtest["Signal_RF"]      = rf_pred_class
backtest["Signal_XGB"]     = xgb_pred_class.astype(int)
backtest["Strat_RF"]       = backtest["Signal_RF"]  * backtest["Rendement_Reel"]
backtest["Strat_XGB"]      = backtest["Signal_XGB"] * backtest["Rendement_Reel"]
backtest["BuyHold"]        = backtest["Rendement_Reel"]

backtest["CumBuyHold"] = (1 + backtest["BuyHold"]).cumprod()
backtest["CumRF"]      = (1 + backtest["Strat_RF"]).cumprod()
backtest["CumXGB"]     = (1 + backtest["Strat_XGB"]).cumprod()


def sharpe(r):
    """Ratio de Sharpe annualisé (base mensuelle)."""
    r = r.dropna()
    return (r.mean() / r.std()) * np.sqrt(12) if r.std() != 0 else 0


print("\n========== BACKTEST SUR LA PÉRIODE DE TEST ==========")
bt_results = pd.DataFrame({
    "Stratégie": ["Buy & Hold", "RF Long/Cash", "XGBoost Long/Cash"],
    "Rendement_Total_%": [
        round((backtest["CumBuyHold"].iloc[-1] - 1) * 100, 2),
        round((backtest["CumRF"].iloc[-1]      - 1) * 100, 2),
        round((backtest["CumXGB"].iloc[-1]     - 1) * 100, 2),
    ],
    "Sharpe_Annualisé": [
        round(sharpe(backtest["BuyHold"]),   3),
        round(sharpe(backtest["Strat_RF"]),  3),
        round(sharpe(backtest["Strat_XGB"]), 3),
    ],
    "Mois_Investis": [
        len(backtest),
        int(backtest["Signal_RF"].sum()),
        int(backtest["Signal_XGB"].sum()),
    ],
})
print(bt_results.to_string(index=False))

# ── Graphique backtest ───────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(backtest["Date"], backtest["CumBuyHold"],
        color="grey",    linewidth=1.2, label="Buy & Hold")
ax.plot(backtest["Date"], backtest["CumRF"],
        color="#2E86AB", linewidth=1.2, label="RF Long/Cash")
ax.plot(backtest["Date"], backtest["CumXGB"],
        color="#E84855", linewidth=1.2, label="XGBoost Long/Cash")
ax.set_xlabel(None)
ax.set_ylabel("Valeur du portefeuille (base 1)")
ax.set_title("Performance cumulée — Backtest (période de test)\n"
             "Stratégie Long/Cash basée sur la prédiction du modèle")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("snp500_backtest.pdf", bbox_inches="tight")
plt.close()
print("Backtest sauvegardé dans 'snp500_backtest.pdf'")


# =============================================================================
# 11. PRÉDICTION DU MOIS SUIVANT (dernier signal)
# =============================================================================

print("\n========== SIGNAL POUR LE MOIS SUIVANT ==========")

last_row  = model_df.tail(1)
last_X    = last_row[feature_cols].values
last_date = last_row["Date"].values[0]

rf_prob_next  = rf_model.predict_proba(last_X)[0, 1]
xgb_prob_next = xgb_model.predict_proba(last_X)[0, 1]

print(f"Données au : {pd.Timestamp(last_date).strftime('%B %Y')}\n")
print(f"Random Forest  → P(Hausse) = {rf_prob_next * 100:.1f}%"
      f"  → Signal : {'LONG ↑' if rf_prob_next > 0.5 else 'CASH ↓'}")
print(f"XGBoost        → P(Hausse) = {xgb_prob_next * 100:.1f}%"
      f"  → Signal : {'LONG ↑' if xgb_prob_next > 0.5 else 'CASH ↓'}")
