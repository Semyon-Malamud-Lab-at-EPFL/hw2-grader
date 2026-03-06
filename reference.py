"""
Reference implementations for Assignment 2 grading.
- Ridge matches professor's ridge_regr exactly (p_ < t_ branching)
- R^2 = 1 - mse / (labels**2).mean()
- Performance: annualised, both ddof=0 and ddof=1 variants
"""
import numpy as np
import pandas as pd


# ═══════ Task 1 ═══════
def load_data(filepath):
    df = pd.read_pickle(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df["yyyymm"] = df["date"].dt.year * 100 + df["date"].dt.month
    return df

def filter_data(df, start_date="2000-01-31"):
    return df.loc[df["date"] >= pd.to_datetime(start_date)].reset_index(drop=True)

def get_feature_columns(df):
    non_feat = {"id", "date", "size_grp", "yyyymm", "r_1"}
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    return sorted([c for c in num if c not in non_feat])


# ═══════ Task 2 ═══════
def count_stocks_per_month(df):
    g = df.groupby("yyyymm")["id"].nunique().reset_index()
    g.columns = ["yyyymm", "cnt"]
    return g.sort_values("yyyymm").values

def mean_return_by_month(df):
    g = df.groupby("yyyymm")["r_1"].mean().reset_index()
    g.columns = ["yyyymm", "mr"]
    return g.sort_values("yyyymm").values

def return_std_by_month(df):
    g = df.groupby("yyyymm")["r_1"].std(ddof=0).reset_index()
    g.columns = ["yyyymm", "sd"]
    return g.sort_values("yyyymm").values

def feature_target_correlation(df, features, target="r_1"):
    tv = df[target].values
    return np.array([np.corrcoef(df[f].values, tv)[0, 1] for f in features])


# ═══════ Task 3 ═══════
def train_test_split(df, features, target, n_train):
    months = np.sort(df["yyyymm"].unique())
    tr_m, te_m = months[:n_train], months[n_train:]
    tr = df["yyyymm"].isin(tr_m)
    te = df["yyyymm"].isin(te_m)
    return (df.loc[tr, features].to_numpy(dtype=np.float64),
            df.loc[tr, target].to_numpy(dtype=np.float64),
            df.loc[te, features].to_numpy(dtype=np.float64),
            df.loc[te, target].to_numpy(dtype=np.float64),
            df.loc[tr, "yyyymm"].to_numpy(),
            df.loc[te, "yyyymm"].to_numpy())


# ═══════ Task 4 — Ridge (professor's ridge_regr verbatim) ═══════
def ridge_regression(S_train, y_train, S_test, shrinkage_list):
    t_ = S_train.shape[0]
    p_ = S_train.shape[1]
    if p_ < t_:
        eigenvalues, eigenvectors = np.linalg.eigh(S_train.T @ S_train / t_)
        means = S_train.T @ y_train.reshape(-1, 1) / t_
        multiplied = eigenvectors.T @ means
        intermed = np.concatenate(
            [(1 / (eigenvalues.reshape(-1, 1) + z)) * multiplied
             for z in shrinkage_list], axis=1)
        betas = eigenvectors @ intermed
    else:
        eigenvalues, eigenvectors = np.linalg.eigh(S_train @ S_train.T / t_)
        means = y_train.reshape(-1, 1) / t_
        multiplied = eigenvectors.T @ means
        intermed = np.concatenate(
            [(1 / (eigenvalues.reshape(-1, 1) + z)) * multiplied
             for z in shrinkage_list], axis=1)
        tmp = eigenvectors.T @ S_train
        betas = tmp.T @ intermed
    predictions = S_test @ betas
    return betas, predictions


# ═══════ Task 5 — Lasso ═══════
def lasso_regression(S_train, y_train, S_test, alpha):
    from sklearn.linear_model import Lasso
    m = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000, tol=1e-6)
    m.fit(S_train, y_train)
    beta = m.coef_.copy()
    y_pred = S_test @ beta
    return beta, y_pred


# ═══════ Task 6 — CV ═══════
def oos_r_squared(y_actual, y_predicted):
    mse = np.mean((y_actual - y_predicted) ** 2)
    return 1.0 - mse / np.mean(y_actual ** 2)

def _cv_folds(dates_train, n_folds=5):
    months = np.sort(np.unique(dates_train))
    N = len(months)
    for k in range(1, n_folds + 1):
        tr_end = int(np.floor(N * k / (n_folds + 1)))
        va_end = int(np.floor(N * (k + 1) / (n_folds + 1)))
        yield (np.isin(dates_train, months[:tr_end]),
               np.isin(dates_train, months[tr_end:va_end]))

def select_ridge_shrinkage(S_train, y_train, dates_train, shrinkage_list):
    K = len(shrinkage_list)
    fold_r2 = []
    for tr_mask, va_mask in _cv_folds(dates_train):
        _, preds = ridge_regression(S_train[tr_mask], y_train[tr_mask],
                                    S_train[va_mask], shrinkage_list)
        fold_r2.append([oos_r_squared(y_train[va_mask], preds[:, j])
                        for j in range(K)])
    cv = np.array(fold_r2).mean(axis=0)
    return shrinkage_list[int(np.argmax(cv))], cv

def select_lasso_alpha(S_train, y_train, dates_train, alpha_list):
    K = len(alpha_list)
    fold_r2 = []
    for tr_mask, va_mask in _cv_folds(dates_train):
        row = []
        for a in alpha_list:
            _, pred = lasso_regression(S_train[tr_mask], y_train[tr_mask],
                                       S_train[va_mask], a)
            row.append(oos_r_squared(y_train[va_mask], pred))
        fold_r2.append(row)
    cv = np.array(fold_r2).mean(axis=0)
    return alpha_list[int(np.argmax(cv))], cv


# ═══════ Task 7 — Portfolio (two ddof variants) ═══════
def compute_managed_returns(y_pred, y_actual, dates):
    managed = y_pred * y_actual
    tmp = pd.DataFrame({"yyyymm": dates, "m": managed})
    g = tmp.groupby("yyyymm")["m"].mean().reset_index().sort_values("yyyymm")
    return g.values

def performance_metrics_ddof0(managed_returns):
    r = managed_returns[:, 1]
    mm = float(np.mean(r));  ms = float(np.std(r, ddof=0))
    return {"mean": mm * 12, "vol": ms * np.sqrt(12),
            "sharpe": (mm * 12) / (ms * np.sqrt(12)) if ms > 0 else 0.0}

def performance_metrics_ddof1(managed_returns):
    r = managed_returns[:, 1]
    mm = float(np.mean(r));  ms = float(np.std(r, ddof=1))
    return {"mean": mm * 12, "vol": ms * np.sqrt(12),
            "sharpe": (mm * 12) / (ms * np.sqrt(12)) if ms > 0 else 0.0}
