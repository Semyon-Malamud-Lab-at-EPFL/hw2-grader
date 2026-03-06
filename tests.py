"""
Test functions for Assignment 2 grading.
Each returns a list of (points_earned, max_points, message) tuples.
"""
import ast, os, numpy as np
from config import (RTOL_EXACT, ATOL_EXACT, RTOL_STRICT, ATOL_STRICT,
                    RTOL_MEDIUM, ATOL_MEDIUM, RTOL_LOOSE, ATOL_LOOSE, POINTS)

def _call(fn, *a, **kw):
    try:    return fn(*a, **kw), None
    except Exception as e: return None, f"{type(e).__name__}: {e}"

def _close(a, b, rtol, atol):
    try:    return np.allclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
    except: return False

def _frac(a, b, rtol, atol):
    try:
        a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
        if len(a) != len(b): return 0.0
        return float(np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True).mean())
    except: return 0.0

def _zeros(keys, msg):
    return [(0, POINTS[k], msg) for k in keys]

def check_no_loops(filepath):
    if not os.path.exists(filepath): return True, None
    with open(filepath) as f:
        try:    tree = ast.parse(f.read())
        except: return True, None
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            return False, getattr(node, "lineno", "?")
    return True, None


# ═══════ Task 1 ═══════
def test_task1(stu, ref, filepath, start_date):
    res = []
    s_df, err = _call(stu.load_data, filepath)
    r_df = ref.load_data(filepath)
    if err:
        return _zeros(["1a_load","1a_dt","1a_yyyymm","1b_filter","1b_reset",
                        "1c_numeric","1c_target","1c_sorted"], err), r_df, None

    res.append((POINTS["1a_load"] if s_df.shape == r_df.shape else 0,
                POINTS["1a_load"], f"shape {s_df.shape}"))
    is_dt = np.issubdtype(s_df["date"].dtype, np.datetime64) if "date" in s_df else False
    res.append((POINTS["1a_dt"] if is_dt else 0, POINTS["1a_dt"], f"dt={is_dt}"))
    ym_ok = "yyyymm" in s_df.columns and np.array_equal(s_df["yyyymm"].values, r_df["yyyymm"].values)
    res.append((POINTS["1a_yyyymm"] if ym_ok else 0, POINTS["1a_yyyymm"], f"yyyymm={ym_ok}"))

    s_f, err = _call(stu.filter_data, s_df, start_date)
    r_f = ref.filter_data(r_df, start_date)
    if err or s_f is None:
        res += _zeros(["1b_filter","1b_reset"], str(err))
    else:
        res.append((POINTS["1b_filter"] if len(s_f)==len(r_f) else 0,
                     POINTS["1b_filter"], f"len {len(s_f)} vs {len(r_f)}"))
        res.append((POINTS["1b_reset"] if (len(s_f)>0 and s_f.index[0]==0) else 0,
                     POINTS["1b_reset"], "reset"))

    df4f = s_f if s_f is not None else r_f
    s_fc, err = _call(stu.get_feature_columns, df4f)
    r_fc = ref.get_feature_columns(r_f)
    if err or s_fc is None:
        res += _zeros(["1c_numeric","1c_target","1c_sorted"], str(err))
    else:
        overlap = len(set(s_fc)&set(r_fc)) / max(len(r_fc),1)
        res.append((POINTS["1c_numeric"]*overlap, POINTS["1c_numeric"], f"overlap {overlap:.0%}"))
        non_feat_check = not any(c in s_fc for c in ["r_1", "id", "yyyymm"])
        res.append((POINTS["1c_target"] if non_feat_check else 0,
                     POINTS["1c_target"], "non-features excluded"))
        res.append((POINTS["1c_sorted"] if s_fc==sorted(s_fc) else 0,
                     POINTS["1c_sorted"], "sorted"))
    return res, r_f, r_fc


# ═══════ Task 2 ═══════
def test_task2(stu, ref, df, features):
    res = []
    for label, fn, kv, kf in [("2a","count_stocks_per_month","2a_vals","2a_fmt"),
                                ("2b","mean_return_by_month","2b_vals","2b_fmt"),
                                ("2c","return_std_by_month","2c_vals","2c_fmt")]:
        s, e = _call(getattr(stu, fn), df)
        r = getattr(ref, fn)(df)
        if e or s is None:
            res += _zeros([kv, kf], str(e))
        else:
            f = _frac(s[:,1], r[:,1], RTOL_STRICT, ATOL_STRICT) if s.shape==r.shape else 0
            res.append((POINTS[kv]*f, POINTS[kv], f"{label} vals {f:.0%}"))
            fmt = s.shape==r.shape and np.array_equal(s[:,0], r[:,0])
            res.append((POINTS[kf] if fmt else 0, POINTS[kf], f"{label} fmt={fmt}"))

    s, e = _call(stu.feature_target_correlation, df, features)
    r = ref.feature_target_correlation(df, features)
    if e or s is None:
        res.append((0, POINTS["2d_corr"], str(e)))
    else:
        f = _frac(s, r, RTOL_STRICT, ATOL_STRICT)
        res.append((POINTS["2d_corr"]*f, POINTS["2d_corr"], f"2d corr {f:.0%}"))
    return res


# ═══════ Task 3 ═══════
def test_task3(stu, ref, df, features, target, n_train):
    keys = ["3_months","3_Str","3_Ste","3_y","3_dates"]
    s, e = _call(stu.train_test_split, df, features, target, n_train)
    r = ref.train_test_split(df, features, target, n_train)
    if e or s is None:
        return _zeros(keys, str(e)), r
    res = []
    ok = set(np.unique(s[4]))==set(np.unique(r[4])) and set(np.unique(s[5]))==set(np.unique(r[5]))
    res.append((POINTS["3_months"] if ok else 0, POINTS["3_months"], f"months={ok}"))
    ok = s[0].shape==r[0].shape and _close(s[0], r[0], RTOL_EXACT, ATOL_EXACT)
    res.append((POINTS["3_Str"] if ok else 0, POINTS["3_Str"], f"S_tr={s[0].shape}"))
    ok = s[2].shape==r[2].shape and _close(s[2], r[2], RTOL_EXACT, ATOL_EXACT)
    res.append((POINTS["3_Ste"] if ok else 0, POINTS["3_Ste"], f"S_te={s[2].shape}"))
    ok = _close(s[1], r[1], RTOL_EXACT, ATOL_EXACT) and _close(s[3], r[3], RTOL_EXACT, ATOL_EXACT)
    res.append((POINTS["3_y"] if ok else 0, POINTS["3_y"], "y"))
    ok = np.array_equal(s[4], r[4]) and np.array_equal(s[5], r[5])
    res.append((POINTS["3_dates"] if ok else 0, POINTS["3_dates"], "dates"))
    return res, r


# ═══════ Task 4 ═══════
def test_task4(stu, ref, S_tr, y_tr, S_te, shrinkage_list):
    keys = ["4_plt_betas","4_pge_betas","4_preds","4_multi","4_no_loop"]
    s, e = _call(stu.ridge_regression, S_tr, y_tr, S_te, shrinkage_list)
    r_b, r_p = ref.ridge_regression(S_tr, y_tr, S_te, shrinkage_list)
    if e or s is None:
        return _zeros(keys, str(e))
    s_b, s_p = s
    res = []
    f = _frac(s_b, r_b, RTOL_MEDIUM, ATOL_MEDIUM)
    res.append((POINTS["4_plt_betas"]*f, POINTS["4_plt_betas"], f"P<T betas {f:.0%}"))
    np.random.seed(42)
    Sb=np.random.randn(50,80); yb=np.random.randn(50); Sbt=np.random.randn(10,80)
    sb2, eb2 = _call(stu.ridge_regression, Sb, yb, Sbt, [1.0, 10.0])
    rb2, _ = ref.ridge_regression(Sb, yb, Sbt, [1.0, 10.0])
    if eb2 or sb2 is None:
        res.append((0, POINTS["4_pge_betas"], f"P>=T error: {eb2}"))
    else:
        f2 = _frac(sb2[0], rb2, RTOL_MEDIUM, ATOL_MEDIUM)
        res.append((POINTS["4_pge_betas"]*f2, POINTS["4_pge_betas"], f"P>=T betas {f2:.0%}"))
    fp = _frac(s_p, r_p, RTOL_MEDIUM, ATOL_MEDIUM)
    res.append((POINTS["4_preds"]*fp, POINTS["4_preds"], f"preds {fp:.0%}"))
    ok = s_b.shape == (S_tr.shape[1], len(shrinkage_list))
    res.append((POINTS["4_multi"] if ok else 0, POINTS["4_multi"], f"shape={s_b.shape}"))
    ok_l, ln = check_no_loops(os.path.join("src","regression.py"))
    res.append((POINTS["4_no_loop"] if ok_l else 0, POINTS["4_no_loop"],
                "no loops" if ok_l else f"loop at line {ln}"))
    return res


# ═══════ Task 5 ═══════
def test_task5(stu, ref, S_tr, y_tr, S_te, alpha):
    keys = ["5_beta","5_pred","5_deterministic"]
    s, e = _call(stu.lasso_regression, S_tr, y_tr, S_te, alpha)
    r_b, r_p = ref.lasso_regression(S_tr, y_tr, S_te, alpha)
    if e or s is None:
        return _zeros(keys, str(e))
    s_b, s_p = s
    res = []
    res.append((POINTS["5_beta"]*_frac(s_b,r_b,RTOL_MEDIUM,ATOL_MEDIUM),
                POINTS["5_beta"], f"beta {_frac(s_b,r_b,RTOL_MEDIUM,ATOL_MEDIUM):.0%}"))
    res.append((POINTS["5_pred"]*_frac(s_p,r_p,RTOL_MEDIUM,ATOL_MEDIUM),
                POINTS["5_pred"], f"pred {_frac(s_p,r_p,RTOL_MEDIUM,ATOL_MEDIUM):.0%}"))
    s2, _ = _call(stu.lasso_regression, S_tr, y_tr, S_te, alpha)
    det = s2 is not None and np.allclose(s[1], s2[1])
    res.append((POINTS["5_deterministic"] if det else 0,
                POINTS["5_deterministic"], f"deterministic={det}"))
    return res


# ═══════ Task 6 ═══════
def test_task6(stu, ref, S_tr, y_tr, d_tr, shrinkage_list, alpha_list):
    res = []
    y = np.array([1.,2.,3.,4.,5.])
    sr2, e = _call(stu.oos_r_squared, y, y)
    rr2 = ref.oos_r_squared(y, y)
    if e or sr2 is None:
        res.append((0, POINTS["6_r2_fn"], str(e)))
    else:
        res.append((POINTS["6_r2_fn"] if np.isclose(sr2, rr2) else 0,
                     POINTS["6_r2_fn"], f"r2={sr2:.4f} vs {rr2:.4f}"))

    s, e = _call(stu.select_ridge_shrinkage, S_tr, y_tr, d_tr, shrinkage_list)
    r_best, r_cv = ref.select_ridge_shrinkage(S_tr, y_tr, d_tr, shrinkage_list)
    if e or s is None:
        res += _zeros(["6a_scores","6a_best"], str(e))
    else:
        s_best, s_cv = s
        f = _frac(s_cv, r_cv, RTOL_LOOSE, ATOL_LOOSE)
        res.append((POINTS["6a_scores"]*f, POINTS["6a_scores"], f"ridge cv {f:.0%}"))
        ok = np.isclose(s_best, r_best, rtol=RTOL_LOOSE)
        res.append((POINTS["6a_best"] if ok else 0, POINTS["6a_best"],
                     f"best z: {s_best:.4g} vs {r_best:.4g}"))

    s, e = _call(stu.select_lasso_alpha, S_tr, y_tr, d_tr, alpha_list)
    r_best, r_cv = ref.select_lasso_alpha(S_tr, y_tr, d_tr, alpha_list)
    if e or s is None:
        res += _zeros(["6b_scores","6b_best"], str(e))
    else:
        s_best, s_cv = s
        f = _frac(s_cv, r_cv, RTOL_LOOSE, ATOL_LOOSE)
        res.append((POINTS["6b_scores"]*f, POINTS["6b_scores"], f"lasso cv {f:.0%}"))
        ok = np.isclose(s_best, r_best, rtol=RTOL_LOOSE)
        res.append((POINTS["6b_best"] if ok else 0, POINTS["6b_best"],
                     f"best a: {s_best:.4g} vs {r_best:.4g}"))

    for fname in ["data_loader.py","eda.py","regression.py","portfolio.py"]:
        ok_l, ln = check_no_loops(os.path.join("src", fname))
        if not ok_l:
            res.append((0, 0, f"WARNING: loop in {fname} line {ln}"))
    return res


# ═══════ Task 7 — accepts BOTH ddof=0 and ddof=1 ═══════
def test_task7(stu, ref, y_pred, y_actual, dates):
    res = []
    s, e = _call(stu.compute_managed_returns, y_pred, y_actual, dates)
    r = ref.compute_managed_returns(y_pred, y_actual, dates)
    if e or s is None:
        res += _zeros(["7a_vals","7a_fmt"], str(e))
    else:
        fmt = s.shape==r.shape and np.array_equal(s[:,0].astype(int), r[:,0].astype(int))
        res.append((POINTS["7a_fmt"] if fmt else 0, POINTS["7a_fmt"], f"fmt={fmt}"))
        f = _frac(s[:,1], r[:,1], RTOL_MEDIUM, ATOL_MEDIUM) if s.shape==r.shape else 0
        res.append((POINTS["7a_vals"]*f, POINTS["7a_vals"], f"vals {f:.0%}"))

    # use REFERENCE managed returns so grading is independent of 7a
    sp, e = _call(stu.performance_metrics, r)
    rp0 = ref.performance_metrics_ddof0(r)
    rp1 = ref.performance_metrics_ddof1(r)
    if e or sp is None:
        res += _zeros(["7b_mean","7b_vol","7b_sharpe"], str(e))
    else:
        # mean is same regardless of ddof
        sv = sp.get("mean")
        ok = sv is not None and np.isclose(sv, rp0["mean"], rtol=RTOL_MEDIUM)
        res.append((POINTS["7b_mean"] if ok else 0, POINTS["7b_mean"],
                     f"mean: {sv} vs {rp0['mean']:.6f}"))
        # vol: accept either ddof
        sv = sp.get("vol")
        ok = sv is not None and (np.isclose(sv, rp0["vol"], rtol=RTOL_MEDIUM)
                                  or np.isclose(sv, rp1["vol"], rtol=RTOL_MEDIUM))
        res.append((POINTS["7b_vol"] if ok else 0, POINTS["7b_vol"],
                     f"vol: {sv}, ref0={rp0['vol']:.6f}, ref1={rp1['vol']:.6f}"))
        # sharpe: accept either ddof
        sv = sp.get("sharpe")
        ok = sv is not None and (np.isclose(sv, rp0["sharpe"], rtol=RTOL_MEDIUM)
                                  or np.isclose(sv, rp1["sharpe"], rtol=RTOL_MEDIUM))
        res.append((POINTS["7b_sharpe"] if ok else 0, POINTS["7b_sharpe"],
                     f"sharpe: {sv}, ref0={rp0['sharpe']:.6f}, ref1={rp1['sharpe']:.6f}"))

    ok_l, ln = check_no_loops(os.path.join("src","portfolio.py"))
    if not ok_l:
        res.append((0, 0, f"WARNING: loop in portfolio.py line {ln}"))
    return res
