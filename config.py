"""Grader configuration: tolerances and point allocations."""

RTOL_EXACT  = 1e-6;  ATOL_EXACT  = 1e-8
RTOL_STRICT = 1e-4;  ATOL_STRICT = 1e-6
RTOL_MEDIUM = 1e-3;  ATOL_MEDIUM = 1e-5
RTOL_LOOSE  = 1e-2;  ATOL_LOOSE  = 1e-4

POINTS = {
    # Task 1  (8)
    "1a_load": 1, "1a_dt": 1, "1a_yyyymm": 1,
    "1b_filter": 1, "1b_reset": 1,
    "1c_numeric": 1, "1c_target": 1, "1c_sorted": 1,
    # Task 2  (12)
    "2a_vals": 1.5, "2a_fmt": 1.5,
    "2b_vals": 1.5, "2b_fmt": 1.5,
    "2c_vals": 1.5, "2c_fmt": 1.5,
    "2d_corr": 3,
    # Task 3  (5)
    "3_months": 1, "3_Str": 1, "3_Ste": 1, "3_y": 1, "3_dates": 1,
    # Task 4  (20)
    "4_plt_betas": 7, "4_pge_betas": 5, "4_preds": 4,
    "4_multi": 2, "4_no_loop": 2,
    # Task 5  (10)
    "5_beta": 4, "5_pred": 4, "5_deterministic": 2,
    # Task 6  (25)
    "6_r2_fn": 2,
    "6a_scores": 8, "6a_best": 5,
    "6b_scores": 6, "6b_best": 4,
    # Task 7  (20)
    "7a_vals": 7, "7a_fmt": 4,
    "7b_mean": 3, "7b_vol": 3, "7b_sharpe": 3,
}
