#!/usr/bin/env python3
"""Assignment 2 Autograder."""
import sys, os, json, traceback, importlib.util

STUDENT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GRADER  = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, STUDENT)
sys.path.insert(0, GRADER)

import numpy as np

# ── Load student config by explicit path (avoids grader/config.py clash) ──
_spec = importlib.util.spec_from_file_location(
    "student_config", os.path.join(STUDENT, "config.py"))
_scfg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_scfg)

N_TRAIN        = _scfg.N_TRAIN
SHRINKAGE_LIST = _scfg.SHRINKAGE_LIST
ALPHA_LASSO    = _scfg.ALPHA_LASSO
DATA_PATH      = _scfg.DATA_PATH
TARGET_COL     = _scfg.TARGET_COL
START_DATE     = _scfg.START_DATE

import reference as ref
from tests import (test_task1, test_task2, test_task3, test_task4,
                   test_task5, test_task6, test_task7)

class _Fake:
    def __getattr__(self, n):
        def _f(*a,**kw): raise NotImplementedError(f"not loaded: {n}")
        return _f

def _load(name):
    from importlib import import_module
    try:    return import_module(name)
    except Exception as e:
        print(f"  WARN: {name}: {e}"); return _Fake()

def main():
    print("="*60)
    print(f"ASSIGNMENT 2 AUTOGRADER   N_TRAIN={N_TRAIN}")
    print("="*60)

    s_dl=_load("src.data_loader"); s_eda=_load("src.eda")
    s_reg=_load("src.regression"); s_ms=_load("src.model_selection")
    s_pf=_load("src.portfolio")
    tasks = []

    # ── Task 1 ────────────────────────────────────────────────────
    print("\n[1] Data loading ...")
    try:
        t1, r_df, r_fc = test_task1(s_dl, ref, DATA_PATH, START_DATE)
    except Exception as e:
        traceback.print_exc()
        r_df = ref.filter_data(ref.load_data(DATA_PATH), START_DATE)
        r_fc = ref.get_feature_columns(r_df)
        t1 = [(0, 8, f"fatal: {e}")]
    tasks.append(("Task 1: Data Loading", t1))
    # Fallback if student code failed to produce features
    if r_fc is None:
        r_fc = ref.get_feature_columns(r_df)

    # ── Task 2 ────────────────────────────────────────────────────
    print("[2] EDA ...")
    try:    t2 = test_task2(s_eda, ref, r_df, r_fc)
    except Exception as e:
        traceback.print_exc(); t2 = [(0, 12, str(e))]
    tasks.append(("Task 2: EDA", t2))

    # ── Task 3 ────────────────────────────────────────────────────
    print("[3] Split ...")
    try:
        t3, r_split = test_task3(s_dl, ref, r_df, r_fc, TARGET_COL, N_TRAIN)
    except Exception as e:
        traceback.print_exc()
        r_split = ref.train_test_split(r_df, r_fc, TARGET_COL, N_TRAIN)
        t3 = [(0, 5, str(e))]
    tasks.append(("Task 3: Split", t3))
    S_tr, y_tr, S_te, y_te, d_tr, d_te = r_split

    # ── Task 4 ────────────────────────────────────────────────────
    print("[4] Ridge ...")
    try:    t4 = test_task4(s_reg, ref, S_tr, y_tr, S_te, SHRINKAGE_LIST)
    except Exception as e:
        traceback.print_exc(); t4 = [(0, 20, str(e))]
    tasks.append(("Task 4: Ridge", t4))

    # ── Task 5 ────────────────────────────────────────────────────
    print("[5] Lasso ...")
    mid_alpha = ALPHA_LASSO[len(ALPHA_LASSO)//2]
    try:    t5 = test_task5(s_reg, ref, S_tr, y_tr, S_te, mid_alpha)
    except Exception as e:
        traceback.print_exc(); t5 = [(0, 10, str(e))]
    tasks.append(("Task 5: Lasso", t5))

    # ── Task 6 ────────────────────────────────────────────────────
    print("[6] CV ...")
    try:    t6 = test_task6(s_ms, ref, S_tr, y_tr, d_tr, SHRINKAGE_LIST, ALPHA_LASSO)
    except Exception as e:
        traceback.print_exc(); t6 = [(0, 25, str(e))]
    tasks.append(("Task 6: Model Selection", t6))

    # ── Predictions for portfolio ─────────────────────────────────
    r_best_z, _ = ref.select_ridge_shrinkage(S_tr, y_tr, d_tr, SHRINKAGE_LIST)
    _, r_preds = ref.ridge_regression(S_tr, y_tr, S_te, [r_best_z])
    r_y_pred = r_preds[:, 0]

    # ── Task 7 ────────────────────────────────────────────────────
    print("[7] Portfolio ...")
    try:    t7 = test_task7(s_pf, ref, r_y_pred, y_te, d_te)
    except Exception as e:
        traceback.print_exc(); t7 = [(0, 20, str(e))]
    tasks.append(("Task 7: Portfolio", t7))

    # ── Report ────────────────────────────────────────────────────
    summary = []
    for name, results in tasks:
        pts = sum(r[0] for r in results)
        mx  = sum(r[1] for r in results)
        summary.append({"name": name, "points": round(pts, 2),
                        "max_points": round(mx, 2)})
        print(f"  {'+'if pts==mx else 'o'} {name}: {pts:.1f}/{mx:.1f}")

    total     = sum(s["points"] for s in summary)
    total_max = sum(s["max_points"] for s in summary)
    print(f"\n  TOTAL: {total:.1f} / {total_max:.1f}")

    report = {"total_points": round(total, 2),
              "max_points": round(total_max, 2),
              "n_train": N_TRAIN, "tasks": summary}
    path = os.path.join(STUDENT, "grading_report.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report -> {path}")

if __name__ == "__main__":
    main()
