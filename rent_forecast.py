#!/usr/bin/env python3
"""Portfolio rent forecasting utility.

This script loads property and Zillow Observed Rent Index (ZORI) datasets,
fits a simple ridge regression model to the current property snapshot and then
projects future rents under multiple inflation scenarios.  It also supports an
optional backtest mode that compares the model against flat and ZORI-only
baselines.
"""

import argparse
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


def _to_month(ts):
    """Round a timestamp-like value down to the first day of its month."""

    return pd.to_datetime(ts).to_period("M").to_timestamp()


def load_inputs(cfg):
    """Load the property snapshot and ZORI data from the configured CSV files."""

    prop = pd.read_csv(cfg["property_csv"])
    zori = pd.read_csv(cfg["zori_csv"])
    prop["as_of_date"] = pd.to_datetime(prop["as_of_date"])
    zori["date"] = pd.to_datetime(zori["date"]).dt.to_period("M").dt.to_timestamp()
    return prop, zori


def prep_zori(zori: pd.DataFrame):
    """Normalise ZORI values and compute a time index."""

    z = zori.sort_values("date").copy()
    z["zori_norm"] = (z["zori"] - z["zori"].mean()) / (z["zori"].std() + 1e-9)
    z["t"] = (z["date"] - z["date"].min()) / np.timedelta64(1, "M")
    return z


def design_matrix(df: pd.DataFrame, X_cols: list):
    """Construct a design matrix with an intercept column and numeric features."""

    X_list = [np.ones(len(df))]
    for c in X_cols:
        X_list.append(df[c].astype(float).values)
    return np.column_stack(X_list)


def ridge_fit(X, y, lam=1.0):
    """Fit a ridge regression model using the closed-form solution."""

    XtX = X.T @ X
    beta = np.linalg.solve(XtX + lam * np.eye(X.shape[1]), X.T @ y)
    return beta


def predict(X, beta):
    """Compute predictions given a design matrix and coefficient vector."""

    return X @ beta


DEFAULT_NUMERIC = [
    "beds",
    "baths",
    "sqft",
    "year_built",
    "amenities_score",
    "condition_score",
    "renovation_year",
    "crime_index",
    "school_score",
]
TREND_COLS = ["t", "zori_norm"]


def build_snapshot(prop, zori):
    """Prepare the design matrix for the current property snapshot."""

    prop = prop.copy()
    prop["as_of_month"] = prop["as_of_date"].dt.to_period("M").dt.to_timestamp()
    snap = (
        pd.merge(
            prop,
            zori[["date", "zori_norm", "t"]],
            left_on="as_of_month",
            right_on="date",
            how="left",
        )
        .drop(columns=["date"])
        .copy()
    )
    num_cols = [c for c in DEFAULT_NUMERIC if c in snap.columns]
    X_cols = TREND_COLS + num_cols
    X = np.column_stack(
        [np.ones(len(snap))]
        + [snap[c].astype(float).fillna(0).values for c in X_cols]
    )
    y = snap["current_rent"].astype(float).values
    return snap, X, y, X_cols


def project_units(snap, zori, beta, X_cols, cfg):
    """Project unit-level rents across the forecast horizon."""

    start = snap["as_of_month"].max()
    end = start + pd.offsets.DateOffset(years=int(cfg["years"]))
    horizon = pd.date_range(start, end, freq=cfg.get("freq", "M"))

    zh = zori.set_index("date").reindex(horizon)
    zh["zori_norm"] = zh["zori_norm"].interpolate().bfill().ffill()
    zh["t"] = ((zh.index - zori["date"].min()) / np.timedelta64(1, "M")).astype(float)

    rows = []
    for _, u in snap.iterrows():
        feat = {
            c: float(u[c]) if c in u and pd.notna(u[c]) else 0.0 for c in X_cols
        }
        Xh = np.column_stack(
            [
                np.ones(len(horizon)),
                zh["t"].values,
                zh["zori_norm"].values,
            ]
            + [
                np.full(len(horizon), feat.get(c, 0.0))
                for c in X_cols
                if c not in TREND_COLS
            ]
        )
        base = predict(Xh, beta)
        df = pd.DataFrame(
            {
                "property_id": u.get("property_id", ""),
                "unit_id": u.get("unit_id", ""),
                "date": horizon,
                "rent_baseline": base,
            }
        )
        df["year"] = df["date"].dt.year
        yearly = (
            df.groupby(["property_id", "unit_id", "year"], as_index=False)[
                "rent_baseline"
            ]
            .mean()
            .copy()
        )
        rows.append(yearly)
    return pd.concat(rows, ignore_index=True)


def apply_scenarios(yearly, cfg):
    """Apply inflation scenarios to the baseline rent projections."""

    cap = cfg.get("annual_growth_cap", None)
    infls = [float(x) for x in cfg.get("inflation_scenarios", [0.02, 0.04, 0.06])]
    out = yearly.sort_values(["property_id", "unit_id", "year"]).copy()
    for r in infls:
        col = f"rent_{int(r * 100)}pct"

        def grow(series):
            vals = series.values.astype(float)
            for i in range(1, len(vals)):
                g = r
                if cap is not None:
                    g = min(g, float(cap))
                vals[i] = vals[i - 1] * (1.0 + g)
            return pd.Series(vals, index=series.index)

        out[col] = out.groupby(["property_id", "unit_id"]) ["rent_baseline"].transform(
            grow
        )
    return out


def backtest(prop, zori, holdout_months=12):
    """Perform a simple time-based backtest against flat and ZORI-only baselines."""

    z = zori.sort_values("date").copy()
    if len(z) <= holdout_months + 6:
        return None
    split_date = z.iloc[-holdout_months]["date"]

    prop = prop.copy()
    prop["as_of_month"] = prop["as_of_date"].dt.to_period("M").dt.to_timestamp()

    records = []
    for _, u in prop.iterrows():
        anchor_date = u["as_of_month"]
        anchor_row = z.loc[z["date"] == anchor_date]
        if anchor_row.empty:
            idx = (z["date"] - anchor_date).abs().argmin()
            anchor_row = z.iloc[[idx]]
        anchor_z = float(anchor_row["zori"])
        for _, zr in z.iterrows():
            rent_t = float(u["current_rent"]) * (zr["zori"] / anchor_z)
            records.append(
                {
                    "property_id": u.get("property_id", ""),
                    "unit_id": u.get("unit_id", ""),
                    "date": zr["date"],
                    "beds": u.get("beds", np.nan),
                    "baths": u.get("baths", np.nan),
                    "sqft": u.get("sqft", np.nan),
                    "year_built": u.get("year_built", np.nan),
                    "amenities_score": u.get("amenities_score", np.nan),
                    "condition_score": u.get("condition_score", np.nan),
                    "renovation_year": u.get("renovation_year", np.nan),
                    "crime_index": u.get("crime_index", np.nan),
                    "school_score": u.get("school_score", np.nan),
                    "rent": rent_t,
                }
            )
    hist = pd.DataFrame(records)
    hist["date"] = pd.to_datetime(hist["date"])
    z = prep_zori(z)

    feat = pd.merge(hist, z[["date", "zori_norm", "t"]], on="date", how="left")
    train = feat[feat["date"] < split_date].copy()
    test = feat[feat["date"] >= split_date].copy()

    X_cols = TREND_COLS + [c for c in DEFAULT_NUMERIC if c in feat.columns]

    def Xy(df):
        X = np.column_stack(
            [np.ones(len(df))]
            + [df[c].astype(float).fillna(0).values for c in X_cols]
        )
        y = df["rent"].astype(float).values
        return X, y

    Xtr, ytr = Xy(train)
    Xte, yte = Xy(test)
    beta = ridge_fit(Xtr, ytr, lam=1.0)
    pred = predict(Xte, beta)

    last_map = (
        train.sort_values("date").groupby(["property_id", "unit_id"])["rent"].last()
    ).to_dict()
    flat_pred = np.array(
        [last_map.get((row["property_id"], row["unit_id"]), ytr.mean()) for _, row in test.iterrows()]
    )

    Xtr_z = np.column_stack([np.ones(len(train)), train["zori_norm"].values])
    Xte_z = np.column_stack([np.ones(len(test)), test["zori_norm"].values])
    beta_z = ridge_fit(Xtr_z, train["rent"].values, lam=1e-6)
    zori_only_pred = Xte_z @ beta_z

    def mae(a, b):
        return float(np.mean(np.abs(a - b)))

    def mape(a, b):
        eps = 1e-9
        return float(np.mean(np.abs((a - b) / (a + eps)))) * 100.0

    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    metrics = {
        "model_mae": mae(yte, pred),
        "model_mape_pct": mape(yte, pred),
        "model_rmse": rmse(yte, pred),
        "flat_mae": mae(yte, flat_pred),
        "flat_mape_pct": mape(yte, flat_pred),
        "flat_rmse": rmse(yte, flat_pred),
        "zori_only_mae": mae(yte, zori_only_pred),
        "zori_only_mape_pct": mape(yte, zori_only_pred),
        "zori_only_rmse": rmse(yte, zori_only_pred),
        "holdout_months": int(holdout_months),
        "split_date": pd.to_datetime(split_date).strftime("%Y-%m-%d"),
    }

    out_test = test[["property_id", "unit_id", "date"]].copy()
    out_test["actual"] = yte
    out_test["pred_model"] = pred
    out_test["pred_flat"] = flat_pred
    out_test["pred_zori"] = zori_only_pred
    return metrics, out_test


def summarize(out_df: pd.DataFrame):
    """Aggregate average rent by year for each scenario column."""

    cols = [c for c in out_df.columns if c.startswith("rent_")]
    return out_df.groupby("year", as_index=False)[cols].mean()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--holdout_months", type=int, default=12)
    parser.add_argument(
        "--open_summary",
        action="store_true",
        help="Open the newest summary CSV after writing outputs",
    )
    parser.add_argument(
        "--export_charts",
        action="store_true",
        help="Export PNG charts of YoY rent by scenario",
    )
    parser.add_argument(
        "--no_backtest",
        action="store_true",
        help="Skip the time-split backtest for faster runs",
    )
    parser.add_argument(
        "--zip_filter",
        type=str,
        default="",
        help="Comma-separated ZIPs to include (e.g., 77007,77008)",
    )
    parser.add_argument(
        "--outdir_suffix",
        type=str,
        default="",
        help="If set, write outputs to outputs/<suffix>. Use AUTO to timestamp automatically.",
    )

    args = parser.parse_args(argv)

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    prop, zori_raw = load_inputs(cfg)
    if args.zip_filter:
        zips = [z.strip() for z in args.zip_filter.split(",") if z.strip()]
        if "zip" in prop.columns:
            prop = prop[prop["zip"].astype(str).isin(zips)].copy()
            if len(prop) == 0:
                print("No rows match the provided zip_filter:", zips)
                return 0
            print(f"Applied zip_filter: {zips} -> {len(prop)} rows")
        else:
            print("zip_filter provided but no `zip` column in property CSV; ignoring.")
    z = prep_zori(zori_raw)
    snap, X, y, X_cols = build_snapshot(prop, z)
    beta = ridge_fit(X, y, lam=1.0)

    yearly = project_units(snap, z, beta, X_cols, cfg)
    out = apply_scenarios(yearly, cfg)
    summ = summarize(out)

    bt = None
    if not args.no_backtest:
        bt = backtest(prop, zori_raw, holdout_months=args.holdout_months)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir_base = Path("outputs")
    suffix = args.outdir_suffix
    if suffix == "AUTO":
        suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = outdir_base / suffix if suffix else outdir_base
    outdir.mkdir(exist_ok=True, parents=True)
    if suffix:
        print("Writing to subfolder:", outdir)

    out.to_csv(outdir / f"forecast_plus_{ts}.csv", index=False)
    summ.to_csv(outdir / f"summary_plus_{ts}.csv", index=False)

    if bt is not None:
        metrics, test_df = bt
        md = pd.DataFrame([metrics])
        md.to_csv(outdir / f"metrics_{ts}.csv", index=False)
        test_df.to_csv(outdir / f"backtest_preds_{ts}.csv", index=False)
        print("Backtest split at:", metrics["split_date"])
        print(
            "Model MAE:",
            metrics["model_mae"],
            " vs Flat MAE:",
            metrics["flat_mae"],
            " vs ZORI-only MAE:",
            metrics["zori_only_mae"],
        )
        print(
            "Wrote:",
            outdir / f"metrics_{ts}.csv",
            outdir / f"backtest_preds_{ts}.csv",
        )
    else:
        print("Not enough history to backtest.")

    print("Wrote:", outdir / f"forecast_plus_{ts}.csv", outdir / f"summary_plus_{ts}.csv")

    if args.export_charts:
        try:
            cols = [c for c in summ.columns if c.startswith("rent_")]
            for c in cols:
                plt.figure()
                plt.plot(summ["year"], summ[c], marker="o")
                plt.title(f"Portfolio Average Monthly Rent – {c}")
                plt.xlabel("Year")
                plt.ylabel("Avg Monthly Rent")
                fig_path = outdir / f"{c}_chart_{ts}.png"
                plt.savefig(fig_path, dpi=150, bbox_inches="tight")
                plt.close()
            print("Exported charts to:", outdir)
        except Exception as e:  # pragma: no cover - best effort logging only
            print("Chart export failed:", e)

    if args.open_summary:
        try:
            newest = str(outdir / f"summary_plus_{ts}.csv")
            if platform.system() == "Darwin":
                subprocess.run(["open", newest], check=False)
            elif platform.system() == "Windows":
                os.startfile(newest)  # type: ignore[attr-defined]
            else:
                subprocess.run(["xdg-open", newest], check=False)
            print("Opened:", newest)
        except Exception as e:  # pragma: no cover - best effort logging only
            print("Open failed:", e)

    return 0


if __name__ == "__main__":
    sys.exit(main())
