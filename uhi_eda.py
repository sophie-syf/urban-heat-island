
import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
import json
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pandas.plotting import scatter_matrix
from scipy.stats import gaussian_kde, spearmanr, probplot, median_abs_deviation
from scipy.spatial import cKDTree
from sklearn.linear_model import LinearRegression

RENAME_MAP = {
    'City Name': 'City_Name',
    'Latitude': 'Latitude',
    'Longitude': 'Longitude',
    'Elevation (m)': 'Elevation_m',
    'Temperature (°C)': 'Temperature_degC',
    'Land Cover': 'Land_Cover',
    'Population Density (people/km²)': 'Population_Density_people_per_km2',
    'Energy Consumption (kWh)': 'Energy_Consumption_kWh',
    'Air Quality Index (AQI)': 'Air_Quality_Index_AQI',
    'Urban Greenness Ratio (%)': 'Urban_Greenness_Ratio',
    'Health Impact (Mortality Rate/100k)': 'Health_Impact_Mortality_Rate_per_100k',
    'Wind Speed (km/h)': 'Wind_Speed_kmph',
    'Humidity (%)': 'Humidity',
    'Annual Rainfall (mm)': 'Annual_Rainfall_mm',
    'GDP per Capita (USD)': 'GDP_per_Capita_USD'
}

DEFAULT_TARGET = 'Temperature_degC'


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p


def load_and_rename(csv_path):
    df = pd.read_csv(csv_path, encoding="utf-8")
    before = df.columns.tolist()
    df = df.rename(columns=RENAME_MAP)
    after = df.columns.tolist()
    print("Original Columns:", before)
    print("Renamed Columns :", after)
    return df


def save_descriptives(df, outdir):
    desc = df.describe(include='all').transpose()
    desc.to_csv(os.path.join(outdir, "descriptives.csv"))
    # top categoricals (value counts)
    cat_cols = [c for c in df.columns if df[c].dtype == 'object']
    topk = {}
    for c in cat_cols:
        vc = df[c].value_counts().head(15)
        topk[c] = vc.to_dict()
    with open(os.path.join(outdir, "top_categories.json"), "w") as f:
        json.dump(topk, f, indent=2)


def plot_missingness(df, outdir):
    miss = df.isna()
    miss_ratio = miss.mean().sort_values(ascending=False)
    # bar chart of missingness ratio
    fig, ax = plt.subplots(figsize=(10,6))
    ax.bar(miss_ratio.index, miss_ratio.values)
    ax.set_title("Missingness per Column (ratio)")
    ax.set_ylabel("Fraction missing")
    plt.setp(ax.get_xticklabels(), rotation=90, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "missingness_bar.pdf"))
    plt.close(fig)

    # heatmap of missingness (first N rows for visibility if very large)
    N = min(500, len(df))
    fig, ax = plt.subplots(figsize=(10,6))
    ax.imshow(miss.iloc[:N, :].values, aspect='auto', interpolation='nearest')
    ax.set_title(f"Missingness Heatmap (first {N} rows)")
    ax.set_xlabel("Columns")
    ax.set_ylabel("Rows")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "missingness_heatmap.pdf"))
    plt.close(fig)


def plot_correlations(df, outdir):
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        print("Not enough numeric columns for correlations.")
        return
    # Pearson
    pearson = numeric.corr(method="pearson")
    fig, ax = plt.subplots(figsize=(12,10))
    im = ax.imshow(pearson.values, aspect='auto', interpolation='nearest')
    ax.set_title("Correlation (Pearson)")
    ax.set_xticks(range(len(pearson.columns)))
    ax.set_yticks(range(len(pearson.columns)))
    ax.set_xticklabels(pearson.columns, rotation=90)
    ax.set_yticklabels(pearson.columns)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "correlation_pearson.pdf"))
    plt.close(fig)

    # Spearman
    spearman = numeric.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(12,10))
    im = ax.imshow(spearman.values, aspect='auto', interpolation='nearest')
    ax.set_title("Correlation (Spearman)")
    ax.set_xticks(range(len(spearman.columns)))
    ax.set_yticks(range(len(spearman.columns)))
    ax.set_xticklabels(spearman.columns, rotation=90)
    ax.set_yticklabels(spearman.columns)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "correlation_spearman.pdf"))
    plt.close(fig)

    # Save top correlations table
    tril_idx = np.tril_indices_from(pearson.values, k=-1)
    pairs = []
    cols = pearson.columns.to_list()
    for i,j in zip(*tril_idx):
        pairs.append((cols[i], cols[j], pearson.values[i,j]))
    pairs = sorted(pairs, key=lambda x: abs(x[2]), reverse=True)
    pd.DataFrame(pairs, columns=["var1","var2","pearson"]).to_csv(
        os.path.join(outdir, "top_correlations.csv"), index=False
    )


def plot_distributions(df, outdir):
    numeric = df.select_dtypes(include=[np.number])
    # hist + KDE for each numeric column
    for col in numeric.columns:
        s = numeric[col].dropna().values
        if len(s) < 5:
            continue
        fig, ax = plt.subplots(figsize=(7,5))
        ax.hist(s, bins=30, density=True, alpha=0.6)
        # KDE
        try:
            kde = gaussian_kde(s)
            xs = np.linspace(np.nanmin(s), np.nanmax(s), 200)
            ax.plot(xs, kde(xs))
        except Exception:
            pass
        ax.set_title(f"Distribution of {col}")
        ax.set_xlabel(col); ax.set_ylabel("Density")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"dist_{col}.pdf"))
        plt.close(fig)

        # QQ plot vs normal
        fig, ax = plt.subplots(figsize=(6,6))
        probplot(s, dist="norm", plot=ax)
        ax.set_title(f"QQ Plot of {col}")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"qq_{col}.pdf"))
        plt.close(fig)


def plot_grouped_boxplots(df, outdir):
    # Boxplots of selected numeric vs Land_Cover (if present)
    if 'Land_Cover' not in df.columns:
        return
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric:
        return
    # Limit to top 6 numerics for compactness
    numeric = numeric[:6]
    for col in numeric:
        fig, ax = plt.subplots(figsize=(9,6))
        groups = df[['Land_Cover', col]].dropna().groupby('Land_Cover')[col]
        labels, data = [], []
        for k, v in groups:
            labels.append(str(k))
            data.append(v.values)
        if len(data) < 2:
            plt.close(fig); continue
        ax.boxplot(data, labels=labels, vert=True, showfliers=False)
        ax.set_title(f"{col} by Land_Cover")
        ax.set_xlabel("Land_Cover"); ax.set_ylabel(col)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"box_{col}_by_landcover.pdf"))
        plt.close(fig)


def plot_scatter_hexbin(df, target, outdir):
    if target not in df.columns:
        print(f"Target {target} not found for scatter/hexbin.")
        return
    numeric = [c for c in df.select_dtypes(include=[np.number]).columns if c != target]
    # Make hexbin plots for up to 8 predictors
    for col in numeric[:8]:
        x = df[col].values
        y = df[target].values
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() < 20:
            continue
        fig, ax = plt.subplots(figsize=(7,5))
        hb = ax.hexbin(x[mask], y[mask], gridsize=40, mincnt=1)
        ax.set_xlabel(col); ax.set_ylabel(target)
        ax.set_title(f"{target} vs {col} (hexbin)")
        cb = fig.colorbar(hb, ax=ax)
        cb.set_label("count")
        # Add linear fit
        try:
            coeffs = np.polyfit(x[mask], y[mask], 1)
            xx = np.linspace(x[mask].min(), x[mask].max(), 200)
            ax.plot(xx, coeffs[0]*xx + coeffs[1])
        except Exception:
            pass
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, f"hex_{target}_vs_{col}.pdf"))
        plt.close(fig)


def plot_spatial_quicklook(df, outdir, target=DEFAULT_TARGET):
    # If Latitude/Longitude exist, show a quick scatter colored by target
    if not {'Latitude', 'Longitude', target}.issubset(df.columns):
        return
    sub = df[['Longitude','Latitude',target]].dropna()
    if len(sub) < 5:
        return
    fig, ax = plt.subplots(figsize=(7,6))
    sc = ax.scatter(sub['Longitude'], sub['Latitude'], c=sub[target], s=20)
    ax.set_title(f"Spatial quicklook: {target}")
    ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    cb = fig.colorbar(sc, ax=ax); cb.set_label(target)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"spatial_quicklook_{target}.pdf"))
    plt.close(fig)


def pair_scatter_matrix(df, outdir, cols=None):
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return
    if cols is None:
        # choose top-4 by absolute Spearman correlation with target
        target = DEFAULT_TARGET
        if target in numeric.columns:
            scores = []
            for c in numeric.columns:
                if c == target: continue
                try:
                    rho, _ = spearmanr(numeric[c], numeric[target], nan_policy='omit')
                    scores.append((c, abs(rho)))
                except Exception:
                    pass
            scores = sorted(scores, key=lambda x: x[1], reverse=True)[:4]
            cols = [target] + [c for c,_ in scores]
        else:
            cols = numeric.columns[:4].tolist()
    if len(cols) < 2:
        return
    fig = plt.figure(figsize=(10,10))
    scatter_matrix(df[cols].dropna(), figsize=(10,10), diagonal='hist', range_padding=0.05)
    fig.suptitle("Scatter Matrix (top related features)")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "scatter_matrix_top_related.pdf"))
    plt.close(fig)


# ==== Geological analysis  ==================================

def _safe_series(df, cols):
    for c in cols:
        if c not in df.columns:
            return None
    return df[cols].dropna()

def geological_descriptives(df, outdir, target=DEFAULT_TARGET):
    # Elevation bands and temperature profiles
    cols = _safe_series(df, ["Elevation_m", target])
    if cols is None or cols.shape[0] < 20:
        return
    tmp = cols.copy()
    qbins = pd.qcut(tmp["Elevation_m"], 10, duplicates="drop")
    prof = tmp.groupby(qbins)[target].agg(["mean", "median", "count"]).reset_index()
    prof.to_csv(os.path.join(outdir, "geo_elevation_deciles_summary.csv"), index=False)

    # Plot mean temp by elevation decile
    fig, ax = plt.subplots(figsize=(7,4))
    ax.plot(range(1, len(prof)+1), prof["mean"].values, marker="o")
    ax.set_xlabel("Elevation decile (low → high)")
    ax.set_ylabel(f"Mean {target}")
    ax.set_title(f"{target} vs Elevation deciles")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "geo_temp_by_elevation_deciles.pdf"))
    plt.close(fig)

    # Hexbin Temperature vs Elevation
    fig, ax = plt.subplots(figsize=(7,5))
    hb = ax.hexbin(tmp["Elevation_m"].values, tmp[target].values, gridsize=40, mincnt=1)
    ax.set_xlabel("Elevation (m)"); ax.set_ylabel(target)
    ax.set_title(f"{target} vs Elevation (hexbin)")
    cb = fig.colorbar(hb, ax=ax); cb.set_label("count")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "geo_hex_temp_vs_elevation.pdf"))
    plt.close(fig)

def geological_lapse_rate(df, outdir, target=DEFAULT_TARGET):
    cols = ["Elevation_m", target]
    controls = []
    if "Latitude" in df.columns and "Longitude" in df.columns:
        controls = ["Latitude", "Longitude"]
    needed = cols + controls
    dat = _safe_series(df, needed)
    if dat is None or dat.shape[0] < 20:
        return

    y = dat[target].values
    X = dat[["Elevation_m"] + controls].values
    model = LinearRegression()
    model.fit(X, y)
    beta_elev = model.coef_[0]
    lapse_per_100m = beta_elev * 100.0  # °C per 100 m

    summ = {
        "n": int(dat.shape[0]),
        "controls": controls,
        "beta_elevation_degC_per_m": float(beta_elev),
        "lapse_rate_degC_per_100m": float(lapse_per_100m),
        "intercept": float(model.intercept_)
    }
    with open(os.path.join(outdir, "geo_lapse_rate_summary.json"), "w") as f:
        json.dump(summ, f, indent=2)

    elev = dat["Elevation_m"].values
    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(elev, y, s=10, alpha=0.6)
    egrid = np.linspace(np.nanmin(elev), np.nanmax(elev), 200)
    if controls:
        ctrl_means = dat[controls].mean().values.reshape(1, -1)
        Xline = np.column_stack([egrid, np.repeat(ctrl_means, repeats=len(egrid), axis=0)])
    else:
        Xline = egrid.reshape(-1,1)
    yline = model.predict(Xline)
    ax.plot(egrid, yline)
    ax.set_xlabel("Elevation (m)"); ax.set_ylabel(target)
    ax.set_title(f"Fitted temperature–elevation trend (controls: {controls if controls else 'none'})")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "geo_lapse_rate_fit.pdf"))
    plt.close(fig)

    if {"Latitude","Longitude"}.issubset(dat.columns):
        X_all = dat[["Elevation_m"] + controls].values
        yhat = model.predict(X_all)
        resid = y - yhat
        fig, ax = plt.subplots(figsize=(7,6))
        sc = ax.scatter(dat["Longitude"].values, dat["Latitude"].values, c=resid, s=20)
        ax.set_title("Residuals after elevation control")
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        cb = fig.colorbar(sc, ax=ax); cb.set_label("residual (°C)")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "geo_residual_map.pdf"))
        plt.close(fig)

def _local_plane_slopes(lat, lon, elev, k=12):
    # Fit local plane space for each point via nearest neighbors
    pts = np.column_stack([lon, lat])
    tree = cKDTree(pts)
    slopes = np.full(len(pts), np.nan)
    aspects = np.full(len(pts), np.nan)
    for i, p in enumerate(pts):
        d, idx = tree.query(p, k=min(k, len(pts)))
        neigh = pts[idx]
        z = elev[idx]
        A = np.column_stack([neigh[:,0], neigh[:,1], np.ones(len(idx))])
        try:
            coefs, *_ = np.linalg.lstsq(A, z, rcond=None)
            ax_, by_ = coefs[0], coefs[1]
            slope = np.arctan(np.sqrt(ax_**2 + by_**2))  # radians
            aspect = np.arctan2(by_, ax_)  # radians
            slopes[i] = slope
            aspects[i] = aspect
        except Exception:
            continue
    return slopes, aspects

def geological_slope_aspect(df, outdir, target=DEFAULT_TARGET):
    needed = ["Latitude","Longitude","Elevation_m"]
    dat = _safe_series(df, needed + ([target] if target in df.columns else []))
    if dat is None or dat.shape[0] < 30:
        return
    lat = dat["Latitude"].values
    lon = dat["Longitude"].values
    elev = dat["Elevation_m"].values
    slopes, aspects = _local_plane_slopes(lat, lon, elev, k=12)

    st = {
        "n_slope": int(np.isfinite(slopes).sum()),
        "slope_mean_deg": float(np.nanmean(slopes)*180/np.pi),
        "slope_median_deg": float(np.nanmedian(slopes)*180/np.pi)
    }
    with open(os.path.join(outdir, "geo_slope_stats.json"), "w") as f:
        json.dump(st, f, indent=2)

    # Histograms
    fig, ax = plt.subplots(figsize=(7,4))
    ax.hist(slopes[np.isfinite(slopes)]*180/np.pi, bins=30)
    ax.set_title("Local slope distribution"); ax.set_xlabel("Slope (degrees)"); ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "geo_slope_hist.pdf"))
    plt.close(fig)

    # Temperature vs slope
    if target in dat.columns:
        s_mask = np.isfinite(slopes) & np.isfinite(dat[target].values)
        if s_mask.sum() > 30:
            fig, ax = plt.subplots(figsize=(7,5))
            hb = ax.hexbin(slopes[s_mask]*180/np.pi, dat[target].values[s_mask], gridsize=40, mincnt=1)
            ax.set_xlabel("Slope (degrees)"); ax.set_ylabel(target)
            ax.set_title(f"{target} vs slope")
            fig.colorbar(hb, ax=ax)
            fig.tight_layout()
            fig.savefig(os.path.join(outdir, "geo_hex_temp_vs_slope.pdf"))
            plt.close(fig)

    finite_aspect = aspects[np.isfinite(aspects)]
    if finite_aspect.size > 10:
        # Polar histogram
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(111, projection="polar")
        bins = np.linspace(-np.pi, np.pi, 37)
        hist, edges = np.histogram(finite_aspect, bins=bins)
        ax.bar((edges[:-1]+edges[1:])/2, hist, width=np.diff(edges), align='center')
        ax.set_title("Aspect distribution")
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "geo_aspect_polar.pdf"))
        plt.close(fig)

        if target in dat.columns:
            d2 = dat.copy()
            d2["aspect"] = aspects
            mask = np.isfinite(d2["aspect"].values) & np.isfinite(d2[target].values)
            if mask.sum() > 30:
                sectors = np.linspace(-np.pi, np.pi, 13)  # 12 sectors
                inds = np.digitize(d2.loc[mask, "aspect"].values, sectors) - 1
                tmeans = []
                centers = []
                for si in range(len(sectors)-1):
                    sel = (inds == si)
                    if sel.sum() == 0:
                        tmeans.append(np.nan); centers.append(0)
                    else:
                        tmeans.append(d2.loc[mask, target].values[sel].mean())
                        centers.append((sectors[si]+sectors[si+1])/2)
                fig = plt.figure(figsize=(6,6))
                ax = fig.add_subplot(111, projection="polar")
                widths = np.diff(sectors)
                ax.bar(centers, np.nan_to_num(tmeans, nan=0.0), width=widths, align='center')
                ax.set_title(f"{target} by aspect sector (mean)")
                fig.tight_layout();
                fig.savefig(os.path.join(outdir, "geo_temp_by_aspect_polar.pdf"))
                plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Enriched EDA with PDF outputs (with geological analysis)")
    ap.add_argument("--csv", default="urban_heat_island_dataset.csv", help="Input CSV path")
    ap.add_argument("--outdir", default="eda_outputs_pdf", help="Output directory for PDFs")
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    df = load_and_rename(args.csv)

    # 1) Tables & missingness
    save_descriptives(df, outdir)
    plot_missingness(df, outdir)

    # 2) Correlations (Pearson & Spearman + top pairs table)
    plot_correlations(df, outdir)

    # 3) Univariate distributions (hist+KDE) + QQ plots
    plot_distributions(df, outdir)

    # 4) Grouped boxplots by Land_Cover (if available)
    plot_grouped_boxplots(df, outdir)

    # 5) Bivariate hexbin with linear fits
    plot_scatter_hexbin(df, DEFAULT_TARGET, outdir)

    # 6) Spatial quicklook (if lat/lon present)
    plot_spatial_quicklook(df, outdir, target=DEFAULT_TARGET)

    # 7) Pairwise scatter matrix (top related to target)
    pair_scatter_matrix(df, outdir)

    # 8) Geological analysis
    geological_descriptives(df, outdir)
    geological_lapse_rate(df, outdir)
    geological_slope_aspect(df, outdir)

    print(f"EDA complete. PDFs written to: {outdir}")
    print("Artifacts (in addition to earlier ones):")
    print(" - geo_elevation_deciles_summary.csv, geo_temp_by_elevation_deciles.pdf, geo_hex_temp_vs_elevation.pdf")
    print(" - geo_lapse_rate_summary.json, geo_lapse_rate_fit.pdf, geo_residual_map.pdf (if lat/lon available)")
    print(" - geo_slope_hist.pdf, geo_hex_temp_vs_slope.pdf, geo_aspect_polar.pdf, geo_temp_by_aspect_polar.pdf (if lat/lon available)")

if __name__ == "__main__":
    main()
