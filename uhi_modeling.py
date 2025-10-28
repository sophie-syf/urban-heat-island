
import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV, learning_curve
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import joblib

# ---------------------- Config ----------------------

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

TARGET = "Temperature_degC"

# ---------------------- Feature engineering helpers (top-level, picklable) ----------------------

def ratio(a, b):
    with np.errstate(divide='ignore', invalid='ignore'):
        z = a / b
        z[~np.isfinite(z)] = np.nan
        return z

def log1p_safe(x):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.asarray(x)
        x = np.clip(x, a_min=0, a_max=None)
        z = np.log1p(x)
        z[~np.isfinite(z)] = np.nan
        return z

def add_derived_columns(df):
    """Create crude Impervious_ratio if missing: 1 - greenness (assuming % if >1)."""
    df2 = df.copy()
    if ('Urban_Greenness_Ratio' in df2.columns) and ('Impervious_ratio' not in df2.columns):
        g = df2['Urban_Greenness_Ratio'].astype(float)
        g01 = np.where(g > 1.0, g / 100.0, g)
        df2['Impervious_ratio'] = np.clip(1.0 - g01, 0, 1)
    return df2

def create_interactions(df):
    """Selected pairwise interactions."""
    out = df.copy()
    if {'Urban_Greenness_Ratio','Population_Density_people_per_km2'}.issubset(out.columns):
        out['Greenness_x_Density'] = out['Urban_Greenness_Ratio'] * out['Population_Density_people_per_km2']
    if {'Energy_Consumption_kWh','Air_Quality_Index_AQI'}.issubset(out.columns):
        out['Energy_x_AQI'] = out['Energy_Consumption_kWh'] * out['Air_Quality_Index_AQI']
    if {'Elevation_m','Humidity'}.issubset(out.columns):
        out['Elevation_x_Humidity'] = out['Elevation_m'] * out['Humidity']
    return out

def create_ratios_logs(df):
    """Log transforms and ratios."""
    out = df.copy()
    if 'Population_Density_people_per_km2' in out.columns:
        out['log_PopDensity'] = log1p_safe(out['Population_Density_people_per_km2'].values)
    if 'Energy_Consumption_kWh' in out.columns:
        out['log_Energy'] = log1p_safe(out['Energy_Consumption_kWh'].values)
    if 'GDP_per_Capita_USD' in out.columns:
        out['log_GDPpc'] = log1p_safe(out['GDP_per_Capita_USD'].values)
    if {'Urban_Greenness_Ratio','Impervious_ratio'}.issubset(out.columns):
        out['Green_to_Imperv'] = ratio(out['Urban_Greenness_Ratio'].values, out['Impervious_ratio'].values)
    return out

def fe_step(df_):
    """Top-level FE function to be used by FunctionTransformer (pickle-safe)."""
    df = df_.copy()
    df = create_interactions(df)
    df = create_ratios_logs(df)
    return df

# ---------------------- Utilities ----------------------

def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    mape = np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-8, None))) * 100.0
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'MAPE%': mape, 'R2': r2}

def save_table_pdf(df, outpath, title="Model Comparison"):
    fig, ax = plt.subplots(figsize=(max(8, df.shape[1]*1.3), max(3, df.shape[0]*0.6)))
    ax.axis('off')
    tbl = ax.table(cellText=np.round(df.values, 4),
                   colLabels=df.columns,
                   rowLabels=df.index,
                   loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.2)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(outpath)
    plt.close(fig)

def get_feature_names(preprocessor, numeric_features, categorical_features):
    """Recover output feature names from ColumnTransformer (post-OHE)."""
    names = []
    # numeric 
    names.extend(numeric_features)
    # categorical 
    cat = preprocessor.named_transformers_.get("cat", None)
    if cat is not None:
        oh = cat.named_steps.get("oh", None)
        if oh is not None:
            try:
                oh_names = oh.get_feature_names_out(categorical_features).tolist()
            except Exception:
                oh_names = ["%s_%s" % (categorical_features[i // 1000], i)  # crude fallback
                            for i in range(oh.n_features_in_)]
            names.extend(oh_names)
    return names


def main():
    ap = argparse.ArgumentParser(description="UHI modeling with FE, tuning, and model comparison (Py37-safe)")
    ap.add_argument("--csv", default="urban_heat_island_dataset.csv", help="Input CSV")
    ap.add_argument("--outdir", default="model_outputs", help="Output directory")
    ap.add_argument("--cv", type=int, default=5, help="CV folds")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load & rename
    df = pd.read_csv(args.csv, encoding="utf-8")
    df = df.rename(columns=RENAME_MAP)
    df = add_derived_columns(df)

    if TARGET not in df.columns:
        raise ValueError(f"Target column {TARGET} not found in the dataset.")

    # Select candidate base features (keep only existing)
    base_numeric_priority = [
        'Elevation_m', 'Population_Density_people_per_km2', 'Energy_Consumption_kWh',
        'Air_Quality_Index_AQI', 'Urban_Greenness_Ratio', 'Wind_Speed_kmph',
        'Humidity', 'Annual_Rainfall_mm', 'GDP_per_Capita_USD', 'Latitude', 'Longitude',
        'Impervious_ratio'
    ]
    numeric_candidates = [c for c in base_numeric_priority if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    categorical_candidates = [c for c in df.columns if df[c].dtype == 'object']

    cols_for_model = numeric_candidates + categorical_candidates + [TARGET]
    df_model = df[cols_for_model].copy()
    df_model = df_model.dropna(subset=[TARGET])

    X = df_model.drop(columns=[TARGET])
    y = df_model[TARGET].values

    # ---------- Feature engineering ----------
    X_fe_preview = fe_step(X.copy())
    numeric_features = [c for c in X_fe_preview.columns if pd.api.types.is_numeric_dtype(X_fe_preview[c])]
    categorical_features = [c for c in X_fe_preview.columns if c not in numeric_features]

    fe = FunctionTransformer(fe_step, validate=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler())
            ]), numeric_features),
            ("cat", Pipeline(steps=[
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse=False))
            ]), categorical_features),
        ],
        remainder="drop"
    )

    models = {
        "Ridge": (Ridge(), {"model__alpha": np.logspace(-3, 3, 20)}),
        "Lasso": (Lasso(max_iter=20000), {"model__alpha": np.logspace(-3, 1, 20)}),
        "ElasticNet": (ElasticNet(max_iter=20000), {
            "model__alpha": np.logspace(-3, 1, 20),
            "model__l1_ratio": np.linspace(0.05, 0.95, 10)
        }),
        "SVR": (SVR(), {
            "model__C": np.logspace(-2, 2, 20),
            "model__gamma": np.logspace(-3, 1, 10),
            "model__epsilon": np.logspace(-3, 0, 10),
            "model__kernel": ["rbf"]
        }),
        "KNN": (KNeighborsRegressor(), {
            "model__n_neighbors": list(range(3, 31)),
            "model__weights": ["uniform", "distance"],
            "model__p": [1, 2]
        }),
        "RandomForest": (RandomForestRegressor(n_jobs=-1, random_state=42), {
            "model__n_estimators": list(range(200, 601, 100)),
            "model__max_depth": [None, 6, 10, 14, 18],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["auto", "sqrt", 0.7]
        }),
        "ExtraTrees": (ExtraTreesRegressor(n_jobs=-1, random_state=42), {
            "model__n_estimators": list(range(200, 601, 100)),
            "model__max_depth": [None, 6, 10, 14, 18],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["auto", "sqrt", 0.7]
        }),
        "GBDT": (GradientBoostingRegressor(random_state=42), {
            "model__n_estimators": list(range(150, 451, 50)),
            "model__learning_rate": np.logspace(-3, -0.3, 10),
            "model__max_depth": [2, 3, 4, 5],
            "model__subsample": [0.6, 0.8, 1.0]
        }),
    }

    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = (XGBRegressor(
            objective="reg:squarederror", tree_method="hist", random_state=42
        ), {
            "model__n_estimators": list(range(200, 601, 100)),
            "model__max_depth": [3, 4, 5, 6, 8],
            "model__learning_rate": np.logspace(-3, -0.1, 10),
            "model__subsample": [0.6, 0.8, 1.0],
            "model__colsample_bytree": [0.6, 0.8, 1.0]
        })
    except Exception:
        pass

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = []
    best_models = {}

    for name, (base_model, param_grid) in models.items():
        print(f"[{name}] tuning...")
        pipe = Pipeline(steps=[
            ("fe", fe),
            ("pre", preprocessor),
            ("model", base_model)
        ])
        cv = KFold(n_splits=args.cv, shuffle=True, random_state=42)

        try:
            breadth = 0
            for v in param_grid.values():
                try:
                    breadth += len(v)
                except TypeError:
                    breadth += 10
            n_iter = min(40, max(10, breadth))
        except Exception:
            n_iter = 30

        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_grid,
            n_iter=n_iter,
            scoring="neg_mean_absolute_error",
            cv=cv,
            n_jobs=-1,
            random_state=42,
            verbose=0,
            refit=True
        )
        search.fit(X_train, y_train)
        best = search.best_estimator_
        best_models[name] = best

        y_pred = best.predict(X_test)
        m = metrics(y_test, y_pred)
        m.update({"Model": name})
        results.append(m)

        joblib.dump(best, os.path.join(args.outdir, f"best_{name}.joblib"))

    res_df = pd.DataFrame(results).set_index("Model").sort_values("MAE")
    res_csv = os.path.join(args.outdir, "model_comparison.csv")
    res_df.to_csv(res_csv)
    save_table_pdf(res_df, os.path.join(args.outdir, "model_comparison.pdf"),
                   title="Model Comparison (Test Set)")

    overall_best_name = res_df.index[0]
    overall_best = best_models[overall_best_name]
    joblib.dump(overall_best, os.path.join(args.outdir, "best_overall.joblib"))

    print(f"Permutation importance for best: {overall_best_name}")
    perm = permutation_importance(
        overall_best, X_test, y_test, n_repeats=10, random_state=42,
        scoring="neg_mean_absolute_error"
    )

    fe_only = overall_best.named_steps['fe']
    pre_only = overall_best.named_steps['pre']
    X_train_fe = fe_only.transform(X_train.copy())
    pre_only.fit(X_train_fe)
    feat_names = get_feature_names(pre_only, numeric_features, categorical_features)

    importances = pd.DataFrame({
        "feature": feat_names[:len(perm.importances_mean)],
        "importance": perm.importances_mean
    }).sort_values("importance", ascending=False).head(20)
    importances.to_csv(os.path.join(args.outdir, "permutation_importance.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, max(4, int(len(importances)*0.4))))
    ax.barh(importances["feature"][::-1], importances["importance"][::-1])
    ax.set_title(f"Permutation Importance (Top 20) — {overall_best_name}")
    ax.set_xlabel("Decrease in MAE")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "permutation_importance.pdf"))
    plt.close(fig)

    try:
        top_feats = importances["feature"].head(4).tolist()
        fig, axes = plt.subplots(len(top_feats), 1, figsize=(7, 3*len(top_feats)))
        if len(top_feats) == 1:
            axes = [axes]
        for ax, f in zip(axes, top_feats):
            try:
                # Works if estimator supports PDP on pipeline with raw column names
                PartialDependenceDisplay.from_estimator(overall_best, X_test, [f], ax=ax)
                ax.set_title(f"PDP: {f}")
            except Exception:
                ax.text(0.5, 0.5, f"PDP not available for {f}", ha="center")
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "partial_dependence.pdf"))
        plt.close(fig)
    except Exception:
        pass

    train_sizes, train_scores, val_scores = learning_curve(
        overall_best, X, y,
        cv=KFold(n_splits=args.cv, shuffle=True, random_state=42),
        scoring="neg_mean_absolute_error",
        train_sizes=np.linspace(0.2, 1.0, 5),
        n_jobs=-1
    )
    fig, ax = plt.subplots(figsize=(7,5))
    ax.plot(train_sizes, -train_scores.mean(axis=1), marker='o', label="Train MAE")
    ax.plot(train_sizes, -val_scores.mean(axis=1), marker='o', label="CV MAE")
    ax.set_xlabel("Training size")
    ax.set_ylabel("MAE")
    ax.set_title(f"Learning Curve — {overall_best_name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "learning_curve.pdf"))
    plt.close(fig)

    y_pred_best = overall_best.predict(X_test)
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred_best}).to_csv(
        os.path.join(args.outdir, "test_predictions_best.csv"), index=False
    )

    print("Done. Outputs written to:", args.outdir)


if __name__ == "__main__":
    main()
