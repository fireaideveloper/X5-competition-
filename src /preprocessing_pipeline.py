import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

TRAIN_PATH = Path(r"C:\Users\Lenovo\Desktop\hacaton x5\ML Demo\train.csv")
TARGET = "РТО"
ID_COL = "new_id"
MONTH_COL = "Месяц"
RANDOM_STATE = 2026

CAT_FEATURES = [
    "Дата открытия, категориальный",
    "Торговая площадь, категориальный",
    "Населенный пункт",
    "Регион"
]

STATIC_NUMERIC = [
    "Численность населения", "Количество домохозяйств",
    "Трафик пеший, в час", "Трафик авто, в час",
    "Маркетплейсы, доставки, постаматы (100 м)",
    "Медицинские уч. и аптеки (300 м)", "Школы (300 м)",
    "Остановки (300 м)", "Продуктовые магазины (500 м)",
    "Пятерочки (500 м)", "Количество касс", "Флаг алкогольной лицензии"
]

def load_data():
    return pd.read_csv(TRAIN_PATH)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates(subset=[ID_COL, MONTH_COL])
    print(f"Дубликатов удалено, размер: {df.shape}")

    static_cols = STATIC_NUMERIC + CAT_FEATURES
    for col in static_cols:
        df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
        mode_map = df.groupby(ID_COL)[col].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
        df[col] = df[ID_COL].map(mode_map)

    initial_len = len(df)
    df = df[df[TARGET] > 0]
    if len(df) < initial_len:
        print(f"Удалено {initial_len - len(df)} строк с РТО <= 0")

    non_negative = STATIC_NUMERIC.copy()
    for col in non_negative:
        if col in df.columns:
            df[col] = df[col].clip(lower=0)

    for col in STATIC_NUMERIC:
        if df[col].nunique() <= 1:
            df.drop(columns=[col], inplace=True)
            print(f"Удален постоянный признак: {col}")

    for col in CAT_FEATURES:
        if col not in df.columns:
            continue
        counts = df.groupby(ID_COL)[col].first().value_counts()
        rare = counts[counts < 10].index.tolist()
        if rare:
            df[col] = df[col].replace(rare, 'Other')
            print(f"Категории {rare} в признаке {col} заменены на 'Other'")

    def winsorize_series(x):
        lower = x.quantile(0.01)
        upper = x.quantile(0.99)
        return x.clip(lower=lower, upper=upper)
    df[TARGET] = df.groupby(ID_COL)[TARGET].transform(winsorize_series)
    print("Винзоризация РТО выполнена")

    def fix_spikes(group):
        group = group.sort_values(MONTH_COL)
        rto = group[TARGET].values.astype(float)
        for i in range(1, len(rto)):
            prev = rto[i-1]
            curr = rto[i]
            if prev > 0:
                pct = (curr - prev) / prev
                if pct > 2.0 or pct < -0.8:
                    if i+1 < len(rto) and rto[i+1] > 0:
                        rto[i] = (prev + rto[i+1]) / 2
                    else:
                        rto[i] = prev
        group[TARGET] = rto
        return group
    df = df.groupby(ID_COL, group_keys=False).apply(fix_spikes)
    print("Скачки исправлены")

    df[TARGET] = df[TARGET].clip(upper=500_000_000)

    num_cols = [c for c in STATIC_NUMERIC if c in df.columns]
    if len(num_cols) > 1:
        corr = df[num_cols].corr().abs()
        upper_tri = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > 0.95)]
        if to_drop:
            df.drop(columns=to_drop, inplace=True)
            print(f"Удалены сильно скоррелированные признаки: {to_drop}")

    feature_for_if = [c for c in num_cols if c in df.columns]
    if feature_for_if:
        X_if = df[feature_for_if].dropna()
        if len(X_if) > 50:
            iso = IsolationForest(contamination=0.001, random_state=RANDOM_STATE, n_jobs=-1)
            preds = iso.fit_predict(X_if)
            mask = preds == 1
            df = df.loc[X_if.index[mask]]
            print(f"Удалено {len(preds) - mask.sum()} глобальных выбросов Isolation Forest")

    for col in df.columns:
        if col in [ID_COL, MONTH_COL, TARGET]:
            continue
        if df[col].dtype in [np.float64, np.int64, float, int]:
            df[col] = df[col].fillna(df[col].median())
        else:
            if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode().iloc[0])
            else:
                df[col] = df[col].fillna("Unknown")
    if df[TARGET].isna().any():
        df[TARGET] = df[TARGET].fillna(df[TARGET].median())
    return df

def generate_features(df):
    df = df.sort_values([ID_COL, MONTH_COL]).copy()
    for lag in range(1, 10):
        df[f"rto_lag_{lag}"] = df.groupby(ID_COL)[TARGET].shift(lag)
    for w in [3, 6, 9]:
        df[f"rto_roll_mean_{w}"] = (
            df.groupby(ID_COL)[TARGET].shift(1)
              .rolling(window=w, min_periods=1).mean()
        )
    df["rto_trend_3m"] = df.groupby(ID_COL)[TARGET].transform(
        lambda x: x.rolling(3, min_periods=1).mean() - x.shift(3).rolling(3, min_periods=1).mean()
    )
    df["rto_trend_6m"] = df.groupby(ID_COL)[TARGET].transform(
        lambda x: x.rolling(6, min_periods=1).mean() - x.shift(6).rolling(6, min_periods=1).mean()
    )
    df["rto_growth_1m"] = df.groupby(ID_COL)[TARGET].pct_change(1)
    df["month_sin"] = np.sin(2 * np.pi * df[MONTH_COL] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df[MONTH_COL] / 12)
    df["months_history"] = df.groupby(ID_COL).cumcount() + 1
    if "Продуктовые магазины (500 м)" in df.columns and "Пятерочки (500 м)" in df.columns:
        df["competitors"] = df["Продуктовые магазины (500 м)"] - df["Пятерочки (500 м)"]
    if "Численность населения" in df.columns and "Количество домохозяйств" in df.columns:
        df["pop_per_household"] = df["Численность населения"] / (df["Количество домохозяйств"] + 1)
    return df

def remove_highly_correlated(df, threshold=0.95):
    num_cols = df.select_dtypes(include=[np.number]).columns
    num_cols = [c for c in num_cols if c not in [TARGET, ID_COL, MONTH_COL]]
    if len(num_cols) < 2:
        return df
    corr_matrix = df[num_cols].corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
    if to_drop:
        df.drop(columns=to_drop, inplace=True)
        print(f"Удалены сильно скоррелированные признаки после генерации: {to_drop}")
    return df

def onehot_encode_features(df, cat_cols, drop='first', sparse_output=False):
    df = df.copy()
    existing = [c for c in cat_cols if c in df.columns]
    if not existing:
        return df
    enc = OneHotEncoder(drop=drop, sparse_output=sparse_output, handle_unknown='ignore')
    encoded = enc.fit_transform(df[existing])
    encoded_df = pd.DataFrame(
        encoded.toarray() if hasattr(encoded, 'toarray') else encoded,
        columns=enc.get_feature_names_out(existing),
        index=df.index
    )
    df = df.drop(columns=existing).join(encoded_df)
    return df

if __name__ == "__main__":
    print("Загрузка исходных данных...")
    raw_data = load_data()
    print("\nОчистка данных")
    cleaned_data = clean_data(raw_data)
    output_dir = TRAIN_PATH.parent
    cleaned_path = output_dir / "cleaned_data.csv"
    cleaned_data.to_csv(cleaned_path, index=False)
    print(f"Очищенные данные сохранены: {cleaned_path}")

    print("\nГенерация признаков")
    featured_data = generate_features(cleaned_data)
    featured_data = remove_highly_correlated(featured_data)

    cat_to_encode = [
        "Дата открытия, категориальный",
        "Торговая площадь, категориальный",
        "Населенный пункт",
        "Регион"
    ]
    featured_data = onehot_encode_features(featured_data, cat_to_encode)

    for col in featured_data.columns:
        if col in [ID_COL, MONTH_COL, TARGET]:
            continue
        if featured_data[col].dtype in [np.float64, np.int64, float, int]:
            featured_data[col] = featured_data[col].fillna(featured_data[col].median())
        else:
            if not featured_data[col].mode().empty:
                featured_data[col] = featured_data[col].fillna(featured_data[col].mode().iloc[0])
            else:
                featured_data[col] = featured_data[col].fillna(0)

    featured_path = output_dir / "cleaned_features_data.csv"
    featured_data.to_csv(featured_path, index=False)
    print(f"Данные с добавленными признаками и кодированием сохранены: {featured_path}")
