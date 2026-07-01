import pandas as pd
import numpy as np

# campaign(접촉 횟수)를 구간화
def add_contact_bucket(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    bins = [0, 1, 2, 3, 5, 100]
    labels = ["1회", "2회", "3회", "4-5회", "6회+"]
    df["campaign_bucket"] = pd.cut(
        df["campaign"], bins=bins, labels=labels, right=True
    )
    return df

# 이전 캠페인 성공 여부 플래그
def add_prev_success(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["prev_success"] = (df["poutcome"] == "success").astype(int)
    return df

# 이전 접촉 여부 플래그 (pdays가 NaN이면 미접촉)
def add_pdays_contacted(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["was_contacted_before"] = df["pdays"].notna().astype(int)
    df["pdays"] = df["pdays"].fillna(-1) 
    return df

# 잔액 기반 고객 세그먼트
def add_balance_segment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["balance_segment"] = pd.cut(
        df["balance"],
        bins=[-np.inf, 0, 500, 2000, np.inf],
        labels=["negative", "low", "mid", "high"]
    )
    return df


def encode_categoricals(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    df = df.copy()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df


def run_feature_engineering(df: pd.DataFrame, cat_cols: list) -> pd.DataFrame:
    df = add_contact_bucket(df)
    df = add_prev_success(df)
    df = add_pdays_contacted(df)
    df = add_balance_segment(df)

    extended_cats = cat_cols + ["campaign_bucket", "balance_segment"]
    df = encode_categoricals(df, extended_cats)

    return df