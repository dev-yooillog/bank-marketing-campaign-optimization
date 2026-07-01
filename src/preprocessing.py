import pandas as pd
import numpy as np
import yaml
from pathlib import Path


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_raw_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 타겟 인코딩
    df["deposit"] = df["deposit"].map({"yes": 1, "no": 0})

    # pdays: -1은 이전 접촉 없음 → 별도 처리
    df["pdays"] = df["pdays"].replace(-1, np.nan)

    # 중복 제거
    df = df.drop_duplicates()

    # 결측치 
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if not missing.empty:
        print("[INFO] 결측치 컬럼:")
        print(missing)

    return df


def save_processed(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"저장: {path} | shape: {df.shape}")


if __name__ == "__main__":
    cfg = load_config()
    df = load_raw_data(cfg["data"]["raw_path"])
    df = clean_data(df)
    save_processed(df, cfg["data"]["processed_path"])