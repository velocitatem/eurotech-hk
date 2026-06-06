from pathlib import Path

import pandas as pd

_RAW = Path("ml/data/raw/查詢2024-03-06至2026-06-06中原城市領先指數.xlsx")


def tidy_cci(raw_path: Path = _RAW) -> pd.DataFrame:
    df = pd.read_excel(raw_path, sheet_name=0)
    dates = df["日期"].str.split(" - ", expand=True)
    df["week_start"] = pd.to_datetime(dates[0], format="%Y/%m/%d")
    df["week_end"] = pd.to_datetime(dates[1], format="%Y/%m/%d")
    return (
        df.rename(columns={"中原城市領先指數": "cci"})[["week_start", "week_end", "cci"]]
        .sort_values("week_start")
        .reset_index(drop=True)
    )
