"""Audit the cleaned point dataset, UTC keys, and deployed ONNX/metrics identity."""
import argparse
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def utc_column(series):
    for value in series.dropna():
        label = datetime.fromisoformat(value)
        if label.utcoffset() != timedelta(0):
            raise ValueError(f"Expected an explicit UTC timestamp, got {value!r}")
    return pd.to_datetime(series, utc=True, errors="raise")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inspect(data):
    report = {}
    hour = pd.Timedelta(np.timedelta64(1, "h"))
    for path in sorted(data.glob("des_moines_*.csv")):
        month = path.stem.removeprefix("des_moines_")
        hrrr, obs, gauge = (pd.read_csv(data / f"{kind}_{month}.csv")
                            for kind in ("des_moines", "observations", "gauge"))
        start = pd.to_datetime(month, format="%Y%m", utc=True)
        index = pd.date_range(start, start + pd.offsets.MonthBegin(), freq="h", inclusive="left")
        for frame in (hrrr, obs, gauge):
            actual = pd.DatetimeIndex(utc_column(frame["valid_utc"]))
            require(actual.equals(index), f"{month}: missing, duplicate, or unordered UTC hour")
        init = pd.DatetimeIndex(utc_column(hrrr["hrrr_init_utc"]))
        require(((index - init) == 2 * hour).all(), f"{month}: HRRR run/valid mismatch")
        for series in (hrrr.hrrr_mm, hrrr.stage4_mm, obs.mrms_mm, gauge.iem_mm):
            require(series.dropna().between(0, 999).all(), f"{month}: invalid rainfall value")
        minutes = gauge.iem_valid_minutes
        require((minutes.between(0, 60) & minutes.eq(minutes.astype(int))).all(),
                f"{month}: invalid gauge minute counts")
        require(minutes[gauge.iem_mm.notna()].eq(60).all(), f"{month}: incomplete gauge labeled")
        require(np.allclose(gauge.iem_mm.dropna() / 0.254,
                            np.round(gauge.iem_mm.dropna() / 0.254), atol=0.0001, rtol=0),
                f"{month}: airport total incompatible with hundredth-inch minute records")
        metar = pd.DatetimeIndex(utc_column(obs["asos_valid_utc"]))
        require((metar.isna() | ((metar > index - hour) & (metar <= index))).all(),
                f"{month}: airport observations outside history hour")
        publication = pd.DatetimeIndex(utc_column(obs["mrms_published_utc"]))
        radar_ok = (publication >= index) & (publication <= index + hour)
        report[month] = {"utc_rows": len(index), "gauge_labeled_hours": int(gauge.iem_mm.notna().sum()),
                         "radar_valid_archive_timestamp_within_one_hour": int((radar_ok & obs.mrms_mm.notna()).sum())}
    require(bool(report), "No monthly dataset found")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--web", type=Path, default=Path("web"))
    args = parser.parse_args()
    comparison = json.loads((args.web / "comparison.json").read_text())
    digests = {}
    for model in comparison["models"]:
        path = args.web / "models" / model["id"] / "model.onnx"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        require(digest == model["model_sha256"], f"{path}: ONNX file and metrics belong to different models")
        digests[model["id"]] = digest
    print(json.dumps({"monthly_alignment": inspect(args.data), "onnx_sha256": digests}, indent=2))
