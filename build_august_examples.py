"""Build all August 2025 Chicago-time dropdown entries using the archived inputs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import eccodes as ec
import pandas as pd

from observations import airport_hours, fetch_mrms
from prepare import check_surface_precipitation, grib_time, hrrr_at, point_mm

TZ = "America/Chicago"
HOUR = timedelta(hours=1)
BEGIN = pd.Timestamp("2025-08-01", tz=TZ).tz_convert("UTC")
END = pd.Timestamp("2025-09-01", tz=TZ).tz_convert("UTC")
HISTORY_FIELDS = ("mrms_mm", "historical_hrrr_mm", "temp_c", "dew_c", "u_ms", "v_ms")
LIMITS = [(0, 999), (0, 999), (-70, 60), (-90, 60), (-100, 100), (-100, 100)]


def load_month(data, month):
    def read(prefix):
        frame = pd.read_csv(data / f"{prefix}_{month}.csv")
        frame["valid_utc"] = pd.to_datetime(frame["valid_utc"], utc=True)
        return frame.set_index("valid_utc", verify_integrity=True)
    return read("des_moines").join(read("observations")).join(read("gauge"))


def boundary(data, stage_cache):
    """Fetch the six UTC hours needed to finish August 31 in Chicago."""
    cache = data / "august_boundary_2025.csv"
    if cache.exists():
        return pd.read_csv(cache).assign(
            valid_utc=lambda f: pd.to_datetime(f.valid_utc, utc=True)).set_index("valid_utc", verify_integrity=True)
    valid = pd.date_range(END.normalize(), END, freq="h", name="valid_utc")

    def stage4(t):
        filename = f"ST4.{t:%Y%m%d%H}.01h.grib"
        local = stage_cache / filename if stage_cache else None
        if local is not None and local.is_file():
            raw = local.read_bytes()
        else:
            url = f"https://mesonet.agron.iastate.edu/archive/data/{t:%Y/%m/%d}/stage4/{filename}"
            with urlopen(url, timeout=30) as response:
                raw = response.read()
        gid = ec.codes_new_from_message(raw)
        try:
            check_surface_precipitation(gid, 0, 1)
            if grib_time(gid, "validityDate", "validityTime") != t:
                raise ValueError(f"Wrong valid time in {filename}")
            return point_mm(gid)
        finally:
            ec.codes_release(gid)

    with ThreadPoolExecutor(max_workers=12) as pool:
        forecasts = pool.map(hrrr_at, valid)
        radar = pool.map(fetch_mrms, valid)
        airport = pool.submit(airport_hours, valid[0], valid[-1] + HOUR, None)
        stage = [stage4(t) for t in valid]
        df = pd.DataFrame({"hrrr_mm": [r[1] for r in forecasts], "stage4_mm": stage}, index=valid)
        df["hrrr_init_utc"] = valid - 2 * HOUR
        df = df.join(pd.DataFrame([(r[0], r[1]) for r in radar], index=valid,
                                  columns=["mrms_mm", "mrms_published_utc"]))
        df = df.join(airport.result()[0])
    # Retrieve any next-month UTC minutes that still belong to local August.
    from urllib.parse import urlencode
    import io
    query = urlencode({"station": "DSM", "vars": "precip", "tz": "UTC", "what": "download",
                       "sts": (valid[0]-HOUR+timedelta(minutes=1)).isoformat(),
                       "ets": (valid[-1]+timedelta(minutes=1)).isoformat()})
    with urlopen("https://mesonet.agron.iastate.edu/cgi-bin/request/asos1min.py?" + query, timeout=30) as response:
        raw = pd.read_csv(io.BytesIO(response.read()))
    if not set(raw.station).issubset({"DSM"}):
        raise ValueError("Unexpected airport in window-boundary data")
    stamp = pd.DatetimeIndex(pd.to_datetime(raw["valid(UTC)"], utc=True))
    value = pd.to_numeric(raw.precip, errors="coerce")
    value = value.where(~stamp.duplicated(keep=False) & value.ge(0))
    mins = pd.Series(value.to_numpy(), index=stamp).dropna()
    minutes = pd.date_range(valid[0]-HOUR+timedelta(minutes=1), valid[-1], freq="min")
    mins = mins.reindex(minutes)
    groups = mins.groupby(mins.index.ceil("h"))
    df["iem_mm"] = groups.sum(min_count=60) * 25.4
    df["iem_valid_minutes"] = groups.count()
    df.to_csv(cache)
    return df


def optional_hour(frame, t):
    if t not in frame.index:
        return [None] * len(HISTORY_FIELDS)
    row = frame.loc[t]
    published = row["mrms_published_utc"]
    asos_time = row["asos_valid_utc"]
    mrms_ok = pd.notna(published) and t <= published <= t + HOUR
    asos_ok = pd.notna(asos_time) and t - HOUR < asos_time <= t
    values = [row.mrms_mm if mrms_ok else None, row.hrrr_mm,
              *[row[k] if asos_ok else None for k in HISTORY_FIELDS[2:]]]
    return [round(float(v), 5) if pd.notna(v) and lo <= v <= hi else None
            for v, (lo, hi) in zip(values, LIMITS)]


def items(frame, keys, test_keys):
    result = []
    for valid in keys:
        row = frame.loc[valid]
        history = [optional_hour(frame, valid - i * HOUR) for i in range(7, 1, -1)]
        target = round(float(row.hrrr_mm), 5) if pd.notna(row.hrrr_mm) else None
        reference = [round(float(row[k]), 5) if pd.notna(row[k]) else None for k in ("stage4_mm", "iem_mm")]
        if row.iem_valid_minutes != 60:
            reference[1] = None
        result.append({"valid_utc": valid.isoformat(), "history": history, "hrrr_mm": target,
                       "reference_mm": reference, "benchmark": valid in test_keys})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--web", type=Path, default=Path("web"))
    parser.add_argument("--stage4-cache", type=Path, help="Optional directory with September boundary GRIBs")
    args = parser.parse_args()
    july = load_month(args.data, "202507").tail(8)
    august = load_month(args.data, "202508")
    # The August archive already includes September 1 00 UTC if provided by a source.
    future = boundary(args.data, args.stage4_cache)
    future = future.loc[~future.index.isin(august.index)]
    frame = pd.concat([july, august, future])
    for column in ("hrrr_init_utc", "mrms_published_utc", "asos_valid_utc"):
        # Concatenated inputs contain both offsets and ISO separators.
        frame[column] = pd.to_datetime(frame[column], utc=True, format="mixed", errors="coerce")
    frame["hrrr_mm"] = frame["hrrr_mm"].where(
        (frame.index - frame.hrrr_init_utc == 2 * HOUR) & frame.hrrr_mm.between(0, 999))
    report = json.loads((args.web / "comparison.json").read_text())
    utc_august = pd.Timestamp("2025-08-01", tz="UTC")
    keys = pd.date_range(utc_august + 7 * HOUR, utc_august + timedelta(days=31), freq="h", inclusive="left")
    prelim = items(frame, keys, set())
    score_keys = [t for t, case in zip(keys, prelim) if case["hrrr_mm"] is not None
                  and all(v is not None for v in case["reference_mm"])
                  and all(all(v is not None for v in row[:2]) for row in case["history"])]
    digest = hashlib.sha256("\n".join(t.isoformat() for t in score_keys).encode()).hexdigest()
    if digest != report["test_hours_sha256"]:
        raise ValueError("Reconstructed benchmark membership differs from the model report")
    full = items(frame, pd.date_range(BEGIN + HOUR, END, freq="h"), set(score_keys))
    (args.web / "august-hours.json").write_text(json.dumps(
        {"timezone": TZ, "hours": full}, separators=(",", ":"), allow_nan=False) + "\n")
    ready = sum(e["hrrr_mm"] is not None and all(r[0] is not None for r in e["history"]) for e in full)
    gauge = sum(e["reference_mm"][1] is not None for e in full)
    print(f"{len(full)} Chicago hours, {ready} inference-ready, {gauge} gauge references, "
          f"{sum(e['benchmark'] for e in full)} benchmark hours")


if __name__ == "__main__":
    main()
