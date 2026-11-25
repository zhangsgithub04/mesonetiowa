"""Append monthly point observations to the existing HRRR/Stage IV pilot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from email.utils import parsedate_to_datetime
import gzip
import io
from pathlib import Path
from threading import Lock
import time
from urllib.parse import urlencode
from urllib.request import urlopen

import eccodes as ec
import numpy as np
import pandas as pd

from prepare import POINT, grib_time

MRMS = "https://noaa-mrms-pds.s3.amazonaws.com"
ASOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
PRODUCT = "RadarOnly_QPE_01H_00.00"
_grid_indices = {}
_decode_lock = Lock()


def fetch_mrms(stamp):
    key = f"CONUS/{PRODUCT}/{stamp:%Y%m%d}/MRMS_{PRODUCT}_{stamp:%Y%m%d-%H}0000.grib2.gz"
    for attempt in range(3):
        try:
            with urlopen(f"{MRMS}/{key}", timeout=14) as response:
                blob = response.read()
                published = parsedate_to_datetime(response.headers["Last-Modified"])
            with _decode_lock:
                gid = ec.codes_new_from_message(gzip.decompress(blob))
                try:
                    if grib_time(gid, "validityDate", "validityTime") != stamp:
                        raise ValueError(f"MRMS valid time mismatch for {key}")
                    codes = tuple(ec.codes_get_long(gid, k) for k in ("discipline", "parameterCategory", "parameterNumber"))
                    if codes != (209, 6, 2):
                        raise ValueError(f"Unexpected MRMS parameter {codes} for {key}")
                    grid = ec.codes_get(gid, "md5GridSection")
                    if grid not in _grid_indices:
                        point = ec.codes_grib_find_nearest(gid, *POINT, npoints=1)[0]
                        if point["distance"] > 2:
                            raise ValueError("Closest radar cell exceeds 2 km")
                        _grid_indices[grid] = point["index"]
                    rain = ec.codes_get_double_element(gid, "values", _grid_indices[grid])
                    return rain if 0 <= rain <= 999 else np.nan, published.isoformat(), len(blob)
                finally:
                    ec.codes_release(gid)
        except (OSError, ValueError, ec.GribInternalError) as exc:
            error = str(exc)
            time.sleep(0.5 * attempt)
    print(f"{stamp}: MRMS unavailable: {error}", flush=True)
    return np.nan, None, 0


def airport_hours(start, end, local_csv):
    dates = {}
    for i, date in enumerate((start - timedelta(days=1), end), 1):
        dates.update({f"year{i}": date.year, f"month{i}": date.month, f"day{i}": date.day})
    query = urlencode(dict(station="DSM", data=["tmpf", "dwpf", "drct", "sknt"], tz="Etc/UTC",
                           format="onlycomma", missing="M", report_type=3, **dates), doseq=True)
    if local_csv:
        raw = local_csv.read_bytes()
    else:
        with urlopen(f"{ASOS}?{query}", timeout=40) as response:
            raw = response.read()
    data = pd.read_csv(io.BytesIO(raw), na_values=["M"])
    data = data.loc[data["station"].isin(["DSM", "KDSM"])].copy()
    if data.empty:
        raise ValueError("No Des Moines airport observations in IEM response")
    data["asos_valid_utc"] = pd.to_datetime(data.pop("valid"), utc=True)
    data["valid_utc"] = data["asos_valid_utc"].dt.ceil("h")
    data = data.sort_values("asos_valid_utc").drop_duplicates("valid_utc", keep="last")
    data["temp_c"] = (data.pop("tmpf") - 32) * 5 / 9
    data["dew_c"] = (data.pop("dwpf") - 32) * 5 / 9
    speed = data.pop("sknt") * 0.514444
    direction = np.deg2rad(data.pop("drct"))
    data["u_ms"] = (-speed * np.sin(direction)).mask(speed == 0, 0)
    data["v_ms"] = (-speed * np.cos(direction)).mask(speed == 0, 0)
    return data.drop(columns="station").set_index("valid_utc"), len(raw)


def process(path, workers, local_csv):
    month = path.stem.removeprefix("des_moines_")
    start = pd.to_datetime(month, format="%Y%m", utc=True)
    end = start + pd.offsets.MonthBegin()
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    asos, asos_bytes = airport_hours(hours[0], hours[-1] + timedelta(hours=1), local_csv)
    total_bytes = 0
    radar = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, result in enumerate(pool.map(fetch_mrms, hours), 1):
            radar.append(result[:2])
            total_bytes += result[2]
            if i % 120 == 0:
                print(f"{path.name}: {i}/{len(hours)} radar hours", flush=True)
    frame = pd.DataFrame(radar, index=hours, columns=["mrms_mm", "mrms_published_utc"]).join(asos)
    frame.index.name = "valid_utc"
    output = path.with_name(path.name.replace("des_moines_", "observations_"))
    temporary = output.with_suffix(".csv.tmp")
    frame.to_csv(temporary, float_format="%.5f")
    temporary.replace(output)
    print(f"{output.name}: MRMS {frame.mrms_mm.notna().sum()}/{len(hours)}, {total_bytes:,} MRMS bytes, {asos_bytes:,} ASOS bytes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("months", nargs="+", help="UTC months YYYYMM")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--asos-csv", type=Path, help="Reuse a local IEM CSV with tmpf,dwpf,drct,sknt and UTC valid columns")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    for month in dict.fromkeys(args.months):
        if args.refresh or not (args.data / f"observations_{month}.csv").exists():
            process(args.data / f"des_moines_{month}.csv", args.workers, args.asos_csv)
