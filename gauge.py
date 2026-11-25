"""Build nominal UTC clock-hour KDSM gauge targets from complete ASOS minute records."""
import argparse
import io
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

HOSTS = ("mesonet.agron.iastate.edu", "mesonet2.agron.iastate.edu", "mesonet3.agron.iastate.edu")
MINUTE = pd.Timedelta(minutes=1)
HOUR = pd.Timedelta(hours=1)
KDSM = "DSM"


def minute_data(start, end, source):
    if source is not None:
        return source.read_bytes()
    # A target ending at month-start needs the preceding 59 minute records.
    query = urlencode(dict(station=KDSM, vars="precip", what="download", tz="UTC",
                           sts=(start - HOUR + MINUTE).isoformat(),
                           ets=(end + MINUTE).isoformat()))
    for attempt, host in enumerate(HOSTS):
        try:
            with urlopen(f"https://{host}/cgi-bin/request/asos1min.py?{query}", timeout=60) as response:
                raw = response.read()
            if not raw.startswith(b"station,"):
                raise ValueError("IEM did not return CSV data")
            return raw
        except (OSError, ValueError):
            if attempt == len(HOSTS) - 1:
                raise
            time.sleep(attempt + 1)


def hourly_totals(raw, start, end):
    frame = pd.read_csv(io.BytesIO(raw), dtype={"precip": str})
    if not set(frame["station"]).issubset({KDSM}):
        raise ValueError("Unexpected station in ASOS precipitation response")
    stamps = pd.DatetimeIndex(pd.to_datetime(frame["valid(UTC)"], utc=True, errors="raise"))
    if stamps.hasnans or (stamps != stamps.floor("min")).any():
        raise ValueError("Expected exact minute labels in UTC")
    amount = pd.to_numeric(frame["precip"], errors="coerce")
    # Duplicate minute stamps are ambiguous and invalidate the affected hour.
    amount = amount.where(amount.between(0, 999 / 25.4) & ~stamps.duplicated(keep=False))
    observed = pd.Series(amount.to_numpy(), index=stamps).dropna()
    hours = pd.date_range(start, end, freq="h", inclusive="left")
    minute_axis = pd.date_range(hours[0] - HOUR + MINUTE, hours[-1], freq="min")
    minutes = observed.reindex(minute_axis)
    group = minutes.groupby(minutes.index.ceil("h"))
    count = group.count().reindex(hours)
    rain = group.sum(min_count=60).reindex(hours) * 25.4
    rain = rain.where(rain.between(0, 999))
    return pd.DataFrame({"iem_mm": rain, "iem_valid_minutes": count}, index=hours).rename_axis("valid_utc")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("months", nargs="+", help="UTC months YYYYMM")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--minute-dir", type=Path, help="Reuse local IEM UTC exports named minutes_YYYYMM.csv")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    args.data.mkdir(parents=True, exist_ok=True)
    for month in dict.fromkeys(args.months):
        output = args.data / f"gauge_{month}.csv"
        if output.exists() and not args.refresh:
            existing = pd.read_csv(output, nrows=0)
            if "iem_valid_minutes" in existing:
                continue
            raise ValueError(f"{output} is a nominal METAR-hour file. Rebuild it with --refresh.")
        start = pd.to_datetime(month, format="%Y%m", utc=True)
        end = start + pd.offsets.MonthBegin()
        source = args.minute_dir / f"minutes_{month}.csv" if args.minute_dir else None
        raw = minute_data(start, end, source)
        totals = hourly_totals(raw, start, end)
        temporary = output.with_suffix(".csv.tmp")
        totals.to_csv(temporary, float_format="%.5f")
        temporary.replace(output)
        print(f"{output.name}: {totals.iem_mm.notna().sum()}/{len(totals)} complete hours; {len(raw):,} source bytes")


if __name__ == "__main__":
    main()
