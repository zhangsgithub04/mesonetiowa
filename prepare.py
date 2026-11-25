"""Build appendable monthly Des Moines point pairs from NOAA archives."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timedelta, timezone
from pathlib import Path
import shutil
import tarfile
import tempfile
from threading import Lock
import time
from urllib.request import Request, urlopen

import eccodes as ec

POINT = (41.5868, -93.6250)
HOUR = timedelta(hours=1)
SOURCES = (
    "https://noaa-hrrr-bdp-pds.s3.us-east-1.amazonaws.com/",
    "https://noaahrrr.blob.core.windows.net/hrrr/",
)
STAGE = "https://data.gdex.ucar.edu/d507005/stage4"
COLUMNS = ("valid_utc", "hrrr_init_utc", "hrrr_mm", "stage4_mm")
TAG = ":APCP:surface:1-2 hour acc fcst:"
_DECODE_LOCK = Lock()


def timestamp(day, hhmm):
    return datetime.strptime(f"{day:08d}{hhmm:04d}", "%Y%m%d%H%M").replace(tzinfo=timezone.utc)


def grib_time(gid, date, hour):
    return timestamp(ec.codes_get_long(gid, date), ec.codes_get_long(gid, hour))


_indices = {}


def point_mm(gid):
    grid = ec.codes_get(gid, "md5GridSection")
    if grid not in _indices:
        _indices[grid] = ec.codes_grib_find_nearest(gid, *POINT, npoints=1)[0]
    p = _indices[grid]
    if p["distance"] > 10:
        raise ValueError(f"Closest source grid cell is {p['distance']:.1f} km away")
    units = ec.codes_get(gid, "units")
    if units not in ("kg m**-2", "mm"):
        raise ValueError(f"Unexpected precipitation units {units}")
    value = ec.codes_get_double_element(gid, "values", p["index"])
    if not 0 <= value <= 999 or value == ec.codes_get(gid, "missingValue"):
        return None
    return round(value, 5)


def check_surface_precipitation(gid, first_hour, last_hour):
    ids = tuple(ec.codes_get_long(gid, key) for key in
                ("discipline", "parameterCategory", "parameterNumber"))
    steps = tuple(ec.codes_get_long(gid, key) for key in ("startStep", "endStep", "stepUnits"))
    if (ids != (0, 1, 8) or ec.codes_get(gid, "typeOfLevel") != "surface"
            or ec.codes_get(gid, "stepType") != "accum" or steps != (first_hour, last_hour, 1)):
        raise ValueError("Expected a surface total-precipitation accumulation for the requested interval")


def hrrr_at(valid):
    init = valid - 2 * HOUR
    key = f"hrrr.{init:%Y%m%d}/conus/hrrr.t{init:%H}z.wrfsfcf02.grib2"
    for attempt in range(4):
        try:
            url = SOURCES[attempt % len(SOURCES)] + key
            with urlopen(url + ".idx", timeout=9) as source:
                index = source.read()
            lines = index.decode().splitlines()
            i, = [i for i, line in enumerate(lines) if TAG in line]
            start = int(lines[i].split(":")[1])
            end = int(lines[i + 1].split(":")[1]) - 1
            request = Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urlopen(request, timeout=12) as source:
                if source.status != 206:
                    raise ValueError("Archive did not honor byte range")
                raw = source.read()
            if len(raw) != end - start + 1 or not raw.startswith(b"GRIB") or not raw.endswith(b"7777"):
                raise ValueError("Incomplete GRIB message")
            with _DECODE_LOCK:
                gid = ec.codes_new_from_message(raw)
                try:
                    decoded = grib_time(gid, "validityDate", "validityTime")
                    actual_init = grib_time(gid, "dataDate", "dataTime")
                    check_surface_precipitation(gid, 1, 2)
                    if (decoded, actual_init) != (valid, init):
                        raise ValueError("Unexpected HRRR date or accumulation interval")
                    return valid, point_mm(gid), len(raw) + len(index), None
                finally:
                    ec.codes_release(gid)
        except (OSError, ValueError, IndexError, ec.GribInternalError) as exc:
            error = str(exc)
            time.sleep(0.25 * attempt)
    return valid, None, 0, error


def stage_month(archive, month):
    values = {}
    with tarfile.open(archive) as outer:
        for day in outer:
            if not day.isfile():
                continue
            with outer.extractfile(day) as data, tarfile.open(fileobj=data, mode="r|*") as inner:
                for member in inner:
                    name = Path(member.name).name
                    if not member.isfile() or not name.startswith("st4_conus.") or not name.endswith(".01h.grb2"):
                        continue
                    with inner.extractfile(member) as source:
                        raw = source.read()
                    marker = raw.find(b"GRIB")
                    if marker < 0:
                        print(f"Missing Stage IV GRIB: {name}", flush=True)
                        continue
                    raw = raw[marker:]
                    length = int.from_bytes(raw[8:16] if raw[7] == 2 else raw[4:7], "big")
                    raw = raw[:length]
                    if not raw.endswith(b"7777"):
                        print(f"Truncated Stage IV GRIB: {name}", flush=True)
                        continue
                    gid = ec.codes_new_from_message(raw)
                    try:
                        check_surface_precipitation(gid, 0, 1)
                        valid = grib_time(gid, "validityDate", "validityTime")
                        begin = grib_time(gid, "dataDate", "dataTime")
                        if (valid - begin != HOUR or valid.strftime("%Y%m") != month
                                or valid.minute != 0 or valid in values):
                            raise ValueError(f"Unexpected Stage IV interval or duplicate {name}")
                        values[valid] = point_mm(gid)
                    finally:
                        ec.codes_release(gid)
    if not values:
        raise ValueError(f"No Stage IV hourly CONUS messages in {archive}")
    return values


def process(month, output, archives, workers):
    start = datetime.strptime(month, "%Y%m").replace(tzinfo=timezone.utc)
    end = (start + timedelta(days=32)).replace(day=1)
    times = [start + i * HOUR for i in range((end - start) // HOUR)]
    with tempfile.TemporaryDirectory() as tmp:
        archive = (archives or Path(tmp)) / f"stage4.{month}.tar"
        if not archive.exists():
            if archives:
                raise FileNotFoundError(archive)
            with urlopen(f"{STAGE}/{archive.name}", timeout=90) as response, archive.open("wb") as dest:
                shutil.copyfileobj(response, dest, 1024 * 1024)
        stage = stage_month(archive, month)
        stage_bytes = archive.stat().st_size
    print(f"{month} Stage IV: {len(stage)} hourly fields, {stage_bytes:,} archive bytes", flush=True)
    target = output / f"des_moines_{month}.csv"
    failures = []
    with tempfile.NamedTemporaryFile(mode="w", newline="", dir=output, delete=False) as file:
        temporary = Path(file.name)
        try:
            writer = csv.writer(file)
            writer.writerow(COLUMNS)
            total_bytes = paired = 0
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for valid, rain, nbytes, error in pool.map(hrrr_at, times):
                    if error:
                        failures.append((valid.isoformat(), error))
                    reference = stage.get(valid)
                    total_bytes += nbytes
                    paired += rain is not None and reference is not None
                    writer.writerow((valid.isoformat(), (valid - 2 * HOUR).isoformat(), rain, reference))
            file.flush()
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    print(f"{month} paired={paired}/{len(times)} hrrr_bytes={total_bytes:,} errors={len(failures)}", flush=True)
    if failures:
        print("First failed HRRR fields:", failures[:3], flush=True)
    return len(failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("months", nargs="+", help="UTC months in YYYYMM format")
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--archives", type=Path, help="Directory with pre-downloaded Stage IV monthly tar files")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--refresh", action="store_true", help="Fetch and replace existing monthly CSVs")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    failures = 0
    for month in dict.fromkeys(args.months):
        if args.refresh or not (args.output / f"des_moines_{month}.csv").exists():
            failures += process(month, args.output, args.archives, args.workers)
    if failures:
        raise SystemExit("Some HRRR fields were unavailable. Blank cells were retained. Rerun with --refresh to retry.")


if __name__ == "__main__":
    main()
