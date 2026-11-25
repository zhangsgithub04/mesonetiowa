"""Compare the nearest native grid point with bilinear interpolation for a sample field."""
import argparse
import gzip
import json
import math
from pathlib import Path

import eccodes as ec
from pyproj import Transformer

from prepare import POINT, check_surface_precipitation, grib_time

AIRPORT = (41.53395, -93.65311)


def coord(gid, name):
    return ec.codes_get_double(gid, name)


def fractional_index(gid, latitude, longitude):
    grid = ec.codes_get(gid, "gridType")
    first_lat = coord(gid, "latitudeOfFirstGridPointInDegrees")
    first_lon = coord(gid, "longitudeOfFirstGridPointInDegrees")
    if grid == "regular_ll":
        delta_lon = (longitude - first_lon + 180) % 360 - 180
        delta_lat = latitude - first_lat
        dx = coord(gid, "iDirectionIncrementInDegrees")
        dy = coord(gid, "jDirectionIncrementInDegrees")
    elif grid in ("lambert", "polar_stereographic"):
        projection = Transformer.from_crs("EPSG:4326", ec.codes_get(gid, "projTargetString"), always_xy=True)
        x0, y0 = projection.transform(first_lon, first_lat)
        x, y = projection.transform(longitude, latitude)
        delta_lon, delta_lat = x - x0, y - y0
        dx = coord(gid, "DxInMetres")
        dy = coord(gid, "DyInMetres")
    else:
        raise ValueError(f"Unsupported audit grid: {grid}")
    return (delta_lon / (-dx if ec.codes_get_long(gid, "iScansNegatively") else dx),
            delta_lat / (dy if ec.codes_get_long(gid, "jScansPositively") else -dy))


def sample(gid, location):
    """Read four enclosing cells in the native coordinate system."""
    if any(ec.codes_get_long(gid, key) for key in ("jPointsAreConsecutive", "alternativeRowScanning")):
        raise ValueError("This audit supports regular row-major grid scans")
    i, j = fractional_index(gid, *location)
    nx, ny = (ec.codes_get_long(gid, dim) for dim in ("Nx", "Ny"))
    ix, iy = math.floor(i), math.floor(j)
    if not (0 <= ix < nx-1 and 0 <= iy < ny-1):
        raise ValueError("Point lacks four enclosing grid cells")
    a, b = i-ix, j-iy
    ll = iy * nx + ix
    idxs = [ll, ll + 1, ll + nx, ll + nx + 1]
    weights = [(1-a)*(1-b), a*(1-b), (1-a)*b, a*b]
    values = ec.codes_get_double_elements(gid, "values", idxs)
    if not all(0 <= v <= 999 for v in values):
        raise ValueError("Invalid rainfall among bilinear corners")
    nearest = ec.codes_grib_find_nearest(gid, *location, npoints=1)[0]
    if nearest["index"] not in idxs:
        raise ValueError("Projected enclosing cell disagrees with ecCodes nearest point")
    return {
        "lat": round(nearest["lat"], 7),
        "lon": round((nearest["lon"] + 180) % 360 - 180, 7),
        "distance_km": round(nearest["distance"], 4),
        "nearest_mm": round(nearest["value"], 5),
        "bilinear_mm": round(sum(w * v for w, v in zip(weights, values)), 5),
    }


def audit(files):
    result = {"study": {"lat": POINT[0], "lon": POINT[1]}, "airport": {"lat": AIRPORT[0], "lon": AIRPORT[1]},
              "valid_utc": None, "fields": {}}
    for key, path in files.items():
        bytes_ = path.read_bytes()
        gid = ec.codes_new_from_message(gzip.decompress(bytes_) if bytes_.startswith(b"\x1f\x8b") else bytes_)
        try:
            if key in ("hrrr", "stage4"):
                steps = (1, 2) if key == "hrrr" else (0, 1)
                check_surface_precipitation(gid, *steps)
            else:
                code = tuple(ec.codes_get_long(gid, name) for name in
                             ("discipline", "parameterCategory", "parameterNumber"))
                if code != (209, 6, 2):
                    raise ValueError("Expected MRMS radar-only one-hour precipitation")
            valid = grib_time(gid, "validityDate", "validityTime").isoformat()
            if result["valid_utc"] not in (None, valid):
                raise ValueError("Grids must share the same valid UTC")
            result["valid_utc"] = valid
            result["fields"][key] = {"grid": ec.codes_get(gid, "gridType"),
                                     "study": sample(gid, POINT), "airport": sample(gid, AIRPORT)}
        finally:
            ec.codes_release(gid)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for product in ("hrrr", "stage4", "mrms"):
        parser.add_argument(product, type=Path, help="Local GRIB2 message, optionally gzipped")
    args = parser.parse_args()
    print(json.dumps(audit(vars(args)), indent=2))
