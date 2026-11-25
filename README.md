# Des Moines rainfall forecast correction · HRRR history comparison

This clock-hour release estimates two signed errors for a one-hour HRRR rainfall forecast for the grid cells nearest 41.5868° N, 93.6250° W. It includes real point data for March–August 2025, the trained ONNX model, the feature specification, a static browser UI, and scripts for adding months and retraining.

## Paired feature comparison

Both models use six hours of MRMS rainfall and KDSM temperature, dew point, and eastward and northward wind. Both also receive the target-hour HRRR forecast. The full model includes two additional history channels, the matched HRRR forecast for each past hour and MRMS minus that past forecast. The reduced model excludes both channels. The two added channels are algebraically related once MRMS rainfall is known, so the comparison tests their joint contribution.

Both runs use the same eligible timestamps, two-layer LSTM design, training and validation periods, holdout, target definitions, loss, and random seed. The input layer sizes and learned parameters differ. Each checkpoint is selected separately on July validation data. The holdout contains 496 paired August hours, including 15 Stage IV wet hours and 17 gauge wet hours. A common baseline is calculated once per target definition. The comparison script checks the exact held-out timestamp digest, the baseline metrics, ONNX hashes, and shared example values.

| Source | Spatial support | Time and units |
|---|---|---|
| HRRR | Nearest point on the native Lambert grid to the Des Moines study coordinate | Surface total precipitation accumulated during forecast hours 1–2. The raw GRIB uses kg/m² of water, numerically equivalent to mm. |
| Stage IV | Nearest point on the native polar stereographic grid to the same study coordinate | One-hour accumulation ending at the join time. Same liquid-water conversion as HRRR. |
| MRMS | Nearest point on the native longitude/latitude grid to the same study coordinate | `RadarOnly_QPE_01H` in mm. The product represents the trailing hour even though the raw GRIB's generic step type can read as instantaneous. The extractor checks the MRMS product identity and valid timestamp. |
| KDSM METAR temperature, dew point, wind | Airport station at 41.53395° N, 93.65311° W | Latest routine report within each input hour. °F is converted to °C and knots to m/s. Wind components describe movement toward east and north. |
| KDSM gauge | Same airport station, about 6.3 km from the study coordinate | Sum of 60 valid minute-ending precipitation records, converted from inches to mm. The ASOS reporting offset is about 23 seconds. |

All join keys use explicit UTC hour-ending timestamps. Chicago time is display-only and follows daylight saving time. The historical discrepancy is computed from MRMS and HRRR for the same interval. The browser now asks for those two amounts separately and computes their difference. Each model ships a feature specification with its own ONNX feature order. Future raw HRRR and Stage IV extractions explicitly check the GRIB parameter, surface level, accumulation type, and lead interval.

The spatial audit uses the four enclosing native grid points and interpolates in native grid coordinates. For 14:00–15:00 UTC on August 7, Stage IV near the study coordinate is 11.375 mm by nearest-point sampling and 11.523 mm by bilinear interpolation. At the airport, those values are 11.125 and 10.772 mm, respectively, versus the airport gauge's 6.350 mm. These are event-specific comparisons. Interpolation changes spatial sampling and cannot turn a grid estimate into a point measurement. A fully collocated or interpolated model experiment would require re-extracting the entire dataset and retraining.

To verify the shipped data and model identity, run `python verify_alignment.py`. Its report is saved in `web/dataset-audit.json`. To reproduce the single-hour spatial audit, install the optional `pyproj` package and run `python spatial_audit.py HRRR.grib2 STAGE_IV.grib2 MRMS.grib2.gz` with the three archived fields ending at 15:00 UTC on August 7. The result is in `web/spatial-audit.json`.

[ecCodes nearest-point documentation](https://sites.ecmwf.int/docs/eccodes/interfaceeccodes_1_1codes__grib__find__nearest.html) · [KDSM station coordinates](https://mesonet.agron.iastate.edu/sites/site.php?network=IA_ASOS&station=DSM)

## Run inference

From the extracted project folder, run:

```bash
python -m http.server 8000 --directory web
```

Open [http://localhost:8000](http://localhost:8000). Choose an August example or edit the input grid. The UI runs both models on the same input and shows two corrections per model, one per reference. Leaving a past HRRR input blank enables inference with the reduced model. Each corrected amount is raw HRRR plus its predicted correction, clipped at zero. The held-out examples show both reference totals. Switch the reference in the benchmark table to see the corresponding scores. The included ONNX Runtime Web runs the model locally. This UI accepts manual input and bundled examples.

## How each sample is aligned

Let **T** be the UTC end of the target hour. The modeled issue time is **T−1 hour**. The target HRRR forecast was initialized at **T−2 hours**, and its forecast-hour 1–2 accumulation covers **T−1 through T**. The shared LSTM has two recurrent layers with 24 hidden units each and 0.15 dropout between layers during training. The final state and target HRRR value feed a 16-unit prediction layer with two outputs. The two training labels are **Stage IV minus target HRRR** and **IEM airport gauge minus target HRRR**. Each label contributes its own Huber loss, with the same wet-hour weight rule. Validation selects the checkpoint by the mean of the two reference-specific mean squared errors.

The six history rows end at **T−7 through T−2**, inclusive. For a 17:00–18:00 UTC forecast issued at 17:00, the latest history row ends at 16:00 and covers 15:00–16:00. Each row contains:

| Column | Unit | Source and meaning |
|---|---|---|
| MRMS rainfall | mm | Hour-ending radar-only QPE at the nearest MRMS cell |
| Recent HRRR error | mm | MRMS rainfall minus the matched one-hour HRRR forecast for that row |
| Air temperature | °C | Latest routine KDSM airport report in the row's hour |
| Dew point | °C | Same KDSM report |
| Eastward wind | m/s | KDSM winds converted from meteorological direction and speed |
| Northward wind | m/s | Same conversion |

MRMS is a NOAA gridded estimate. The near-surface airport observations come through Iowa Environmental Mesonet. Hourly Stage IV and a reconstructed airport-gauge total are the two target references. Stage IV samples the nearest grid cell to the study coordinate. The DSM airport gauge is roughly 6.3 km from that coordinate and samples rainfall at the station.

For a target ending at 15:00 UTC, `gauge.py` sums the 60 one-minute DSM precipitation values labeled 14:01 through 15:00 UTC, then converts inches to millimeters. Per IEM's ASOS timing notes, each minute ending at HH:MM covers approximately the prior minute shifted by 23 seconds, so these sums represent nominal UTC clock hours. This replaces the previous routine METAR accumulation ending near :54. A duplicate, missing, or invalid minute invalidates the gauge hour. No missing minutes are treated as zero.

[ASOS timing conventions](https://mesonet.agron.iastate.edu/ASOS/precipnote.phtml) · [IEM one-minute archive](https://mesonet.agron.iastate.edu/request/asos/1min.phtml)

### Availability and missing data

- MRMS uses the exact top-of-hour `RadarOnly_QPE_01H` field. The AWS object's Last-Modified time is used as a publication proxy. Only fields dated between their valid hour and one hour later enter a sample.
- The extra hour between latest history and issue time provides a buffer. The historical airport feed contains observation timestamps, but these files lack original receipt times. HRRR inputs were extracted from the original pilot and their original publication timestamps were not audited here. This is a retrospective experiment with a modeled issue time.
- Missing radar and missing matched forecast values make a six-row sample unusable. Airport values may be blank. The network uses a training-only median for missing airport values, plus a separate missingness flag for each airport variable. A blank is distinct from a measured zero. The historical residual and target residual intentionally use different observational references. Training and validation can retain Stage IV examples with a missing gauge target. A missing gauge target is excluded from its loss, and label weights normalize the training contributions by observed count for each reference. Validation averages the two reference-specific mean squared errors using available labels. August evaluation uses only the common hours with both targets. Each baseline fits each reference on the training hours available for that reference. No target-hour observation is passed into the model.
- March–June is training, July is validation, and August is test. The requested six-month interval starts March 1 and ends immediately before September 1, 2025. Validation and test each omit the first seven hours at the boundary so their input windows do not reuse earlier split hours. All baselines are scored on the same test samples. Ridge regression using the same six hours checks whether the LSTM adds value beyond a linear model. The latest-hour regression uses only the most recent row.

## Retrain and expand

Use Python 3.11 or 3.12. On Linux or Windows use the PyTorch CPU wheel index. On macOS install `torch` from the regular package index.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python train.py --history-mode with-history
python train.py --history-mode without-history
python compare.py
```

The bundled point CSVs allow immediate retraining. To rebuild the six-month data from NOAA and IEM or append future months, run all three download stages for the desired months:

```bash
python prepare.py 202503 202504 202505 202506 202507 202508
python observations.py 202503 202504 202505 202506 202507 202508
python gauge.py 202503 202504 202505 202506 202507 202508
python train.py --history-mode with-history
python train.py --history-mode without-history
python compare.py
```

The download scripts skip bundled months. To add future months:

```bash
python prepare.py 202509 202510 202511
python observations.py 202509 202510 202511
python gauge.py 202509 202510 202511
python train.py --history-mode with-history --validation-start 2025-09-01 --test-start 2025-11-01
python train.py --history-mode without-history --validation-start 2025-09-01 --test-start 2025-11-01
python compare.py
```

The download scripts skip existing months unless given `--refresh`. If IEM rate-limits requests, obtain a single UTC routine METAR export with columns `station,valid,tmpf,dwpf,drct,sknt`, then pass it with `--asos-csv path.csv` to reuse it across months. `prepare.py` fetches HRRR by byte range and reads Stage IV monthly tar files without unpacking them. The observation script retains only point data, using one MRMS file per hour rather than every two-minute MRMS update. Allow about 1 GB working disk for weather files when processing months serially, separate from the CPU training environment. Download volume depends on compressed radar grid size and storm coverage.

The full ONNX graph expects a `[1,6,7]` float32 sequence and the reduced graph expects `[1,6,5]`. Both take a separate `[1,1]` target HRRR tensor. It returns two `[1,2]` tensors, the signed residuals and the corrected rainfall amounts. Both outputs are ordered Stage IV first, IEM gauge second. The browser consumes the matching JSON feature spec, metrics, and held-out examples. Each evaluation report ships as `web/models/<history-mode>/metrics.json` and is also embedded as JSON in the ONNX model metadata under `evaluation_metrics`. The sidecar includes the ONNX SHA-256 so the report can be matched to its model. The UI reports MAE, RMSE, wet-hour MAE, and wet-hour CSI for each reference. `train.py` verifies ONNX output against PyTorch on every test sample.

[NOAA HRRR archive](https://registry.opendata.aws/noaa-hrrr-pds/) · [NOAA MRMS archive](https://registry.opendata.aws/noaa-mrms-pds/) · [MRMS product table](https://www.nssl.noaa.gov/projects/mrms/operational/tables.php) · [IEM airport observations](https://mesonet.agron.iastate.edu/request/download.phtml?network=IA) · [NCAR Stage IV archive](https://gdex.ucar.edu/datasets/d507005/)

## Redeploy

Replace the served `web/` directory with the entire `web/` directory in this release. Keep the model subfolders and `comparison.json` inside `web/`. Browser inference uses the bundled runtime. Host the files over HTTP and reload the page after replacing them.

## Paired August results

| Reference | Model | MAE (mm) | RMSE (mm) | Wet-hour MAE (mm) | Wet-hour CSI |
|---|---|---:|---:|---:|---:|
| Stage IV | Raw HRRR | 0.116 | 1.310 | 3.699 | 0.500 |
| Stage IV | Scaled HRRR | 0.080 | 0.730 | 2.585 | 0.375 |
| Stage IV | With HRRR history | 0.150 | 1.137 | 3.414 | 0.235 |
| Stage IV | Without HRRR history | 0.167 | 1.169 | 3.445 | 0.182 |
| Airport gauge | Raw HRRR | 0.117 | 1.244 | 3.316 | 0.444 |
| Airport gauge | Scaled HRRR | 0.079 | 0.632 | 2.271 | 0.235 |
| Airport gauge | With HRRR history | 0.178 | 1.007 | 3.139 | 0.057 |
| Airport gauge | Without HRRR history | 0.177 | 1.063 | 3.193 | 0.227 |

For August 7, 14:00–15:00 UTC, both models clipped their corrected rainfall to zero for both references. The Stage IV and airport totals were 11.375 and 6.350 mm, respectively. The small wet-hour sample and single seed limit generalization. The ONNX predictions matched their PyTorch counterparts on the held-out set, and the browser interface was checked with the bundled WebAssembly runtime in a simulated browser.

## Browse the full local August calendar

The dropdown groups 744 one-hour intervals into 31 Chicago calendar days. The first interval is 00:00–01:00 CDT August 1, and the last is 23:00 CDT August 31 to midnight September 1. The underlying data remain indexed by hour-ending UTC. Both models can run for every listed hour. A gauge reference is available for 496 hours. Other entries show the gauge as unavailable and do not substitute a zero.

The published benchmark remains the original 496 held-out test hours. Each entry declares its benchmark membership. Rebuild the dropdown with `python build_august_examples.py`. It uses the July/August point CSVs and the bundled early-September UTC boundary records. The documented data endpoints are used if the boundary cache is absent. The builder verifies the exact held-out timestamp digest against both model reports.
# mesonetiowa
