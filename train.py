"""Fit and export an observation-informed HRRR error model."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pandas as pd
import torch
from torch import nn

STEPS = 6
HIDDEN_SIZE = 24
RECURRENT_LAYERS = 2
RECURRENT_DROPOUT = 0.15
MAX_MM = 999
MISSING = -9999.0
REFERENCES = {"stage4": "Stage IV grid", "iem": "IEM airport gauge (clock-hour)"}
FEATURES = [
    ("mrms_mm", "MRMS rainfall", "mm", 0, MAX_MM),
    ("recent_error_mm", "MRMS minus HRRR", "mm", -MAX_MM, MAX_MM),
    ("historical_hrrr_mm", "Historical HRRR", "mm", 0, MAX_MM),
    ("temp_c", "Air temperature", "°C", -70, 60),
    ("dew_c", "Dew point", "°C", -90, 60),
    ("u_ms", "Eastward wind", "m/s", -100, 100),
    ("v_ms", "Northward wind", "m/s", -100, 100),
]


AIRPORT_FEATURES = 4
HISTORY_MODES = {"with-history": "With historical HRRR and error",
                 "without-history": "Without historical HRRR or error"}


def windows(data):
    arrays = []
    for paired in sorted(data.glob("des_moines_*.csv")):
        observed = paired.with_name(paired.name.replace("des_moines_", "observations_"))
        gauge = paired.with_name(paired.name.replace("des_moines_", "gauge_"))
        if not observed.is_file() or not gauge.is_file():
            raise ValueError(f"{paired.name} requires both {observed.name} and {gauge.name}")
        pair, obs, station = (pd.read_csv(p, parse_dates=["valid_utc"]) for p in (paired, observed, gauge))
        if "iem_valid_minutes" not in station:
            raise ValueError(f"Rebuild {gauge.name} from ASOS minute observations")
        station["iem_mm"] = station["iem_mm"].where(station["iem_valid_minutes"].eq(60))
        arrays.append(pair.merge(obs, on="valid_utc", how="left", validate="one_to_one")
                      .merge(station, on="valid_utc", how="left", validate="one_to_one"))
    if not arrays:
        raise ValueError("No paired monthly CSVs with observation files")
    frame = pd.concat(arrays, ignore_index=True)
    frame["valid_utc"] = pd.to_datetime(frame["valid_utc"], utc=True)
    frame = frame.set_index("valid_utc", verify_integrity=True).sort_index()
    frame = frame.reindex(pd.date_range(frame.index[0], frame.index[-1], freq="h"))
    hours = frame.index
    init = pd.to_datetime(frame["hrrr_init_utc"], utc=True)
    forecast = frame["hrrr_mm"].where(((hours - init) == pd.Timedelta(hours=2)) & frame["hrrr_mm"].between(0, MAX_MM))
    rain = frame["mrms_mm"].where(frame["mrms_mm"].between(0, MAX_MM))
    frame["historical_hrrr_mm"] = forecast
    frame["recent_error_mm"] = rain - forecast
    pub = pd.to_datetime(frame["mrms_published_utc"], utc=True, errors="coerce")
    # A product must be usable by its own one-hour cutoff. This is a stricter
    # criterion than only checking publication against the target issue time.
    radar_ok = (pub >= hours) & (pub <= hours + pd.Timedelta(hours=1))
    asos = pd.to_datetime(frame["asos_valid_utc"], utc=True, errors="coerce")
    station_ok = (asos <= hours) & (asos > hours - pd.Timedelta(hours=1))
    frame.loc[~station_ok, [f[0] for f in FEATURES[-AIRPORT_FEATURES:]]] = np.nan
    features = []
    for key, _, _, low, high in FEATURES:
        raw = frame["mrms_mm"] if key == "mrms_mm" else frame[key]
        features.append(raw.where(raw.between(low, high)).to_numpy(dtype=np.float32))
    features = np.stack(features, axis=-1)
    labels = frame[[f"{key}_mm" for key in REFERENCES]].to_numpy(dtype=np.float32)
    rows = []
    for t in range(STEPS + 1, len(frame)):
        history = slice(t - STEPS - 1, t - 1)
        x = features[history].copy()
        h = forecast.iloc[t]
        y = labels[t]
        if not (0 <= h <= MAX_MM and 0 <= y[0] <= MAX_MM and np.isfinite(x[:, :-AIRPORT_FEATURES]).all()
                and radar_ok.iloc[history].all()):
            continue
        y = np.where((y >= 0) & (y <= MAX_MM), y, np.nan)
        x[:, -AIRPORT_FEATURES:] = np.nan_to_num(x[:, -AIRPORT_FEATURES:], nan=MISSING)
        rows.append((hours[t], x, h, y))
    if not rows:
        raise ValueError("No valid hourly windows")
    stamps, x, forecast, y = zip(*rows)
    return (pd.DatetimeIndex(stamps), np.stack(x), np.asarray(forecast, dtype=np.float32)[:, None],
            np.asarray(y, dtype=np.float32), int(frame["hrrr_init_utc"].notna().sum()))


class ResidualModel(nn.Module):
    def __init__(self, center, spread):
        super().__init__()
        self.register_buffer("center", torch.from_numpy(center))
        self.register_buffer("spread", torch.from_numpy(spread))
        self.lstm = nn.LSTM(len(center) + AIRPORT_FEATURES, HIDDEN_SIZE, num_layers=RECURRENT_LAYERS,
                            dropout=RECURRENT_DROPOUT, batch_first=True)
        self.head = nn.Sequential(nn.Linear(HIDDEN_SIZE + 1, 16), nn.Tanh(), nn.Linear(16, len(REFERENCES)))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, history, hrrr):
        missing = history == MISSING
        normal = (torch.where(missing, self.center, history) - self.center) / self.spread
        normal = torch.sign(normal) * torch.log1p(torch.abs(normal))
        sequence, _ = self.lstm(torch.cat((normal, missing[:, :, -AIRPORT_FEATURES:].float()), dim=-1))
        error = self.head(torch.cat((sequence[:, -1], torch.log1p(hrrr)), dim=1))
        return error, (hrrr + error).clamp_min(0)


def metrics(p, y):
    err = p - y
    wet = y >= 0.25
    dry = ~wet
    detected = p >= 0.25
    union = np.count_nonzero(detected | wet)
    return dict(mae_mm=float(np.abs(err).mean()), rmse_mm=float(np.sqrt(np.mean(err**2))),
                wet_mae_mm=float(np.abs(err[wet]).mean()) if wet.any() else None,
                csi_0_25=float(np.count_nonzero(detected & wet) / union) if union else None,
                dry_false_alarm_rate=float(detected[dry].mean()) if dry.any() else None)


def regression(x, h, train_x, train_h, train_y, center, spread):
    def design(seq, hr):
        mask = seq == MISSING
        standard = (np.where(mask, center, seq) - center) / spread
        standard = np.sign(standard) * np.log1p(np.abs(standard))
        return np.column_stack([np.ones(len(hr)), np.log1p(hr), standard.reshape(len(hr), -1),
                                mask[:, :, -AIRPORT_FEATURES:].reshape(len(hr), -1)])
    a, b = design(train_x, train_h), design(x, h)
    penalty = np.eye(a.shape[1]) * 15
    penalty[0, 0] = 0
    beta = []
    for reference in train_y.T:
        valid = np.isfinite(reference)
        design_valid = a[valid]
        residual = reference[valid] - train_h[valid, 0]
        beta.append(np.linalg.solve(design_valid.T @ design_valid + penalty, design_valid.T @ residual))
    beta = np.stack(beta, axis=1)
    return np.maximum(0, h + b @ beta)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data", type=Path, default=Path("data"))
    cli.add_argument("--history-mode", choices=HISTORY_MODES, default="with-history")
    cli.add_argument("--web", type=Path, help="Export directory, defaults to web/models/<history-mode>")
    cli.add_argument("--validation-start", default="2025-07-01")
    cli.add_argument("--test-start", default="2025-08-01")
    cli.add_argument("--epochs", type=int, default=70)
    args = cli.parse_args()
    if args.epochs < 1:
        cli.error("--epochs must be positive")
    torch.manual_seed(2025)
    torch.set_num_threads(2)
    times, x, h, y, input_hours = windows(args.data)
    # Form the same eligible cases before removing either feature.
    selected = [i for i, field in enumerate(FEATURES)
                if args.history_mode == "with-history"
                or field[0] not in {"historical_hrrr_mm", "recent_error_mm"}]
    features = [FEATURES[i] for i in selected]
    x = x[:, :, selected]
    if args.web is None:
        args.web = Path("web") / "models" / args.history_mode
    va, te = [pd.Timestamp(s, tz="UTC") for s in (args.validation_start, args.test_start)]
    # Purge seven hours at each boundary so no history overlaps an earlier split.
    gap = pd.Timedelta(hours=STEPS + 1)
    masks = [times < va, (times >= va + gap) & (times < te), (times >= te + gap) & np.isfinite(y).all(axis=1)]
    groups = [(x[m], h[m], y[m]) for m in masks]
    if min(map(len, (times[m] for m in masks))) < 12 or va >= te:
        raise ValueError("Each chronological split needs at least 12 valid samples")
    train_x, train_h, train_y = groups[0]
    for label, (_, _, targets) in zip(("training", "validation", "test"), groups):
        if (np.isfinite(targets).sum(axis=0) < 12).any():
            raise ValueError(f"Each reference needs at least 12 usable {label} labels")
    clean = np.where(train_x == MISSING, np.nan, train_x)
    center = np.nan_to_num(np.nanmedian(clean, axis=(0, 1))).astype(np.float32)
    spread = np.maximum(np.nan_to_num(np.nanstd(clean, axis=(0, 1))), 0.25).astype(np.float32)
    model = ResidualModel(center, spread)
    opt = torch.optim.AdamW(model.parameters(), lr=0.005)
    xt, ht, yt = map(torch.from_numpy, groups[0])
    xv, hv, yv = map(torch.from_numpy, groups[1])
    best_score, best_epoch = float("inf"), 0
    available_train = torch.isfinite(yt)
    available_val = torch.isfinite(yv)
    label_weight = len(yt) / available_train.sum(dim=0)
    val_counts = available_val.sum(dim=0)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for i in torch.randperm(len(xt)).split(64):
            opt.zero_grad()
            error, _ = model(xt[i], ht[i])
            valid = available_train[i]
            target = torch.where(valid, yt[i], ht[i])
            loss = torch.nn.functional.huber_loss(error, target - ht[i], reduction="none")
            (loss * (1 + 2 * (target >= 0.25)) * valid * label_weight).mean().backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            prediction = model(xv, hv)[1]
            delta = prediction - torch.where(available_val, yv, prediction)
            score = (delta.square().sum(dim=0) / val_counts).mean().item()
        if score < best_score:
            best_score, best_epoch, best = score, epoch, deepcopy(model.state_dict())
        if epoch - best_epoch >= 12:
            break
    model.load_state_dict(best)
    model.eval()
    args.web.mkdir(parents=True, exist_ok=True)
    path = args.web / "model.onnx"
    torch.onnx.export(model, (torch.zeros(1, STEPS, len(features)), torch.zeros(1, 1)), path,
                      input_names=["history", "target_hrrr_mm"], output_names=["error_mm", "corrected_mm"],
                      opset_version=17, dynamo=False)
    onnx.checker.check_model(path)
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    runtime = ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])
    tx, th, ty = groups[-1]
    with torch.no_grad():
        native = [a.numpy() for a in model(torch.from_numpy(tx), torch.from_numpy(th))]
    exported = [runtime.run(None, {"history": a[None], "target_hrrr_mm": b[None]}) for a, b in zip(tx, th)]
    residual, corrected = [np.concatenate(a) for a in zip(*exported)]
    for a, b in zip(native, (residual, corrected)):
        np.testing.assert_allclose(a, b, atol=1e-5, rtol=1e-5)
    slope = np.maximum(0, np.nansum(train_h * train_y, axis=0) /
                       np.maximum(1e-9, np.where(np.isfinite(train_y), train_h**2, 0).sum(axis=0)))
    predictions = {
        "Raw HRRR": np.repeat(th, len(REFERENCES), axis=1),
        "Scaled HRRR": th * slope,
        "Regression · HRRR only": regression(tx[:, :0], th, train_x[:, :0], train_h, train_y, center, spread),
        "Regression · latest hour": regression(tx[:, -1:], th, train_x[:, -1:], train_h, train_y, center, spread),
        "Regression · six hours": regression(tx, th, train_x, train_h, train_y, center, spread),
        "LSTM · six hours": corrected,
    }
    scores = {key: {name: metrics(pred[:, i], ty[:, i]) for name, pred in predictions.items()}
              for i, key in enumerate(REFERENCES)}
    test_times = times[masks[-1]]
    split = {label: {"hours": int(m.sum()),
                     "available_labels": {key: int(np.isfinite(y[m, i]).sum()) for i, key in enumerate(REFERENCES)},
                     "wet_hours": {key: int((y[m, i] >= 0.25).sum()) for i, key in enumerate(REFERENCES)},
                     "first_utc": times[m][0].isoformat(), "last_utc": times[m][-1].isoformat()}
             for label, m in zip(("train", "validation", "test"), masks)}
    architecture = {"history_steps": STEPS, "hidden_size": HIDDEN_SIZE,
                    "recurrent_layers": RECURRENT_LAYERS, "inter_layer_dropout": RECURRENT_DROPOUT,
                    "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    months = sorted(p.stem.removeprefix("des_moines_") for p in args.data.glob("des_moines_*.csv"))
    gauge_coverage = {}
    for month in months:
        gauge = pd.read_csv(args.data / f"gauge_{month}.csv")
        gauge_coverage[month] = {"hours": len(gauge), "complete_valid_hours": int(gauge.iem_mm.notna().sum())}
    test_keys = "\n".join(stamp.isoformat() for stamp in test_times)
    report = dict(history_mode=args.history_mode, label=HISTORY_MODES[args.history_mode],
                  test_hours_sha256=hashlib.sha256(test_keys.encode()).hexdigest(),
                  input_hours=input_hours, data_months_utc=months, gauge_coverage=gauge_coverage, architecture=architecture,
                  split=split, metrics=scores, epoch=best_epoch, wet_threshold_mm=0.25,
                  model_file=path.name, validation_start_utc=va.isoformat(), test_start_utc=te.isoformat(),
                  onnx_max_difference=float(max(np.max(np.abs(native[0]-residual)), np.max(np.abs(native[1]-corrected)))))
    spec = dict(gauge_reference="KDSM ASOS, 60 minute-ending totals per nominal UTC clock hour", output_shape=[1, len(REFERENCES)], references=REFERENCES, history_shape=[1, STEPS, len(features)], target_shape=[1, 1], unit="mm", missing=MISSING,
                order="oldest first", issue_hours_before_target_end=1, latest_history_hours_before_target_end=2,
                forecast_accumulation_hours=[1, 2],
                features=[dict(name=k, label=l, unit=u, low=a, high=b, optional=i >= len(features) - AIRPORT_FEATURES)
                          for i, (k, l, u, a, b) in enumerate(features)])
    picks = list(dict.fromkeys(
        [int(i) for col in ty.T for i in np.argsort(col)[-3:][::-1]]
        + [int(i) for i in np.flatnonzero((ty == 0).all(axis=1))[:1]]
    ))
    examples = [dict(valid_utc=test_times[i].isoformat(), history=tx[i].tolist(), hrrr_mm=float(th[i, 0]),
                     reference_mm=ty[i].tolist(), error_mm=residual[i].tolist(), corrected_mm=corrected[i].tolist())
                for i in picks]
    # The metric report travels with the model even when the ONNX is copied alone.
    graph = onnx.load(path)
    onnx.helper.set_model_props(graph, {
        "model_description": "Des Moines hourly HRRR rainfall correction, two reference targets",
        "reference_order": json.dumps(list(REFERENCES)),
        "history_mode": args.history_mode,
        "history_feature_order": json.dumps([field[0] for field in features]),
        "evaluation_metrics": json.dumps(report, separators=(",", ":"), allow_nan=False),
    })
    onnx.checker.check_model(graph)
    onnx.save(graph, path)
    report["model_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, payload in (("model-spec", spec), ("metrics", report), ("examples", examples)):
        (args.web / f"{name}.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
