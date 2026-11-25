"use strict";

const byId = id => document.getElementById(id);
const form = byId("predict");
const button = form.querySelector("button");
const select = byId("examples");
const status = byId("status");
const targetInput = byId("forecast");
const format = number => Number.isFinite(number) ? number.toFixed(3) : "—";
const mm = number => `${format(number)} mm`;
const stamp = iso => iso.slice(0, 16).replace("T", " ");
const local = new Intl.DateTimeFormat("en-US", {
  timeZone: "America/Chicago", month: "short", day: "numeric", hour: "numeric",
  minute: "2-digit", timeZoneName: "short"
});
const signedMM = number => (number > 0 && format(number) !== "0.000" ? "+" : "") + mm(number);

async function readJSON(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Could not read ${path} (HTTP ${response.status})`);
  return response.json();
}

async function start() {
  const [comparison, audit, archive] = await Promise.all([
    readJSON("comparison.json"), readJSON("spatial-audit.json"), readJSON("august-hours.json")
  ]);
  const models = await Promise.all(comparison.models.map(async ({ id, label }) => {
    const directory = `models/${id}/`;
    const spec = await readJSON(directory + "model-spec.json");
    return { id, label, directory, spec };
  }));
  const { spec } = models[0];
  const examples = archive.hours;
  const report = comparison;
  for (const [key, label] of [["hrrr", "HRRR"], ["stage4", "Stage IV"], ["mrms", "MRMS"]]) {
    const row = byId("sampling-audit").insertRow();
    const heading = document.createElement("th");
    heading.scope = "row";
    heading.textContent = label;
    row.append(heading);
    const { study, airport } = audit.fields[key];
    for (const value of [study.nearest_mm, study.bilinear_mm, airport.nearest_mm, airport.bilinear_mm]) {
      row.insertCell().textContent = format(value);
    }
  }
  const header = byId("history-head").insertRow();
  const displayFeatures = ["mrms_mm", "historical_hrrr_mm", "temp_c", "dew_c", "u_ms", "v_ms"]
    .map(name => spec.features.find(feature => feature.name === name));
  const names = ["Hour ends (UTC)", ...displayFeatures.slice(0, 2).map(f => `${f.label} (${f.unit})`),
    "MRMS minus HRRR (mm)", ...displayFeatures.slice(2).map(f => `${f.label} (${f.unit})`)];
  const residualCells = [];
  for (const name of names) {
    const cell = document.createElement("th");
    cell.scope = "col";
    cell.textContent = name;
    header.append(cell);
  }
  const labels = [];
  const history = Array.from({ length: spec.history_shape[1] }, (_, i) => {
    const row = byId("history").insertRow();
    const cell = document.createElement("th");
    cell.scope = "row";
    const offset = spec.history_shape[1] + spec.latest_history_hours_before_target_end - 1 - i;
    cell.textContent = `−${offset} h`;
    row.append(cell);
    labels.push({ cell, offset });
    return displayFeatures.map((feature, j) => {
      const input = document.createElement("input");
      Object.assign(input, { type: "number", min: feature.low, max: feature.high,
                            step: "any", required: feature.name === "mrms_mm" });
      input.setAttribute("aria-label", `${feature.label}, target minus ${offset} hours`);
      row.insertCell().append(input);
      if (j === 1) {
        const difference = row.insertCell();
        difference.className = "difference";
        residualCells.push(difference);
      }
      return input;
    });
  });
  const references = Object.entries(spec.references);
  const benchmarkReference = byId("benchmark-reference");
  for (const [key, name] of references) benchmarkReference.add(new Option(name, key));
  const resultCells = references.map(([, name]) => models.map(model => {
    const row = byId("outputs").insertRow();
    const label = document.createElement("th");
    label.scope = "row";
    label.textContent = name;
    row.append(label);
    row.insertCell().textContent = model.label;
    return { correction: row.insertCell(), applied: row.insertCell(), corrected: row.insertCell(), reference: row.insertCell() };
  }));
  const showScores = () => {
    const key = benchmarkReference.value;
    const test = report.split.test;
    const train = report.split.train;
    byId("test-period").textContent =
      `${test.first_utc.slice(0, 10)} to ${test.last_utc.slice(0, 10)} UTC · ${test.hours} hours · ${test.wet_hours[key]} wet hours. Trained on ${train.hours.toLocaleString()} windows.`;
    byId("metrics").replaceChildren();
    for (const [name, result] of Object.entries(report.metrics[key])) {
      const row = byId("metrics").insertRow();
      for (const value of [name, ...["mae_mm", "rmse_mm", "wet_mae_mm", "csi_0_25"].map(k => format(result[k]))]) {
        row.insertCell().textContent = value;
      }
    }
  };
  benchmarkReference.addEventListener("change", showScores);
  showScores();

  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = new URL("runtime/", document.baseURI).href;
  await Promise.all(models.map(async model => {
    model.session = await ort.InferenceSession.create(model.directory + "model.onnx", { executionProviders: ["wasm"] });
  }));
  select.replaceChildren(new Option("Custom values", ""));
  const days = new Map();
  const localDay = new Intl.DateTimeFormat("en-US", {
    timeZone: archive.timezone, weekday: "short", month: "short", day: "numeric", year: "numeric"
  });
  const localTime = new Intl.DateTimeFormat("en-GB", {
    timeZone: archive.timezone, hour: "2-digit", minute: "2-digit", hourCycle: "h23"
  });
  for (const [i, sample] of examples.entries()) {
    const end = new Date(sample.valid_utc);
    const start = new Date(end.getTime() - 3600000);
    const day = localDay.format(start);
    if (!days.has(day)) {
      const group = document.createElement("optgroup");
      group.label = day;
      days.set(day, group);
      select.append(group);
    }
    const gauge = sample.reference_mm[1] === null ? " · gauge unavailable" : "";
    days.get(day).append(new Option(`${localTime.format(start)}–${localTime.format(end)} CDT${gauge}`, i));
  }

  let revision = 0;
  function clearResults() {
    revision++;
    byId("raw").textContent = "—";
    for (const cells of resultCells.flat()) {
      for (const cell of Object.values(cells)) cell.textContent = "—";
    }
    status.textContent = "";
    byId("valid").textContent = "";
    byId("coverage").textContent = "";
    for (const { cell, offset } of labels) cell.textContent = `−${offset} h`;
  }

  function updateDifferences() {
    history.forEach(([mrms, hrrr], i) => {
      residualCells[i].textContent = Number.isFinite(mrms.valueAsNumber - hrrr.valueAsNumber)
        ? format(mrms.valueAsNumber - hrrr.valueAsNumber)
        : "—";
    });
  }

  function loadExample() {
    clearResults();
    if (select.value === "") return;
    const sample = examples[+select.value];
    targetInput.value = sample.hrrr_mm ?? "";
    for (const [i, row] of sample.history.entries()) {
      for (let j = 0; j < displayFeatures.length; j++) {
        history[i][j].value = row[j] ?? "";
      }
      const { cell, offset } = labels[i];
      const date = new Date(Date.parse(sample.valid_utc) - offset * 3600000);
      cell.textContent = stamp(date.toISOString());
    }
    updateDifferences();
    byId("raw").textContent = sample.hrrr_mm === null ? "Unavailable" : mm(sample.hrrr_mm);
    resultCells.forEach((rows, i) => rows.forEach(cells => {
      cells.reference.textContent = sample.reference_mm[i] === null ? "Unavailable" : mm(sample.reference_mm[i]);
    }));
    const messages = [sample.benchmark ? `Included in the ${report.split.test.hours}-hour benchmark.` : `Outside the ${report.split.test.hours}-hour benchmark.`];
    if (sample.reference_mm[1] === null) messages.push("No complete airport gauge total.");
    if (sample.hrrr_mm === null || sample.history.some(row => row[0] === null)) {
      messages.push("Target HRRR or required MRMS history is missing. Fill the blank fields to run inference.");
    }
    byId("coverage").textContent = messages.join(" ");
    const end = new Date(sample.valid_utc);
    const start = new Date(end.getTime() - 3600000);
    const init = new Date(end.getTime() - spec.forecast_accumulation_hours[1] * 3600000);
    byId("valid").textContent = `Target ${stamp(start.toISOString())} to ${stamp(end.toISOString())} UTC (${local.formatRange(start, end)}). HRRR initialized ${stamp(init.toISOString())} UTC.`;
  }

  select.addEventListener("change", loadExample);
  form.addEventListener("input", event => {
    if (event.target === select) return;
    select.value = "";
    clearResults();
    updateDifferences();
  });
  form.addEventListener("submit", async event => {
    event.preventDefault();
    if (!form.reportValidity()) return;
    const includeHistory = history.every(row => Number.isFinite(row[1].valueAsNumber));
    const inputs = history.map(row => {
      const values = Object.fromEntries(displayFeatures.map((feature, i) => [
        feature.name, row[i].value === "" ? spec.missing : row[i].valueAsNumber
      ]));
      values.recent_error_mm = values.mrms_mm - values.historical_hrrr_mm;
      return values;
    });
    const forecast = targetInput.valueAsNumber;
    const version = revision;
    button.disabled = true;
    status.textContent = "Running model…";
    try {
      const results = await Promise.all(models.map(model => {
        if (model.id === "with-history" && !includeHistory) return null;
        const modelSpec = model.spec;
        const values = inputs.flatMap(row => modelSpec.features.map(feature => row[feature.name]));
        return model.session.run({
          history: new ort.Tensor("float32", Float32Array.from(values), modelSpec.history_shape),
          target_hrrr_mm: new ort.Tensor("float32", Float32Array.of(forecast), modelSpec.target_shape)
        });
      }));
      if (version !== revision) return;
      byId("raw").textContent = mm(forecast);
      resultCells.forEach((rows, i) => rows.forEach((cells, j) => {
        const output = results[j];
        if (!output) {
          cells.correction.textContent = cells.applied.textContent = cells.corrected.textContent = "—";
          return;
        }
        const error = output.error_mm.data[i];
        const corrected = output.corrected_mm.data[i];
        const wet = corrected >= report.wet_threshold_mm;
        cells.correction.textContent = signedMM(error);
        cells.applied.textContent = signedMM(corrected - forecast);
        cells.corrected.textContent = `${mm(corrected)} · ${wet ? "wet" : "dry"}`;
      }));
      status.textContent = includeHistory ? "" : "Only the model without HRRR history ran. Enter six historical HRRR values to compare both.";
    } catch (error) {
      if (version === revision) status.textContent = error.message;
    } finally {
      button.disabled = false;
    }
  });

  select.disabled = false;
  button.disabled = false;
  select.value = "0";
  loadExample();
}
start().catch(error => { status.textContent = `${error.message}. Serve this folder over HTTP.`; });
