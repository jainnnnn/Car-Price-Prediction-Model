import json
from pathlib import Path

import numpy as np
import pandas as pd
import pickle
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)

# Load trained pipeline and the metadata written by model.py
with open(BASE_DIR / "model_pipeline.pkl", "rb") as f:
    model = pickle.load(f)
with open(BASE_DIR / "model_meta.json") as f:
    meta = json.load(f)
print("Model loaded")

# Column order the pipeline was trained with
FEATURES = list(model.feature_names_in_)

# ------------------------------------------------------------------ form layout
# label / unit / help for each input. Options and ranges come from model_meta.json
FIELD_INFO = {
    "brand": ("Make", None, None),
    "carbody": ("Body style", None, None),
    "doornumber": ("Doors", None, None),
    "drivewheel": ("Drive", None, "fwd = front wheels, rwd = rear wheels, 4wd = all four"),
    "enginelocation": ("Engine position", None, None),
    "symboling": ("Insurance risk rating", None, "-2 is safest, 3 is riskiest"),
    "wheelbase": ("Wheelbase", "in", None),
    "carlength": ("Length", "in", None),
    "carwidth": ("Width", "in", None),
    "carheight": ("Height", "in", None),
    "curbweight": ("Curb weight", "lb", "Weight with fluids, no passengers"),
    "fueltype": ("Fuel", None, None),
    "aspiration": ("Aspiration", None, "Turbo or standard"),
    "enginetype": ("Engine type", None, "Valve layout, e.g. ohc = overhead cam"),
    "cylindernumber": ("Cylinders", None, None),
    "enginesize": ("Engine size", "cu in", None),
    "fuelsystem": ("Fuel system", None, "mpfi = multi-point injection, 2bbl = two-barrel carburettor"),
    "boreratio": ("Bore ratio", None, None),
    "stroke": ("Stroke", "in", None),
    "compressionratio": ("Compression ratio", None, None),
    "horsepower": ("Horsepower", "hp", None),
    "peakrpm": ("Peak RPM", "rpm", None),
    "citympg": ("City mileage", "mpg", None),
    "highwaympg": ("Highway mileage", "mpg", None),
}

GROUPS = [
    ("The car", ["brand", "carbody", "doornumber", "drivewheel", "enginelocation", "symboling"]),
    ("Size and weight", ["wheelbase", "carlength", "carwidth", "carheight", "curbweight"]),
    ("Engine", ["fueltype", "aspiration", "enginetype", "cylindernumber", "enginesize",
                "fuelsystem", "boreratio", "stroke", "compressionratio"]),
    ("Performance and economy", ["horsepower", "peakrpm", "citympg", "highwaympg"]),
]


OPTION_LABELS = {
    "drivewheel": {"fwd": "Front-wheel (fwd)", "rwd": "Rear-wheel (rwd)", "4wd": "Four-wheel (4wd)"},
    "aspiration": {"std": "Standard", "turbo": "Turbo"},
}


def option_label(field, value):
    custom = OPTION_LABELS.get(field, {}).get(value)
    if custom:
        return custom
    if field == "brand":
        return "BMW" if value == "bmw" else value.replace("-", " ").title()
    if field in {"carbody", "doornumber", "enginelocation", "fueltype", "cylindernumber"}:
        return value.capitalize()
    return value


def build_form_groups():
    groups = []
    for title, names in GROUPS:
        fields = []
        for name in names:
            if name not in FEATURES:
                continue
            label, unit, help_text = FIELD_INFO.get(name, (name, None, None))
            field = {"name": name, "label": label, "unit": unit, "help": help_text}
            if name in meta["categorical"]:
                info = meta["categorical"][name]
                field["kind"] = "select"
                field["default"] = info["default"]
                field["options"] = [
                    {"value": v, "label": option_label(name, v)} for v in info["options"]
                ]
            else:
                info = meta["numeric"][name]
                field["kind"] = "number"
                field["min"] = int(info["min"]) if info["is_int"] else info["min"]
                field["max"] = int(info["max"]) if info["is_int"] else info["max"]
                field["step"] = 1 if info["is_int"] else "any"
                field["default"] = int(info["median"]) if info["is_int"] else round(info["median"], 2)
            fields.append(field)
        groups.append({"title": title, "fields": fields})
    return groups


FORM_GROUPS = build_form_groups()


# routes
@app.route("/")
def home():
    return render_template("index.html", groups=FORM_GROUPS, meta=meta)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    row, errors, notes = {}, [], []

    for name in FEATURES:
        raw = data.get(name)
        if raw is None or str(raw).strip() == "":
            errors.append(f"{FIELD_INFO.get(name, (name,))[0]} is required.")
            continue

        if name in meta["categorical"]:
            if str(raw) not in meta["categorical"][name]["options"]:
                errors.append(f"{FIELD_INFO.get(name, (name,))[0]}: '{raw}' is not a known option.")
                continue
            row[name] = str(raw)
        else:
            label = FIELD_INFO.get(name, (name,))[0]
            try:
                value = float(raw)
            except (TypeError, ValueError):
                errors.append(f"{label} must be a number.")
                continue
            if not np.isfinite(value):
                errors.append(f"{label} must be a number.")
                continue
            lo, hi = meta["numeric"][name]["min"], meta["numeric"][name]["max"]
            if value < lo or value > hi:
                notes.append(
                    f"{label} is outside the range the model saw in training ({lo:g} to {hi:g}), "
                    "so this estimate is less reliable."
                )
            row[name] = value

    if errors:
        return jsonify({"ok": False, "errors": errors}), 400

    X_new = pd.DataFrame([row], columns=FEATURES)
    log_price = float(model.predict(X_new)[0])

    # The model predicts log(1 + price), so convert back to dollars.
    # The range uses the model's typical error (RMSE on the log scale).
    rmse = meta["metrics"]["rmse_log"]
    return jsonify({
        "ok": True,
        "price": float(np.expm1(log_price)),
        "low": float(np.expm1(log_price - rmse)),
        "high": float(np.expm1(log_price + rmse)),
        "notes": notes,
    })


if __name__ == "__main__":
    app.run(debug=True)