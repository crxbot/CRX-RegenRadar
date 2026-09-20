#!/usr/bin/env python3
"""RV-Composite (DWD, ODIM-HDF5) -> farbiges WebP in EPSG:3857."""
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import numpy as np
from PIL import Image
from pyproj import Transformer

# --------------------------------------------------------------------------- #
# Konfiguration
# --------------------------------------------------------------------------- #
SRC_DIR = Path("data/rv")
OUT_DIR = Path("output/rv")

FILENAME_RE = re.compile(r"composite_rv_(\d{8})_(\d{4})_(\d{3})-hd5")

# Falls das Dataset als Akkumulation (quantity=ACRR) vorliegt: Laenge des
# Akkumulationsintervalls in Minuten -> Umrechnung auf mm/h (RV = 5 min).
RV_ACCUM_MINUTES = 5
# Fallbacks, falls die Attribute (gain/nodata) in der Datei fehlen
RV_DEFAULT_GAIN = 0.01
RV_DEFAULT_NODATA = 65535

MIN_VISIBLE_MMH = 0.01      # darunter: transparent
WHITE_MAX_MMH = 0.09        # ab hier volle Weiß-Deckkraft der Rampe
WHITE_ALPHA_MIN = 20        # Alpha bei 0.01 mm/h (0-255)
WHITE_ALPHA_MAX = 160       # Alpha bei 0.09 mm/h (0-255)

# Farbtabelle: (mm/h, (R, G, B))
# Diskrete Stufen: jeder Wert bekommt die Farbe der letzten Schwelle <= Wert.
MIN_VISIBLE_MMH = 0.01      # darunter: transparent

COLOR_TABLE: list[tuple[float, tuple[int, int, int]]] = [
    (0.1,   (0, 221, 238)),
    (0.12,  (0, 206, 240)),
    (0.14,  (1, 190, 242)),
    (0.16,  (1, 175, 244)),
    (0.18,  (1, 160, 246)),
    (0.2,   (1, 160, 246)),
    (0.24,  (1, 120, 246)),
    (0.28,  (1, 80, 246)),
    (0.32,  (0, 40, 246)),
    (0.36,  (0, 0, 246)),

    # Grün-Phase: leichter Regen 0.4-4.5 mm/h
    (0.4,   (0, 255, 0)),
    (0.52,  (0, 246, 0)),
    (0.64,  (0, 237, 0)),
    (0.76,  (0, 228, 0)),
    (0.88,  (0, 218, 0)),
    (1.0,   (0, 209, 0)),
    (1.26,  (0, 200, 0)),
    (1.52,  (0, 200, 0)),
    (1.78,  (0, 192, 0)),
    (2.04,  (0, 184, 0)),
    (2.3,   (0, 176, 0)),
    (2.86,  (0, 168, 0)),
    (3.42,  (0, 160, 0)),
    (3.98,  (0, 152, 0)),
    (4.56,  (0, 144, 0)),

    # Gelb-Phase:
    (5.1,   (255, 255, 0)),
    (6.48,  (249, 239, 0)),
    (7.86,  (243, 224, 0)),
    (9.24,  (237, 208, 0)),
    (10.62, (231, 192, 0)),
    (12.0,  (231, 192, 0)),
    (14.4,  (237, 180, 0)),
    (16.8,  (243, 168, 0)),
    (19.2,  (249, 156, 0)),
    (21.6,  (255, 144, 0)),

    # Orange-Phase:
    (24.0,  (255, 144, 0)),
    (28.0,  (255, 108, 0)),
    (32.0,  (255, 72, 0)),

    # Rot-Phase:
    (36.0,  (255, 36, 0)),
    (40.0,  (255, 0, 0)),
    (44.0,  (255, 0, 0)),
    (51.2,  (245, 0, 0)),
    (58.4,  (235, 0, 0)),
    (65.6,  (224, 0, 0)),
    (72.8,  (214, 0, 0)),
    (80.0,  (214, 0, 0)),
    (93.2,  (201, 0, 0)),
    (106.4, (187, 0, 0)),
    (119.6, (174, 0, 0)),
    (132.8, (160, 0, 0)),

    # Violett-Phase:
    (146.0, (255, 200, 255)),
    (170.4, (244, 180, 244)),
    (194.8, (232, 160, 232)),
    (219.2, (221, 140, 221)),
    (243.6, (209, 120, 209)),
    (268.0, (198, 100, 198)),
    (312.6, (186, 80, 186)),
    (357.2, (175, 60, 175)),
    (401.8, (163, 40, 163)),
    (446.4, (152, 20, 152)),
    (491.0, (140, 0, 140)),
]

# Geometrie / Ausgabe
BERLIN = ZoneInfo("Europe/Berlin")
WEBMERCATOR_OUT_WIDTH = 1400
EDGE_SAMPLES = 200
BBOX_MARGIN_DEG = 0.02
EARTH_RADIUS = 6378137.0


# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #
def lonlat_to_webmercator(lon_deg, lat_deg):
    x = EARTH_RADIUS * np.radians(lon_deg)
    y = EARTH_RADIUS * np.log(np.tan(np.pi / 4 + np.radians(lat_deg) / 2))
    return x, y


def webmercator_to_lonlat(x, y):
    lon = np.degrees(x / EARTH_RADIUS)
    lat = np.degrees(2 * np.arctan(np.exp(y / EARTH_RADIUS)) - np.pi / 2)
    return lon, lat


def parse_filename(filename: str) -> tuple[datetime, int]:
    """Liefert (Basiszeit UTC, Vorhersageschritt in Minuten) aus dem RV-Dateinamen."""
    m = FILENAME_RE.match(filename)
    if not m:
        raise ValueError(
            "Dateiname passt nicht zum Schema "
            f"'composite_rv_yyyymmdd_HHMM_LLL-hd5': {filename}"
        )
    date_str, time_str, lead_str = m.groups()
    naive = datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
    return naive.replace(tzinfo=timezone.utc), int(lead_str)


# --------------------------------------------------------------------------- #
# HDF5 lesen
# --------------------------------------------------------------------------- #
def find_rate_dataset(h5file: h5py.File) -> h5py.Dataset:
    """Sucht das 2D-Niederschlags-Dataset; Fallback: erstes 2D-Dataset."""
    candidates: list[h5py.Dataset] = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and name.endswith("/data") and obj.ndim == 2:
            candidates.append(obj)

    h5file.visititems(visitor)
    if not candidates:
        raise RuntimeError("Kein 2D-Datensatz in der RV-Datei gefunden.")

    for ds in candidates:
        what = ds.parent.get("what")
        if what is not None and "quantity" in what.attrs:
            q = _attr_str(what.attrs["quantity"]).upper()
            if any(k in q for k in ("ACRR", "RATE", "PRECIP", "RR")):
                return ds
    return candidates[0]


def _attr_str(v) -> str:
    return v.decode(errors="ignore") if isinstance(v, bytes) else str(v)


def _scalar(v):
    return v.item() if isinstance(v, np.ndarray) and v.size == 1 else v


def read_rate_mmh(ds: h5py.Dataset) -> np.ndarray:
    """Liest das Dataset und gibt mm/h als float64 zurueck (NaN = kein Datum)."""
    # 'what'-Gruppe der Daten, sonst eine Ebene hoeher, sonst Root
    what = None
    for grp in (ds.parent, ds.parent.parent, ds.file):
        w = grp.get("what")
        if w is not None and "gain" in w.attrs:
            what = w
            break
    attrs = what.attrs if what is not None else {}

    gain = float(_scalar(attrs.get("gain", RV_DEFAULT_GAIN)))
    offset = float(_scalar(attrs.get("offset", 0.0)))
    nodata = _scalar(attrs.get("nodata", RV_DEFAULT_NODATA))
    undetect = _scalar(attrs["undetect"]) if "undetect" in attrs else None
    quantity = _attr_str(attrs.get("quantity", "")).upper()

    raw = ds[()]
    values = raw.astype(np.float64) * gain + offset

    factor = 60.0 / RV_ACCUM_MINUTES if "ACRR" in quantity else 1.0
    values *= factor

    values[raw == nodata] = np.nan
    if undetect is not None and undetect != nodata:
        values[raw == undetect] = 0.0

    print(
        f"RV-Dataset: quantity='{quantity}', gain={gain}, offset={offset}, "
        f"nodata={nodata}, undetect={undetect}, Faktor->mm/h={factor:g}"
    )
    valid = values[np.isfinite(values)]
    if valid.size:
        print(f"Wertebereich: {valid.min():.2f} .. {valid.max():.2f} mm/h")
    return values


def find_where_group(h5file: h5py.File) -> h5py.Group | None:
    required = ("projdef", "xsize", "ysize", "xscale", "yscale", "LL_lon", "LL_lat")

    def complete(grp) -> bool:
        return all(k in grp.attrs for k in required)

    root_where = h5file.get("where")
    if root_where is not None and complete(root_where):
        return root_where

    found: list[h5py.Group] = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Group) and name.split("/")[-1] == "where" and complete(obj):
            found.append(obj)

    h5file.visititems(visitor)
    return found[0] if found else None


def extract_grid_info(where: h5py.Group) -> dict:
    def as_str(key: str) -> str:
        return _attr_str(where.attrs[key])

    def as_float(key: str) -> float:
        return float(_scalar(where.attrs[key]))

    return {
        "projdef": as_str("projdef"),
        "xsize": int(as_float("xsize")),
        "ysize": int(as_float("ysize")),
        "xscale": as_float("xscale"),
        "yscale": as_float("yscale"),
        "ll_lon": as_float("LL_lon"),
        "ll_lat": as_float("LL_lat"),
    }


# --------------------------------------------------------------------------- #
# Geometrie / Warp
# --------------------------------------------------------------------------- #
def native_origin_and_extent(grid: dict, to_proj: Transformer):
    ll_x, ll_y = to_proj.transform(grid["ll_lon"], grid["ll_lat"])
    x_max = ll_x + grid["xsize"] * grid["xscale"]
    y_max = ll_y + grid["ysize"] * grid["yscale"]
    return ll_x, ll_y, x_max, y_max


def wgs84_bbox_from_perimeter(ll_x, ll_y, x_max, y_max, to_wgs84: Transformer):
    """WGS84-Bounding-Box durch Abtasten des nativen Rasterrands."""
    t = np.linspace(0.0, 1.0, EDGE_SAMPLES)
    xs_span = ll_x + t * (x_max - ll_x)
    ys_span = ll_y + t * (y_max - ll_y)
    xs = np.concatenate([xs_span, xs_span, np.full_like(ys_span, ll_x), np.full_like(ys_span, x_max)])
    ys = np.concatenate([np.full_like(xs_span, ll_y), np.full_like(xs_span, y_max), ys_span, ys_span])
    lons, lats = (np.asarray(a) for a in to_wgs84.transform(xs, ys))
    return (
        float(lons.min()) - BBOX_MARGIN_DEG,
        float(lons.max()) + BBOX_MARGIN_DEG,
        float(lats.min()) - BBOX_MARGIN_DEG,
        float(lats.max()) + BBOX_MARGIN_DEG,
    )


def webmercator_target_grid(lon_min, lon_max, lat_min, lat_max):
    x_min, y_min = lonlat_to_webmercator(lon_min, lat_min)
    x_max, y_max = lonlat_to_webmercator(lon_max, lat_max)
    aspect = (y_max - y_min) / (x_max - x_min)
    out_h = max(int(round(WEBMERCATOR_OUT_WIDTH * aspect)), 1)
    x_new = np.linspace(x_min, x_max, WEBMERCATOR_OUT_WIDTH)
    y_new = np.linspace(y_min, y_max, out_h)
    return x_new, y_new, [x_min, y_min, x_max, y_max]


def nearest_neighbor_warp(
    data: np.ndarray,
    grid: dict,
    to_proj: Transformer,
    x_new: np.ndarray,
    y_new: np.ndarray,
    fill_value: float = np.nan,
) -> np.ndarray:
    """Nearest-Neighbor-Warp des nativen Rasters auf das EPSG:3857-Zielraster."""
    xx, yy = np.meshgrid(x_new, y_new)
    lon, lat = webmercator_to_lonlat(xx, yy)
    x_nat, y_nat = to_proj.transform(lon.ravel(), lat.ravel())
    x_nat = np.asarray(x_nat).reshape(xx.shape)
    y_nat = np.asarray(y_nat).reshape(xx.shape)

    ll_x, ll_y = to_proj.transform(grid["ll_lon"], grid["ll_lat"])

    # ODIM: LL_lon/LL_lat = untere linke Ecke des Rasters -> floor
    # (bei 1 km Aufloesung ist der Unterschied zu round() nur ein halber Pixel;
    #  falls das Bild gegenueber frueher verschoben wirkt, hier wieder round() nehmen)
    col = np.floor((x_nat - ll_x) / grid["xscale"]).astype(np.int64)
    row = (grid["ysize"] - 1 - np.floor((y_nat - ll_y) / grid["yscale"])).astype(np.int64)
    valid = (col >= 0) & (col < grid["xsize"]) & (row >= 0) & (row < grid["ysize"])

    out = np.full(xx.shape, fill_value, dtype=np.float64)
    out[valid] = data[row[valid], col[valid]]
    return out


# --------------------------------------------------------------------------- #
# Einfärben
# --------------------------------------------------------------------------- #
def colorize(rate: np.ndarray) -> np.ndarray:
    """mm/h -> RGBA: weiße Alpha-Rampe (0.01-0.09), darüber diskrete Farbstufen."""
    thresholds = np.array([t for t, _ in COLOR_TABLE], dtype=np.float64)
    colors = np.array([c for _, c in COLOR_TABLE], dtype=np.uint8)   # (N, 3)

    rgba = np.zeros((*rate.shape, 4), dtype=np.uint8)
    visible = np.isfinite(rate) & (rate >= MIN_VISIBLE_MMH)
    if not visible.any():
        return rgba

    faint = visible & (rate < thresholds[0])   # 0.01 .. <0.1 -> weiß mit Alpha
    strong = visible & ~faint

    # Weiß-Rampe: Alpha steigt linear von WHITE_ALPHA_MIN auf WHITE_ALPHA_MAX
    t = (rate[faint] - MIN_VISIBLE_MMH) / (WHITE_MAX_MMH - MIN_VISIBLE_MMH)
    t = np.clip(t, 0.0, 1.0)
    rgba[faint, :3] = 255
    rgba[faint, 3] = (WHITE_ALPHA_MIN + t * (WHITE_ALPHA_MAX - WHITE_ALPHA_MIN)).astype(np.uint8)

    # Ab der ersten Schwelle: diskrete Farbstufen wie bisher
    idx = np.searchsorted(thresholds, rate[strong], side="right") - 1
    idx = np.clip(idx, 0, len(thresholds) - 1)
    rgba[strong, :3] = colors[idx]
    rgba[strong, 3] = 255
    return rgba


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    # Nur Analysedateien (_000), neueste zuletzt
    candidates = sorted(
        p for p in SRC_DIR.glob("composite_rv_*-hd5")
        if FILENAME_RE.match(p.name) and parse_filename(p.name)[1] == 0
    )
    if not candidates:
        sys.exit(f"Keine RV-Datei (composite_rv_*_000-hd5) in {SRC_DIR} gefunden.")
    src_path = candidates[-1]
    base_ts, lead_min = parse_filename(src_path.name)
    ts = base_ts + timedelta(minutes=lead_min)
    print(f"Quelle: {src_path.name}  ({ts:%Y-%m-%d %H:%M} UTC)")

    with h5py.File(src_path, "r") as f:
        rate_native = read_rate_mmh(find_rate_dataset(f))
        where = find_where_group(f)
        if where is None:
            sys.exit("Keine 'where'-Projektionsinfo in der HD5-Datei gefunden - Warp nicht möglich.")
        grid = extract_grid_info(where)

    if rate_native.shape != (grid["ysize"], grid["xsize"]):
        sys.exit(f"Rastergröße {rate_native.shape} passt nicht zu where-Info "
                 f"({grid['ysize']} x {grid['xsize']}).")

    to_proj = Transformer.from_crs("EPSG:4326", grid["projdef"], always_xy=True)
    to_wgs84 = Transformer.from_crs(grid["projdef"], "EPSG:4326", always_xy=True)

    ll_x, ll_y, x_max, y_max = native_origin_and_extent(grid, to_proj)
    lon_min, lon_max, lat_min, lat_max = wgs84_bbox_from_perimeter(ll_x, ll_y, x_max, y_max, to_wgs84)
    x_new, y_new, extent = webmercator_target_grid(lon_min, lon_max, lat_min, lat_max)
    print(f"WGS84-BBox: lon [{lon_min:.4f}, {lon_max:.4f}], lat [{lat_min:.4f}, {lat_max:.4f}]")
    print(f"EPSG:3857-Extent [xmin, ymin, xmax, ymax]: {extent}")
    print(f"Zielraster: {len(x_new)} x {len(y_new)} px")

    rate_merc = nearest_neighbor_warp(rate_native, grid, to_proj, x_new, y_new, fill_value=np.nan)
    rgba = colorize(rate_merc)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"liveanalyse_{ts.astimezone(BERLIN):%Y%m%d_%H%M}.webp"
    # Zeilen umdrehen: y_new läuft von Süd nach Nord, Bilder von oben nach unten
    Image.fromarray(rgba[::-1], mode="RGBA").save(out_path, format="WEBP", lossless=True)
    print(f"Gespeichert: {out_path}")


if __name__ == "__main__":
    main()
