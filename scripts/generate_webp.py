#!/usr/bin/env python3
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
from scipy.signal import fftconvolve
import base64
import zlib

import h5py
import numpy as np
import requests
from PIL import Image
from pyproj import Transformer

# --------------------------------------------------------------------------- #
# Konfiguration
# --------------------------------------------------------------------------- #
SRC_DIR = Path("data/hymecng")
OUT_DIR = Path("output/hymecng")

FILENAME_RE = re.compile(r"composite_HymecNG_(\d{8})_(\d{4})_(\d{3})-hd5")

# Normale HymecNG-Codes -> Farbe.
# Code 2 wurde entfernt (wird ausgeblendet, siehe refine_precipitation_classes).
# Code 3 bleibt als Fallback-Farbe erhalten fuer Pixel ohne RV-Abdeckung.
# Codes 31/32/33 sind die RV-verfeinerten Regen-Intensitaeten.
PRECIP_COLORS: dict[int, str] = {
    2: "#43FF43",
    31: "#43FF43",  # Regen leicht
    32: "#34C134",  # Regen maessig
    33: "#008200",  # Regen stark
    4: "#FF4343",
    5: "#FF4343",
    6: "#FFA500",
    7: "#47F0FF",
    71: "#47F0FF",
    72: "#478CFF",
    73: "#3568BD",
    8: "#3568BD",
    9: "#008000",   # Hagel
    10: "#008000",  # Hagel
}
HAIL_CLASSES = {9, 10}

# mm/h-Schwellen fuer Regen (Code 3 -> 31/32/33). [lower, upper, neuer_code]
# upper ist exklusiv. PLATZHALTER - bitte pruefen/anpassen!
RAIN_MMH_THRESHOLDS: list[tuple[float, float, int]] = [
    (0.01, 1.3, 31),           # leicht
    (1.3, 13.0, 32),          # maessig
    (13.0, float("inf"), 33), # stark
]

# mm/h-Schwellen fuer Schnee - PLATZHALTER, bitte pruefen/anpassen!
# Schnee hat i.d.R. ein geringeres fluessiges Aequivalent pro Zeiteinheit,
# daher tendenziell niedrigere Schwellen als bei Regen.
SNOW_MMH_THRESHOLDS: list[tuple[float, float, int]] = [
    (0.1, 1.0, 71),           # leicht
    (1.0, 4.0, 72),           # maessig
    (4.0, float("inf"), 73),  # stark
]

# Welche HymecNG-Basis-Codes werden ueber RV verfeinert?
# Nur Regen (3) ist aktuell aktiv. Sobald klar ist, welcher HymecNG-Code
# tatsaechlich "Schnee" ist (vermutlich 7, evtl. auch 8), einfach eine
# Zeile hinzufuegen, z.B.:
#   REFINEMENT_CONFIG[7] = SNOW_MMH_THRESHOLDS
REFINEMENT_CONFIG: dict[int, list[tuple[float, float, int]]] = {
    3: RAIN_MMH_THRESHOLDS,
    7: SNOW_MMH_THRESHOLDS,
}

# Blitze
THUNDER_COLOR = "#FD5FFF"
STRONG_THUNDER_COLOR = "#BA1ABC"  # Blitz in Hagelzone
LIGHTNING_BASE_URL = "https://radar.wetterstation-neustadt.de/blitze/archive/"
LIGHTNING_BACKUP_URL = "https://nowsky.vercel.app/api/lightning"
LIGHTNING_WINDOW_MINUTES = 5
LIGHTNING_MARKER_RADIUS_PX = 8

# Geometrie / Ausgabe
BERLIN = ZoneInfo("Europe/Berlin")
WEBMERCATOR_OUT_WIDTH = 1400
EDGE_SAMPLES = 200
BBOX_MARGIN_DEG = 0.02
EARTH_RADIUS = 6378137.0
NODATA_CLASS = -1
INVISIBLE_CLASS = 1   # Regen ohne RY-Wert -> nicht dargestellt


BRIGHTSKY_RADAR_URL = "https://api.brightsky.dev/radar"
BRIGHTSKY_PROJ = (
    "+proj=stere +lat_0=90 +lat_ts=60 +lon_0=10 +a=6378137 "
    "+b=6356752.3142451802 +no_defs "
    "+x_0=543196.83521776402 +y_0=3622588.8619310018"
)
BRIGHTSKY_XSIZE = 1100
BRIGHTSKY_YSIZE = 1200
BRIGHTSKY_RES_M = 1000.0
BRIGHTSKY_NODATA = 65535

# Code 1 Nearest
FILL_UNCLASSIFIABLE_CODE = 1
FILL_RADIUS_PX = 8          # Suchradius um jeden Code-1-Pixel
FILL_MIN_NEIGHBORS = 4      # mind. so viele Niederschlags-Pixel im Radius, sonst bleibt 1
PRECIP_SOURCE_CODES = [2, 3, 4, 5, 6, 7, 8, 9, 10]   # nur echte Niederschlagsklassen


# --------------------------------------------------------------------------- #
# Hilfsfunktionen
# --------------------------------------------------------------------------- #
def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def lonlat_to_webmercator(lon_deg, lat_deg):
    x = EARTH_RADIUS * np.radians(lon_deg)
    y = EARTH_RADIUS * np.log(np.tan(np.pi / 4 + np.radians(lat_deg) / 2))
    return x, y


def webmercator_to_lonlat(x, y):
    lon = np.degrees(x / EARTH_RADIUS)
    lat = np.degrees(2 * np.arctan(np.exp(y / EARTH_RADIUS)) - np.pi / 2)
    return lon, lat


def parse_timestamp(filename: str) -> datetime:
    """Zeitstempel aus dem HymecNG-Dateinamen (UTC)."""
    m = FILENAME_RE.match(filename)
    if not m:
        raise ValueError(
            "Dateiname passt nicht zum Schema "
            f"'composite_HymecNG_yyyymmdd_HHMM_000-hd5': {filename}"
        )
    date_str, time_str, _ = m.groups()
    naive = datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
    return naive.replace(tzinfo=timezone.utc)

# --------------------------------------------------------------------------- #
# HDF5 lesen
# --------------------------------------------------------------------------- #
def _find_2d_dataset_by_quantity(h5file: h5py.File, keywords: tuple[str, ...], error_msg: str) -> h5py.Dataset:
    candidates: list[h5py.Dataset] = []

    def visitor(name, obj):
        if isinstance(obj, h5py.Dataset) and name.endswith("/data") and obj.ndim == 2:
            candidates.append(obj)

    h5file.visititems(visitor)
    if not candidates:
        raise RuntimeError(error_msg)

    for ds in candidates:
        what = ds.parent.get("what")
        if what is not None and "quantity" in what.attrs:
            quantity = what.attrs["quantity"]
            if isinstance(quantity, bytes):
                quantity = quantity.decode(errors="ignore")
            if any(k in str(quantity).upper() for k in keywords):
                return ds
    return candidates[0]


def find_classification_dataset(h5file: h5py.File) -> h5py.Dataset:
    """Sucht das 2D-Klassifikations-Dataset (HymecNG); Fallback: erstes 2D-Dataset."""
    return _find_2d_dataset_by_quantity(
        h5file,
        keywords=("CLASS", "PRECIP", "HCLASS", "TYPE"),
        error_msg="Kein 2D-Datensatz in der HymecNG-Datei gefunden.",
    )

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
        v = where.attrs[key]
        return v.decode() if isinstance(v, bytes) else str(v)

    def as_float(key: str) -> float:
        return float(where.attrs[key])

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
    fill_value: float,
) -> np.ndarray:
    """Nearest-Neighbor-Warp eines beliebigen nativen Rasters auf EPSG:3857.

    Arbeitet ausschliesslich mit dem uebergebenen 'grid' (eigene
    xsize/ysize/xscale/yscale/Ursprung). Dadurch ist es unerheblich, ob das
    RV-Quellraster eine andere Aufloesung/Groesse als das HymecNG-Raster hat
    - beide werden unabhaengig voneinander korrekt in das gemeinsame
    Zielraster (x_new/y_new) gesampled.
    """
    xx, yy = np.meshgrid(x_new, y_new)
    lon, lat = webmercator_to_lonlat(xx, yy)
    x_nat, y_nat = to_proj.transform(lon.ravel(), lat.ravel())
    x_nat = np.asarray(x_nat).reshape(xx.shape)
    y_nat = np.asarray(y_nat).reshape(xx.shape)

    ll_x, ll_y = to_proj.transform(grid["ll_lon"], grid["ll_lat"])
    
    col = np.round((x_nat - ll_x) / grid["xscale"]).astype(np.int64)
    row = np.round(grid["ysize"] - 1 - (y_nat - ll_y) / grid["yscale"]).astype(np.int64)
    valid = (col >= 0) & (col < grid["xsize"]) & (row >= 0) & (row < grid["ysize"])

    out = np.full(xx.shape, fill_value, dtype=np.float64)
    out[valid] = data[row[valid], col[valid]]
    return out


def warp_classification_to_webmercator(
    class_array: np.ndarray,
    grid: dict,
    to_proj: Transformer,
    x_new: np.ndarray,
    y_new: np.ndarray,
) -> np.ndarray:
    """Wrapper um nearest_neighbor_warp fuer Integer-Klassifikationscodes."""
    warped = nearest_neighbor_warp(
        class_array.astype(np.float64), grid, to_proj, x_new, y_new, fill_value=float(NODATA_CLASS)
    )
    return np.round(warped).astype(np.int32)


# --------------------------------------------------------------------------- #
# Klassifikation verfeinern (HymecNG-Codes + RV mm/h)
# --------------------------------------------------------------------------- #
def fill_unclassifiable(class_merc: np.ndarray,
                        code: int = FILL_UNCLASSIFIABLE_CODE,
                        radius: int = FILL_RADIUS_PX,
                        min_neighbors: int = FILL_MIN_NEIGHBORS) -> np.ndarray:
    """Ersetzt 'code' durch den häufigsten Niederschlagscode im Kreis um den Pixel."""
    bad = class_merc == code
    if not bad.any():
        return class_merc

    # Kreis-Kernel
    off = np.arange(-radius, radius + 1)
    dr, dc = np.meshgrid(off, off, indexing="ij")
    kernel = (dr * dr + dc * dc <= radius * radius).astype(np.float32)

    # Pro Code zählen, wie oft er im Kreis vorkommt
    codes = [c for c in PRECIP_SOURCE_CODES if (class_merc == c).any()]
    if not codes:
        return class_merc

    counts = np.stack([
        fftconvolve((class_merc == c).astype(np.float32), kernel, mode="same")
        for c in codes
    ])                                   # Form: (n_codes, H, W)
    counts = np.rint(counts)             # FFT-Rundungsrauschen entfernen

    best_idx = counts.argmax(axis=0)
    best_cnt = counts.max(axis=0)
    best_code = np.asarray(codes)[best_idx]

    fill = bad & (best_cnt >= min_neighbors)
    out = class_merc.copy()
    out[fill] = best_code[fill]
    return out


def refine_precipitation_classes(
    class_merc: np.ndarray,
    rate_merc: np.ndarray | None,
) -> np.ndarray:
    refined = class_merc.copy()

    for base_class, thresholds in REFINEMENT_CONFIG.items():
        mask_base = class_merc == base_class
        if not np.any(mask_base):
            continue

        # Niedrigste Stufe der jeweiligen Klasse (Regen -> 31, Schnee -> 71)
        fallback_code = thresholds[0][2]

        if rate_merc is None:
            # Keine RV-Daten: Basis-Klasse in ihre niedrigste Stufe umwandeln
            refined[mask_base] = fallback_code
            continue

        has_rate = ~np.isnan(rate_merc)

        # 1) Pixel MIT RV-Wert: nach Schwellen einteilen
        for lower, upper, new_code in thresholds:
            m = mask_base & has_rate & (rate_merc >= lower) & (rate_merc < upper)
            refined[m] = new_code

        # 2) Pixel OHNE RV-Wert (Nodata / außerhalb des Rasters): Fallback-Stufe
        refined[mask_base & ~has_rate] = fallback_code

        # 3) Pixel MIT RV-Wert, aber unter der untersten Schwelle: ausblenden
        refined[mask_base & (refined == base_class)] = INVISIBLE_CLASS

    return refined

# --------------------------------------------------------------------------- #
# Einfärben
# --------------------------------------------------------------------------- #
def colorize(class_merc: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*class_merc.shape, 4), dtype=np.uint8)
    for cls, hex_color in PRECIP_COLORS.items():
        r, g, b = hex_to_rgb(hex_color)
        rgba[class_merc == cls] = (r, g, b, 255)
    return rgba


# --------------------------------------------------------------------------- #
# Blitze
# --------------------------------------------------------------------------- #
def _window_ms(ts: datetime, minutes: int) -> tuple[int, int]:
    end = int(ts.timestamp() * 1000)
    return end - minutes * 60_000, end


def _fetch_strikes_primary(ts: datetime, minutes: int) -> list[tuple[float, float]]:
    ts_local = ts.astimezone(BERLIN)
    url = f"{LIGHTNING_BASE_URL}{ts_local:%Y-%m-%d-%H%M}.json"
    resp = requests.get(url, timeout=30)
    if resp.status_code == 404:
        print(f"Warnung: Primärer Blitz-Feed liefert 404 ({url}) - nutze Backup-API.", file=sys.stderr)
        raise FileNotFoundError(url)
    resp.raise_for_status()

    start_ms, end_ms = _window_ms(ts, minutes)
    return [
        (s["lat"], s["lon"])
        for s in resp.json().get("strikes", [])
        if start_ms <= s.get("t", 0) <= end_ms
    ]


def _fetch_strikes_backup(ts: datetime, minutes: int) -> list[tuple[float, float]]:
    resp = requests.get(LIGHTNING_BACKUP_URL, timeout=30)
    resp.raise_for_status()

    start_ms, end_ms = _window_ms(ts, minutes)
    strikes = []
    for s in resp.json().get("strikes", []):
        t_ms = int(datetime.fromisoformat(s["time"].replace("Z", "+00:00")).timestamp() * 1000)
        if start_ms <= t_ms <= end_ms:
            strikes.append((s["lat"], s["lon"]))
    return strikes


def fetch_recent_strikes(ts: datetime, minutes: int = LIGHTNING_WINDOW_MINUTES) -> list[tuple[float, float]]:
    try:
        return _fetch_strikes_primary(ts, minutes)
    except FileNotFoundError:
        return _fetch_strikes_backup(ts, minutes)


def apply_lightning_overlay(
    rgba: np.ndarray,
    class_merc: np.ndarray,
    ts: datetime,
    x_new: np.ndarray,
    y_new: np.ndarray,
) -> int:
    """Färbt Blitze innerhalb von Niederschlagsflächen ein. Gibt die Trefferzahl zurück."""
    strikes = fetch_recent_strikes(ts)
    print(f"{len(strikes)} Blitze in den letzten {LIGHTNING_WINDOW_MINUTES} Minuten geladen.")

    out_h, out_w = class_merc.shape
    radius = LIGHTNING_MARKER_RADIUS_PX
    color_normal = (*hex_to_rgb(THUNDER_COLOR), 255)
    color_strong = (*hex_to_rgb(STRONG_THUNDER_COLOR), 255)

    precip_mask = np.isin(class_merc, PRECIP_SOURCE_CODES)   # enthält auch 3
    hail_mask = np.isin(class_merc, list(HAIL_CLASSES))

    offsets = np.arange(-radius, radius + 1)
    dr, dc = np.meshgrid(offsets, offsets, indexing="ij")
    circle = dr * dr + dc * dc <= radius * radius

    x_min, x_max = x_new[0], x_new[-1]
    y_min, y_max = y_new[0], y_new[-1]

    hits = 0
    for lat, lon in strikes:
        sx, sy = lonlat_to_webmercator(lon, lat)
        if not (x_min <= sx <= x_max and y_min <= sy <= y_max):
            continue
        col = int(round((sx - x_min) / (x_max - x_min) * (out_w - 1)))
        row = int(round((sy - y_min) / (y_max - y_min) * (out_h - 1)))
        if not precip_mask[row, col]:
            continue

        r0, r1 = max(0, row - radius), min(out_h, row + radius + 1)
        c0, c1 = max(0, col - radius), min(out_w, col + radius + 1)
        circ = circle[r0 - (row - radius): r1 - (row - radius),
                      c0 - (col - radius): c1 - (col - radius)]

        area = precip_mask[r0:r1, c0:c1] & circ
        hail = hail_mask[r0:r1, c0:c1]
        target = rgba[r0:r1, c0:c1]
        target[area & hail] = color_strong
        target[area & ~hail] = color_normal
        hits += 1
    return hits


# --------------------------------------------------------------------------- #
# RY laden + auf Zielraster warpen
# --------------------------------------------------------------------------- #
def load_brightsky_rate_on_target_grid(ts: datetime, x_new: np.ndarray, y_new: np.ndarray) -> np.ndarray:
    """Holt RV-Radar (5 min) von Bright Sky und warpt es als mm/h auf das Zielraster."""
    resp = requests.get(
        BRIGHTSKY_RADAR_URL,
        params={"date": ts.isoformat(), "format": "compressed"},
        timeout=30,
    )
    resp.raise_for_status()
    radar = resp.json().get("radar", [])
    if not radar:
        raise RuntimeError("Bright Sky lieferte keine Radardaten.")
    entry = radar[0]
    print(f"Bright-Sky-Radar: {entry.get('source')} ({entry.get('timestamp')})")

    raw = np.frombuffer(
        zlib.decompress(base64.b64decode(entry["precipitation_5"])), dtype=np.uint16
    )
    if raw.size != BRIGHTSKY_XSIZE * BRIGHTSKY_YSIZE:
        raise RuntimeError(f"Unerwartete Arraygröße: {raw.size}")
    raw = raw.reshape(BRIGHTSKY_YSIZE, BRIGHTSKY_XSIZE)

    # 0.01 mm pro 5 min  ->  mm/h
    rate = raw.astype(np.float64) * 0.01 * 12.0
    rate[raw == BRIGHTSKY_NODATA] = np.nan

    to_proj = Transformer.from_crs("EPSG:4326", BRIGHTSKY_PROJ, always_xy=True)

    xx, yy = np.meshgrid(x_new, y_new)
    lon, lat = webmercator_to_lonlat(xx, yy)
    x_p, y_p = to_proj.transform(lon.ravel(), lat.ravel())
    x_p = np.asarray(x_p).reshape(xx.shape)
    y_p = np.asarray(y_p).reshape(xx.shape)

    col = np.floor(x_p / BRIGHTSKY_RES_M).astype(np.int64)
    row = np.floor(-y_p / BRIGHTSKY_RES_M).astype(np.int64)
    valid = (col >= 0) & (col < BRIGHTSKY_XSIZE) & (row >= 0) & (row < BRIGHTSKY_YSIZE)

    out = np.full(xx.shape, np.nan, dtype=np.float64)
    out[valid] = rate[row[valid], col[valid]]
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main() -> None:
    candidates = sorted(p for p in SRC_DIR.glob("composite_HymecNG_*-hd5") if FILENAME_RE.match(p.name))
    if not candidates:
        sys.exit(f"Keine HymecNG-Datei in {SRC_DIR} gefunden.")
    src_path = candidates[-1]
    ts = parse_timestamp(src_path.name)

    with h5py.File(src_path, "r") as f:
        class_array = find_classification_dataset(f)[()].astype(np.int32)
        where = find_where_group(f)
        if where is None:
            sys.exit("Keine 'where'-Projektionsinfo in der HD5-Datei gefunden - Warp nicht möglich.")
        grid = extract_grid_info(where)

    to_proj = Transformer.from_crs("EPSG:4326", grid["projdef"], always_xy=True)
    to_wgs84 = Transformer.from_crs(grid["projdef"], "EPSG:4326", always_xy=True)

    ll_x, ll_y, x_max, y_max = native_origin_and_extent(grid, to_proj)
    lon_min, lon_max, lat_min, lat_max = wgs84_bbox_from_perimeter(ll_x, ll_y, x_max, y_max, to_wgs84)
    x_new, y_new, extent = webmercator_target_grid(lon_min, lon_max, lat_min, lat_max)
    print(f"WGS84-BBox: lon [{lon_min:.4f}, {lon_max:.4f}], lat [{lat_min:.4f}, {lat_max:.4f}]")
    print(f"EPSG:3857-Extent [xmin, ymin, xmax, ymax]: {extent}")
    print(f"Zielraster: {len(x_new)} x {len(y_new)} px")

    class_merc = warp_classification_to_webmercator(class_array, grid, to_proj, x_new, y_new)
    class_merc = fill_unclassifiable(class_merc)

    # RY (mm/h) laden und unabhängig auf dasselbe Zielraster warpen
    rate_merc = None
    try:
        rate_merc = load_brightsky_rate_on_target_grid(ts, x_new, y_new)
    except (RuntimeError, ValueError, zlib.error, requests.RequestException) as e:
        print(f"Warnung: Bright-Sky-Radar nicht verfügbar ({e}). Regen bleibt ohne Intensitätsstufe.", file=sys.stderr)

    class_refined = refine_precipitation_classes(class_merc, rate_merc)
    rgba = colorize(class_refined)

    try:
        hits = apply_lightning_overlay(rgba, class_merc, ts, x_new, y_new)
        print(f"{hits} Blitz-Treffer eingefärbt (normal: {THUNDER_COLOR}, mit Hagel: {STRONG_THUNDER_COLOR}).")
    except requests.RequestException as e:
        print(f"Warnung: Blitzdaten konnten nicht geladen werden ({e}). Überspringe Overlay.", file=sys.stderr)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"liveanalyse_{ts.astimezone(BERLIN):%Y%m%d_%H%M}.webp"
    # Zeilen umdrehen: y_new läuft von Süd nach Nord, Bilder von oben nach unten
    Image.fromarray(rgba[::-1], mode="RGBA").save(out_path, format="WEBP", lossless=True)
    print(f"Gespeichert: {out_path}")


if __name__ == "__main__":
    main()
