from __future__ import annotations

import logging
import math
import os
import tempfile
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse
from weakref import WeakKeyDictionary

import psycopg2
import requests
from psycopg2.extras import DictCursor
from dotenv import load_dotenv
from pyproj import Transformer
from shapely import wkt as wkt_loader
from shapely.geometry import Point, Polygon, MultiPolygon, box, mapping
from shapely.strtree import STRtree
from ultralytics import YOLO

load_dotenv()

log = logging.getLogger(__name__)

_transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)

DEFAULT_DISTANCE_M = 100.0

# Сектор обзора камеры по умолчанию (полный, в градусах).
# Половина (fov_deg / 2) — допуск отклонения азимута на полигон от направления камеры.
DEFAULT_FOV_DEG = 90.0

# ─── YOLO (ансамбль классификаторов) ────────────────────────────────────────
# Папка с моделями (как в processing_job.py); можно переопределить через .env
YOLO_MODELS_DIR = os.getenv("YOLO_MODELS_DIR", "result_modelv3")

# Пути к моделям ансамбля — те же 10 *.pt, что и в processing_job.py.
YOLO_MODEL_PATHS: List[str] = [
    f"{YOLO_MODELS_DIR}/model{i}.pt" for i in range(1, 11)
]

# Таймаут скачивания изображения по URL.
IMAGE_DOWNLOAD_TIMEOUT_S = 60

# Кэш загруженных YOLO‑моделей в памяти процесса.
_YOLO_MODELS_CACHE: Optional[List[YOLO]] = None

# Тот же запрос списка WKT, что в processing_job.py / map_test.py / build_map.py
ACTIVE_ORDERS_WKT_SQL = """
    SELECT DISTINCT wkt
    FROM renovation_ii.table_oati_uved_order_raskopki
    WHERE "Виды работ" IS NOT NULL
      AND "Статус" = 'Действует';
"""

# Кэш: есть ли расширение postgis в этой БД (на соединение)
_POSTGIS_EXT_FOR_CONN: WeakKeyDictionary = WeakKeyDictionary()

_POSTGIS_SAVEPOINT = "_cam_order_match_postgis"


def _postgis_extension_available(conn) -> bool:
    if conn in _POSTGIS_EXT_FOR_CONN:
        cached = _POSTGIS_EXT_FOR_CONN[conn]
        log.debug("PostGIS: взято из кэша для соединения: %s", cached)
        return cached
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'postgis');"
            )
            ok = bool(cur.fetchone()[0])
    except psycopg2.Error as exc:
        conn.rollback()
        log.debug("Проверка расширения postgis: %s", exc)
        ok = False
    _POSTGIS_EXT_FOR_CONN[conn] = ok
    log.info(
        "PostGIS: проверка pg_extension (postgis) → %s (результат закэширован на это соединение)",
        "установлено" if ok else "не установлено / ошибка проверки",
    )
    return ok


def _wkt_cell_to_str(value) -> Optional[str]:
    """
    Значение колонки wkt из psycopg2: обычно str с WKT, иногда bytes/memoryview.
    Как в других файлах — дальше передаём строку в shapely.wkt.loads.
    """
    if value is None:
        return None
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, (bytes, bytearray)):
        try:
            value = bytes(value).decode("utf-8")
        except Exception:
            return None
    s = str(value).strip()
    return s or None


def get_db_conn():
    db_name = os.getenv("DB_NAME", "")
    db_user = os.getenv("DB_USER", "")
    if not db_name or not db_user:
        raise RuntimeError("DB_NAME/DB_USER не заданы в .env")
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=db_name,
        user=db_user,
        password=os.getenv("DB_PASSWORD", ""),
    )


def parse_cameras_lat_lng_azimuth(cameras_value) -> Optional[Tuple[float, float, Optional[float]]]:
    """
    Из поля ``cameras`` (json/jsonb в echd_camera_solr_dds): lat, lng и опционально ``azimuth_delta``.
    Координаты: ``precise_latitude`` / ``precise_longitude`` или ``lat`` / ``lng``.
    Азимут: ``azimuth_delta`` (градусы, компас), если ключа нет — третий элемент ``None``.
    """
    if isinstance(cameras_value, dict):
        data = cameras_value
    else:
        import json
        import ast as _ast

        s = str(cameras_value)
        try:
            data = json.loads(s)
        except Exception:
            try:
                data = _ast.literal_eval(s)
            except Exception as e:
                log.error("Не удалось распарсить cameras: %s (value=%r)", e, cameras_value)
                return None

    lat = data.get("precise_latitude") or data.get("lat")
    lng = data.get("precise_longitude") or data.get("lng")
    if lat is None or lng is None:
        log.error("В cameras нет lat/lng: %s", data)
        return None

    az = data.get("azimuth_delta")
    azimuth: Optional[float] = None
    if az is not None:
        try:
            azimuth = float(az)
        except (TypeError, ValueError):
            log.warning("cameras.azimuth_delta не число: %r — азимут не передаём", az)

    return float(lat), float(lng), azimuth


def _match_orders_postgis(conn, lat: float, lng: float, distance_m: float) -> bool:
    """
    Одна выборка EXISTS по ордерам в БД (ST_DWithin в метрах на geography).
    Имена ST_* без схемы — подхватываются из search_path (postgis в public и т.д.).
    """
    log.info(
        "PostGIS: запрос EXISTS — точка камеры SRID=4326 ST_MakePoint(lon=%s, lat=%s), "
        "порог ST_DWithin=%s м (geography, эллипсоид WGS84), полигоны из wkt через ST_GeomFromText(..., 4326)",
        lng,
        lat,
        distance_m,
    )
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM renovation_ii.table_oati_uved_order_raskopki AS t
                WHERE t."Виды работ" IS NOT NULL
                  AND t."Статус" = 'Действует'
                  AND t.wkt IS NOT NULL
                  AND btrim(t.wkt::text) <> ''
                  AND ST_IsValid(ST_GeomFromText(t.wkt::text, 4326))
                  AND ST_DWithin(
                        geography(ST_SetSRID(ST_MakePoint(%s, %s), 4326)),
                        geography(ST_SetSRID(ST_GeomFromText(t.wkt::text, 4326), 4326)),
                        %s
                      )
            );
            """,
            (lng, lat, distance_m),
        )
        row = cur.fetchone()
        out = bool(row and row[0])
    log.info("PostGIS: результат EXISTS → %s", out)
    return out


def load_active_order_geometries(conn) -> Tuple[List[object], Optional[STRtree]]:
    log.info("Загрузка ордеров: выполняется SELECT DISTINCT wkt (действующие ордера с видами работ)")
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(ACTIVE_ORDERS_WKT_SQL)
        rows = cur.fetchall()

    n_rows = len(rows)
    skipped_empty = 0
    parse_errors = 0
    geoms: List[object] = []
    for row in rows:
        wkt_str = _wkt_cell_to_str(row["wkt"])
        if not wkt_str:
            skipped_empty += 1
            continue
        try:
            g = wkt_loader.loads(wkt_str)
            geoms.append(g)
            log.debug(
                "WKT распарсен: тип=%s, bounds(lon/lat)=%s, длина строки=%s",
                getattr(g, "geom_type", type(g)),
                getattr(g, "bounds", None),
                len(wkt_str),
            )
        except Exception as e:
            parse_errors += 1
            log.error("Ошибка парсинга WKT (фрагмент): %s… — %s", wkt_str[:120], e)
            continue

    log.info(
        "Загрузка ордеров: строк из БД=%s, распознано полигонов=%s, пропущено пустых wkt=%s, ошибок парсинга=%s",
        n_rows,
        len(geoms),
        skipped_empty,
        parse_errors,
    )

    if not geoms:
        log.warning("Не удалось распарсить ни один WKT для ордеров — STRtree не строится")
        return [], None

    tree = STRtree(geoms)
    log.info("Построен STRtree по %s геометриям (индекс по bbox в WGS84)", len(geoms))
    return geoms, tree


def _lonlat_meters_per_degree(lat: float) -> Tuple[float, float]:
    lat_rad = math.radians(lat)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * max(math.cos(lat_rad), 1e-6)
    return m_per_deg_lon, m_per_deg_lat


def _search_box_around_point(lng: float, lat: float, distance_m: float):
    m_lon, m_lat = _lonlat_meters_per_degree(lat)
    pad_lon = distance_m / m_lon
    pad_lat = distance_m / m_lat
    return box(lng - pad_lon, lat - pad_lat, lng + pad_lon, lat + pad_lat)


def _strtree_query_geoms(tree: STRtree, query_geom, geoms: List[object]) -> List[object]:
    """
    Shapely 2: STRtree.query часто возвращает ndarray индексов, а не геометрии.
    Учитываем также кортеж (индексы, …) и «сырые» геометрии (старые версии).
    """
    raw = tree.query(query_geom)
    log.debug(
        "STRtree.query: тип сырого ответа=%s, repr(начало)=%s",
        type(raw).__name__,
        repr(raw)[:200] if raw is not None else None,
    )
    if raw is None:
        return []

    if isinstance(raw, tuple) and len(raw) > 0:
        raw = raw[0]

    try:
        n = len(raw)
    except TypeError:
        log.debug("STRtree.query: результат без len() — считаем пустым")
        return []
    if n == 0:
        return []

    first = raw[0]
    if isinstance(first, Integral):
        resolved = [geoms[int(i)] for i in raw]
        log.debug("STRtree.query: интерпретация как индексы целых → %s геометрий", len(resolved))
        return resolved

    try:
        import numpy as np

        if isinstance(raw, np.ndarray):
            if raw.ndim == 2 and raw.shape[0] > 0:
                raw = raw[0]
            if raw.size and np.issubdtype(raw.dtype, np.integer):
                resolved = [geoms[int(i)] for i in raw.tolist()]
                log.debug(
                    "STRtree.query: numpy целочисленные индексы → %s геометрий",
                    len(resolved),
                )
                return resolved
    except ImportError:
        pass

    if hasattr(first, "geom_type"):
        resolved = list(raw)
        log.debug("STRtree.query: ответ уже геометрии → %s шт.", len(resolved))
        return resolved
    resolved = [geoms[int(i)] for i in raw]
    log.debug("STRtree.query: fallback индексы → %s геометрий", len(resolved))
    return resolved


def _transform_lonlat_to_m(lon: float, lat: float) -> Tuple[float, float]:
    x, y = _transformer.transform(lon, lat)
    return x, y


def _geom_to_metric(geom) -> Optional[object]:
    if geom.geom_type == "Polygon":
        exterior = [_transform_lonlat_to_m(x, y) for x, y in geom.exterior.coords]
        interiors = [
            [_transform_lonlat_to_m(x, y) for x, y in ring.coords]
            for ring in geom.interiors
        ]
        return Polygon(exterior, interiors)
    if geom.geom_type == "MultiPolygon":
        parts = []
        for poly in geom.geoms:
            exterior = [_transform_lonlat_to_m(x, y) for x, y in poly.exterior.coords]
            interiors = [
                [_transform_lonlat_to_m(x, y) for x, y in ring.coords]
                for ring in poly.interiors
            ]
            parts.append(Polygon(exterior, interiors))
        return MultiPolygon(parts)
    return None


# ─── Азимут / сектор обзора ─────────────────────────────────────────────────

def _normalize_deg(value: float) -> float:
    """Приводит угол к [0, 360)."""
    return value % 360.0


def _bearing_deg_metric(cam_xy: Tuple[float, float],
                        target_xy: Tuple[float, float]) -> float:
    """
    Азимут (компасный) из точки cam_xy в target_xy в EPSG:3857.
    0° — север (+Y), 90° — восток (+X), 180° — юг, 270° — запад.
    """
    dx_east = target_xy[0] - cam_xy[0]
    dy_north = target_xy[1] - cam_xy[1]
    if dx_east == 0.0 and dy_north == 0.0:
        return 0.0
    return _normalize_deg(math.degrees(math.atan2(dx_east, dy_north)))


def _polygon_vertices_metric(geom_m) -> List[Tuple[float, float]]:
    """Все вершины внешних колец Polygon/MultiPolygon в EPSG:3857."""
    coords: List[Tuple[float, float]] = []
    if geom_m.geom_type == "Polygon":
        coords.extend(tuple(p) for p in geom_m.exterior.coords)
    elif geom_m.geom_type == "MultiPolygon":
        for part in geom_m.geoms:
            coords.extend(tuple(p) for p in part.exterior.coords)
    return coords


def _polygon_bearing_arc(cam_xy: Tuple[float, float],
                         geom_m) -> Optional[Tuple[float, float]]:
    """
    Возвращает (start_deg, end_deg) — наименьшую дугу, покрывающую полигон,
    как он виден из cam_xy. Дуга идёт по часовой стрелке от start к end
    (через 0° при необходимости).

    Возвращает ``None``, если камера находится внутри полигона
    (тогда направление не имеет смысла).
    """
    try:
        if geom_m.contains(Point(cam_xy)):
            return None
    except Exception:
        pass

    pts = _polygon_vertices_metric(geom_m)
    if not pts:
        return None

    bearings = sorted({_bearing_deg_metric(cam_xy, p) for p in pts})
    n = len(bearings)
    if n == 1:
        return bearings[0], bearings[0]

    # Ищем наибольший «пробел» между соседними азимутами на окружности.
    # Дуга, покрываемая полигоном, — это дополнение этого пробела.
    max_gap = -1.0
    gap_idx = 0
    for i in range(n):
        a = bearings[i]
        b = bearings[(i + 1) % n]
        gap = (b - a) % 360.0
        if gap > max_gap:
            max_gap = gap
            gap_idx = i

    start = bearings[(gap_idx + 1) % n]
    end = bearings[gap_idx]
    return start, end


def _arc_contains(start: float, end: float, value: float) -> bool:
    """Лежит ли value в дуге start→end по часовой стрелке (с переходом через 0°)."""
    start = _normalize_deg(start)
    end = _normalize_deg(end)
    value = _normalize_deg(value)
    if start <= end:
        return start <= value <= end
    return value >= start or value <= end


def _arcs_overlap(a_start: float, a_end: float,
                  b_start: float, b_end: float) -> bool:
    """Пересекаются ли две дуги на окружности."""
    return (
        _arc_contains(a_start, a_end, b_start)
        or _arc_contains(a_start, a_end, b_end)
        or _arc_contains(b_start, b_end, a_start)
        or _arc_contains(b_start, b_end, a_end)
    )


def _fov_arc(azimuth_deg: float, fov_deg: float) -> Tuple[float, float]:
    """Сектор обзора камеры: (azimuth - fov/2, azimuth + fov/2), оба в [0, 360)."""
    half = fov_deg / 2.0
    return _normalize_deg(azimuth_deg - half), _normalize_deg(azimuth_deg + half)


def _camera_fov_sees_polygon(cam_xy: Tuple[float, float],
                             geom_m,
                             azimuth_deg: float,
                             fov_deg: float,
                             *,
                             candidate_idx: Optional[int] = None) -> bool:
    """
    Видит ли камера (азимут + сектор fov_deg) хотя бы часть полигона.
    Камера внутри полигона — считаем, что видит.
    """
    # Сектор обзора ≥ 360° — камера «видит во все стороны».
    if fov_deg >= 360.0:
        log.debug("Сектор #%s: fov_deg=%s ≥ 360 → True", candidate_idx, fov_deg)
        return True

    arc = _polygon_bearing_arc(cam_xy, geom_m)
    if arc is None:
        log.info(
            "Сектор #%s: камера внутри полигона — направление не важно → True",
            candidate_idx,
        )
        return True

    p_start, p_end = arc
    f_start, f_end = _fov_arc(azimuth_deg, fov_deg)
    overlap = _arcs_overlap(f_start, f_end, p_start, p_end)
    log.info(
        "Сектор #%s: дуга полигона=[%s°, %s°], дуга обзора=[%s°, %s°] "
        "(азимут=%s°, fov=%s°) → пересечение=%s",
        candidate_idx,
        round(p_start, 2),
        round(p_end, 2),
        round(f_start, 2),
        round(f_end, 2),
        round(azimuth_deg, 2),
        round(fov_deg, 2),
        overlap,
    )
    return overlap


# ─── YOLO + изображение ─────────────────────────────────────────────────────

def load_yolo_models() -> List[YOLO]:
    """Загружает все .pt из YOLO_MODEL_PATHS и кэширует на время жизни процесса."""
    global _YOLO_MODELS_CACHE
    if _YOLO_MODELS_CACHE is not None:
        log.debug("YOLO: используем кэш моделей (%s шт.)", len(_YOLO_MODELS_CACHE))
        return _YOLO_MODELS_CACHE

    if not YOLO_MODEL_PATHS:
        raise RuntimeError("Список YOLO_MODEL_PATHS пуст")

    models: List[YOLO] = []
    skipped: List[str] = []
    for path in YOLO_MODEL_PATHS:
        if not path:
            continue
        if not os.path.exists(path):
            log.warning("YOLO: файл модели не найден, пропуск: %s", path)
            skipped.append(path)
            continue
        log.info("YOLO: загружаем модель %s", path)
        models.append(YOLO(path))

    if not models:
        raise RuntimeError(
            f"Не удалось загрузить ни одной YOLO‑модели. Пропущены: {skipped}"
        )

    log.info(
        "YOLO: загружено моделей=%s, пропущено=%s (директория YOLO_MODELS_DIR=%s)",
        len(models),
        len(skipped),
        YOLO_MODELS_DIR,
    )
    _YOLO_MODELS_CACHE = models
    return models


def _is_url(image_link: str) -> bool:
    parsed = urlparse(image_link)
    return parsed.scheme in ("http", "https")


def _resolve_image_to_local(image_link: str) -> Tuple[Path, bool]:
    """
    Локальный путь к изображению + флаг «временный (нужно удалить)».
    Для http/https — скачивает во временный файл, иначе использует как есть.
    """
    if _is_url(image_link):
        suffix = Path(urlparse(image_link).path).suffix or ".jpg"
        log.info(
            "Изображение: HTTP(S) URL, скачиваем во временный файл (suffix=%s, timeout=%sс): %s",
            suffix,
            IMAGE_DOWNLOAD_TIMEOUT_S,
            image_link,
        )
        resp = requests.get(image_link, stream=True, timeout=IMAGE_DOWNLOAD_TIMEOUT_S)
        resp.raise_for_status()
        fd, tmp_name = tempfile.mkstemp(suffix=suffix, prefix="cam_order_")
        os.close(fd)
        tmp_path = Path(tmp_name)
        size = 0
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)
                    size += len(chunk)
        log.info("Изображение скачано: %s (%s байт)", tmp_path, size)
        return tmp_path, True

    local = Path(image_link)
    if not local.exists():
        raise FileNotFoundError(f"Файл изображения не найден на диске: {image_link}")
    log.info("Изображение: локальный путь %s (%s байт)", local, local.stat().st_size)
    return local, False


def _classify_image_ensemble(image_link: str) -> Tuple[Optional[int], List[Dict[str, Any]]]:
    """
    Прогон через ансамбль YOLO‑моделей (логика голосования как в processing_job.py).
    Возвращает (label, details). label = 0 / 1 / None (если все модели не дали предсказания).
    """
    models = load_yolo_models()

    local_path, is_tmp = _resolve_image_to_local(image_link)
    try:
        votes_0 = 0
        votes_1 = 0
        details: List[Dict[str, Any]] = []

        for idx, model in enumerate(models, start=1):
            log.info(
                "YOLO модель #%s: прогон на %s",
                idx,
                local_path,
            )
            results = model(str(local_path))

            if not results:
                log.warning("YOLO модель #%s: пустой результат", idx)
                continue

            res = results[0]
            probs = getattr(res, "probs", None)
            if probs is None:
                log.error(
                    "YOLO модель #%s: в результате нет .probs (модель, похоже, не классификационная)",
                    idx,
                )
                continue

            cls_id = int(probs.top1)
            conf = float(probs.top1conf)
            cls_name = (
                model.names.get(cls_id, str(cls_id))
                if hasattr(model, "names")
                else str(cls_id)
            )

            details.append({"cls_id": cls_id, "cls_name": cls_name, "conf": conf})
            if cls_id == 1:
                votes_1 += 1
            else:
                votes_0 += 1

            log.info(
                "YOLO модель #%s: cls=%s (id=%s), conf=%.3f",
                idx,
                cls_name,
                cls_id,
                conf,
            )

        if not details:
            log.warning("YOLO: ни одна модель не дала валидного предсказания")
            return None, []

        if votes_1 > votes_0:
            label = 1
        elif votes_0 > votes_1:
            label = 0
        else:
            label = 0
            log.info(
                "YOLO: ничья по голосам (0=%s, 1=%s) — выбираем label=0 (нет разрытия) по умолчанию",
                votes_0,
                votes_1,
            )

        log.info(
            "YOLO итог: label=%s (голоса 0=%s, 1=%s, моделей с ответом=%s)",
            label,
            votes_0,
            votes_1,
            len(details),
        )
        return label, details
    finally:
        if is_tmp:
            try:
                local_path.unlink()
                log.debug("Удалён временный файл изображения: %s", local_path)
            except OSError as e:
                log.warning("Не удалось удалить временный файл %s: %s", local_path, e)


# ─── Поиск ордера: возвращает геометрию и статус ────────────────────────────

# Возможные статусы:
#   "inside"         — точка внутри какого-то ордера (азимут не учитываем)
#   "in_radius_in_fov"   — точка в радиусе и попала в сектор обзора (azimuth учтён)
#   "in_radius_no_azimuth" — точка в радиусе, азимут не задан
#   "wrong_azimuth"  — точка в радиусе, но ни один кандидат не в секторе обзора
#   "no_orders_nearby" — нет ни одного кандидата по радиусу
#   "no_orders_loaded" — geoms пуст / tree None
ORDER_MATCH_LEGAL_STATUSES = ("inside", "in_radius_in_fov", "in_radius_no_azimuth")


def _find_order_match(
    lat: float,
    lng: float,
    geoms: List[object],
    tree: Optional[STRtree],
    *,
    distance_m: float = DEFAULT_DISTANCE_M,
    azimuth_deg: Optional[float] = None,
    fov_deg: float = DEFAULT_FOV_DEG,
) -> Tuple[Optional[object], str]:
    """
    Ищет первый подходящий ордер; возвращает (геометрия_WGS84_или_None, статус).
    Полная пошаговая логика та же, что прежняя в is_camera_near_loaded_orders.
    """
    azimuth_set = azimuth_deg is not None
    log.info(
        "Поиск ордера: lat=%s lng=%s distance_m=%s азимут=%s FOV=%s° загружено_геометрий=%s",
        lat,
        lng,
        distance_m,
        azimuth_deg if azimuth_set else "None (не учитываем)",
        fov_deg if azimuth_set else "—",
        len(geoms) if geoms else 0,
    )

    if not geoms or tree is None:
        log.warning("Поиск ордера: нет геометрий/STRtree → no_orders_loaded")
        return None, "no_orders_loaded"

    point_wgs = Point(lng, lat)
    m_lon, m_lat = _lonlat_meters_per_degree(lat)
    pad_lon = distance_m / m_lon
    pad_lat = distance_m / m_lat
    search = _search_box_around_point(lng, lat, distance_m)
    minx, miny, maxx, maxy = search.bounds
    log.info(
        "Поисковый bbox: pad_lon=%s°, pad_lat=%s° (distance_m=%s м, "
        "≈%s м/°lon, ≈%s м/°lat); bounds min=(%s, %s) max=(%s, %s)",
        round(pad_lon, 8),
        round(pad_lat, 8),
        round(distance_m, 2),
        round(m_lon, 2),
        round(m_lat, 2),
        round(minx, 6),
        round(miny, 6),
        round(maxx, 6),
        round(maxy, 6),
    )

    candidates = _strtree_query_geoms(tree, search, geoms)
    log.info(
        "STRtree: кандидатов с пересекающимся bbox=%s (из %s ордеров)",
        len(candidates),
        len(geoms),
    )
    if not candidates:
        log.info("Итог: no_orders_nearby — bbox-кандидатов нет")
        return None, "no_orders_nearby"

    # Фаза 1: contains/intersects
    matched: List[Tuple[int, object, Optional[object], bool]] = []
    for idx, geom in enumerate(candidates):
        if not hasattr(geom, "geom_type"):
            log.debug("Кандидат #%s: не геометрия (%s), пропуск", idx, type(geom))
            continue
        c = geom.contains(point_wgs)
        i = geom.intersects(point_wgs)
        log.debug(
            "Кандидат #%s: тип=%s, bounds=%s, contains=%s, intersects=%s",
            idx,
            geom.geom_type,
            getattr(geom, "bounds", None),
            c,
            i,
        )
        if c or i:
            log.info(
                "Кандидат #%s: точка внутри/на границе (тип=%s) — добавлен",
                idx,
                geom.geom_type,
            )
            matched.append((idx, geom, None, True))

    if matched:
        idx0, geom0, _, _ = matched[0]
        log.info(
            "Итог: inside — точка внутри %s ордеров (первый: #%s, тип=%s)",
            len(matched),
            idx0,
            geom0.geom_type,
        )
        return geom0, "inside"

    if distance_m <= 0:
        log.info("Итог: no_orders_nearby — distance_m<=0 и внутрь полигона не попали")
        return None, "no_orders_nearby"

    # Фаза 2: расстояние ≤ distance_m
    x_p, y_p = _transform_lonlat_to_m(lng, lat)
    cam_xy = (x_p, y_p)
    point_m = Point(x_p, y_p)
    log.info(
        "Точка в EPSG:3857: x=%s, y=%s (pyproj WGS84→3857, always_xy)",
        round(x_p, 2),
        round(y_p, 2),
    )

    for idx, geom in enumerate(candidates):
        if any(m[0] == idx for m in matched):
            continue
        geom_m = _geom_to_metric(geom)
        if geom_m is None:
            log.debug(
                "Кандидат #%s: тип %s не Polygon/MultiPolygon — пропуск",
                idx,
                getattr(geom, "geom_type", type(geom)),
            )
            continue
        dist = geom_m.distance(point_m)
        log.debug(
            "Кандидат #%s: distance в 3857=%s м (порог %s)",
            idx,
            round(dist, 3),
            distance_m,
        )
        if dist <= distance_m:
            log.info(
                "Кандидат #%s проходит по радиусу: distance=%s м ≤ %s м (тип=%s)",
                idx,
                round(dist, 3),
                distance_m,
                geom.geom_type,
            )
            matched.append((idx, geom, geom_m, False))

    if not matched:
        log.info("Итог: no_orders_nearby — никто не уложился в distance ≤ %s м", distance_m)
        return None, "no_orders_nearby"

    log.info("Радиус-кандидатов всего: %s", len(matched))

    if not azimuth_set:
        idx0, geom0, _, _ = matched[0]
        log.info(
            "Итог: in_radius_no_azimuth — азимут не задан, берём первого кандидата #%s (тип=%s)",
            idx0,
            geom0.geom_type,
        )
        return geom0, "in_radius_no_azimuth"

    # Фаза 3: сектор обзора
    azimuth_norm = _normalize_deg(float(azimuth_deg))
    f_start, f_end = _fov_arc(azimuth_norm, fov_deg)
    log.info(
        "Проверка направления: азимут=%s°, FOV=%s°, сектор обзора=[%s°, %s°] (компас)",
        round(azimuth_norm, 2),
        round(fov_deg, 2),
        round(f_start, 2),
        round(f_end, 2),
    )

    for idx, geom, geom_m, was_contains in matched:
        if was_contains:
            log.info("Кандидат #%s: contains/intersects — видимым считаем безусловно", idx)
            return geom, "in_radius_in_fov"
        if geom_m is None:
            geom_m = _geom_to_metric(geom)
            if geom_m is None:
                continue
        if _camera_fov_sees_polygon(cam_xy, geom_m, azimuth_norm, fov_deg, candidate_idx=idx):
            log.info(
                "Итог: in_radius_in_fov — камера смотрит на кандидат #%s (тип=%s)",
                idx,
                geom.geom_type,
            )
            return geom, "in_radius_in_fov"

    # Не попали в сектор обзора — возвращаем ПЕРВОГО радиус-кандидата для информации
    idx0, geom0, _, _ = matched[0]
    log.info(
        "Итог: wrong_azimuth — %s радиус-кандидатов не в секторе [%s°, %s°] "
        "(возвращаем первого #%s для order_coord)",
        len(matched),
        round(f_start, 2),
        round(f_end, 2),
        idx0,
    )
    return geom0, "wrong_azimuth"


# ─── Основные публичные функции ─────────────────────────────────────────────

# Тексты description для итогового словаря
_DESCRIPTION_NO_OPENING = "No opening found"
_DESCRIPTION_OPENING_LEGAL = "Opening matches active order"
_DESCRIPTION_INSIDE = "Camera is inside an active order polygon"
_DESCRIPTION_NO_NEARBY = "There are no orders nearby"
_DESCRIPTION_WRONG_AZIMUTH = "Wrong azimuth direction"
_DESCRIPTION_YOLO_FAIL = "YOLO classification failed"


def _build_result(
    *,
    opening: Optional[bool],
    legal: Optional[bool],
    description: str,
    image_link: str,
    lat: float,
    lng: float,
    azimuth_deg: Optional[float],
    order_geom: Optional[object],
    yolo_label: Optional[int] = None,
    yolo_details: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Формирует итоговый словарь (обычный dict в Python)."""
    photo_coordinate = {"lat": lat, "lng": lng}
    if azimuth_deg is not None:
        photo_coordinate["azimuth_deg"] = float(azimuth_deg)

    order_coord = None
    if order_geom is not None:
        try:
            order_coord = mapping(order_geom)
        except Exception as e:
            log.error("Не удалось сериализовать геометрию ордера в GeoJSON: %s", e)
            order_coord = {"wkt": order_geom.wkt}

    result: Dict[str, Any] = {
        "opening": opening,
        "legal": legal,
        "description": description,
        "image": image_link,
        "photo_coordinate": photo_coordinate,
        "order_coord": order_coord,
    }
    if yolo_label is not None:
        result["yolo"] = {"label": yolo_label, "votes": yolo_details or []}
    return result


def camera_coordinate_matches_order(
    image_link: str,
    lat: float,
    lng: float,
    azimuth_deg: Optional[float] = None,
    *,
    conn=None,
    distance_m: float = DEFAULT_DISTANCE_M,
    fov_deg: float = DEFAULT_FOV_DEG,
    close_conn: bool = True,
) -> Dict[str, Any]:
    """
    Высокоуровневая проверка фотографии камеры на «законное разрытие».

    Шаги:
      1) Прогон изображения через ансамбль YOLO‑моделей (как в ``processing_job.py``).
         ``image_link`` может быть HTTP(S)-URL или локальным путём.
      2) Если YOLO дал label != 1 (разрытия нет) — сразу
         ``opening=False, legal=True, description='No opening found'``.
      3) Иначе ищем действующий ордер по координатам:
         - есть полигон в радиусе и (азимут не задан ИЛИ полигон в секторе обзора) →
           ``opening=True, legal=True``;
         - есть полигоны в радиусе, но не в секторе обзора →
           ``opening=True, legal=False, description='Wrong azimuth direction'``;
         - вообще нет полигонов в радиусе →
           ``opening=True, legal=False, description='There are no orders nearby'``.

    :param image_link: URL фотографии или путь в файловой системе
    :param lat: широта камеры (WGS84)
    :param lng: долгота камеры (WGS84)
    :param azimuth_deg: компасный азимут (0=север, 90=восток), None — не проверять
    :param conn: psycopg2-соединение; None — создаётся из .env
    :param distance_m: радиус поиска ордеров в метрах
    :param fov_deg: ширина сектора обзора камеры (используется при заданном azimuth_deg)
    :param close_conn: закрывать ли соединение, если создано внутри функции
    :return: словарь с полями
        ``opening``, ``legal``, ``description``, ``image``,
        ``photo_coordinate``, ``order_coord`` (None или GeoJSON-полигон),
        и дополнительно ``yolo`` (классификатор: label + голоса моделей).
    """
    log.info(
        "=== camera_coordinate_matches_order: image=%s lat=%s lng=%s "
        "azimuth_deg=%s distance_m=%s fov_deg=%s ===",
        image_link,
        lat,
        lng,
        azimuth_deg if azimuth_deg is not None else "None",
        distance_m,
        fov_deg if azimuth_deg is not None else "—",
    )

    # 1) YOLO
    try:
        yolo_label, yolo_details = _classify_image_ensemble(image_link)
    except Exception as e:
        log.error("YOLO: ошибка прогона/загрузки изображения: %s", e)
        return _build_result(
            opening=False,
            legal=True,
            description=f"{_DESCRIPTION_YOLO_FAIL}: {e}",
            image_link=image_link,
            lat=lat,
            lng=lng,
            azimuth_deg=azimuth_deg,
            order_geom=None,
        )

    if yolo_label is None:
        log.warning("YOLO не дал валидного label — возвращаем opening=False, legal=True")
        return _build_result(
            opening=False,
            legal=True,
            description=_DESCRIPTION_YOLO_FAIL,
            image_link=image_link,
            lat=lat,
            lng=lng,
            azimuth_deg=azimuth_deg,
            order_geom=None,
            yolo_label=None,
            yolo_details=yolo_details,
        )

    if yolo_label != 1:
        log.info("YOLO label=%s ≠ 1 — разрытия нет, ордера не проверяем", yolo_label)
        return _build_result(
            opening=False,
            legal=True,
            description=_DESCRIPTION_NO_OPENING,
            image_link=image_link,
            lat=lat,
            lng=lng,
            azimuth_deg=azimuth_deg,
            order_geom=None,
            yolo_label=yolo_label,
            yolo_details=yolo_details,
        )

    # 2) Разрытие есть — ищем ордер
    own_conn = conn is None
    if own_conn:
        conn = get_db_conn()
    try:
        geoms, tree = load_active_order_geometries(conn)
    finally:
        if own_conn and close_conn:
            conn.close()

    matched_geom, status = _find_order_match(
        lat,
        lng,
        geoms,
        tree,
        distance_m=distance_m,
        azimuth_deg=azimuth_deg,
        fov_deg=fov_deg,
    )

    if status == "inside":
        return _build_result(
            opening=True,
            legal=True,
            description=_DESCRIPTION_INSIDE,
            image_link=image_link,
            lat=lat,
            lng=lng,
            azimuth_deg=azimuth_deg,
            order_geom=matched_geom,
            yolo_label=yolo_label,
            yolo_details=yolo_details,
        )
    if status in ("in_radius_in_fov", "in_radius_no_azimuth"):
        return _build_result(
            opening=True,
            legal=True,
            description=_DESCRIPTION_OPENING_LEGAL,
            image_link=image_link,
            lat=lat,
            lng=lng,
            azimuth_deg=azimuth_deg,
            order_geom=matched_geom,
            yolo_label=yolo_label,
            yolo_details=yolo_details,
        )
    if status == "wrong_azimuth":
        return _build_result(
            opening=True,
            legal=False,
            description=_DESCRIPTION_WRONG_AZIMUTH,
            image_link=image_link,
            lat=lat,
            lng=lng,
            azimuth_deg=azimuth_deg,
            order_geom=matched_geom,  # ближайший по радиусу — для контекста
            yolo_label=yolo_label,
            yolo_details=yolo_details,
        )
    # no_orders_nearby / no_orders_loaded
    return _build_result(
        opening=True,
        legal=False,
        description=_DESCRIPTION_NO_NEARBY,
        image_link=image_link,
        lat=lat,
        lng=lng,
        azimuth_deg=azimuth_deg,
        order_geom=None,
        yolo_label=yolo_label,
        yolo_details=yolo_details,
    )


def is_camera_near_loaded_orders(
    lat: float,
    lng: float,
    geoms: List[object],
    tree: Optional[STRtree],
    *,
    distance_m: float = DEFAULT_DISTANCE_M,
    azimuth_deg: Optional[float] = None,
    fov_deg: float = DEFAULT_FOV_DEG,
) -> bool:
    """
    Низкоуровневая bool-проверка для уже загруженных геометрий
    (удобно при пакетной обработке без повторных SQL и без YOLO).
    Возвращает ``True``, если статус совпадения легальный
    (``inside`` / ``in_radius_in_fov`` / ``in_radius_no_azimuth``).
    """
    _, status = _find_order_match(
        lat,
        lng,
        geoms,
        tree,
        distance_m=distance_m,
        azimuth_deg=azimuth_deg,
        fov_deg=fov_deg,
    )
    out = status in ORDER_MATCH_LEGAL_STATUSES
    log.info("is_camera_near_loaded_orders → %s (status=%s)", out, status)
    return out


if __name__ == "__main__":
    import json as _json

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Пример: image_link — URL или путь, координаты камеры, азимут (опционально).
    result = camera_coordinate_matches_order(
        image_link="отправить/1000.jpg",
        lat=55.764524,
        lng=37.730552,
        azimuth_deg=180,
    )
    print(_json.dumps(result, ensure_ascii=False, indent=2, default=str))
