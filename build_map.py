#!/usr/bin/env python3
"""
Карта с click‑подсказками, прогресс‑баром и подробным логгированием.
"""

import os
import logging
import math
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv

import folium
from shapely import wkt as wkt_loader
from shapely.geometry import Point, Polygon, MultiPolygon
from pyproj import Transformer
from tqdm import tqdm  # прогресс‑бар

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,  # измените на INFO, если нужно меньше деталей
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ------------------------------------------------------------
# Параметры подключения к БД
# ------------------------------------------------------------
DB_HOST = os.getenv("DB_HOST", "")
DB_PORT = os.getenv("DB_PORT", "")
DB_NAME = os.getenv("DB_NAME", "")
DB_USER = os.getenv("DB_USER", "")
DB_PASS = os.getenv("DB_PASSWORD", "")

OUTPUT_HTML = Path("db_map.html")
DEFAULT_DISTANCE_M = 50.0          # порог расстояния для совпадения
_TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


# ------------------------------------------------------------
# Вспомогательные функции
# ------------------------------------------------------------
def get_db_conn():
    log.info("Устанавливаем соединение с БД …")
    start = time.time()
    if not DB_NAME or not DB_USER:
        raise RuntimeError("DB_NAME/DB_USER не заданы в .env")
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )
    log.info(f"Соединение с БД установлено за {time.time() - start:.2f}s")
    return conn


def _lonlat_meters_per_degree(lat: float) -> Tuple[float, float]:
    """Приблизительное количество метров в одном градусе широты/долготы."""
    lat_rad = math.radians(lat)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * max(math.cos(lat_rad), 1e-6)
    return m_per_deg_lon, m_per_deg_lat


def _transform_lonlat_to_m(lon: float, lat: float) -> Tuple[float, float]:
    """WGS84 → Web Mercator (EPSG:3857) в метрах."""
    return _TRANSFORMER.transform(lon, lat)


def _geom_to_metric(geom):
    """Преобразует Shapely‑геометрию из WGS84 в метры (EPSG:3857)."""
    if geom.geom_type == "Polygon":
        ext = [_transform_lonlat_to_m(x, y) for x, y in geom.exterior.coords]
        inters = [
            [_transform_lonlat_to_m(x, y) for x, y in ring.coords]
            for ring in geom.interiors
        ]
        return Polygon(ext, inters)
    if geom.geom_type == "MultiPolygon":
        parts = []
        for poly in geom.geoms:
            ext = [_transform_lonlat_to_m(x, y) for x, y in poly.exterior.coords]
            inters = [
                [_transform_lonlat_to_m(x, y) for x, y in ring.coords]
                for ring in poly.interiors
            ]
            parts.append(Polygon(ext, inters))
        return MultiPolygon(parts)
    return None


# ------------------------------------------------------------
# Загрузка данных из БД
# ------------------------------------------------------------
def load_active_orders(conn) -> List[object]:
    """Возвращает список Shapely‑геометрий действующих ордеров."""
    log.info("Загрузка ордеров из БД …")
    start = time.time()
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute("""
            SELECT DISTINCT wkt
            FROM renovation_ii.table_oati_uved_order_raskopki
            WHERE "Виды работ" IS NOT NULL
              AND "Статус" = 'Действует';
        """)
        rows = cur.fetchall()
    log.debug(f"Получено {len(rows)} строк из таблицы renovation_ii.table_oati_uved_order_raskopki")

    geoms: List[object] = []
    parsed_ok = 0
    parse_err = 0
    for idx, row in enumerate(rows, start=1):
        wkt_str = row["wkt"]
        if not wkt_str:
            parse_err += 1
            continue
        try:
            geom = wkt_loader.loads(wkt_str)
            geoms.append(geom)
            parsed_ok += 1
            if idx % 500 == 0 or idx == len(rows):
                log.debug(f"Обработано WKT: {idx}/{len(rows)} (OK={parsed_ok}, ERR={parse_err})")
        except Exception as e:
            parse_err += 1
            log.error(f"Ошибка парсинга WKT (строка {idx}): {e} – фрагмент: {wkt_str[:120]!r}")

    log.info(
        f"Ордеров для карты: {len(geoms)} (успешно распарсено={parsed_ok}, ошибок={parse_err}) "
        f"– загрузка заняла {time.time() - start:.2f}s"
    )
    return geoms


def parse_cameras_lat_lng(cameras_value) -> Optional[Tuple[float, float]]:
    """Извлекает lat/lng из поля cameras (JSON, dict или строка)."""
    if isinstance(cameras_value, dict):
        data = cameras_value
    else:
        import json, ast as _ast
        s = str(cameras_value)
        try:
            data = json.loads(s)
        except Exception:
            try:
                data = _ast.literal_eval(s)
            except Exception as e:
                log.error(f"Не удалось распарсить cameras: {e} (value={cameras_value!r})")
                return None

    lat = data.get("precise_latitude") or data.get("lat")
    lng = data.get("precise_longitude") or data.get("lng")
    if lat is None or lng is None:
        log.error(f"В cameras нет lat/lng: {data}")
        return None
    return float(lat), float(lng)


def load_checked_photos(conn) -> List[dict]:
    """Фото с разрытием и незаконным статусом (is_opening / is_legal), включая координаты."""
    log.info("Загрузка фото из cam_photos …")
    start = time.time()
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute("""
            SELECT id, name, cam_name, is_opening, is_legal
            FROM renovation_ii.cam_photos
            WHERE is_opening IS TRUE
              AND is_legal IS FALSE;
        """)
        photo_rows = cur.fetchall()
    log.debug(f"Получено {len(photo_rows)} строк из cam_photos (is_opening=true, is_legal=false)")

    if not photo_rows:
        log.warning("Нет фото с is_opening=true и is_legal=false в cam_photos")
        return []

    cam_names = {row["cam_name"] for row in photo_rows}
    log.debug(f"Уникальных cam_name в выборке: {len(cam_names)}")

    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(
            """
            SELECT shortname, cameras
            FROM renovation_ii.echd_camera_solr_dds
            WHERE shortname = ANY(%s);
            """,
            (list(cam_names),),
        )
        camera_rows = cur.fetchall()
    log.debug(f"Получено {len(camera_rows)} строк из echd_camera_solr_dds для выбранных камер")

    cam_map: Dict[str, dict] = {}
    for row in camera_rows:
        cam_map[row["shortname"]] = row["cameras"]

    result: List[dict] = []
    skipped_no_cam = 0
    skipped_no_latlon = 0
    for row in photo_rows:
        name = row["name"]
        cam_name = row["cam_name"]
        is_opening = row["is_opening"]
        is_legal = row["is_legal"]

        cameras_value = cam_map.get(cam_name)
        if cameras_value is None:
            skipped_no_cam += 1
            if skipped_no_cam <= 5:  # логируем только первые几个, чтобы не заливать лог
                log.warning(f"Нет cameras для cam_name={cam_name}, фото {name} пропущено")
            continue

        lat_lng = parse_cameras_lat_lng(cameras_value)
        if lat_lng is None:
            skipped_no_latlon += 1
            if skipped_no_latlon <= 5:
                log.warning(f"Не удалось получить lat/lng для cam_name={cam_name}, фото {name} пропущено")
            continue

        lat, lng = lat_lng
        result.append(
            {
                "name": name,
                "cam_name": cam_name,
                "is_opening": bool(is_opening),
                "is_legal": bool(is_legal),
                "lat": lat,
                "lng": lng,
            }
        )

    log.info(
        f"Фото для карты (is_opening=true, is_legal=false): {len(result)} "
        f"– пропущено из‑за отсутствия cameras: {skipped_no_cam}, "
        f"из‑за отсутствия lat/lng: {skipped_no_latlon} "
        f"– загрузка заняла {time.time() - start:.2f}s"
    )
    return result


# ------------------------------------------------------------
# Построение карты
# ------------------------------------------------------------
def build_map(photos: List[dict], geoms: List[object]):
    log.info("Начало построения карты …")
    map_start = time.time()

    if photos:
        avg_lat = sum(p["lat"] for p in photos) / len(photos)
        avg_lng = sum(p["lng"] for p in photos) / len(photos)
        log.debug(f"Средние координаты по фото: lat={avg_lat:.6f}, lng={avg_lng:.6f}")
    else:
        avg_lat, avg_lng = 55.75, 37.62  # fallback – Москва
        log.debug("Фото отсутствуют, используем координаты Москвы по умолчанию")

    m = folium.Map(location=[avg_lat, avg_lng], zoom_start=12, tiles="OpenStreetMap")
    log.debug("Базовый объект folium.Map создан")

    # Предварительно переводим все полигоны в метры для проверки расстояния
    log.debug("Преобразуем геометрии ордеров в метрическую систему (EPSG:3857) …")
    metric_start = time.time()
    metric_geoms = [_geom_to_metric(g) for g in geoms if _geom_to_metric(g) is not None]
    log.debug(
        f"Преобразовано {len(metric_geoms)} из {len(geoms)} геометрий "
        f"– заняло {time.time() - metric_start:.2f}s"
    )

    # --------------------------------------------------------
    # 1. Отрисовка ордеров (полигоны) с popup‑Координатами + прогресс‑бар
    # --------------------------------------------------------
    log.info("Начало отрисовки ордеров (полигонов)…")
    poly_start = time.time()
    for idx, geom in enumerate(tqdm(geoms, desc="Отрисовка ордеров", unit="poly"), start=1):
        try:
            if geom.geom_type == "Polygon":
                coords = list(geom.exterior.coords)
                locations = [(y, x) for x, y in coords]
            elif geom.geom_type == "MultiPolygon":
                locations = []
                for poly in geom.geoms:
                    locations.extend([(y, x) for x, y in poly.exterior.coords])
            else:
                log.debug(f"Пропущен ордер #{idx} – неподдерживаемый тип геометрии: {geom.geom_type}")
                continue

            # WKT как строка для popup
            wkt_text = wkt_loader.dumps(geom).replace("\n", " ").strip()
            popup = folium.Popup(
                f"<b>Ордер #{idx}</b><br>WKT:<br>{wkt_text}", max_width=350
            )

            folium.Polygon(
                locations=locations,
                color="blue",
                weight=2,
                fill=True,
                fill_opacity=0.2,
                popup=popup,
            ).add_to(m)

            if idx % 500 == 0 or idx == len(geoms):
                log.debug(f"Отрисовано полигонов: {idx}/{len(geoms)}")
        except Exception as e:
            log.error(f"Ошибка при отрисовке полигона #{idx}: {e}", exc_info=True)
            continue
    log.info(
        f"Отрисовка ордеров завершена за {time.time() - poly_start:.2f}s"
    )

    # --------------------------------------------------------
    # 2. Точки камер/фото (незаконные разрытия) с popup‑совпадением + прогресс‑бар
    # --------------------------------------------------------
    log.info("Начало отрисовки точек камер …")
    photo_start = time.time()
    for p_idx, p in enumerate(tqdm(photos, desc="Отрисовка точек камер", unit="photo"), start=1):
        if not p.get("is_opening"):
            continue

        lat, lng = p["lat"], p["lng"]
        cam_name = p["cam_name"]
        name = p["name"]
        is_legal = p["is_legal"]

        # Незаконные точки — красные (is_legal=false по выборке)
        color = "green" if is_legal else "red"

        # Определяем, к какому ордеру относится точка
        point_wgs = Point(lng, lat)
        match_idx: Optional[int] = None

        # 1) Прямое попадание внутри/на границе (WGS84)
        for i, g in enumerate(geoms):
            if g.contains(point_wgs) or g.intersects(point_wgs):
                match_idx = i
                break

        # 2) Если не внутри – проверяем расстояние в метрах (EPSG:3857)
        if match_idx is None:
            x_m, y_m = _transform_lonlat_to_m(lng, lat)
            point_m = Point(x_m, y_m)
            for i, g_m in enumerate(metric_geoms):
                if g_m.distance(point_m) <= DEFAULT_DISTANCE_M:
                    match_idx = i
                    break

        match_text = (
            f"Ордер: #{match_idx}" if match_idx is not None else "Ордер: не найден"
        )

        popup_html = (
            f"file: {name}<br>"
            f"cam: {cam_name}<br>"
            f"lat={lat:.6f}, lng={lng:.6f}<br>"
            f"is_opening={p['is_opening']}, is_legal={p['is_legal']}<br>"
            f"{match_text}"
        )

        folium.CircleMarker(
            location=[lat, lng],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_html, max_width=350),
        ).add_to(m)

        if p_idx % 500 == 0 or p_idx == len(photos):
            log.debug(f"Отрисовано точек камер: {p_idx}/{len(photos)}")
    log.info(
        f"Отрисовка точек камер завершена за {time.time() - photo_start:.2f}s"
    )

    # --------------------------------------------------------
    # Сохранение карты
    # --------------------------------------------------------
    log.info("Сохранение карты в файл …")
    save_start = time.time()
    m.save(str(OUTPUT_HTML))
    log.info(
        f"Карта сохранена в {OUTPUT_HTML.resolve()} "
        f"(сохранение заняло {time.time() - save_start:.2f}s)"
    )
    log.info(
        f"Весь процесс построения карты завершён за {time.time() - map_start:.2f}s"
    )


# ------------------------------------------------------------
# Точка входа
# ------------------------------------------------------------
def main():
    log.info("=== Запуск скрипта построения карты ===")
    conn = None
    try:
        conn = get_db_conn()
        geoms = load_active_orders(conn)
        photos = load_checked_photos(conn)
    finally:
        if conn:
            conn.close()
            log.info("Соединение с БД закрыто")

    if not geoms and not photos:
        log.warning("Нет данных для отображения – карта будет пустой")
    build_map(photos, geoms)
    log.info("=== Скрипт завершён ===")


if __name__ == "__main__":
    main()