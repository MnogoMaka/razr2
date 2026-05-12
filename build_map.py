import os
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict

import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv

import folium
from shapely import wkt as wkt_loader
from shapely.geometry import Point
from shapely.strtree import STRtree

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

DB_HOST = os.getenv("DB_HOST", "")
DB_PORT = os.getenv("DB_PORT", "")
DB_NAME = os.getenv("DB_NAME", "")
DB_USER = os.getenv("DB_USER", "")
DB_PASS = os.getenv("DB_PASSWORD", "")

OUTPUT_HTML = Path("db_map.html")

def get_db_conn():
    if not DB_NAME or not DB_USER:
        raise RuntimeError("DB_NAME/DB_USER не заданы в .env")
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )

def load_active_orders(conn) -> Tuple[List[object], Optional[STRtree]]:
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute("""
            SELECT DISTINCT wkt
            FROM renovation_ii.table_oati_uved_order_raskopki
            WHERE "Виды работ" IS NOT NULL
              AND "Статус" = 'Действует';
        """)
        rows = cur.fetchall()

    geoms: List[object] = []
    for row in rows:
        wkt_str = row["wkt"]
        if not wkt_str:
            continue
        try:
            geom = wkt_loader.loads(wkt_str)
        except Exception as e:
            log.error(f"Ошибка парсинга WKT: {e}")
            continue
        geoms.append(geom)

    if not geoms:
        log.warning("Не удалось распарсить ни один WKT для ордеров")
        return [], None

    tree = STRtree(geoms)
    log.info(f"Ордеров для карты: {len(geoms)}")
    return geoms, tree

def parse_cameras_lat_lng(cameras_value) -> Optional[Tuple[float, float]]:
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
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute("""
            SELECT id, name, cam_name, label, decision
            FROM renovation_ii.cam_photos
            WHERE label IS NOT NULL
              AND decision IS NOT NULL AND label != 0 AND decision != 0;
        """)
        photo_rows = cur.fetchall()

    if not photo_rows:
        log.warning("Нет фото с заполненными label и decision в cam_photos")
        return []

    cam_names = {row["cam_name"] for row in photo_rows}

    # Тянем cameras по этим cam_name
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute("""
            SELECT shortname, cameras
            FROM renovation_ii.echd_camera_solr_dds
            WHERE shortname = ANY(%s);
        """, (list(cam_names),))
        camera_rows = cur.fetchall()

    cam_map: Dict[str, dict] = {}
    for row in camera_rows:
        shortname = row["shortname"]
        cameras_value = row["cameras"]
        cam_map[shortname] = cameras_value

    result: List[dict] = []
    for row in photo_rows:
        name = row["name"]
        cam_name = row["cam_name"]
        label = row["label"]
        decision = row["decision"]

        cameras_value = cam_map.get(cam_name)
        if cameras_value is None:
            log.warning(f"Нет cameras для cam_name={cam_name}, фото {name} пропущено")
            continue

        lat_lng = parse_cameras_lat_lng(cameras_value)
        if lat_lng is None:
            log.warning(f"Не удалось получить lat/lng для cam_name={cam_name}, фото {name} пропущено")
            continue

        lat, lng = lat_lng

        result.append({
            "name": name,
            "cam_name": cam_name,
            "label": int(label),
            "decision": int(decision),
            "lat": lat,
            "lng": lng,
        })

    log.info(f"Фото для карты (label/decision заполнены): {len(result)}")
    return result

def build_map(photos: List[dict], geoms: List[object]):
    if photos:
        avg_lat = sum(p["lat"] for p in photos) / len(photos)
        avg_lng = sum(p["lng"] for p in photos) / len(photos)
    else:
        avg_lat, avg_lng = 55.75, 37.62  # Москва по умолчанию

    m = folium.Map(location=[avg_lat, avg_lng], zoom_start=12, tiles="OpenStreetMap")

    # 1. Ордеры
    for geom in geoms:
        try:
            if geom.geom_type == "Polygon":
                coords = list(geom.exterior.coords)
                folium.Polygon(
                    locations=[(y, x) for x, y in coords],
                    color="blue",
                    weight=2,
                    fill=True,
                    fill_opacity=0.2,
                ).add_to(m)
            elif geom.geom_type == "MultiPolygon":
                for poly in geom.geoms:
                    coords = list(poly.exterior.coords)
                    folium.Polygon(
                        locations=[(y, x) for x, y in coords],
                        color="blue",
                        weight=2,
                        fill=True,
                        fill_opacity=0.2,
                    ).add_to(m)
        except Exception as e:
            log.error(f"Ошибка при отрисовке полигона: {e}")
            continue

    # 2. Камеры/фото: рисуем только label = 1
    for p in photos:
        label = p["label"]
        decision = p["decision"]

        # пропускаем записи без разрытия
        if label == 0:
            continue

        lat = p["lat"]
        lng = p["lng"]
        cam_name = p["cam_name"]
        name = p["name"]

        # label == 1:
        #  - decision = 0 → зелёный (законно)
        #  - decision = 1 → красный (незаконно)
        if decision == 0:
            color = "green"
        else:
            color = "red"

        popup_text = (
            f"file: {name}<br>"
            f"cam: {cam_name}<br>"
            f"lat={lat}, lng={lng}<br>"
            f"label={label}, decision={decision}"
        )

        folium.CircleMarker(
            location=[lat, lng],
            radius=6,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            popup=folium.Popup(popup_text, max_width=350),
        ).add_to(m)

    m.save(str(OUTPUT_HTML))
    log.info(f"Карта по данным БД сохранена в {OUTPUT_HTML.resolve()}")


def main():
    conn = get_db_conn()
    try:
        geoms, _ = load_active_orders(conn)
        photos = load_checked_photos(conn)
    finally:
        conn.close()

    build_map(photos, geoms)


if __name__ == "__main__":
    main()