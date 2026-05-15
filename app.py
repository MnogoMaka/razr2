#!/usr/bin/env python3
"""
Карта разрытий: данные только из PostgreSQL (ордера + cam_photos + echd).
Фото по ссылке из БД (link), превью через /cam-thumb/<id>.

Запуск: python app.py  →  http://127.0.0.1:8050/

.env: DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
"""

from __future__ import annotations

import html as html_module
import json
import logging
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote

import folium
import psycopg2
import requests
from dash import Dash, Input, Output, dcc, html, no_update
from dotenv import load_dotenv
from folium.plugins import MarkerCluster
from psycopg2.extras import DictCursor
from shapely import wkt as wkt_loader
from shapely.geometry import mapping
from flask import Response

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("app")

# ─── Конфиг ────────────────────────────────────────────────────────────────

MAP_CENTER = [55.75, 37.62]
MAP_ZOOM = 10
BATCH_SIZE = 250
ARROW_LEN_M = 35.0
THUMB_TIMEOUT_S = 15

ACTIVE_ORDERS_WKT_SQL = """
    SELECT DISTINCT wkt
    FROM renovation_ii.table_oati_uved_order_raskopki
    WHERE "Виды работ" IS NOT NULL
      AND "Статус" = 'Действует';
"""

OPENING_PHOTOS_SQL = """
    SELECT
        p.id,
        p.name,
        p.link,
        p.is_legal,
        p.description,
        p.order_coord,
        p.cam_name,
        d.cameras
    FROM renovation_ii.cam_photos p
    LEFT JOIN renovation_ii.echd_camera_solr_dds d
        ON d.shortname = p.cam_name
    WHERE p.is_opening IS TRUE;
"""

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)
MAP_FILE = os.path.join(ASSETS_DIR, "map.html")

# Глобально после загрузки из БД (для батчевой отрисовки без огромного dcc.Store)
_ORDERS: List[object] = []
_ORDERS_FC: Dict[str, Any] = {"type": "FeatureCollection", "features": []}
_CAMERAS: List[dict] = []
_LOAD_ERROR: Optional[str] = None
_LAST_BUILT_END: int = -1  # сколько камер уже отрисовано в текущем map.html


def get_db_conn():
    db_name = os.getenv("DB_NAME", "").strip()
    db_user = os.getenv("DB_USER", "").strip()
    if not db_name or not db_user:
        raise RuntimeError("DB_NAME/DB_USER не заданы в .env")
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=db_name,
        user=db_user,
        password=os.getenv("DB_PASSWORD", ""),
    )


def _wkt_cell_to_str(value) -> Optional[str]:
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


def parse_cameras_lat_lng_azimuth(cameras_value) -> Optional[Tuple[float, float, Optional[float]]]:
    if isinstance(cameras_value, dict):
        data = cameras_value
    else:
        import ast as _ast

        s = str(cameras_value)
        try:
            data = json.loads(s)
        except Exception:
            try:
                data = _ast.literal_eval(s)
            except Exception:
                return None

    lat = data.get("precise_latitude") or data.get("lat")
    lng = data.get("precise_longitude") or data.get("lng")
    if lat is None or lng is None:
        return None

    azimuth: Optional[float] = None
    az = data.get("azimuth_delta")
    if az is not None:
        try:
            azimuth = float(az)
        except (TypeError, ValueError):
            pass

    return float(lat), float(lng), azimuth


def _meters_per_degree(lat: float) -> Tuple[float, float]:
    lat_rad = math.radians(lat)
    m_per_deg_lat = 111_320.0
    m_per_deg_lon = 111_320.0 * max(math.cos(lat_rad), 1e-6)
    return m_per_deg_lon, m_per_deg_lat


def azimuth_arrow_tip(lat: float, lng: float, azimuth_deg: float, dist_m: float) -> Tuple[float, float]:
    """Азимут: 0° — север, 90° — восток (как compass bearing)."""
    br = math.radians(azimuth_deg)
    m_lon, m_lat = _meters_per_degree(lat)
    dlat = (dist_m * math.cos(br)) / m_lat
    dlng = (dist_m * math.sin(br)) / m_lon
    return lat + dlat, lng + dlng


def polygon_vertex_summary(geom) -> str:
    """Текст координат для popup при клике по полигону."""
    try:
        if geom.geom_type == "Polygon":
            coords = list(geom.exterior.coords)
        elif geom.geom_type == "MultiPolygon":
            coords = list(next(iter(geom.geoms)).exterior.coords)
        else:
            return str(geom)[:2000]
    except Exception:
        return str(geom)[:2000]

    head = coords[:25]
    lines = [f"{lat:.6f}, {lon:.6f}" for lon, lat in head]
    extra = len(coords) - len(head)
    body = "\n".join(lines)
    if extra > 0:
        body += f"\n… ещё {extra} вершин(ы)"
    return body


def load_active_order_geometries(conn) -> List[object]:
    geoms: List[object] = []
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(ACTIVE_ORDERS_WKT_SQL)
        rows = cur.fetchall()
    for row in rows:
        wkt_str = _wkt_cell_to_str(row["wkt"])
        if not wkt_str:
            continue
        try:
            geoms.append(wkt_loader.loads(wkt_str))
        except Exception as e:
            log.warning("WKT ордера не распарсился: %s", e)
    log.info("Ордеров (полигонов) на карте: %s", len(geoms))
    return geoms


def orders_to_feature_collection(geoms: List[object]) -> Dict[str, Any]:
    """Один GeoJSON для всех ордеров — иначе тысячи folium.GeoJson зависают на минуты."""
    features: List[dict] = []
    for i, geom in enumerate(geoms):
        try:
            gj = mapping(geom)
            coord_raw = polygon_vertex_summary(geom)
            # Жёсткий лимит: при 2000+ фичах большие тексты раздувают map.html на десятки МБ
            if len(coord_raw) > 800:
                coord_raw = coord_raw[:800] + "\n…(обрезано)"
            features.append(
                {
                    "type": "Feature",
                    "geometry": gj,
                    "properties": {
                        "coord_text": coord_raw,
                        "idx": i,
                    },
                }
            )
        except Exception as e:
            log.debug("Пропуск геометрии %s в FC: %s", i, e)
    return {"type": "FeatureCollection", "features": features}


def load_opening_cameras(conn) -> List[dict]:
    out: List[dict] = []
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(OPENING_PHOTOS_SQL)
        rows = cur.fetchall()

    for row in rows:
        parsed = parse_cameras_lat_lng_azimuth(row["cameras"])
        if not parsed:
            continue
        lat, lng, az = parsed
        legal = row["is_legal"]
        with_order = legal is True
        out.append(
            {
                "id": int(row["id"]),
                "name": row["name"] or "",
                "link": row["link"] or "",
                "cam_name": row["cam_name"] or "",
                "lat": lat,
                "lng": lng,
                "azimuth_deg": az,
                "with_order": with_order,
                "description": (row["description"] or "").strip(),
                "order_coord": row["order_coord"],
            }
        )
    log.info("Точек камер (is_opening=true, есть lat/lng): %s", len(out))
    return out


def bootstrap_db() -> None:
    global _ORDERS, _ORDERS_FC, _CAMERAS, _LOAD_ERROR
    try:
        conn = get_db_conn()
        try:
            _ORDERS = load_active_order_geometries(conn)
            _ORDERS_FC = orders_to_feature_collection(_ORDERS)
            _CAMERAS = load_opening_cameras(conn)
            _LOAD_ERROR = None
        finally:
            conn.close()
    except Exception as e:
        log.exception("Ошибка загрузки данных из БД")
        _ORDERS = []
        _ORDERS_FC = {"type": "FeatureCollection", "features": []}
        _CAMERAS = []
        _LOAD_ERROR = str(e)


def _order_coord_snippet(order_coord: Any) -> str:
    if not order_coord:
        return ""
    if isinstance(order_coord, dict):
        try:
            return json.dumps(order_coord, ensure_ascii=False)[:800]
        except Exception:
            return str(order_coord)[:800]
    return str(order_coord)[:800]


def marker_tooltip_html(cam: dict) -> str:
    """Текст для всплывающей подсказки (без картинки — фото грузится только по клику)."""
    if cam["with_order"]:
        desc = cam["description"] or "Разрытие с ордером"
        return (
            f"<div style='max-width:260px;font-size:12px'>"
            f"<b>{html_module.escape(desc)}</b></div>"
        )
    return (
        "<div style='max-width:260px;font-size:12px'>"
        "<b>Незаконное разрытие</b></div>"
    )


def _safe_photo_url(link: str) -> str:
    """URL фото из БД может содержать пробелы — кодируем безопасно."""
    if not link:
        return ""
    return quote(link, safe=":/?&=#%")


def marker_popup_html(cam: dict) -> str:
    """Popup по клику: превью + кнопка открыть оригинал в новой вкладке.
    `<img>` в Leaflet popup создаётся в DOM только при первом открытии — поэтому
    фото грузится именно по клику."""
    safe_url = _safe_photo_url(cam.get("link", ""))
    pid = cam["id"]
    name = html_module.escape(cam.get("name") or "")
    cam_name = html_module.escape(cam.get("cam_name") or "")
    az = cam.get("azimuth_deg")
    az_str = f"{az:.0f}°" if az is not None else "—"
    title = "Разрытие с ордером" if cam["with_order"] else "Незаконное разрытие"
    title_color = "#1565c0" if cam["with_order"] else "#c62828"

    if safe_url:
        photo_block = (
            f"<a href='{safe_url}' target='_blank' rel='noopener' "
            f"style='text-decoration:none;display:block;margin-top:8px'>"
            f"<img src='{safe_url}' "
            f"style='display:block;max-width:280px;max-height:200px;width:100%;"
            f"height:auto;border-radius:6px;border:1px solid #ccc' "
            f"onerror=\"this.style.display='none';"
            f"this.parentNode.insertAdjacentHTML('beforeend',"
            f"'<div style=&quot;color:#999;font-size:11px;padding:8px 0&quot;>"
            f"Не удалось загрузить фото</div>');\"/>"
            f"<div style='background:{title_color};color:#fff;text-align:center;"
            f"padding:6px;border-radius:4px;margin-top:6px;font-size:12px;"
            f"font-weight:bold'>Открыть в новой вкладке</div></a>"
        )
    else:
        photo_block = (
            "<div style='margin-top:8px;padding:8px;background:#f5f5f5;"
            "border-radius:4px;color:#999;font-size:12px;text-align:center'>"
            "Нет ссылки на фото в БД</div>"
        )

    return (
        f"<div style='min-width:280px;font-size:12px'>"
        f"<div style='font-weight:bold;color:{title_color};font-size:13px'>{title}</div>"
        f"<hr style='margin:4px 0;border:0;border-top:1px solid #eee'/>"
        f"<div><b>Камера:</b> {cam_name}</div>"
        f"<div><b>Файл:</b> {name}</div>"
        f"<div><b>Азимут:</b> {az_str}</div>"
        f"<div><b>photo_id:</b> {pid}</div>"
        f"{photo_block}</div>"
    )


def _map_center_from_geoms(order_geoms: List[object]) -> Optional[List[float]]:
    if not order_geoms:
        return None
    try:
        minx = miny = math.inf
        maxx = maxy = -math.inf
        for g in order_geoms:
            b = g.bounds
            minx, miny = min(minx, b[0]), min(miny, b[1])
            maxx, maxy = max(maxx, b[2]), max(maxy, b[3])
        return [(miny + maxy) / 2, (minx + maxx) / 2]
    except Exception:
        return None


def build_and_save_map(cameras: List[dict]) -> None:
    t0 = time.perf_counter()
    center = _map_center_from_geoms(_ORDERS)
    if center is None and cameras:
        center = [
            sum(c["lat"] for c in cameras) / len(cameras),
            sum(c["lng"] for c in cameras) / len(cameras),
        ]
    elif center is None:
        center = MAP_CENTER

    m = folium.Map(location=center, zoom_start=MAP_ZOOM, tiles="OpenStreetMap")

    # Все ордера одним слоем (иначе Dash-колбэк не успевает → «Updating…»)
    fc = _ORDERS_FC.get("features") or []
    if fc:
        folium.GeoJson(
            _ORDERS_FC,
            name="Ордера",
            style_function=lambda _feature: {
                "fillColor": "#1565c0",
                "color": "#0d47a1",
                "weight": 1.5,
                "fillOpacity": 0.45,
            },
            highlight_function=lambda _feature: {"weight": 2.5, "fillOpacity": 0.65},
            popup=folium.GeoJsonPopup(
                fields=["coord_text"],
                aliases=["Координаты полигона"],
                labels=True,
            ),
        ).add_to(m)

    green = folium.FeatureGroup(name="С ордером")
    red = folium.FeatureGroup(name="Без ордера")
    cluster_green = MarkerCluster(name="Кластер (с ордером)").add_to(green)
    cluster_red = MarkerCluster(name="Кластер (без ордера)").add_to(red)

    for cam in cameras:
        color = "#2e7d32" if cam["with_order"] else "#c62828"
        target = cluster_green if cam["with_order"] else cluster_red
        tooltip_text = marker_tooltip_html(cam)
        popup_html = marker_popup_html(cam)
        folium.CircleMarker(
            location=[cam["lat"], cam["lng"]],
            radius=7,
            color=color,
            weight=2,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            tooltip=folium.Tooltip(tooltip_text, sticky=True),
            popup=folium.Popup(popup_html, max_width=320, lazy=True),
        ).add_to(target)

        if cam["azimuth_deg"] is not None:
            lat2, lng2 = azimuth_arrow_tip(
                cam["lat"], cam["lng"], float(cam["azimuth_deg"]), ARROW_LEN_M
            )
            folium.PolyLine(
                locations=[[cam["lat"], cam["lng"]], [lat2, lng2]],
                color="#424242",
                weight=2,
                opacity=0.75,
                tooltip=f"азимут {cam['azimuth_deg']:.0f}°",
            ).add_to(target)

    green.add_to(m)
    red.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m.save(MAP_FILE)
    dt = time.perf_counter() - t0
    if dt > 1.0:
        log.info("Сборка map.html: %.2f с (камер %s, полигонов %s)", dt, len(cameras), len(fc))


# ─── Dash + Flask ───────────────────────────────────────────────────────────

bootstrap_db()

# Если камер меньше или равно одному батчу — рисуем сразу всё, прогрессивная подгрузка не нужна.
_INITIAL_END = 0 if _LOAD_ERROR else min(BATCH_SIZE, len(_CAMERAS))
try:
    build_and_save_map(_CAMERAS[:_INITIAL_END])
    _LAST_BUILT_END = _INITIAL_END
except Exception:
    log.exception("Первичная сборка map.html")

app = Dash(__name__, assets_folder="assets")
server = app.server


@server.route("/cam-thumb/<int:photo_id>")
def cam_thumb(photo_id: int):
    """Прокси превью по link из cam_photos (без CORS)."""
    try:
        conn = get_db_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT link FROM renovation_ii.cam_photos WHERE id = %s LIMIT 1;",
                    (photo_id,),
                )
                row = cur.fetchone()
        finally:
            conn.close()
        if not row or not row[0]:
            return Response("Нет ссылки", mimetype="text/plain", status=404)
        url = str(row[0]).strip()
        r = requests.get(url, timeout=THUMB_TIMEOUT_S, stream=True)
        r.raise_for_status()
        ct = r.headers.get("Content-Type", "image/jpeg")
        return Response(r.content, mimetype=ct.split(";")[0].strip())
    except Exception as e:
        log.warning("cam-thumb %s: %s", photo_id, e)
        return Response("Ошибка загрузки", mimetype="text/plain", status=502)


_total_cams = len(_CAMERAS)
_need_progressive = _total_cams > BATCH_SIZE
_initial_status = (
    f"Ошибка БД: {_LOAD_ERROR}"
    if _LOAD_ERROR
    else f"Камеры на карте: {_INITIAL_END}/{_total_cams}  |  ордеров: {len(_ORDERS)}"
)

app.layout = html.Div(
    [
        html.Div(
            _initial_status,
            id="map-status",
            style={
                "position": "absolute",
                "top": "8px",
                "left": "50px",
                "zIndex": "1000",
                "background": "rgba(255,255,255,0.92)",
                "padding": "6px 12px",
                "borderRadius": "6px",
                "fontSize": "13px",
                "boxShadow": "0 1px 4px rgba(0,0,0,0.2)",
            },
        ),
        html.Iframe(
            id="map-frame",
            src=f"/assets/map.html?v={int(time.time())}",
            style={
                "width": "100vw",
                "height": "100vh",
                "border": "none",
                "display": "block",
            },
        ),
        dcc.Interval(
            id="map-tick",
            interval=2000,
            n_intervals=0,
            max_intervals=-1,
            disabled=not _need_progressive,
        ),
    ],
    style={"margin": 0, "padding": 0, "position": "relative"},
)


@app.callback(
    Output("map-frame", "src"),
    Output("map-status", "children"),
    Output("map-tick", "disabled"),
    Input("map-tick", "n_intervals"),
    prevent_initial_call=True,
)
def progressive_map(n_intervals: int):
    global _LAST_BUILT_END

    if _LOAD_ERROR:
        return no_update, f"Ошибка БД: {_LOAD_ERROR}", True

    total = len(_CAMERAS)
    n = int(n_intervals or 0)
    # Стартовый батч уже отрисован при старте → следующий = (n+1)+1 батч
    end = min((_INITIAL_END) + n * BATCH_SIZE, total)

    status = f"Камеры на карте: {end}/{total}  |  ордеров: {len(_ORDERS)}"
    done = end >= total

    if end == _LAST_BUILT_END:
        return no_update, status, done

    slice_cams = _CAMERAS[:end]
    build_and_save_map(slice_cams)
    _LAST_BUILT_END = end
    src = f"/assets/map.html?v={int(time.time())}"
    return src, status, done


if __name__ == "__main__":
    print("http://127.0.0.1:8050/")
    app.run(debug=False, port=8050)
