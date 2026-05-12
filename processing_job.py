import os
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import psycopg2
from psycopg2.extras import DictCursor
from dotenv import load_dotenv
from ultralytics import YOLO

from shapely import wkt as wkt_loader
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.strtree import STRtree

from pyproj import Transformer

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ─── Конфиг ──────────────────────────────────────────────────────────────────

DB_HOST = os.getenv("DB_HOST", "")
DB_PORT = os.getenv("DB_PORT", "")
DB_NAME = os.getenv("DB_NAME", "")
DB_USER = os.getenv("DB_USER", "")
DB_PASS = os.getenv("DB_PASSWORD", "")

# Список путей к YOLO‑моделям (ансамбль)
PATH = "result_modelv2"
YOLO_MODEL_PATHS = [
    f"{PATH}/model1.pt",
    f"{PATH}/model2.pt",
    f"{PATH}/model3.pt",
    f"{PATH}/model4.pt",
    f"{PATH}/model5.pt",
    f"{PATH}/model6.pt",
    f"{PATH}/model7.pt",
    f"{PATH}/model8.pt",
    f"{PATH}/model9.pt",
    f"{PATH}/model10.pt"
]

PROCESSING_DIR = Path("processing")

# Допуск по расстоянию до ордера, м.
DISTANCE_TOLERANCE_M = 100.0

# Трансформер: WGS84 (EPSG:4326) -> Web Mercator (EPSG:3857), метры
_transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


# ─── БД ──────────────────────────────────────────────────────────────────────

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


def get_photo_by_name(conn, name: str) -> Optional[dict]:
    """
    Забираем также label и decision, чтобы понимать, обрабатывать ли фото.
    """
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute("""
            SELECT id, name, cam_name, label, decision
            FROM renovation_ii.cam_photos
            WHERE name = %s
            LIMIT 1;
        """, (name,))
        row = cur.fetchone()
        if row is None:
            return None
        return dict(row)


def get_cameras_by_cam_name(conn, cam_name: str):
    """
    Возвращает значение поля cameras как есть (обычно уже dict, если тип json/jsonb).
    """
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute("""
            SELECT cameras
            FROM renovation_ii.echd_camera_solr_dds
            WHERE shortname = %s
            LIMIT 1;
        """, (cam_name,))
        row = cur.fetchone()
        if row is None:
            return None
        return row["cameras"]


def update_photo_result(conn, photo_id: int, label: int, decision: int):
    """
    Обновляет поля label и decision для записи cam_photos.
    """
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE renovation_ii.cam_photos
            SET label = %s,
                decision = %s
            WHERE id = %s;
        """, (label, decision, photo_id))
    conn.commit()
    log.info(f"Обновлена cam_photos id={photo_id}: label={label}, decision={decision}")


# ─── Ордеры ──────────────────────────────────────────────────────────────────

def load_active_orders(conn) -> Tuple[List[object], Optional[STRtree]]:
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute("""
            SELECT DISTINCT wkt
            FROM renovation_ii.table_oati_uved_order_raskopki
            WHERE "Виды работ" IS NOT NULL
              AND "Статус" = 'Действует';
        """)
        rows = cur.fetchall()

    geoms_wgs84: List[object] = []

    for row in rows:
        wkt_str = row["wkt"]
        if not wkt_str:
            continue
        try:
            geom = wkt_loader.loads(wkt_str)  # WGS84, lon/lat
        except Exception as e:
            log.error(f"Ошибка парсинга WKT: {e}")
            continue
        geoms_wgs84.append(geom)

    if not geoms_wgs84:
        log.warning("Не удалось распарсить ни один WKT для ордеров")
        return [], None

    tree_wgs84 = STRtree(geoms_wgs84)
    log.info(f"Загружено законных ордеров: {len(geoms_wgs84)}")
    return geoms_wgs84, tree_wgs84


def _transform_lonlat_to_m(lon: float, lat: float) -> Tuple[float, float]:
    x, y = _transformer.transform(lon, lat)
    return x, y


def is_point_legal(lat: float, lng: float,
                   geoms_wgs84: List[object],
                   tree_wgs84: Optional[STRtree],
                   distance_tolerance_m: float) -> bool:
    """
    Законно, если:
      - точка внутри любого ордера (contains/intersects в WGS84),
      - ИЛИ расстояние до ордера в метрах <= distance_tolerance_m.
    """
    if not geoms_wgs84 or tree_wgs84 is None:
        return False

    point_wgs84 = Point(lng, lat)

    # 1. быстрый bbox-фильтр
    candidate_geoms = tree_wgs84.query(point_wgs84)

    try:
        if len(candidate_geoms) == 0:
            return False
    except TypeError:
        return False

    # 2. сначала проверяем попадание внутрь полигона / на границу
    for geom in candidate_geoms:
        # фильтруем сразу всё, что не geometry
        if not hasattr(geom, "geom_type"):
            continue

        if geom.contains(point_wgs84) or geom.intersects(point_wgs84):
            return True

    if distance_tolerance_m <= 0:
        return False

    # 3. считаем расстояние в метрах
    x_p, y_p = _transform_lonlat_to_m(lng, lat)
    point_m = Point(x_p, y_p)

    for geom in candidate_geoms:
        # снова фильтруем не‑геометрии
        if not hasattr(geom, "geom_type"):
            continue

        if geom.geom_type == "Polygon":
            exterior = [_transform_lonlat_to_m(x, y) for x, y in geom.exterior.coords]
            interiors = [
                [_transform_lonlat_to_m(x, y) for x, y in ring.coords]
                for ring in geom.interiors
            ]
            geom_m = Polygon(exterior, interiors)
        elif geom.geom_type == "MultiPolygon":
            parts = []
            for poly in geom.geoms:
                exterior = [_transform_lonlat_to_m(x, y) for x, y in poly.exterior.coords]
                interiors = [
                    [_transform_lonlat_to_m(x, y) for x, y in ring.coords]
                    for ring in poly.interiors
                ]
                parts.append(Polygon(exterior, interiors))
            geom_m = MultiPolygon(parts)
        else:
            # игнорируем LineString/Point/иные типы
            continue

        dist = geom_m.distance(point_m)
        if dist <= distance_tolerance_m:
            return True

    return False

# ─── YOLO (ансамбль классификаторов) ────────────────────────────────────────

def load_yolo_models() -> List[YOLO]:
    models = []
    if not YOLO_MODEL_PATHS:
        raise RuntimeError("Список YOLO_MODEL_PATHS пуст")

    for path in YOLO_MODEL_PATHS:
        if not path:
            continue
        log.info(f"Загружаем YOLO модель (классификация) из: {path}")
        models.append(YOLO(path))

    if not models:
        raise RuntimeError("Не удалось загрузить ни одной YOLO‑модели")
    return models


def run_ensemble_classification(models: List[YOLO], image_path: Path) -> Optional[Tuple[int, List[Dict[str, Any]]]]:
    """
    Прогоняет изображение через несколько моделей и возвращает:
      - итоговый label по большинству голосов (0 или 1),
      - список предсказаний по моделям:
          [{"cls_id": int, "cls_name": str, "conf": float}, ...]
    """
    if not image_path.exists():
        log.error(f"Файл не найден: {image_path}")
        return None

    votes_0 = 0
    votes_1 = 0
    details: List[Dict[str, Any]] = []

    for idx, model in enumerate(models, start=1):
        log.info(f"Запускаем YOLO (classification) модель #{idx} на: {image_path}")
        results = model(str(image_path))

        if not results:
            log.warning(f"YOLO модель #{idx} не вернула результатов")
            continue

        res = results[0]
        probs = getattr(res, "probs", None)
        if probs is None:
            log.error(f"В результате модели #{idx} нет .probs — возможно, это не классификационная модель")
            continue

        cls_id = int(probs.top1)
        conf = float(probs.top1conf)

        if hasattr(model, "names"):
            cls_name = model.names.get(cls_id, str(cls_id))
        else:
            cls_name = str(cls_id)

        details.append({"cls_id": cls_id, "cls_name": cls_name, "conf": conf})

        if cls_id == 1:
            votes_1 += 1
        else:
            votes_0 += 1

        log.info(f"  Модель #{idx}: cls={cls_name} (id={cls_id}), conf={conf:.3f}")

    if not details:
        log.warning("Ни одна модель не дала валидного предсказания")
        return None

    # большинством голосов
    if votes_1 > votes_0:
        label = 1
    elif votes_0 > votes_1:
        label = 0
    else:
        # ничья: можно выбрать, что считать дефолтом
        # здесь по умолчанию считаем 'нет разрытия' (0) более безопасным
        label = 0
        log.info(f"Ничья по голосам (0={votes_0}, 1={votes_1}), выбираем label=0 по умолчанию")

    log.info(f"Итоговый ансамблевый label: {label} (голоса: 0={votes_0}, 1={votes_1})")
    return label, details


# ─── Парсинг cameras ─────────────────────────────────────────────────────────

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


# ─── Основная логика ─────────────────────────────────────────────────────────

def main():
    if not PROCESSING_DIR.exists():
        log.error(f"Папка {PROCESSING_DIR} не существует")
        return

    conn = get_db_conn()

    # 1. Загружаем ордера (WGS84 + STRtree)
    geoms_wgs84, tree_wgs84 = load_active_orders(conn)

    # 2. Загружаем ансамбль YOLO моделей
    models = load_yolo_models()

    try:
        files = sorted(
            [p for p in PROCESSING_DIR.iterdir() if p.is_file()],
            key=lambda p: p.name,
        )

        if not files:
            log.info(f"В папке {PROCESSING_DIR} нет файлов для обработки")
            return

        log.info(f"Найдено файлов для обработки: {len(files)}")

        for image_path in files:
            name = image_path.name
            log.info(f"Обрабатываем файл: {name}")

            photo = get_photo_by_name(conn, name)
            if photo is None:
                log.warning(f"В cam_photos не найдена запись с name={name}")
                continue

            cam_name = photo["cam_name"]
            photo_id = photo["id"]
            existing_label = photo.get("label")
            existing_decision = photo.get("decision")

            if existing_label is not None and existing_decision is not None:
                log.info(
                    f"Фото уже обработано (id={photo_id}, label={existing_label}, "
                    f"decision={existing_decision}), пропускаем."
                )
                continue

            log.info(f"Найдена запись cam_photos id={photo_id}, cam_name={cam_name}")

            try:
                cameras_value = get_cameras_by_cam_name(conn, cam_name)
            except psycopg2.Error as e:
                log.error(f"Ошибка при запросе cameras для cam_name={cam_name}: {e}")
                continue

            if cameras_value is None:
                log.warning(
                    f"Для shortname={cam_name} не найдено поле cameras "
                    f"в renovation_ii.echd_camera_solr_dds"
                )
                continue

            lat_lng = parse_cameras_lat_lng(cameras_value)
            if lat_lng is None:
                continue

            lat, lng = lat_lng

            print("=" * 80)
            print(f"Файл: {name}")
            print(f"cam_name: {cam_name}")
            print(f"Координаты камеры: lat={lat}, lng={lng}")
            print(f"DISTANCE_TOLERANCE_M = {DISTANCE_TOLERANCE_M}")
            print("Поле cameras:")
            print(cameras_value)

            ensemble_result = run_ensemble_classification(models, image_path)
            if ensemble_result is None:
                print("Ансамбль моделей не смог выдать классификацию для изображения.")
                print("=" * 80)
                continue

            label, details = ensemble_result
            print("Предсказания моделей:")
            for idx, d in enumerate(details, start=1):
                print(f"  Модель #{idx}: cls={d['cls_name']} (id={d['cls_id']}), conf={d['conf']:.3f}")
            print(f"Итоговый label по большинству: {label}")

            if label == 0:
                decision = 0
                print("Ансамбль: разрытие не детектировано (label=0), decision = 0.")
            else:
                legal = is_point_legal(
                    lat,
                    lng,
                    geoms_wgs84,
                    tree_wgs84,
                    distance_tolerance_m=DISTANCE_TOLERANCE_M,
                )
                if legal:
                    decision = 0
                    print(
                        "РЕЗУЛЬТАТ: разрытие ЗАКОННО "
                        "(точка внутри ордера или не дальше допуска по расстоянию). "
                        "decision = 0."
                    )
                else:
                    decision = 1
                    print(
                        "РЕЗУЛЬТАТ: разрытие НЕЗАКОННО "
                        "(точка вне ордеров и дальше допуска по расстоянию). "
                        "decision = 1."
                    )

            try:
                update_photo_result(conn, photo_id, label=label, decision=decision)
            except psycopg2.Error as e:
                log.error(f"Ошибка при обновлении cam_photos id={photo_id}: {e}")

            print("=" * 80)

    finally:
        conn.close()
        log.info("Соединение с БД закрыто")


if __name__ == "__main__":
    main()