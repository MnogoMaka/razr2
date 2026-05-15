import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any

import psycopg2
from psycopg2.extras import DictCursor, Json
from dotenv import load_dotenv

from camera_order_match import (
    camera_coordinate_matches_order,
    parse_cameras_lat_lng_azimuth,
)

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

PROCESSING_DIR = Path("processing")

# Радиус поиска ордера (метры) — передаётся в camera_order_match
DISTANCE_TOLERANCE_M = 100.0


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
    Запись cam_photos: is_opening / is_legal — чтобы понимать, обрабатывать ли фото.
    """
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute("""
            SELECT id, name, cam_name, link, is_opening, is_legal
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
    Значение поля cameras (json/jsonb → dict).
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


def update_photo_from_match(conn, photo_id: int, match: Dict[str, Any]) -> None:
    """
    Записывает результат ``camera_coordinate_matches_order`` в cam_photos.
    """
    opening = match.get("opening")
    legal = match.get("legal")
    description = match.get("description") or ""
    order_coord = match.get("order_coord")

    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE renovation_ii.cam_photos
            SET is_opening = %s,
                is_legal = %s,
                order_coord = %s,
                description = %s
            WHERE id = %s;
            """,
            (opening, legal, Json(order_coord) if order_coord is not None else None, description, photo_id),
        )
    conn.commit()
    log.info(
        "Обновлена cam_photos id=%s: is_opening=%s is_legal=%s",
        photo_id,
        opening,
        legal,
    )


# ─── Основная логика ─────────────────────────────────────────────────────────

def main():
    if not PROCESSING_DIR.exists():
        log.error("Папка %s не существует", PROCESSING_DIR)
        return

    conn = get_db_conn()
    try:
        files = sorted(
            [p for p in PROCESSING_DIR.iterdir() if p.is_file()],
            key=lambda p: p.name,
        )

        if not files:
            log.info("В папке %s нет файлов для обработки", PROCESSING_DIR)
            return

        log.info("Найдено файлов для обработки: %s", len(files))

        for image_path in files:
            name = image_path.name
            log.info("Обрабатываем файл: %s", name)

            photo = get_photo_by_name(conn, name)
            if photo is None:
                log.warning("В cam_photos не найдена запись с name=%s", name)
                continue

            cam_name = photo["cam_name"]
            photo_id = photo["id"]
            link = photo.get("link")
            existing_opening = photo.get("is_opening")
            existing_legal = photo.get("is_legal")

            if existing_opening is not None and existing_legal is not None:
                log.info(
                    "Фото уже обработано (id=%s, is_opening=%s, is_legal=%s), пропускаем.",
                    photo_id,
                    existing_opening,
                    existing_legal,
                )
                continue

            log.info("Запись cam_photos id=%s, cam_name=%s", photo_id, cam_name)

            try:
                cameras_value = get_cameras_by_cam_name(conn, cam_name)
            except psycopg2.Error as e:
                log.error("Ошибка при запросе cameras для cam_name=%s: %s", cam_name, e)
                continue

            if cameras_value is None:
                log.warning(
                    "Для shortname=%s не найдено поле cameras в renovation_ii.echd_camera_solr_dds",
                    cam_name,
                )
                continue

            parsed = parse_cameras_lat_lng_azimuth(cameras_value)
            if parsed is None:
                continue

            lat, lng, azimuth_deg = parsed

            print("=" * 80)
            print(f"Файл: {name}")
            print(f"cam_name: {cam_name}")
            print(f"Координаты: lat={lat}, lng={lng}, azimuth_delta={azimuth_deg}")
            print(f"DISTANCE_TOLERANCE_M = {DISTANCE_TOLERANCE_M}")
            print("Поле cameras:")
            print(cameras_value)

            image_link = link if link else str(image_path.resolve())

            match = camera_coordinate_matches_order(
                image_link,
                lat,
                lng,
                azimuth_deg,
                conn=conn,
                distance_m=DISTANCE_TOLERANCE_M,
                close_conn=False,
            )

            print("Результат camera_order_match:")
            for k, v in match.items():
                if k == "yolo" and isinstance(v, dict):
                    print(f"  {k}: label={v.get('label')} (голосов: {len(v.get('votes') or [])})")
                else:
                    print(f"  {k}: {v}")
            print("=" * 80)

            try:
                update_photo_from_match(conn, photo_id, match)
            except psycopg2.Error as e:
                log.error("Ошибка при обновлении cam_photos id=%s: %s", photo_id, e)

    finally:
        conn.close()
        log.info("Соединение с БД закрыто")


if __name__ == "__main__":
    main()
