import os
import sys
import logging
from pathlib import Path

import psycopg2
import requests
from psycopg2.extras import DictCursor
from dotenv import load_dotenv

load_dotenv()

# ─── Конфиг ──────────────────────────────────────────────────────────────────

DB_HOST = os.getenv("DB_HOST", "")
DB_PORT = os.getenv("DB_PORT", "")
DB_NAME = os.getenv("DB_NAME", "")
DB_USER = os.getenv("DB_USER", "")
DB_PASS = os.getenv("DB_PASSWORD", "")

PROCESSING_DIR = Path("processing")
DOWNLOAD_TIMEOUT = 30      # секунд
CHUNK_SIZE = 1024 * 1024   # 1 MB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


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


def ensure_processing_dir():
    PROCESSING_DIR.mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest: Path) -> bool:
    try:
        with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT) as resp:
            resp.raise_for_status()
            with dest.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    f.write(chunk)
        return True
    except Exception as e:
        log.error(f"Ошибка скачивания {url} -> {dest}: {e}")
        return False


# ─── Основная логика ─────────────────────────────────────────────────────────

def fetch_rows(conn, limit: int | None):
    with conn.cursor(cursor_factory=DictCursor) as cur:
        if limit is None:
            cur.execute("""
                SELECT name, link
                FROM renovation_ii.cam_photos
                WHERE label IS NULL
                  AND decision IS NULL
                ORDER BY id;
            """)
        else:
            cur.execute("""
                SELECT name, link
                FROM renovation_ii.cam_photos
                WHERE label IS NULL
                  AND decision IS NULL
                ORDER BY id
                LIMIT %s;
            """, (limit,))
        return cur.fetchall()


def sync_from_db(limit: int | None = None):
    log.info("=== Загрузка файлов (label IS NULL, decision IS NULL) в ./processing ===")
    ensure_processing_dir()

    conn = get_db_conn()
    try:
        rows = fetch_rows(conn, limit)
    finally:
        conn.close()

    log.info(f"Найдено записей для скачивания: {len(rows)}")

    downloaded = 0
    skipped = 0

    for row in rows:
        name = row["name"]
        link = row["link"]

        dest_path = PROCESSING_DIR / name

        if dest_path.exists():
            log.info(f"Уже существует, пропускаем: {dest_path}")
            skipped += 1
            continue

        log.info(f"Скачиваем {link} -> {dest_path}")
        ok = download_file(link, dest_path)
        if ok:
            downloaded += 1

    log.info(f"Готово. Скачано: {downloaded}, пропущено (уже есть): {skipped}")
    log.info("=== Завершено ===")


if __name__ == "__main__":
    # Парсим необязательный лимит из аргументов CLI
    arg_limit: int | None = None
    if len(sys.argv) >= 2:
        try:
            arg_limit = int(sys.argv[1])
        except ValueError:
            log.error(f"Некорректный лимит: {sys.argv[1]!r}, ожидаю целое число")
            sys.exit(1)

    sync_from_db(limit=arg_limit)