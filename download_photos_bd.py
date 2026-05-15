#!/usr/bin/env python3
"""
download_illegal_photos.py

Скачивает все фотографии из renovation_ii.cam_photos,
у которых is_opening = true и is_legal = false, в папку ./downloaded_illegal.

Имя файла на диске = поле name из таблицы.
Скачивание идёт по URL из поля link.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import psycopg2
from psycopg2.extras import DictCursor
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ─── Конфиг ──────────────────────────────────────────────────────────────────

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "")
DB_USER = os.getenv("DB_USER", "")
DB_PASS = os.getenv("DB_PASSWORD", "")

# Таблица и поля можно при необходимости поправить
TABLE_NAME = "renovation_ii.cam_photos"
FIELD_NAME = "name"
FIELD_LINK = "link"
FIELD_IS_OPENING = "is_opening"
FIELD_IS_LEGAL = "is_legal"

# Куда скачиваем
OUTPUT_DIR = Path("downloaded_illegal")


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


def fetch_illegal_photos(conn) -> list[dict]:
    """
    Возвращает все записи с разрытием (is_opening) и незаконными (is_legal = false).
    """
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(f"""
            SELECT {FIELD_NAME} AS name,
                   {FIELD_LINK} AS link,
                   {FIELD_IS_OPENING} AS is_opening,
                   {FIELD_IS_LEGAL} AS is_legal
            FROM {TABLE_NAME}
            WHERE {FIELD_IS_OPENING} IS TRUE
              AND {FIELD_IS_LEGAL} IS FALSE;
        """)
        rows = cur.fetchall()
    return [dict(r) for r in rows]


# ─── Скачивание ──────────────────────────────────────────────────────────────

def download_file(url: str, dest: Path, timeout: int = 60) -> bool:
    """
    Скачивает файл по URL в dest. Возвращает True при успехе.
    """
    try:
        resp = requests.get(url, stream=True, timeout=timeout)
        resp.raise_for_status()
    except Exception as e:
        log.error(f"Ошибка при скачивании {url}: {e}")
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if not chunk:
                    continue
                f.write(chunk)
        os.replace(tmp, dest)
        return True
    except Exception as e:
        log.error(f"Ошибка записи файла {dest}: {e}")
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        return False


# ─── Основная логика ─────────────────────────────────────────────────────────

def main():
    conn = get_db_conn()
    try:
        rows = fetch_illegal_photos(conn)
        total = len(rows)
        log.info(f"Найдено записей с is_opening=true и is_legal=false: {total}")

        if total == 0:
            return

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        downloaded = 0
        skipped = 0
        errors = 0

        for idx, row in enumerate(rows, start=1):
            name: str = row["name"]
            link: Optional[str] = row["link"]

            log.info(f"[{idx}/{total}] name={name}, link={link}")

            if not link:
                log.warning(f"У записи name={name} пустой link, пропускаю")
                skipped += 1
                continue

            # имя файла как в поле name
            dest_path = OUTPUT_DIR / name

            if dest_path.exists():
                log.info(f"Файл уже существует, пропускаю: {dest_path}")
                skipped += 1
                continue

            ok = download_file(link, dest_path)
            if ok:
                downloaded += 1
                log.info(f"Скачано: {dest_path}")
            else:
                errors += 1
                log.error(f"Ошибка скачивания для name={name}")

        log.info(
            f"Готово. Всего: {total}, скачано: {downloaded}, "
            f"пропущено: {skipped}, ошибок: {errors}"
        )

    finally:
        conn.close()
        log.info("Соединение с БД закрыто")


if __name__ == "__main__":
    main()