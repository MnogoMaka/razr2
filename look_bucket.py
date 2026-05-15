import os
import re
import time
import threading
import logging
from datetime import date

import requests
import xml.etree.ElementTree as ET
import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

load_dotenv()

# ─── Конфиг ──────────────────────────────────────────────────────────────────

OAUTH_TOKEN = os.getenv("OAUTH_TOKEN", "").strip()

DB_HOST = os.getenv("DB_HOST", "")
DB_PORT = os.getenv("DB_PORT", "")
DB_NAME = os.getenv("DB_NAME", "")
DB_USER = os.getenv("DB_USER", "")
DB_PASS = os.getenv("DB_PASSWORD", "")

BUCKET      = "kube-cxm-cni"
PREFIX      = "pvc-a6daa919-5f6c-4bca-8838-7d3f103e5fae/cctv/day/"
STORAGE_URL = "https://storage.yandexcloud.net"
IAM_URL     = "https://iam.api.cloud.yandex.net/iam/v1/tokens"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
S3NS       = "{http://s3.amazonaws.com/doc/2006-03-01/}"
DATE_RE    = re.compile(r"\d{4}-\d{2}-\d{2}")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ─── IAM токен ───────────────────────────────────────────────────────────────

class IamToken:
    def __init__(self):
        self.token = None
        self.lock = threading.Lock()
        self.expires = 0

    def get(self) -> str | None:
        if not OAUTH_TOKEN:
            return None
        with self.lock:
            if time.time() < self.expires and self.token:
                return self.token
            try:
                resp = requests.post(
                    IAM_URL,
                    json={"yandexPassportOauthToken": OAUTH_TOKEN},
                    timeout=10,
                )
                resp.raise_for_status()
                self.token = resp.json()["iamToken"]
                self.expires = time.time() + 36000
                log.info("IAM-токен обновлён")
            except Exception as e:
                log.error(f"Ошибка получения IAM-токена: {e}")
                self.token = None
        return self.token


_iam = IamToken()


def yandex_session() -> requests.Session | None:
    token = _iam.get()
    if not token:
        return None
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}"})
    return s


# ─── S3 — разбор ключей ──────────────────────────────────────────────────────

def parse_date_from_key_or_filename(key: str) -> date | None:
    parts = key.split("/")
    for part in parts:
        if DATE_RE.fullmatch(part):
            try:
                return date.fromisoformat(part)
            except ValueError:
                pass

    filename = os.path.basename(key)
    m = DATE_RE.search(filename)
    if m:
        try:
            return date.fromisoformat(m.group())
        except ValueError:
            pass

    return None


def make_name(cam: str, photo_date: date | None, key: str) -> str:
    if photo_date is None:
        photo_date = parse_date_from_key_or_filename(key)

    date_str = photo_date.isoformat() if photo_date else "unknown"
    ext = os.path.splitext(key)[-1].lower() or ".jpg"
    return f"{cam}_{date_str}{ext}"


def make_link(key: str) -> str:
    return f"{STORAGE_URL}/{BUCKET}/{key}"


def find_new_image_items(
    session: requests.Session,
    existing_names: set[str],
    limit_new: int | None = None,
):
    """
    Генератор: обходит S3 под PREFIX и yield'ит только новые фото.

    Элемент:
      { "key": str, "cam": str, "date": date | None, "name": str, "link": str }
    """
    new_count = 0
    continuation_token: str | None = None

    while True:
        params: dict[str, str] = {
            "prefix": PREFIX,
            "list-type": "2",
            "max-keys": "1000",
        }
        if continuation_token:
            params["continuation-token"] = continuation_token

        try:
            resp = session.get(f"{STORAGE_URL}/{BUCKET}", params=params, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            log.error(f"Ошибка запроса к S3: {e}")
            break

        root = ET.fromstring(resp.text)

        for contents in root.findall(f"{S3NS}Contents"):
            key_node = contents.find(f"{S3NS}Key")
            if key_node is None or not key_node.text:
                continue
            key = key_node.text

            ext = os.path.splitext(key)[-1].lower()
            if ext not in IMAGE_EXTS:
                continue

            relative = key[len(PREFIX):]
            parts = relative.split("/")

            if len(parts) < 2:
                continue

            raw_cam = parts[0]
            cam_name = raw_cam.strip()
            if not cam_name:
                continue

            photo_date = None
            if len(parts) >= 3 and DATE_RE.fullmatch(parts[1]):
                try:
                    photo_date = date.fromisoformat(parts[1])
                except ValueError:
                    photo_date = None
            else:
                photo_date = parse_date_from_key_or_filename(key)

            name = make_name(cam_name, photo_date, key)
            if name in existing_names:
                continue

            existing_names.add(name)
            new_count += 1

            yield {
                "key":  key,
                "cam":  cam_name,
                "date": photo_date,
                "name": name,
                "link": make_link(key),
            }

            if limit_new is not None and new_count >= limit_new:
                log.info(f"Набрали {limit_new} новых фото, останавливаем обход S3")
                return

        is_truncated = (root.findtext(f"{S3NS}IsTruncated") or "").lower()
        if is_truncated == "true":
            continuation_token = root.findtext(f"{S3NS}NextContinuationToken")
        else:
            break


# ─── PostgreSQL ──────────────────────────────────────────────────────────────

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


def _table_columns(conn, table: str) -> set[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'renovation_ii' AND table_name = %s;
            """,
            (table,),
        )
        return {r[0] for r in cur.fetchall()}


def ensure_table(conn):
    """
    Создаёт/обновляет renovation_ii.cam_photos:
      is_opening, is_legal (BOOLEAN), actual, order_coord (JSONB), description (TEXT).
    Мигрирует данные из устаревших label/decision (INTEGER), если они ещё есть.
    """
    with conn.cursor() as cur:
        cur.execute("CREATE SCHEMA IF NOT EXISTS renovation_ii;")

        cur.execute("""
            CREATE TABLE IF NOT EXISTS renovation_ii.cam_photos (
                id       SERIAL PRIMARY KEY,
                name     TEXT NOT NULL UNIQUE,
                cam_name TEXT NOT NULL DEFAULT '',
                date     DATE,
                link     TEXT NOT NULL
            );
        """)

        cur.execute("""
            ALTER TABLE renovation_ii.cam_photos
                ADD COLUMN IF NOT EXISTS cam_name TEXT NOT NULL DEFAULT '';
        """)

        for ddl in (
            "ALTER TABLE renovation_ii.cam_photos ADD COLUMN IF NOT EXISTS is_opening BOOLEAN",
            "ALTER TABLE renovation_ii.cam_photos ADD COLUMN IF NOT EXISTS is_legal BOOLEAN",
            "ALTER TABLE renovation_ii.cam_photos ADD COLUMN IF NOT EXISTS actual BOOLEAN",
            "ALTER TABLE renovation_ii.cam_photos ADD COLUMN IF NOT EXISTS order_coord JSONB",
            "ALTER TABLE renovation_ii.cam_photos ADD COLUMN IF NOT EXISTS description TEXT",
        ):
            cur.execute(ddl)

        cols = _table_columns(conn, "cam_photos")
        if "label" in cols:
            log.info("Миграция cam_photos: label/decision → is_opening/is_legal")
            cur.execute("""
                UPDATE renovation_ii.cam_photos
                SET is_opening = (label = 1)
                WHERE label IS NOT NULL;
            """)
            cur.execute("""
                UPDATE renovation_ii.cam_photos
                SET is_legal = (decision = 0)
                WHERE decision IS NOT NULL;
            """)
            cur.execute("ALTER TABLE renovation_ii.cam_photos DROP COLUMN IF EXISTS label;")
            cur.execute("ALTER TABLE renovation_ii.cam_photos DROP COLUMN IF EXISTS decision;")

    conn.commit()
    log.info(
        "Таблица renovation_ii.cam_photos готова "
        "(is_opening, is_legal, actual, order_coord, description)"
    )


def apply_actual_status(conn, *, update: bool = False) -> None:
    """
    Проставляет actual по полю date:
      - date в пределах последних 60 дней от текущей даты → TRUE;
      - иначе → FALSE;
      - date NULL → actual NULL.

    Выполняется только если ``update=True`` (иначе только лог и выход).
    """
    if not update:
        log.info("apply_actual_status: update=False — поле actual не трогаем")
        return

    with conn.cursor() as cur:
        cur.execute("""
            UPDATE renovation_ii.cam_photos
            SET actual = CASE
                WHEN date IS NULL THEN NULL
                WHEN date >= (CURRENT_DATE - INTERVAL '60 days') THEN TRUE
                ELSE FALSE
            END;
        """)
        n = cur.rowcount
    conn.commit()
    log.info("apply_actual_status: обновлено строк (actual): %s", n)


def get_existing_names(conn) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT name FROM renovation_ii.cam_photos;")
        return {row[0] for row in cur.fetchall()}


def insert_new_photos(conn, new_rows: list[tuple]):
    """
    Каждый элемент new_rows: (name, cam_name, date, is_opening, is_legal, link)
    """
    if not new_rows:
        return
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO renovation_ii.cam_photos
                (name, cam_name, date, is_opening, is_legal, link)
            VALUES %s
            ON CONFLICT (name) DO NOTHING;
            """,
            new_rows,
        )
    conn.commit()
    log.info(f"Вставлено новых записей: {len(new_rows)}")


# ─── Основная логика ─────────────────────────────────────────────────────────

def sync(
    limit_new: int | None = None,
    batch_size: int = 50,
    *,
    apply_actual: bool = False,
):
    """
    Синхронизация бакета → БД.

    limit_new:
      * None  — найти и вставить все новые изображения
      * число — найти не более указанного количества новых изображений

    batch_size:
      * размер батча для вставки в БД

    apply_actual:
      * если True — после синхронизации для всех строк cam_photos пересчитывается
        поле ``actual`` по правилу «дата снимка не старше 60 дней от сегодня».
    """
    log.info("=== Синхронизация бакета → БД ===")

    session = yandex_session()
    if session is None:
        log.error("Нет IAM-токена. Проверь OAUTH_TOKEN в .env")
        return

    conn = get_db_conn()
    ensure_table(conn)

    existing = get_existing_names(conn)
    log.info(f"Уже в БД: {len(existing)} записей")

    batch: list[tuple] = []
    total_new = 0

    for item in find_new_image_items(session, existing_names=existing, limit_new=limit_new):
        batch.append((
            item["name"],
            item["cam"],
            item["date"],
            None,
            None,
            item["link"],
        ))
        total_new += 1

        if len(batch) >= batch_size:
            log.info(f"Батч заполнен ({batch_size}), пишем в БД")
            insert_new_photos(conn, batch)
            batch.clear()

    if batch:
        log.info(f"Финальный батч на {len(batch)} записей, пишем в БД")
        insert_new_photos(conn, batch)

    apply_actual_status(conn, update=apply_actual)

    log.info(f"Всего новых фоток записано: {total_new}")
    conn.close()
    log.info("=== Готово ===")


if __name__ == "__main__":
    sync(batch_size=10000)
