import sqlite3
import csv
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "shop.db"
DATA_DIR = Path(__file__).parent.parent / "data"


def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS products (
        id INTEGER PRIMARY KEY, name TEXT,
        category TEXT, price REAL, rating REAL, tags TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS users (
        user_id INTEGER PRIMARY KEY, username TEXT, label TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS behaviors (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, product_id INTEGER,
        action TEXT, ts TEXT)""")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_user_ts ON behaviors(user_id, ts DESC)")

    cur.execute("SELECT COUNT(*) FROM products")
    if cur.fetchone()[0] == 0:
        with open(DATA_DIR / "products.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cur.execute("INSERT INTO products VALUES (?,?,?,?,?,?)",
                    (int(row["id"]), row["name"], row["category"],
                     float(row["price"]), float(row["rating"]), row["tags"]))

    cur.execute("SELECT COUNT(*) FROM users")
    if cur.fetchone()[0] == 0:
        with open(DATA_DIR / "users.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cur.execute("INSERT INTO users VALUES (?,?,?)",
                    (int(row["user_id"]), row["username"], row["label"]))

    cur.execute("SELECT COUNT(*) FROM behaviors")
    if cur.fetchone()[0] == 0:
        with open(DATA_DIR / "behaviors.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                cur.execute("INSERT INTO behaviors (user_id,product_id,action,ts) VALUES (?,?,?,?)",
                    (int(row["user_id"]), int(row["product_id"]),
                     row["action"], row["ts"]))

    conn.commit()
    conn.close()


def record_action(user_id: int, product_id: int, action: str):
    ts = datetime.now().isoformat(timespec="seconds")
    conn = get_conn()
    conn.execute("INSERT INTO behaviors (user_id,product_id,action,ts) VALUES (?,?,?,?)",
                 (user_id, product_id, action, ts))
    conn.commit()
    conn.close()


def load_user_history(user_id: int, max_len: int = 20) -> list:
    conn = get_conn()
    rows = conn.execute("""
        SELECT product_id FROM behaviors
        WHERE user_id=? ORDER BY ts DESC LIMIT ?
    """, (user_id, max_len)).fetchall()
    conn.close()
    return [r[0] for r in reversed(rows)]


def load_all_products() -> dict:
    conn = get_conn()
    rows = conn.execute("SELECT id,name,category,price,rating,tags FROM products").fetchall()
    conn.close()
    return {r[0]: dict(r) for r in rows}


def load_all_users() -> list:
    conn = get_conn()
    rows = conn.execute("SELECT user_id,username,label FROM users").fetchall()
    conn.close()
    return [dict(r) for r in rows]
