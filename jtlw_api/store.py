"""作業與逐字稿的持久化（SQLite）

jtlw 重啟後佇列要能接續、已產出的 segments 不可重算，所以全部落地。
每一次寫入都在單一連線上以鎖序列化（SQLite 本身不適合多執行緒同時寫）。
"""
import json
import os
import sqlite3
import threading
import time

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    client TEXT NOT NULL,
    status TEXT NOT NULL,
    stage TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    terminal_at REAL,
    acknowledged INTEGER NOT NULL DEFAULT 0,
    content_available INTEGER NOT NULL DEFAULT 1,
    last_event_seq INTEGER NOT NULL DEFAULT 0,
    doc TEXT NOT NULL              -- 作業的完整 JSON（對外欄位 + 內部欄位）
);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_client ON jobs(client);

CREATE TABLE IF NOT EXISTS segments (
    job_id TEXT NOT NULL,
    layer TEXT NOT NULL,           -- raw / final / speakers
    seq INTEGER NOT NULL,
    doc TEXT NOT NULL,
    PRIMARY KEY (job_id, layer, seq)
);

CREATE TABLE IF NOT EXISTS events (
    job_id TEXT NOT NULL,
    event_seq INTEGER NOT NULL,
    doc TEXT NOT NULL,
    PRIMARY KEY (job_id, event_seq)
);

CREATE TABLE IF NOT EXISTS webhooks (
    endpoint_id TEXT PRIMARY KEY,
    client TEXT NOT NULL,
    url TEXT NOT NULL,
    secrets TEXT NOT NULL,         -- JSON 陣列，輪替期間會有兩組
    created_at TEXT NOT NULL
);

-- 作業產出的檔案（api_revision 2.4：會議摘要 summary.json / summary.md）。**內容的一部分**：
-- ACK、過期、刪除作業時與逐字稿一起清掉（jtvc 的保留政策靠刪除作業執行）
CREATE TABLE IF NOT EXISTS artifacts (
    job_id TEXT NOT NULL,
    name TEXT NOT NULL,
    content_type TEXT NOT NULL,
    body BLOB NOT NULL,
    PRIMARY KEY (job_id, name)
);
-- 上傳的來源檔（api_revision 2.4，source.type="upload"）：檔案在 work_dir/uploads/，這裡只記錄
CREATE TABLE IF NOT EXISTS uploads (
    upload_id TEXT PRIMARY KEY,
    client TEXT NOT NULL,
    path TEXT NOT NULL,
    filename TEXT,
    size_bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    created_at REAL NOT NULL,
    job_id TEXT                    -- 用來送件之後填上；一個上傳只能給一件作業
);
CREATE TABLE IF NOT EXISTS idempotency (
    client TEXT NOT NULL,
    key TEXT NOT NULL,
    job_id TEXT NOT NULL,
    body_hash TEXT NOT NULL,
    body TEXT NOT NULL,
    PRIMARY KEY (client, key)
);
"""


class Store:
    def __init__(self, db_path):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        with self._lock:
            self._db.executescript(SCHEMA)
            self._db.commit()

    def close(self):
        with self._lock:
            self._db.close()

    # ── 作業 ──────────────────────────────────────────────
    def put_job(self, job):
        """新增或整筆覆蓋一件作業"""
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO jobs (job_id, client, status, stage, created_at,"
                " updated_at, terminal_at, acknowledged, content_available, last_event_seq, doc)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (job["job_id"], job["_client"], job["status"], (job.get("progress") or {}).get("stage"),
                 job["created_at"], job["updated_at"], job.get("_terminal_at"),
                 int(job.get("acknowledged", False)), int(job.get("content_available", True)),
                 job.get("last_event_seq", 0), json.dumps(job, ensure_ascii=False)))
            self._db.commit()

    def get_job(self, job_id):
        with self._lock:
            row = self._db.execute("SELECT doc FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return json.loads(row["doc"]) if row else None

    def list_jobs(self, status=None):
        sql = "SELECT doc FROM jobs"
        args = ()
        if status:
            sql += " WHERE status=?"
            args = (status,)
        sql += " ORDER BY created_at"
        with self._lock:
            rows = self._db.execute(sql, args).fetchall()
        return [json.loads(r["doc"]) for r in rows]

    def jobs_ahead(self, job):
        """排在這一件前面的作業數：正在跑的（含取消中）＋比它早送出、還在排隊的（依送件時間）。
        `queue_position` 的意思是「前面還有幾件」——要每次讀取時算，存下來的值不會跟著前面的作業完成而減少
        （2026-09-28 發現：原本送件時寫一次 queued+1 就不再更新，JTDT 畫面的「前面還有 N 件」一直不動）"""
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*) c FROM jobs WHERE job_id<>? AND "
                "(status IN ('running','cancelling') OR "
                " (status='queued' AND (created_at<? OR (created_at=? AND job_id<?))))",
                # 同一毫秒送出的兩件用 job_id 分先後（ULID 依時間排序），不會兩件都說前面沒人
                (job["job_id"], job["created_at"], job["created_at"], job["job_id"])).fetchone()
        return row["c"]

    def count_status(self, status):
        with self._lock:
            row = self._db.execute("SELECT COUNT(*) c FROM jobs WHERE status=?", (status,)).fetchone()
        return row["c"]

    def delete_job(self, job_id):
        with self._lock:
            for table in ("segments", "events", "artifacts"):
                self._db.execute(f"DELETE FROM {table} WHERE job_id=?", (job_id,))
            self._db.execute("DELETE FROM jobs WHERE job_id=?", (job_id,))
            self._db.execute("DELETE FROM idempotency WHERE job_id=?", (job_id,))
            self._db.execute("DELETE FROM uploads WHERE job_id=?", (job_id,))    # 檔案處理完就刪了，這裡清紀錄
            self._db.commit()

    def clear_content(self, job_id):
        """ACK 或過期：刪掉內容，保留作業紀錄"""
        with self._lock:
            self._db.execute("DELETE FROM segments WHERE job_id=?", (job_id,))
            self._db.execute("DELETE FROM artifacts WHERE job_id=?", (job_id,))
            self._db.commit()

    # ── 作業產出的檔案（會議摘要）───────────────────────────
    def put_artifact(self, job_id, name, content_type, body):
        if isinstance(body, str):
            body = body.encode("utf-8")
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO artifacts (job_id, name, content_type, body)"
                             " VALUES (?,?,?,?)", (job_id, name, content_type, body))
            self._db.commit()

    def get_artifact(self, job_id, name):
        """回傳 (content_type, bytes) 或 None"""
        with self._lock:
            row = self._db.execute("SELECT content_type, body FROM artifacts WHERE job_id=? AND name=?",
                                   (job_id, name)).fetchone()
        return (row["content_type"], bytes(row["body"])) if row else None

    # ── 上傳的來源檔 ──────────────────────────────────────
    def put_upload(self, upload_id, client, path, filename, size_bytes, sha256):
        with self._lock:
            self._db.execute("INSERT INTO uploads (upload_id, client, path, filename, size_bytes, sha256,"
                             " created_at, job_id) VALUES (?,?,?,?,?,?,?,NULL)",
                             (upload_id, client, path, filename, size_bytes, sha256, time.time()))
            self._db.commit()

    def get_upload(self, upload_id):
        with self._lock:
            row = self._db.execute("SELECT * FROM uploads WHERE upload_id=?", (upload_id,)).fetchone()
        return dict(row) if row else None

    def claim_upload(self, upload_id, job_id):
        """把上傳交給一件作業；已經被別件用掉時回 False（同一個上傳不能送兩次）"""
        with self._lock:
            cur = self._db.execute("UPDATE uploads SET job_id=? WHERE upload_id=? AND job_id IS NULL",
                                   (job_id, upload_id))
            self._db.commit()
        return cur.rowcount == 1

    def delete_upload(self, upload_id):
        with self._lock:
            self._db.execute("DELETE FROM uploads WHERE upload_id=?", (upload_id,))
            self._db.commit()

    def stale_uploads(self, ttl):
        """沒有被任何作業用掉、放超過 ttl 秒的上傳"""
        with self._lock:
            rows = self._db.execute("SELECT * FROM uploads WHERE job_id IS NULL AND created_at < ?",
                                    (time.time() - ttl,)).fetchall()
        return [dict(r) for r in rows]

    # ── 逐字稿三層 ────────────────────────────────────────
    def add_segments(self, job_id, layer, items):
        if not items:
            return
        with self._lock:
            self._db.executemany(
                "INSERT OR REPLACE INTO segments (job_id, layer, seq, doc) VALUES (?,?,?,?)",
                [(job_id, layer, it["seq"], json.dumps(it, ensure_ascii=False)) for it in items])
            self._db.commit()

    def replace_layer(self, job_id, layer, items):
        """重跑校正時整層換掉（raw 與 speakers 永遠不動）"""
        with self._lock:
            self._db.execute("DELETE FROM segments WHERE job_id=? AND layer=?", (job_id, layer))
            self._db.commit()
        self.add_segments(job_id, layer, items)

    def get_segments(self, job_id, layer, after_seq=0, limit=500):
        with self._lock:
            rows = self._db.execute(
                "SELECT doc FROM segments WHERE job_id=? AND layer=? AND seq>? ORDER BY seq LIMIT ?",
                (job_id, layer, after_seq, limit + 1)).fetchall()
        items = [json.loads(r["doc"]) for r in rows]
        has_more = len(items) > limit
        return items[:limit], has_more

    def layer_stats(self, job_id, layer):
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*) c, COALESCE(MAX(seq),0) last FROM segments WHERE job_id=? AND layer=?",
                (job_id, layer)).fetchone()
        return {"count": row["c"], "last_seq": row["last"]}

    def all_segments(self, job_id, layer):
        with self._lock:
            rows = self._db.execute(
                "SELECT doc FROM segments WHERE job_id=? AND layer=? ORDER BY seq",
                (job_id, layer)).fetchall()
        return [json.loads(r["doc"]) for r in rows]

    # ── 事件 ──────────────────────────────────────────────
    def add_event(self, job_id, event):
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO events (job_id, event_seq, doc) VALUES (?,?,?)",
                             (job_id, event["event_seq"], json.dumps(event, ensure_ascii=False)))
            self._db.commit()

    def get_events(self, job_id, after_seq=0, limit=1000):
        with self._lock:
            rows = self._db.execute(
                "SELECT doc FROM events WHERE job_id=? AND event_seq>? ORDER BY event_seq LIMIT ?",
                (job_id, after_seq, limit)).fetchall()
        return [json.loads(r["doc"]) for r in rows]

    # ── Webhook ───────────────────────────────────────────
    def put_webhook(self, endpoint_id, client, url, secrets_list, created_at):
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO webhooks (endpoint_id, client, url, secrets, created_at)"
                " VALUES (?,?,?,?,?)",
                (endpoint_id, client, url, json.dumps(secrets_list), created_at))
            self._db.commit()

    def get_webhook(self, endpoint_id):
        with self._lock:
            row = self._db.execute("SELECT * FROM webhooks WHERE endpoint_id=?",
                                   (endpoint_id,)).fetchone()
        if not row:
            return None
        return {"endpoint_id": row["endpoint_id"], "client": row["client"], "url": row["url"],
                "secrets": json.loads(row["secrets"]), "created_at": row["created_at"]}

    def list_webhooks(self, client):
        with self._lock:
            rows = self._db.execute("SELECT endpoint_id FROM webhooks WHERE client=? ORDER BY created_at",
                                    (client,)).fetchall()
        return [self.get_webhook(r["endpoint_id"]) for r in rows]

    def delete_webhook(self, endpoint_id):
        with self._lock:
            self._db.execute("DELETE FROM webhooks WHERE endpoint_id=?", (endpoint_id,))
            self._db.commit()

    # ── Idempotency-Key ───────────────────────────────────
    def get_idempotent(self, client, key):
        with self._lock:
            row = self._db.execute("SELECT * FROM idempotency WHERE client=? AND key=?",
                                   (client, key)).fetchone()
        if not row:
            return None
        return {"job_id": row["job_id"], "body_hash": row["body_hash"], "body": json.loads(row["body"])}

    def put_idempotent(self, client, key, job_id, body_hash, body):
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO idempotency (client, key, job_id, body_hash, body)"
                " VALUES (?,?,?,?,?)",
                (client, key, job_id, body_hash, json.dumps(body, ensure_ascii=False)))
            self._db.commit()

    # ── 維護 ──────────────────────────────────────────────
    def expired_jobs(self, unacked_ttl, record_ttl):
        """回傳 (要清內容的作業, 要整筆刪除的作業)"""
        now = time.time()
        to_clear, to_delete = [], []
        for job in self.list_jobs():
            t = job.get("_terminal_at")
            if not t:
                continue
            if job.get("content_available") and not job.get("acknowledged") and now - t > unacked_ttl:
                to_clear.append(job)
            elif not job.get("content_available") and now - t > unacked_ttl + record_ttl:
                to_delete.append(job)
        return to_clear, to_delete
