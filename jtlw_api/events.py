"""事件與 webhook 投遞

- 事件外框：event_id（全域唯一，去重用）＋ event_seq（同一件作業內連續，漏收判斷用）
- 簽章：HMAC-SHA256 對 "{timestamp}.{原始 body 位元組}"，標頭 X-JTLW-Signature: v1=<hex>
  密鑰輪替期間兩組同時簽章，任一組符合即可
- 投遞失敗最多重試 5 次（1、2、4、8…上限 30 秒）
"""
import hashlib
import hmac
import json
import random
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone

from . import config
from . import log as jlog

_ULID_CHARS = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def ulid():
    """時間可排序的唯一 ID（Crockford Base32）"""
    ms = int(time.time() * 1000)
    out = []
    for _ in range(10):
        ms, rem = divmod(ms, 32)
        out.append(_ULID_CHARS[rem])
    body = "".join(reversed(out))
    rand = "".join(random.choice(_ULID_CHARS) for _ in range(16))
    return body + rand


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") + \
        f"{datetime.now(timezone.utc).microsecond // 1000:03d}Z"


def sign(secret, ts, raw_body):
    return hmac.new(secret.encode("utf-8"), f"{ts}.".encode("utf-8") + raw_body,
                    hashlib.sha256).hexdigest()


class EventBus:
    """產生事件、寫入儲存、投遞 webhook"""

    def __init__(self, store, settings):
        self.store = store
        self.settings = settings
        self._lock = threading.Lock()
        self._listeners = []          # 測試用：直接收事件，不經 webhook

    def add_listener(self, fn):
        self._listeners.append(fn)

    def emit(self, job, etype, data, deliver=True):
        """建立事件並寫入；job 會就地更新 last_event_seq 與 updated_at"""
        with self._lock:
            job["last_event_seq"] = job.get("last_event_seq", 0) + 1
            event = {
                "event_id": "evt_" + ulid(),
                "event_seq": job["last_event_seq"],
                "type": etype,
                "job_id": job["job_id"],
                "external_ref": job.get("external_ref") or {},
                "occurred_at": now_iso(),
                "data": data,
            }
            job["updated_at"] = event["occurred_at"]
        self.store.add_event(job["job_id"], event)
        for fn in self._listeners:
            try:
                fn(event)
            except Exception:
                pass
        if deliver and job.get("_webhook"):
            self.deliver(job["_webhook"], event)
        return event

    def deliver(self, endpoint_id, event):
        ep = self.store.get_webhook(endpoint_id)
        if not ep:
            return
        raw = json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        threading.Thread(target=self._deliver_loop, args=(ep, event, raw), daemon=True).start()

    def _deliver_loop(self, ep, event, raw):
        """投遞並記錄每一次嘗試。

        原本這裡把所有例外吞掉、也不留任何紀錄——webhook 沒送到時兩邊都不知道，
        只能靠對方回報「我沒收到」再從頭猜。現在每次嘗試都記結果，
        用 event_id 就能查到某一則事件投了幾次、為什麼失敗。
        """
        delay = 1.0
        last = None
        for attempt in range(1, config.WEBHOOK_MAX_ATTEMPTS + 1):
            ts = str(int(time.time()))     # 每次投遞重新產生時間戳
            sigs = ",".join("v1=" + sign(s, ts, raw) for s in ep["secrets"])
            req = urllib.request.Request(ep["url"], data=raw, method="POST", headers={
                "Content-Type": "application/json",
                "X-JTLW-Event-Id": event["event_id"],
                "X-JTLW-Timestamp": ts,
                "X-JTLW-Signature": sigs,
                "X-JTLW-Delivery": str(attempt),
            })
            t0 = time.monotonic()
            try:
                with urllib.request.urlopen(req, timeout=10) as resp:
                    ms = int((time.monotonic() - t0) * 1000)
                    if 200 <= resp.status < 300:
                        jlog.log("webhook.delivered", event_id=event["event_id"],
                                 job_id=event.get("job_id"), type=event.get("type"),
                                 endpoint_id=ep.get("endpoint_id"), attempt=attempt,
                                 status=resp.status, ms=ms)
                        return
                    last = f"HTTP {resp.status}"
            except urllib.error.HTTPError as e:
                last = f"HTTP {e.code}"
            except (urllib.error.URLError, OSError, ValueError) as e:
                last = f"{type(e).__name__}: {str(e)[:120]}"
            jlog.warn("webhook.retry", event_id=event["event_id"],
                      job_id=event.get("job_id"), type=event.get("type"),
                      endpoint_id=ep.get("endpoint_id"), attempt=attempt,
                      reason=last, ms=int((time.monotonic() - t0) * 1000),
                      next_retry_s=round(delay, 1))
            time.sleep(delay / max(self.settings.speed, 0.001))
            delay = min(delay * 2, 30)
        jlog.error("webhook.failed", event_id=event["event_id"],
                   job_id=event.get("job_id"), type=event.get("type"),
                   endpoint_id=ep.get("endpoint_id"),
                   attempts=config.WEBHOOK_MAX_ATTEMPTS, last_reason=last,
                   url_host=urllib.parse.urlsplit(ep["url"]).netloc)
