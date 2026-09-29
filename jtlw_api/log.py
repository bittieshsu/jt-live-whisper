"""結構化日誌：出事時要查得出「哪一件、哪一步、多久、為什麼」。

一行一個 JSON，同時寫到檔案與 stderr（systemd 會收進 journald）。
用 JSON 是為了事後能用 jq 過濾，例如：

    jq 'select(.job_id=="job_01ABC")' api.log            # 某一件作業的全部軌跡
    jq 'select(.event=="stage.done" and .ms>10000)' api.log   # 慢的階段
    jq 'select(.level=="error")' api.log

**不可記錄的東西**：API Key、Authorization 標頭、詞彙庫內容、逐字稿內容。
逐字稿是客戶的會議內容，日誌不是存放它的地方；要看內容請查 API 的分層端點。
"""
import json
import os
import sys
import threading
import time
from logging.handlers import RotatingFileHandler

_lock = threading.Lock()
_handler = None
_to_stderr = True

# 這些欄位名稱一旦出現就整個遮蔽，避免哪天有人手滑把它們傳進來
_REDACT = {"authorization", "api_key", "key", "secret", "token", "password"}


def init(data_dir, to_stderr=True, max_bytes=20 * 1024 * 1024, backups=5):
    """開檔；data_dir 通常就是 settings.data_dir"""
    global _handler, _to_stderr
    _to_stderr = to_stderr
    os.makedirs(data_dir, exist_ok=True)
    _handler = RotatingFileHandler(os.path.join(data_dir, "api.log"),
                                   maxBytes=max_bytes, backupCount=backups,
                                   encoding="utf-8")


def _clean(v):
    if isinstance(v, dict):
        return {k: ("***" if k.lower() in _REDACT else _clean(x)) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_clean(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def log(event, level="info", **fields):
    """寫一筆。event 用「名詞.動詞」命名，例如 request.done、stage.done、webhook.failed"""
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()) +
           f".{int(time.time() * 1000) % 1000:03d}",
           "level": level, "event": event}
    for k, v in fields.items():
        if v is not None:
            rec[k] = "***" if k.lower() in _REDACT else _clean(v)
    line = json.dumps(rec, ensure_ascii=False)
    with _lock:
        if _handler is not None:
            try:
                _handler.stream.write(line + "\n")
                _handler.stream.flush()
                if _handler.shouldRollover(type("R", (), {"getMessage": lambda s: line})()):
                    _handler.doRollover()
            except Exception:
                pass
        if _to_stderr:
            try:
                sys.stderr.write(line + "\n")
                sys.stderr.flush()
            except Exception:
                pass


def warn(event, **f):
    log(event, level="warn", **f)


def error(event, **f):
    log(event, level="error", **f)


class Timer:
    """用來量一段工作花多久：with Timer() as t: ...  然後讀 t.ms"""

    def __enter__(self):
        self._t0 = time.monotonic()
        return self

    def __exit__(self, *exc):
        self.ms = int((time.monotonic() - self._t0) * 1000)
        return False
