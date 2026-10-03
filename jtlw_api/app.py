"""jtlw REST API v1（正式版）

端點與交付給 JTDT 的 mock server 相同，差別在於這裡會真的做語音處理。
契約以 jtlw_api/schemas/jtlw-api-v1.schema.json 為準。
"""
import asyncio
import contextlib
import hashlib
import json
import os
import queue
import secrets
import shutil
import threading
import time

import jsonschema
from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response

from . import config
from .config import Settings
from .engine import STAGES, STAGE_FOR_TASK, Engine, EngineError, glossary_from_entries, set_glossary
from . import log as jlog
from .events import EventBus, now_iso, ulid
from .store import Store

SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "schemas", "jtlw-api-v1.schema.json")
with open(SCHEMA_PATH, encoding="utf-8") as _f:
    SCHEMA = json.load(_f)


def _validator(name):
    return jsonschema.Draft202012Validator(
        {"$schema": SCHEMA["$schema"], "$defs": SCHEMA["$defs"], "$ref": f"#/$defs/{name}"},
        format_checker=jsonschema.Draft202012Validator.FORMAT_CHECKER)


REQUEST_VALIDATOR = _validator("JobCreateRequest")
RETRY_VALIDATOR = _validator("JobRetryRequest")
REQUEST_FIELDS = set(SCHEMA["$defs"]["JobCreateRequest"]["properties"])
RETRY_FIELDS = set(SCHEMA["$defs"]["JobRetryRequest"]["properties"])
TASKS = tuple(SCHEMA["$defs"]["Task"]["enum"])
TERMINAL = ("succeeded", "partially_succeeded", "failed", "cancelled")

ERROR_META = {
    "summary_not_ready": ("request", True),
    "source_unreachable": ("source", True), "source_auth_failed": ("source", False),
    "source_not_allowed": ("source", False), "source_checksum_mismatch": ("source", True),
    "glossary_unreachable": ("source", True),
    "unsupported_media": ("media", False), "audio_too_long": ("media", False),
    "source_too_large": ("media", False),
    "queue_full": ("capacity", True), "worker_unavailable": ("dependency", True),
    "disk_full": ("capacity", True),
    "asr_failed": ("processing", True), "diarization_degraded": ("processing", False),
    "llm_unavailable": ("dependency", True), "llm_failed": ("dependency", True),
    "profile_not_found": ("request", False), "glossary_too_large": ("request", False),
    "cancelled": ("processing", False), "not_found": ("request", False),
    "invalid_request": ("request", False), "task_not_supported": ("request", False),
    "language_not_supported": ("request", False),
    "unauthorized": ("auth", False), "forbidden": ("auth", False),
    "internal_error": ("internal", True),
}
ERROR_MESSAGES = {
    "summary_not_ready": "會議摘要還沒做好",
    "source_not_allowed": "網址不在允許的主機清單內",
    "glossary_unreachable": "無法取得詞彙表檔案",
    "glossary_too_large": "詞彙表筆數超過上限",
    "queue_full": "佇列已滿，請稍後再送件",
    "asr_failed": "語音辨識失敗",
    "llm_unavailable": "LLM 伺服器沒有回應",
    "llm_failed": "LLM 校正失敗",
    "profile_not_found": "找不到指定的 profile",
    "not_found": "找不到指定的作業",
    "invalid_request": "請求內容不正確",
    "unauthorized": "缺少或無效的 API Key",
    "forbidden": "API Key 沒有這個操作的權限",
    "cancelled": "作業已取消",
    "task_not_supported": "所選 profile 不支援部分任務",
    "language_not_supported": "所選 profile 不支援指定的語言",
    "audio_too_long": "音訊長度超過上限",
    "source_too_large": "檔案大小超過上限",
    "unsupported_media": "無法解碼的音訊格式",
    "source_unreachable": "無法取得音訊檔",
    "source_auth_failed": "音訊來源認證失敗",
    "source_checksum_mismatch": "音訊檔與 sha256 或大小不符",
    "internal_error": "內部錯誤",
}

PROFILE_VERSION = "2026-09-29.1"   # 09-23：會議 profile 加韓文（2.3）；09-24：名稱與說明補日文、mock 對齊；09-28：加 summarize（2.4）；
                                   # 09-29：meeting.detailed 標為停用（從來沒有與 balanced 不同的處理）；台語說明寫明適用情境
PROFILES = [
    {"id": "meeting.balanced", "version": PROFILE_VERSION, "default": True, "deprecated": False,
     "replacement_profile_id": None,
     "name": {"zh-Hant": "會議（平衡）", "en": "Meeting (balanced)", "ja": "会議（バランス）"},
     "description": {"zh-Hant": "一般會議錄音，速度與準確度平衡",
                     "en": "General meetings, balanced speed and accuracy",
                     "ja": "一般的な会議録音。速度と精度のバランス"},
     "capabilities": ["transcribe", "diarize", "correct", "summarize"],
     "languages": ["zh-Hant", "en", "ja", "ko", "und"]},
    # 停用（v2.25.3）：說明寫「較慢但更準」，程式卻從來沒有對應的處理，結果與 balanced 完全相同。
    # 仍然接受、照 balanced 處理（送件的回應帶 profile_deprecated 警告），不讓已經選了它的呼叫端失敗
    {"id": "meeting.detailed", "version": PROFILE_VERSION, "default": False, "deprecated": True,
     "replacement_profile_id": "meeting.balanced",
     "name": {"zh-Hant": "會議（精細，已停用）", "en": "Meeting (detailed, deprecated)", "ja": "会議（詳細・廃止）"},
     "description": {"zh-Hant": "已停用：處理方式與「會議（平衡）」相同，請改用平衡",
                     "en": "Deprecated: processed exactly like Meeting (balanced); use that instead",
                     "ja": "廃止：「会議（バランス）」と同じ処理です。そちらを使ってください"},
     "capabilities": ["transcribe", "diarize", "correct", "summarize"],
     "languages": ["zh-Hant", "en", "ja", "ko", "und"]},
    {"id": "transcribe.taiwanese", "version": PROFILE_VERSION, "default": False, "deprecated": False,
     "replacement_profile_id": None,
     "name": {"zh-Hant": "台語轉錄", "en": "Taiwanese transcription", "ja": "台湾語の文字起こし"},
     # 適用情境寫在這裡（JTDT 回覆 v2.21：呼叫端的設定頁直接顯示這段，不另外寫）；依據見 BENCHMARKS.md
     "description": {"zh-Hant": "台語（台灣閩南語）轉錄，也能處理夾雜的華語，文字寫成華語用字；英文大多會被翻成中文；"
                                "不分講者。華語為主、只偶爾一兩句台語的會議，一般模式比較好",
                     "en": "Taiwanese Hokkien transcription; also handles mixed-in Mandarin, written in Mandarin wording. "
                           "English is mostly translated into Chinese. No speaker separation. For mostly-Mandarin "
                           "meetings with only occasional Taiwanese, the general meeting profile works better",
                     "ja": "台湾語（台湾閩南語）の文字起こし。混ざった華語も認識し、華語の表記で出力します。"
                           "英語はほとんど中国語に翻訳されます。話者は区別しません。華語が中心で台湾語が時々入る程度の会議は、"
                           "一般の会議モードのほうが適しています"},
     "capabilities": ["transcribe", "correct", "summarize"],
     "languages": ["nan-Hant", "zh-Hant"]},
]
PROFILES_ETAG = '"' + hashlib.sha256(
    json.dumps(PROFILES, sort_keys=True).encode()).hexdigest()[:16] + '"'

STATE = None


class ApiState:
    """伺服器的共用狀態：設定、儲存、事件、佇列與工作執行緒"""

    def __init__(self, settings=None):
        self.settings = settings or Settings()
        self.store = Store(self.settings.db_path)
        self.bus = EventBus(self.store, self.settings)
        self.engine = Engine(self.settings, self.store, self.bus)
        self.queue = queue.Queue()
        # 抓檔佇列：送件後**立刻**開始抓音檔，抓完才進處理佇列。
        # 為什麼不在 worker 裡抓：對方常用短效簽章網址（JTDT 是 2 小時），
        # 而 worker 只有一個、序列處理，一小時的會議要跑十幾分鐘——
        # 排在第 10 件之後才抓就會拿到「網址已過期」的 404，
        # 而那看起來像「檔案不見了」不像「網址過期」。
        self.fetch_queue = queue.Queue()
        # 會議摘要佇列（api_revision 2.4）：摘要要跑幾十次 LLM，放在自己的執行緒，
        # 辨識的工作執行緒做完逐字稿就去接下一件（JTDT 的辨識不會被 jtvc 的摘要卡住）
        self.summary_queue = queue.Queue()
        self.cancelled = set()
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self._threads = []

    def start(self):
        self._requeue_unfinished()
        t = threading.Thread(target=self._fetcher, name="jtlw-fetcher", daemon=True)
        t.start()
        self._threads.append(t)
        t = threading.Thread(target=self._summary_worker, name="jtlw-summary", daemon=True)
        t.start()
        self._threads.append(t)
        for i in range(max(1, self.settings.workers)):
            t = threading.Thread(target=self._worker, name=f"jtlw-worker-{i}", daemon=True)
            t.start()
            self._threads.append(t)
        t = threading.Thread(target=self._expire_loop, daemon=True)
        t.start()
        self._threads.append(t)

    def stop(self):
        self._stop.set()

    def _requeue_unfinished(self):
        """重啟後接續：queued / running 的作業重新排隊（已完成的階段不重跑）"""
        # 先掃掉已經結束的作業留下的工作檔（舊版的失敗路徑不會清，會一直累積）
        alive = {j["job_id"] for j in self.store.list_jobs()
                 if j["status"] not in TERMINAL}
        try:
            for name in os.listdir(self.settings.work_dir):
                if name.split(".")[0] not in alive:
                    with contextlib.suppress(OSError):
                        os.unlink(os.path.join(self.settings.work_dir, name))
        except OSError:
            pass
        for job in self.store.list_jobs():
            if job["status"] in ("queued", "running", "cancelling"):
                if (job.get("progress") or {}).get("stage") == "summary" \
                        and job["tasks"].get("transcribe") == "succeeded":
                    # 逐字稿已經做完、正在摘要：只重跑摘要，不重跑辨識
                    job["status"] = "running"
                    self.store.put_job(job)
                    self.summary_queue.put(job["job_id"])
                    continue
                job["status"] = "queued"
                self.store.put_job(job)
                # 已經抓到檔而且檔案還在 → 直接進處理佇列，不要重抓
                # （重抓對短效網址來說多半會失敗，而檔案本來就還在）
                sp = job.get("_source_path")
                if sp and (sp == "(fake)" or os.path.isfile(sp)):
                    self.queue.put(job["job_id"])
                else:
                    job.pop("_source_path", None)
                    self.fetch_queue.put(job["job_id"])

    def _fetcher(self):
        """送件後立刻抓音檔，抓完才放進處理佇列。

        磁碟是這裡唯一的限制：預抓等於把還沒輪到的作業的檔案先堆在磁碟上。
        空間不夠時**不硬撐也不等**——直接放行讓 worker 照舊在輪到時才抓，
        並在 `warnings` 說明原因。對方用短效網址時可能因此拿到過期錯誤，
        但那總比整台機器磁碟寫滿、所有作業一起失敗好。
        """
        while not self._stop.is_set():
            try:
                job_id = self.fetch_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            job = self.store.get_job(job_id)
            if not job or job["status"] in TERMINAL or job_id in self.cancelled:
                continue
            if "fetch" not in (job.get("_stages") or []):
                # 重試時逐字稿已經做好（不重跑拉檔、辨識）：不必再抓一次來源
                self.queue.put(job_id)
                continue
            try:
                need = int((job.get("_source") or {}).get("size_bytes") or 0)
                free = shutil.disk_usage(self.settings.work_dir).free
                if need and free - need < config.FETCH_DISK_RESERVE_BYTES:
                    # **只記日誌，不寫進 job 的 warnings**：schema 的 Warning.code
                    # 是列舉，新增值會讓已經照現有 schema 寫契約測試的用戶端變紅。
                    # 要讓對方看得到的話，得先談好 schema 改版。
                    jlog.warn("prefetch.skipped", job_id=job_id, reason="disk_reserve",
                              need_bytes=need, free_bytes=free,
                              reserve_bytes=config.FETCH_DISK_RESERVE_BYTES)
                else:
                    with jlog.Timer() as t:
                        job["_source_path"] = self.engine._fetch(job)
                    job["_stage_ms"] = {**(job.get("_stage_ms") or {}), "fetch": t.ms}
                    self.store.put_job(job)
                    jlog.log("prefetch.done", job_id=job_id, ms=t.ms)
            except EngineError as e:
                # 抓檔失敗就是作業失敗，不必等排到才發現
                self._fail(job, e.code, e.stage or "fetch", e.details)
                continue
            except Exception as e:
                self._fail(job, "internal_error", "fetch", {"reason": str(e)[:200]})
                continue
            self.queue.put(job_id)

    def _worker(self):
        while not self._stop.is_set():
            try:
                job_id = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._run_job(job_id)
            except Exception as e:                      # 不讓工作執行緒死掉
                job = self.store.get_job(job_id)
                if job:
                    self._fail(job, "internal_error", None, {"reason": str(e)[:200]})

    def _run_job(self, job_id):
        job = self.store.get_job(job_id)
        if not job or job["status"] in TERMINAL:
            return
        job["status"] = "running"
        job["queue_position"] = None
        self.store.put_job(job)

        def cancelled():
            return job_id in self.cancelled

        try:
            outcome = self.engine.run(job, cancelled)
        except EngineError as e:
            self._fail(job, e.code, e.stage, e.details)
            return
        if outcome == "cancelled" or cancelled():
            self.cancelled.discard(job_id)
            self._finish(job, "cancelled")
            return
        if job["tasks"].get("summarize") == "pending" and job["tasks"].get("transcribe") == "succeeded":
            self.store.put_job(job)
            self.summary_queue.put(job_id)       # 摘要交給摘要執行緒，這條執行緒去接下一件
            return
        self._complete(job)

    def _summary_worker(self):
        while not self._stop.is_set():
            try:
                job_id = self.summary_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            job = self.store.get_job(job_id)
            try:
                if not job or job["status"] in TERMINAL:
                    continue

                def cancelled():
                    return job_id in self.cancelled
                outcome = self.engine.summarize(job, cancelled)
                if outcome == "cancelled" or cancelled():
                    self.cancelled.discard(job_id)
                    self._finish(job, "cancelled")
                    continue
                self._complete(job)
            except Exception as e:                      # 不讓摘要執行緒死掉；逐字稿已經做好，只讓摘要失敗
                if job:
                    job["tasks"]["summarize"] = "failed"
                    job.setdefault("errors", []).append(
                        {"code": "internal_error", "category": "internal", "retryable": False,
                         "message": "internal_error", "stage": "summary", "task": "summarize",
                         "details": {"reason": str(e)[:200]}})
                    self._complete(job)

    def _complete(self, job):
        """依各任務結果決定終態：逐字稿失敗＝failed；其他任務失敗＝partially_succeeded"""
        failed = [t for t, s in job["tasks"].items() if s in ("failed",)]
        if failed and "transcribe" not in failed:
            self._finish(job, "partially_succeeded")
        elif failed:
            self._finish(job, "failed")
        else:
            self._finish(job, "succeeded")

    def _sweep_work_files(self, job_id):
        """刪掉這件作業的工作檔。

        `engine.run()` 的 finally 本來就會清，但**預抓之後、還沒輪到就失敗或
        取消的作業不會經過那裡**——預抓把這個窗口變長了，不補會漏檔案。
        """
        wd = self.settings.work_dir
        try:
            for name in os.listdir(wd):
                if name.startswith(job_id):
                    with contextlib.suppress(OSError):
                        os.unlink(os.path.join(wd, name))
        except OSError:
            pass

    def _finish(self, job, status):
        self._sweep_work_files(job["job_id"])
        job["status"] = status
        job["progress"] = None
        job["_terminal_at"] = time.time()
        job["result_url"] = (f"/api/v1/jobs/{job['job_id']}/result"
                             if status in ("succeeded", "partially_succeeded") else None)
        self.store.put_job(job)
        if not self._keeps_upload(job):
            self._release_upload(job)
        # 每種終態事件要帶的欄位不同（見 schema 的 Event）
        if status == "failed":
            data = {"tasks": dict(job["tasks"]), "errors": job.get("errors") or []}
        elif status == "cancelled":
            data = {"segment_count": job.get("segment_count", 0)}
        else:
            data = {"tasks": dict(job["tasks"]), "result_url": job["result_url"],
                    "segment_count": job.get("segment_count", 0)}
        self.bus.emit(job, f"job.{status}", data)
        self.store.put_job(job)

    # ── 上傳的來源檔（v2.25.3）─────────────────────────────
    # 2.4 時作業一結束就刪（engine._cleanup），辨識失敗後 retry 必定再失敗一次（upload_consumed），
    # 錯誤卻標 retryable: true——只用上傳的 jtvc 只能重新上傳。現在：逐字稿失敗、而且錯誤可以重試時保留，
    # 直到 retry 成功、ACK、DELETE 或內容到期；_expire_loop 每分鐘再掃一次，任何一條路漏了都會被收掉。
    @staticmethod
    def _keeps_upload(job):
        return (job.get("status") == "failed" and job.get("content_available")
                and not job.get("acknowledged")
                and (job.get("tasks") or {}).get("transcribe") == "failed"
                and any(e.get("retryable") for e in job.get("errors") or []))

    def _release_upload(self, job):
        src = job.get("_source") or {}
        if src.get("type") != "upload":
            return
        up = self.store.get_upload(src.get("upload_id") or "")
        if up and os.path.isfile(up["path"]):
            with contextlib.suppress(OSError):
                os.unlink(up["path"])
            jlog.log("upload.released", upload_id=up["upload_id"], job_id=job.get("job_id"))

    def _sweep_uploads(self):
        """保險：已交給作業的上傳檔，作業不在了、或已結束而且不需要留給 retry 的就刪"""
        for up in self.store.claimed_uploads():
            if not os.path.isfile(up["path"]):
                continue
            job = self.store.get_job(up["job_id"])
            if job is None or (job["status"] in TERMINAL and not self._keeps_upload(job)):
                with contextlib.suppress(OSError):
                    os.unlink(up["path"])
                jlog.log("upload.released", upload_id=up["upload_id"], job_id=up["job_id"], by="sweep")

    def _fail(self, job, code, stage, details=None):
        self._sweep_work_files(job["job_id"])
        err = make_error(code, stage=stage, details=details)
        job.setdefault("errors", []).append(err)
        for t, s in job["tasks"].items():
            if s == "pending":
                job["tasks"][t] = "failed" if STAGE_FOR_TASK.get(t) == stage or t == "transcribe" else "skipped"
        self._finish(job, "failed")

    def _expire_loop(self):
        while not self._stop.wait(60):
            to_clear, to_delete = self.store.expired_jobs(config.UNACKED_TTL_SEC, config.RECORD_TTL_SEC)
            for job in to_clear:
                self.store.clear_content(job["job_id"])
                job["content_available"] = False
                self.store.put_job(job)
                self.bus.emit(job, "job.expired", {"expired_at": now_iso(), "reason": "unacked_ttl"})
                self.store.put_job(job)
            for job in to_delete:
                self.store.delete_job(job["job_id"])          # 留著的上傳檔由 store 一起刪
            self._sweep_uploads()        # 內容過期（上面清掉的）、ACK 漏刪等，留著的上傳檔都在這裡收
            for up in self.store.stale_uploads(config.UPLOAD_TTL_SEC):
                with contextlib.suppress(OSError):
                    os.unlink(up["path"])
                self.store.delete_upload(up["upload_id"])
                jlog.log("upload.expired", upload_id=up["upload_id"], client=up["client"])


def make_error(code, stage=None, task=None, retry_after_ms=None, details=None):
    category, retryable = ERROR_META.get(code, ("internal", False))
    if (details or {}).get("reason") == "upload_consumed":
        retryable = False       # 上傳的錄影已經刪了：retry 一定再失敗，要重新上傳送新的一件
    err = {"code": code, "category": category, "retryable": retryable,
           "message": ERROR_MESSAGES.get(code, code)}
    if stage:
        err["stage"] = stage
    if task:
        err["task"] = task
    if retry_after_ms is not None:
        err["retry_after_ms"] = retry_after_ms
    if details:
        err["details"] = details
    return err


def error_response(status, code, **kw):
    return JSONResponse({"error": make_error(code, **kw)}, status_code=status)


def auth(authorization, scope):
    if STATE is None:
        return None, error_response(500, "internal_error",
                                    details={"reason": "not_initialized"})
    if not authorization or not authorization.startswith("Bearer "):
        return None, error_response(401, "unauthorized")
    presented = authorization[len("Bearer "):].strip()
    info = STATE.settings.key_info(presented)
    if not info:
        # 金鑰本身又帶著前綴時講清楚是哪一種錯。照文件複製「Authorization: Bearer <金鑰>」
        # 整串貼進設定欄位是最自然的動作，而原本的訊息「缺少或無效的 API Key」指向金鑰，
        # 管理員會去要一把新金鑰——但金鑰是對的，錯的只是多了前綴（JTDT 2026-09-22 回報）。
        # **必須要求前綴後面真的有空白**：寫成「開頭是 bearer 就算」的話，
        # 真的以那幾個字母開頭的金鑰會被誤判。不回傳金鑰內容。
        low = presented.lower()
        reason = None
        if low.startswith("bearer ") or low.startswith("bearer\t"):
            reason = "key_has_bearer_prefix"
        elif low.startswith("authorization:"):
            reason = "key_has_header_name_prefix"
        if reason:
            return None, error_response(401, "unauthorized",
                                        details={"reason": reason})
        return None, error_response(401, "unauthorized")
    if scope not in info["scopes"] and "admin" not in info["scopes"]:
        return None, error_response(403, "forbidden", details={"required_scope": scope})
    return info["client"], None


def public_job(job):
    """對外的作業物件：去掉底線開頭的內部欄位。排隊中的 `queue_position`（前面還有幾件）每次重算"""
    out = {k: v for k, v in job.items() if not k.startswith("_")}
    if job.get("status") == "queued" and STATE is not None and getattr(STATE, "store", None) is not None:
        out["queue_position"] = STATE.store.jobs_ahead(job)
    out.setdefault("glossary", None)
    out.setdefault("correction_level", None)
    return json.loads(json.dumps(out, ensure_ascii=False))


def get_job(client, job_id):
    job = STATE.store.get_job(job_id)
    if not job or job["_client"] != client:
        return None
    return job



#: source 是 oneOf（url／upload）：type 對應第幾支
SOURCE_BRANCH = {SCHEMA["$defs"][r["$ref"].rsplit("/", 1)[-1]]["properties"]["type"]["const"]: i
                 for i, r in enumerate(SCHEMA["$defs"]["Source"]["oneOf"])}


def _schema_error(body, validator):
    """最相關的一個錯誤。source 的 oneOf 要挑 type 對得上的那一支，否則 best_match 會回
    「source.type 不是 url」——送的明明是 upload，錯在 upload_id 格式（api_revision 2.4）"""
    picked = []
    for e in validator.iter_errors(body):
        if e.validator == "oneOf" and list(e.absolute_path) == ["source"]:
            i = SOURCE_BRANCH.get((body.get("source") or {}).get("type"))
            sub = [c for c in e.context if c.schema_path and c.schema_path[0] == i]
            if sub:
                picked.append(jsonschema.exceptions.best_match(sub))
                continue
        picked.append(e)
    return jsonschema.exceptions.best_match(picked)

def validate_body(body, kind="JobCreateRequest"):
    fields, validator = ((REQUEST_FIELDS, REQUEST_VALIDATOR) if kind == "JobCreateRequest"
                         else (RETRY_FIELDS, RETRY_VALIDATOR))
    unknown = sorted(set(body) - fields)
    if unknown:
        return error_response(400, "invalid_request",
                              details={"field": unknown[0], "reason": "unknown_field"})
    err = _schema_error(body, validator)
    if err is None:
        return None
    field = ".".join(str(p) for p in err.absolute_path) or "body"
    return error_response(400, "invalid_request", details={"field": field, "reason": err.validator})


@contextlib.asynccontextmanager
async def lifespan(_app):
    global STATE
    if STATE is None:
        STATE = ApiState()
    jlog.init(STATE.settings.data_dir)
    jlog.log("api.start", data_dir=STATE.settings.data_dir,
             correction_model=STATE.settings.correction_model,
             summary_model=STATE.settings.summary_model,
             asr_model=STATE.settings.asr_model,
             remote_whisper=(STATE.settings.remote_whisper or {}).get("host"),
             llm=f"{STATE.settings.llm_host}:{STATE.settings.llm_port}",
             allowed_hosts=sorted(STATE.settings.allowed_hosts) or "（未限制）")
    STATE.start()
    yield
    STATE.stop()


app = FastAPI(title="jt-live-whisper API", version=config.API_REVISION, lifespan=lifespan)


@app.middleware("http")
async def request_log(request, call_next):
    """配一個 X-Request-Id、記錄結果、把同一個 id 回給用戶端。

    產生 id 與回傳 id 必須在**同一個**中介層做——拆成兩個的話，用戶端沒帶
    X-Request-Id 時兩邊會各產生一個，日誌裡的 id 和回給對方的 id 就對不起來。

    用戶端可以自己帶 X-Request-Id 進來（我們照原樣回傳），這樣他們的日誌
    與我們的日誌就能對上；回報問題時引用這個 id，我們查得到那一次。
    """
    rid = request.headers.get("X-Request-Id") or ("req_" + ulid())
    t0 = time.monotonic()
    status = 500
    resp = None
    try:
        resp = await call_next(request)
        status = resp.status_code
        resp.headers["Date"] = time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime())
        resp.headers["X-Request-Id"] = rid
        return resp
    finally:
        ms = int((time.monotonic() - t0) * 1000)
        # 健康檢查每幾秒一次，記了只會淹沒日誌
        if request.url.path != "/api/v1/health":
            jlog.log("request.done" if status < 400 else "request.failed",
                     level="info" if status < 400 else "warn",
                     request_id=rid, method=request.method, path=request.url.path,
                     status=status, ms=ms,
                     client_ip=request.client.host if request.client else None)


# ── 基本資訊 ──────────────────────────────────────────────
@app.get("/api/v1/health")
async def health():
    return {"status": "ok"}


@app.get("/api/v1/capabilities")
async def capabilities(authorization: str = Header(None)):
    _client, err = auth(authorization, "profiles:read")
    if err:
        return err
    s = STATE.settings
    running = STATE.store.count_status("running")
    return {
        "version": _app_version(), "api_version": config.API_VERSION,
        "api_revision": config.API_REVISION,
        "capabilities": list(TASKS),
        "limits": s.limits,
        "queue": {"depth": STATE.store.count_status("queued"), "running": running,
                  "max_depth": s.max_queue_depth},
        "workers": await asyncio.to_thread(_workers_info),
    }


def _app_version():
    import translate_meeting as tm
    return tm.APP_VERSION


def _probe_json(url, timeout=2.0):
    """GET 一個 JSON 端點；連不上或格式不對回 None（不拋例外）"""
    import urllib.request
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def _workers_info():
    """後端各服務的**實際**狀態。

    2026-09-23 之前這裡的 health 是寫死的 "ok"、GPU 的 version 是寫死的 "remote"，
    從來沒有去問過後面那台——GPU 伺服器曾經停了 29 天，這裡照樣回 ok。
    JTDT 想把它接進「測試連線」，那樣寫死的值就是在騙人。

    每次呼叫都即時探測（各 2 秒逾時），不快取：這個端點是給人按「測試連線」用的。
    """
    s = STATE.settings
    out = []
    rw = s.remote_whisper or {}
    if rw.get("host"):
        base = f"http://{rw['host']}:{rw.get('whisper_port', 8978)}"
        h = _probe_json(f"{base}/health")
        w = {"id": "asr-remote", "kind": "asr",
             "endpoint": f"{rw['host']}:{rw.get('whisper_port', 8978)}"}
        if not h or h.get("status") != "ok":
            w.update(health="down", version="unknown")
        else:
            # 伺服器退回 CPU 時仍能用但慢很多 → degraded
            w.update(health="ok" if h.get("gpu") else "degraded",
                     version=str(h.get("version") or "unknown"))
            st = _probe_json(f"{base}/v1/status")
            q = (st or {}).get("queue")
            if isinstance(q, dict):
                # 只算離線線（我們的作業走那條）；v2.21.7 起的伺服器才有 queue
                b = q.get("batch") or {}
                w["load"] = {"running": 1 if b.get("running") else 0,
                             "queued": len(b.get("waiting") or [])}
            elif st is not None:
                w["load"] = {"running": 1 if st.get("busy") else 0, "queued": 0}
        out.append(w)
    else:
        out.append({"id": "asr-local", "kind": "asr", "endpoint": "local",
                    "health": "ok", "version": "faster-whisper"})
    if s.llm_host:
        tags = _probe_json(f"http://{s.llm_host}:{s.llm_port}/api/tags")
        if tags is None:
            tags = _probe_json(f"http://{s.llm_host}:{s.llm_port}/v1/models")
        out.append({"id": "llm", "kind": "llm", "endpoint": f"{s.llm_host}:{s.llm_port}",
                    "health": "ok" if tags is not None else "down",
                    "version": s.correction_model})
    return out


@app.get("/api/v1/profiles")
async def profiles(request: Request, authorization: str = Header(None)):
    _client, err = auth(authorization, "profiles:read")
    if err:
        return err
    if request.headers.get("if-none-match") == PROFILES_ETAG:
        return Response(status_code=304, headers={"ETag": PROFILES_ETAG})
    return JSONResponse(PROFILES, headers={"ETag": PROFILES_ETAG})


# ── Webhook ───────────────────────────────────────────────
@app.post("/api/v1/webhooks")
async def create_webhook(body: dict, authorization: str = Header(None)):
    client, err = auth(authorization, "jobs:write")
    if err:
        return err
    url = (body or {}).get("url", "")
    if not url.startswith(("http://", "https://")):
        return error_response(400, "invalid_request", details={"field": "url"})
    eid = "wh_" + ulid()
    secret = "whsec_" + secrets.token_urlsafe(32)
    STATE.store.put_webhook(eid, client, url, [secret], now_iso())
    return JSONResponse(_webhook_view(eid, secret), status_code=201)


def _webhook_view(eid, secret=None):
    ep = STATE.store.get_webhook(eid)
    out = {"endpoint_id": eid, "url": ep["url"], "created_at": ep["created_at"],
           "active_secrets": len(ep["secrets"])}
    if secret:
        out["secret"] = secret
    return out


@app.get("/api/v1/webhooks")
async def list_webhooks(authorization: str = Header(None)):
    client, err = auth(authorization, "jobs:read")
    if err:
        return err
    return [_webhook_view(ep["endpoint_id"]) for ep in STATE.store.list_webhooks(client)]


@app.delete("/api/v1/webhooks/{eid}")
async def delete_webhook(eid: str, authorization: str = Header(None)):
    client, err = auth(authorization, "jobs:write")
    if err:
        return err
    ep = STATE.store.get_webhook(eid)
    if ep and ep["client"] == client:
        STATE.store.delete_webhook(eid)
    return Response(status_code=204)


@app.post("/api/v1/webhooks/{eid}/rotate")
async def rotate_webhook(eid: str, body: dict = None, authorization: str = Header(None)):
    client, err = auth(authorization, "jobs:write")
    if err:
        return err
    ep = STATE.store.get_webhook(eid)
    if not ep or ep["client"] != client:
        return error_response(404, "not_found")
    if (body or {}).get("retire_old"):
        STATE.store.put_webhook(eid, client, ep["url"], ep["secrets"][-1:], ep["created_at"])
        return _webhook_view(eid)
    secret = "whsec_" + secrets.token_urlsafe(32)
    STATE.store.put_webhook(eid, client, ep["url"], ep["secrets"][-1:] + [secret], ep["created_at"])
    return _webhook_view(eid, secret)


# ── 作業 ──────────────────────────────────────────────────
@app.post("/api/v1/jobs")
async def create_job(request: Request, authorization: str = Header(None),
                     idempotency_key: str = Header(None)):
    client, err = auth(authorization, "jobs:write")
    if err:
        return err
    try:
        body = await request.json()
    except Exception:
        return error_response(400, "invalid_request", details={"field": "body"})
    if not idempotency_key:
        return error_response(400, "invalid_request", details={"field": "Idempotency-Key"})

    body_hash = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    prev = STATE.store.get_idempotent(client, idempotency_key)
    if prev:
        job = STATE.store.get_job(prev["job_id"])
        if job:
            out = public_job(job)
            if prev["body_hash"] != body_hash:
                diff = sorted(_diff_fields(prev["body"], body))
                out["warnings"] = out.get("warnings", []) + [
                    {"code": "idempotency_body_mismatch", "fields": diff}]
            return JSONResponse(out, status_code=200)

    for field in ("profile_id", "tasks", "source"):
        if field not in body:
            return error_response(400, "invalid_request", details={"field": field})
    profile = next((p for p in PROFILES if p["id"] == body["profile_id"]), None)
    if not profile:
        return error_response(404, "profile_not_found", details={"profile_id": body["profile_id"]})
    tasks = list(dict.fromkeys(body["tasks"]))
    if (body.get("source") or {}).get("type") in ("url", "upload") and "transcribe" not in tasks:
        tasks.insert(0, "transcribe")
    unsupported = [t for t in tasks if t not in TASKS or t not in profile["capabilities"]]
    if unsupported:
        return error_response(422, "task_not_supported",
                              details={"profile_id": profile["id"], "tasks": ",".join(unsupported)})
    if "glossary" in body and "glossary_url" in body:
        return error_response(400, "invalid_request",
                              details={"field": "glossary_url",
                                       "reason": "mutually_exclusive_with_glossary"})
    entries = (body.get("glossary") or {}).get("entries") if isinstance(body.get("glossary"), dict) else None
    max_inline = STATE.settings.limits["max_glossary_inline_entries"]
    if isinstance(entries, list) and len(entries) > max_inline:
        return error_response(422, "glossary_too_large",
                              details={"max_inline_entries": max_inline, "entries": len(entries),
                                       "hint": "use glossary_url"})
    bad = validate_body(body)
    if bad:
        return bad
    lang = body.get("language", "auto")
    if lang != "auto" and lang not in profile["languages"]:
        return error_response(422, "language_not_supported",
                              details={"field": "language", "languages": lang})
    if "summarize" in tasks and lang.split("-")[0] in ("ja", "ko"):
        # 會議摘要的引用驗證只認中文與英文（韓文會整條被丟掉），送件當下就講，不要跑完才失敗
        return error_response(422, "language_not_supported",
                              details={"field": "tasks", "task": "summarize", "languages": lang})
    src = body["source"]
    upload = None
    if src.get("type") == "upload":
        upload = STATE.store.get_upload(src.get("upload_id") or "")
        if not upload or upload["client"] != client:
            return error_response(404, "not_found", details={"field": "source.upload_id"})
        if upload["job_id"]:
            return error_response(409, "invalid_request",
                                  details={"field": "source.upload_id", "reason": "already_used",
                                           "job_id": upload["job_id"]})
        if not os.path.isfile(upload["path"]):
            return error_response(404, "not_found", details={"field": "source.upload_id", "reason": "expired"})
    ok, host = STATE.settings.host_allowed(src["url"]) if not upload else (True, None)
    if not ok:
        # **被擋的主機一定要進 log**。回應裡雖然有 details.host，但對方的介面
        # 不一定會顯示（JTDT 2026-09-22 的畫面只印「欄位 source.url」），
        # 管理者就只能猜要把哪個主機加進 allowed_hosts。
        jlog.log("source.rejected", level="warn", host=host,
            allowed=sorted(STATE.settings.allowed_hosts))
        return error_response(422, "source_not_allowed", details={"field": "source.url", "host": host})
    if (upload["size_bytes"] if upload else src["size_bytes"]) > STATE.settings.limits["max_source_bytes"]:
        return error_response(422, "source_too_large",
                              details={"size_bytes": upload["size_bytes"] if upload else src["size_bytes"],
                                       "max_source_bytes": STATE.settings.limits["max_source_bytes"]})
    if body.get("glossary_url"):
        ok, host = STATE.settings.host_allowed(body["glossary_url"]["url"])
        if not ok:
            return error_response(422, "source_not_allowed",
                                  details={"field": "glossary_url", "host": host})
    wh = (body.get("webhook") or {}).get("endpoint_id")
    if wh:
        ep = STATE.store.get_webhook(wh)
        if not ep or ep["client"] != client:
            return error_response(400, "invalid_request", details={"field": "webhook.endpoint_id"})

    queued = STATE.store.count_status("queued")
    if queued >= STATE.settings.max_queue_depth:
        return error_response(429, "queue_full", retry_after_ms=30000)

    stages = [s for s in STAGES
              if s in ("fetch", "normalize", "finalize") or any(STAGE_FOR_TASK[t] == s for t in tasks)]
    # 一筆寫好幾個詞（`Proxmox VE / PVE`）時拆開；統計回報呼叫端送來的筆數（v2.26.8）；錯寫法（v2.26.9）
    gl = glossary_from_entries(entries)
    if gl["problems"]:
        return error_response(400, "invalid_request", details=gl["problems"][0])
    job_id = "job_" + ulid()
    if upload:
        if not STATE.store.claim_upload(upload["upload_id"], job_id):      # 同時送兩件用同一個上傳
            return error_response(409, "invalid_request",
                                  details={"field": "source.upload_id", "reason": "already_used"})
        src = {"type": "upload", "upload_id": upload["upload_id"], "size_bytes": upload["size_bytes"],
               "sha256": upload["sha256"], "filename": upload.get("filename")}
    job = {
        "job_id": job_id, "status": "queued", "queue_position": queued + 1,
        "created_at": now_iso(), "updated_at": now_iso(),
        "profile_id": profile["id"], "profile_version": profile["version"],
        "external_ref": body.get("external_ref") or {},
        "progress": None,
        "tasks": {t: "pending" for t in TASKS if t in tasks},
        "last_event_seq": 0, "segment_count": 0, "acknowledged": False,
        "content_available": True, "result_url": None,
        "glossary": None,                   # 下面算（要用到 _glossary_terms）
        "correction_level": (body.get("correction_level") or "standard") if "correct" in tasks else None,
        "errors": [],
        "warnings": ([{"code": "profile_deprecated", "fields": ["profile_id"]}] if profile.get("deprecated") else []),
        "_language": lang, "_hints": body.get("hints") or {},
        "_client": client, "_stages": stages, "_source": src,
        "_glossary_url": body.get("glossary_url"),
        "_webhook": wh, "_total_audio_ms": None, "_processed_ms": 0,
    }
    set_glossary(job, gl)
    STATE.store.put_job(job)
    job["queue_position"] = STATE.store.jobs_ahead(job)      # 前面還有幾件（沒有就是 0）
    STATE.store.put_idempotent(client, idempotency_key, job_id, body_hash, body)
    STATE.bus.emit(job, "job.accepted", {"status": job["status"],
                                        "queue_position": job["queue_position"],
                                        "profile_version": job["profile_version"]})
    if upload:
        job["_source_path"] = upload["path"]      # 檔案已經在這台：不必抓，直接排進處理佇列
        STATE.store.put_job(job)
        STATE.queue.put(job_id)
    else:
        STATE.store.put_job(job)
        STATE.fetch_queue.put(job_id)
    return JSONResponse(public_job(job), status_code=202)


def _diff_fields(a, b, prefix=""):
    out = set()
    for k in set(a) | set(b):
        pa, pb = a.get(k), b.get(k)
        if isinstance(pa, dict) and isinstance(pb, dict):
            out |= _diff_fields(pa, pb, f"{prefix}{k}.")
        elif pa != pb:
            out.add(f"{prefix}{k}")
    return out


@app.get("/api/v1/jobs/{job_id}")
async def read_job(job_id: str, authorization: str = Header(None)):
    client, err = auth(authorization, "jobs:read")
    if err:
        return err
    job = get_job(client, job_id)
    if not job:
        return error_response(404, "not_found")
    return public_job(job)


@app.get("/api/v1/jobs/{job_id}/events")
async def read_events(job_id: str, after_seq: int = 0, authorization: str = Header(None)):
    client, err = auth(authorization, "jobs:read")
    if err:
        return err
    job = get_job(client, job_id)
    if not job:
        return error_response(404, "not_found")
    return {"job_id": job_id, "events": STATE.store.get_events(job_id, after_seq)}


@app.get("/api/v1/jobs/{job_id}/segments")
async def read_segments(job_id: str, after_seq: int = 0, limit: int = config.SEGMENT_PAGE_DEFAULT,
                        layer: str = "raw", authorization: str = Header(None)):
    client, err = auth(authorization, "jobs:read")
    if err:
        return err
    job = get_job(client, job_id)
    if not job:
        return error_response(404, "not_found")
    if layer not in ("raw", "final", "speakers"):
        return error_response(400, "invalid_request", details={"field": "layer"})
    need_task = {"final": "correct", "speakers": "diarize"}.get(layer)
    if need_task and need_task not in job["tasks"]:
        return error_response(400, "invalid_request",
                              details={"field": "layer", "reason": "task_not_requested",
                                       "task": need_task})
    limit = max(1, min(limit, STATE.settings.limits["max_segments_page"]))
    items, has_more = STATE.store.get_segments(job_id, layer, after_seq, limit)
    stats = STATE.store.layer_stats(job_id, layer)
    done_states = {"raw": ("succeeded", "failed", "skipped"),
                   "final": ("succeeded", "failed", "skipped"),
                   "speakers": ("succeeded", "degraded", "failed", "skipped")}[layer]
    task = {"raw": "transcribe", "final": "correct", "speakers": "diarize"}[layer]
    complete = job["tasks"].get(task) in done_states and not has_more
    return {"job_id": job_id, "layer": layer, "segments": items,
            "next_after_seq": items[-1]["seq"] if has_more and items else None,
            "last_seq": stats["last_seq"], "has_more": has_more, "complete": bool(complete)}


@app.get("/api/v1/jobs/{job_id}/result")
async def read_result(job_id: str, authorization: str = Header(None)):
    client, err = auth(authorization, "jobs:read")
    if err:
        return err
    job = get_job(client, job_id)
    if not job or job["status"] == "failed":
        return error_response(404, "not_found")
    raw = STATE.store.all_segments(job_id, "raw")
    speakers = {}
    for a in STATE.store.all_segments(job_id, "speakers"):
        sp = speakers.setdefault(a["speaker_id"],
                                 {"speaker_id": a["speaker_id"], "segments": 0, "first_seq": a["seq"]})
        sp["segments"] += 1
    return {
        "result_schema_version": config.RESULT_SCHEMA_VERSION, "job_id": job_id,
        "status": job["status"], "tasks": dict(job["tasks"]),
        "duration_ms": job.get("_total_audio_ms"),
        "languages": job.get("_languages") or list(dict.fromkeys(s["language"] for s in raw)),
        "speakers": sorted(speakers.values(), key=lambda x: x["first_seq"]),
        "layers": {
            "raw": STATE.store.layer_stats(job_id, "raw"),
            "final": STATE.store.layer_stats(job_id, "final") if "correct" in job["tasks"] else None,
            "speakers": STATE.store.layer_stats(job_id, "speakers") if "diarize" in job["tasks"] else None,
        },
        "glossary": job.get("glossary"),
        "correction_level": job.get("correction_level"),
        "correction": job.get("_correction"),
        # 每個階段花了多久，讓下游看得出時間花在哪（拉檔？辨識？校正？）
        # 而不是只知道「總共 N 秒」。GPU 伺服器是共用的，排隊時會反映在 asr 上。
        "stage_ms": job.get("_stage_ms") or None,
        "warnings": job.get("warnings") or [],
        # api_revision 2.4：要求 summarize 而且成功時才有值
        "summary_url": (f"/api/v1/jobs/{job_id}/summary"
                        if job["tasks"].get("summarize") == "succeeded" else None),
        # api_revision 2.8（JTDT 要求）：語音辨識用的模型、在哪裡跑（整場一個值；辨識失敗或升級前的作業為 null）
        "asr": job.get("_asr") if job["tasks"].get("transcribe") == "succeeded" else None,
        # api_revision 2.5：講者辨識要求的與實際用的方法（沒有要求 diarize 時為 null）
        # api_revision 2.7：saturated（升級前做完的作業沒有這個欄位，補 False）
        "diarization": ({"saturated": False, **job["_diarization"]} if job.get("_diarization") else job.get("_diarization"))
                       if "diarize" in job["tasks"] else None,
    }


class _TooLarge(Exception):
    pass


@app.post("/api/v1/uploads")
async def create_upload(request: Request, authorization: str = Header(None)):
    """上傳來源檔（api_revision 2.4）：給拿不出下載網址的呼叫端（例如 jtvc 的會議錄影）。
    接受 multipart/form-data 的 file 欄位，或直接把檔案當成 body（檔名放 ?filename= 或 X-Filename）。
    回傳 upload_id，之後送件用 source={"type":"upload","upload_id":…}；一個上傳只能給一件作業，
    沒用掉的 24 小時後刪除，用掉的在作業處理完就刪除。影片可以直接傳（mp4 等），這邊會取出音軌"""
    client, err = auth(authorization, "jobs:write")
    if err:
        return err
    limit = STATE.settings.limits["max_source_bytes"]
    declared = int(request.headers.get("content-length") or 0)
    if declared > limit + 1024 * 1024:
        return error_response(413, "source_too_large", details={"size_bytes": declared, "max_source_bytes": limit})
    updir = os.path.join(STATE.settings.work_dir, "uploads")
    os.makedirs(updir, exist_ok=True)
    free = shutil.disk_usage(updir).free
    if free - max(declared, 0) < config.FETCH_DISK_RESERVE_BYTES:
        return error_response(507, "disk_full", details={"free_bytes": free})
    upload_id = "upl_" + ulid()
    path = os.path.join(updir, upload_id)
    filename = request.query_params.get("filename") or request.headers.get("x-filename")
    h, size = hashlib.sha256(), 0

    def put(out, chunk):
        nonlocal size
        size += len(chunk)
        if size > limit:
            raise _TooLarge()
        h.update(chunk)
        out.write(chunk)
    try:
        with open(path, "wb") as out:
            if (request.headers.get("content-type") or "").startswith("multipart/form-data"):
                form = await request.form()
                f = form.get("file")
                if not hasattr(f, "read"):
                    raise ValueError("file")
                filename = filename or getattr(f, "filename", None)
                while True:
                    chunk = await f.read(1024 * 1024)
                    if not chunk:
                        break
                    put(out, chunk)
            else:
                async for chunk in request.stream():
                    if chunk:
                        put(out, chunk)
    except _TooLarge:
        with contextlib.suppress(OSError):
            os.unlink(path)
        return error_response(413, "source_too_large", details={"max_source_bytes": limit})
    except ValueError:
        with contextlib.suppress(OSError):
            os.unlink(path)
        return error_response(400, "invalid_request", details={"field": "file"})
    except Exception as e:                            # 對方斷線、磁碟滿……不留半個檔
        with contextlib.suppress(OSError):
            os.unlink(path)
        jlog.warn("upload.failed", client=client, reason=str(e)[:200])
        return error_response(400, "invalid_request", details={"field": "body", "reason": "upload_interrupted"})
    if size == 0:
        with contextlib.suppress(OSError):
            os.unlink(path)
        return error_response(400, "invalid_request", details={"field": "file", "reason": "empty"})
    filename = os.path.basename(filename)[:200] if filename else None
    STATE.store.put_upload(upload_id, client, path, filename, size, h.hexdigest())
    jlog.log("upload.done", upload_id=upload_id, client=client, size_bytes=size)
    exp = time.time() + config.UPLOAD_TTL_SEC
    expires = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(exp)) + f".{int(exp * 1000) % 1000:03d}Z"
    return JSONResponse({"upload_id": upload_id, "size_bytes": size, "sha256": h.hexdigest(),
                         "filename": filename, "expires_at": expires}, status_code=201)


def _summary_artifact(client, job_id, name):
    """回傳 (artifact, None) 或 (None, 錯誤回應)"""
    job = get_job(client, job_id)
    if not job:
        return None, error_response(404, "not_found")
    state = job["tasks"].get("summarize")
    if state is None:
        return None, error_response(404, "not_found", details={"reason": "summarize_not_requested"})
    if state == "pending":
        return None, error_response(409, "summary_not_ready", retry_after_ms=10000,
                                    details={"status": job["status"]})
    if state == "failed":
        return None, error_response(404, "not_found", details={"reason": "summarize_failed"})
    art = STATE.store.get_artifact(job_id, name)
    if not art:
        return None, error_response(404, "not_found", details={"reason": "content_cleared"})
    return art, None


@app.get("/api/v1/jobs/{job_id}/summary")
async def read_summary(job_id: str, authorization: str = Header(None)):
    """會議摘要（api_revision 2.4，結構化 JSON；格式見 schema 的 MeetingSummary）"""
    client, err = auth(authorization, "jobs:read")
    if err:
        return err
    art, err = _summary_artifact(client, job_id, "summary.json")
    if err:
        return err
    return Response(art[1], media_type="application/json")


@app.get("/api/v1/jobs/{job_id}/summary.md")
async def read_summary_md(job_id: str, authorization: str = Header(None)):
    """會議摘要（api_revision 2.4，Markdown；與 jtlw 命令列產生的會議記錄同一種寫法）"""
    client, err = auth(authorization, "jobs:read")
    if err:
        return err
    art, err = _summary_artifact(client, job_id, "summary.md")
    if err:
        return err
    return Response(art[1], media_type="text/markdown; charset=utf-8")


@app.post("/api/v1/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, authorization: str = Header(None)):
    client, err = auth(authorization, "jobs:cancel")
    if err:
        return err
    job = get_job(client, job_id)
    if not job:
        return error_response(404, "not_found")
    if job["status"] in TERMINAL:
        return error_response(409, "invalid_request", details={"status": job["status"]})
    STATE.cancelled.add(job_id)
    job["status"] = "cancelling"
    STATE.store.put_job(job)
    return JSONResponse(public_job(job), status_code=202)


@app.post("/api/v1/jobs/{job_id}/retry")
async def retry_job(request: Request, job_id: str, authorization: str = Header(None)):
    client, err = auth(authorization, "jobs:write")
    if err:
        return err
    job = get_job(client, job_id)
    if not job:
        return error_response(404, "not_found")
    try:
        body = await request.json()
    except Exception:
        body = {}
    if not isinstance(body, dict):
        body = {}
    if "glossary" in body and "glossary_url" in body:
        return error_response(400, "invalid_request",
                              details={"field": "glossary_url",
                                       "reason": "mutually_exclusive_with_glossary"})
    entries = (body.get("glossary") or {}).get("entries") if isinstance(body.get("glossary"), dict) else None
    max_inline = STATE.settings.limits["max_glossary_inline_entries"]
    if isinstance(entries, list) and len(entries) > max_inline:
        return error_response(422, "glossary_too_large",
                              details={"max_inline_entries": max_inline, "entries": len(entries),
                                       "hint": "use glossary_url"})
    if body.get("glossary_url"):
        ok, host = STATE.settings.host_allowed(body["glossary_url"]["url"])
        if not ok:
            return error_response(422, "source_not_allowed",
                                  details={"field": "glossary_url", "host": host})
    bad = validate_body(body, "JobRetryRequest")
    if bad:
        return bad

    updates_correction = bool(entries is not None or body.get("glossary_url")
                              or body.get("correction_level"))
    if not job.get("content_available"):
        # ACK 之後（或 7 天到期）逐字稿已清掉，沒有東西可以重跑（v2.26.7，JTDT 要求講清楚）：
        # 呼叫端要讓使用者重新送件。想補專有名詞重跑校正的，要在 ACK 之前 retry
        return error_response(409, "invalid_request", details={"status": job["status"], "reason": "content_cleared"})
    if job["status"] not in ("failed", "partially_succeeded") and not (
            updates_correction and job["status"] == "succeeded" and "correct" in job["tasks"]):
        return error_response(409, "invalid_request", details={"status": job["status"]})

    if entries is not None:
        gl = glossary_from_entries(entries)
        if gl["problems"]:
            return error_response(400, "invalid_request", details=gl["problems"][0])
        set_glossary(job, gl)
    elif body.get("glossary_url"):
        job["_glossary_url"] = body["glossary_url"]
        # 只重跑校正時不會經過「拉檔」階段，詞彙表要在校正前另外下載（v2.26.7 以前新的詞彙表不會生效）
        job["_glossary_reload"] = True
    if body.get("correction_level"):
        job["correction_level"] = body["correction_level"]

    for t, s in job["tasks"].items():
        if s in ("failed", "skipped"):
            job["tasks"][t] = "pending"
    if updates_correction and "correct" in job["tasks"]:
        job["tasks"]["correct"] = "pending"      # 換詞彙庫 → 只重跑校正
        if "summarize" in job["tasks"]:
            job["tasks"]["summarize"] = "pending"    # 逐字稿會變，摘要跟著重做
    job["errors"] = []
    job["status"] = "queued"
    job["result_url"] = None
    job["_terminal_at"] = None
    # 已完成辨識的作業只重跑還沒完成的任務：不重跑拉檔、轉檔、ASR（raw 與 seq 不變），
    # 也不重跑已經成功的語者分離與校正——只有摘要失敗時重送，final 不能因此被重新校正一次（2.4）
    if job["tasks"].get("transcribe") == "succeeded":
        job["_stages"] = [s for s in STAGES if s == "finalize" or any(
            STAGE_FOR_TASK.get(t) == s and st == "pending" for t, st in job["tasks"].items())]
    STATE.store.put_job(job)
    STATE.fetch_queue.put(job_id)
    return JSONResponse(public_job(job), status_code=202)


@app.post("/api/v1/jobs/{job_id}/ack")
async def ack_job(job_id: str, authorization: str = Header(None)):
    client, err = auth(authorization, "jobs:write")
    if err:
        return err
    job = get_job(client, job_id)
    if not job:
        return error_response(404, "not_found")
    if job["status"] not in TERMINAL:
        return error_response(409, "invalid_request", details={"status": job["status"]})
    if not job.get("acknowledged"):
        job["acknowledged"] = True
        job["content_available"] = False
        job["_terminal_at"] = job.get("_terminal_at") or time.time()
        STATE.store.clear_content(job_id)
        STATE.store.put_job(job)
        STATE._release_upload(job)
    return public_job(job)          # 冪等：重送一律回 200


@app.delete("/api/v1/jobs/{job_id}")
async def delete_job(job_id: str, authorization: str = Header(None)):
    client, err = auth(authorization, "jobs:write")
    if err:
        return err
    job = get_job(client, job_id)
    if job:
        STATE.store.delete_job(job_id)       # 留給 retry 的上傳檔也一起刪（store.delete_job）
    return Response(status_code=204)     # 不存在也回 204，可安全重送
