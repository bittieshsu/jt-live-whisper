"""Core Engine：聲音 → 逐字稿

階段：fetch → normalize → asr → diarization → correction → finalize
沿用 translate_meeting 既有的底層函式（轉檔、遠端／本機辨識、語者分離、LLM 校正），
不重寫辨識邏輯，也不影響 CLI 與 WebUI。

校正等級：
- standard：沿用 translate_meeting 的逐行把關
- punctuation_only：再加一層，只接受標點、大小寫、空白的變動，
  以及詞彙庫詞彙的替換（公司名、產品名這類）
"""
import difflib
import json
import hashlib
import os
import re
import shutil
import sys
import time
import unicodedata
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import translate_meeting as tm   # noqa: E402

from . import config             # noqa: E402
from . import log as jlog        # noqa: E402
from .events import now_iso      # noqa: E402

STAGES = ["fetch", "normalize", "asr", "diarization", "correction", "finalize", "summary"]
STAGE_FOR_TASK = {"transcribe": "asr", "diarize": "diarization", "correct": "correction",
                  "summarize": "summary"}
#: 會議摘要（api_revision 2.4）的結果格式版本
SUMMARY_SCHEMA_VERSION = "1.0"
#: 台語模式：用 GPU 伺服器的 Breeze-ASR-26（見 Engine._taiwanese_asr）
TAIWANESE_PROFILE = "transcribe.taiwanese"


def _to_tw(text):
    """辨識結果一律轉台灣繁體，與 CLI 同一條規則。

    2026-09-20 拿真實中文音檔實測才發現：CLI 有 28 處呼叫 `_s2twp_safe()`，
    但 v3 API 是另外寫的一條路徑，這條規則從來沒跟過來——raw 層直接吐簡體。
    先前沒看出來是因為 `standard` 校正的 LLM 會順手轉成繁體，把問題蓋住了；
    選 `punctuation_only` 的客戶（JTDT 的預設）就會直接拿到簡體逐字稿。

    日文行不能轉（會動到漢字），判斷方式與 CLI 一致：有假名就跳過。
    """
    text = (text or "").strip()
    if not text or tm._KANA_RE.search(text):
        return text
    return tm._s2twp_safe(text)
_PUNCT_RE = re.compile(r"[\s\W_]+", re.UNICODE)


class EngineError(Exception):
    """帶錯誤碼的失敗，會直接對應到 API 的 error.code"""

    def __init__(self, code, stage=None, details=None):
        super().__init__(code)
        self.code = code
        self.stage = stage
        self.details = details or {}


def _normalize_for_compare(text):
    """比較用：去掉標點、空白與大小寫差異；全形轉半形"""
    text = unicodedata.normalize("NFKC", text)
    return _PUNCT_RE.sub("", text).lower()


def punctuation_only_ok(original, corrected, glossary_terms=()):
    """punctuation_only：只允許標點／大小寫／空白的變動，或詞彙庫詞彙的替換。

    詞彙庫替換要逐段驗證：把兩邊正規化之後做差異比對，**每一個**差異段落
    都必須是「原文的誤聽被換成某個詞彙庫詞彙」才放行。
    只看「長度沒變長」是不夠的——那會讓「已經接好 → 還沒接好」這種
    整句反意的改寫通過（旗標等於沒作用）。
    """
    a = _normalize_for_compare(original)
    b = _normalize_for_compare(corrected)
    if a == b:
        return True
    terms = [t for t in (_normalize_for_compare(x) for x in glossary_terms) if t]
    if not terms:
        return False
    # 校正後的文字裡，詞彙庫詞彙佔到的位置（替換只能發生在這些位置上）
    covered = [False] * len(b)
    for t in terms:
        start = b.find(t)
        while start >= 0:
            for k in range(start, start + len(t)):
                covered[k] = True
            start = b.find(t, start + 1)

    # 合併差異段落：中間相同的部分太短（誤聽的詞彙常被切成好幾塊）就併成同一段
    ops = difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes()
    regions, pending = [], None
    for tag, i1, i2, j1, j2 in ops:
        if tag == "equal":
            # 相同的部分夠長才切段；短的先跳過，之後真的還有差異時會一起被併進來
            if (i2 - i1) >= 4 and pending:
                regions.append(pending)
                pending = None
            continue
        pending = (pending[0], i2, pending[2], j2) if pending else (i1, i2, j1, j2)
    if pending:
        regions.append(pending)

    for i1, i2, j1, j2 in regions:
        old, new = a[i1:i2], b[j1:j2]
        if not new:
            return False                      # 只刪不補，不是詞彙庫替換
        if not all(covered[k] for k in range(j1, j2)):
            return False                      # 改到了詞彙庫範圍以外的地方
        if len(old) > len(new) + 4:
            return False                      # 被換掉的原文太長，是整句改寫不是換詞
    return True


class Engine:
    def __init__(self, settings, store, bus):
        self.settings = settings
        self.store = store
        self.bus = bus
        os.makedirs(settings.work_dir, exist_ok=True)

    # ── 對外：跑完一件作業 ────────────────────────────────
    def run(self, job, cancelled):
        """依作業的階段清單逐一執行。cancelled() 回 True 時盡快收手。"""
        wav_path = None
        jid = job["job_id"]
        timings = job.setdefault("_stage_ms", {})
        jlog.log("job.start", job_id=jid, client=job.get("_client"),
                 tasks=sorted(job.get("tasks") or {}), stages=job.get("_stages"),
                 correction_level=job.get("correction_level"),
                 external_ref=job.get("external_ref"))
        t_all = time.monotonic()
        try:
            for stage in job["_stages"]:
                if stage == "summary":
                    # 會議摘要要跑幾十次 LLM（一小時的會議約十幾分鐘）：交給另一條執行緒（app.py 的
                    # _summary_worker），辨識的工作執行緒做完逐字稿就去接下一件，不讓別人的辨識等摘要
                    continue
                if cancelled():
                    jlog.warn("job.cancelled", job_id=jid, stage=stage)
                    return "cancelled"
                self._set_stage(job, stage)
                if self.settings.fake_engine:
                    # 測試模式下每個階段停一下，讓「執行中」這個狀態真的存在。
                    # 不這樣做的話假引擎會瞬間完成，用輪詢觀察進度的人什麼都看不到。
                    # （2026-09-21 JTDT 正是因為他們的假伺服器第一次就回成功，
                    #   才讓「progress 接錯、進度條不動」整個被測試漏掉。）
                    # 契約測試本身已改為驗事件流，不依賴這個停頓——
                    # 靠時序的測試會時好時壞，那比空過好不了多少。
                    time.sleep(0.05 / max(self.settings.speed, 0.001))
                t0 = time.monotonic()
                try:
                    if stage == "fetch":
                        # 送件當下可能已經先抓過了（見 app.py 的 _fetcher）。
                        # 抓檔提前是為了對方的短效簽章網址——那種網址的有效期
                        # 只涵蓋幾分鐘到幾小時，而作業可能在佇列裡等更久，
                        # 排到才抓會拿到「網址已過期」的 404。
                        if not job.get("_source_path"):
                            job["_source_path"] = self._fetch(job)
                    elif stage == "normalize":
                        wav_path, job["_total_audio_ms"] = self._normalize(job)
                    elif stage == "asr":
                        self._asr(job, wav_path, cancelled)
                    elif stage == "diarization":
                        self._optional(job, "diarize", lambda: self._diarize(job, wav_path))
                    elif stage == "correction":
                        self._optional(job, "correct", lambda: self._correct(job))
                except EngineError as e:
                    timings[stage] = int((time.monotonic() - t0) * 1000)
                    jlog.error("stage.failed", job_id=jid, stage=stage, ms=timings[stage],
                               code=e.code, details=e.details)
                    raise
                timings[stage] = int((time.monotonic() - t0) * 1000)
                jlog.log("stage.done", job_id=jid, stage=stage, ms=timings[stage])
            jlog.log("job.done", job_id=jid, ms=int((time.monotonic() - t_all) * 1000),
                     stage_ms=dict(timings), tasks=job.get("tasks"),
                     audio_ms=job.get("_total_audio_ms"))
            return "succeeded"
        finally:
            self._cleanup(job, wav_path)

    def _optional(self, job, task, fn):
        """語者分離與校正是選配：失敗只讓該任務失敗，逐字稿仍然交付"""
        try:
            fn()
        except EngineError as e:
            jlog.error("task.failed", job_id=job["job_id"], task=task,
                       code=e.code, stage=e.stage, details=e.details)
            job["tasks"][task] = "failed"
            job.setdefault("errors", []).append(
                {"code": e.code, "category": "dependency", "retryable": True,
                 "message": e.code, "stage": e.stage, "task": task, "details": e.details})
            self.bus.emit(job, "task.completed",
                          {"task": task, "status": "failed",
                           "error": {"code": e.code, "category": "dependency", "retryable": True,
                                     "message": e.code}})
            self.store.put_job(job)

    # ── 階段 ─────────────────────────────────────────────
    def _set_stage(self, job, stage):
        idx = job["_stages"].index(stage)
        previous = (job.get("progress") or {}).get("stage")
        job["progress"] = {"stage": stage, "stage_index": idx + 1,
                           "stage_count": len(job["_stages"]),
                           "processed_audio_ms": job.get("_processed_ms", 0),
                           "total_audio_ms": job.get("_total_audio_ms"),
                           "percent": round((idx / max(len(job["_stages"]), 1)) * 100, 1),
                           # api_revision 2.2：在 GPU 伺服器排隊時才有值，其餘為 null
                           "waiting": None}
        job["_progress_saved"] = 0.0
        self.bus.emit(job, "job.stage_changed", {"stage": stage, "previous_stage": previous,
                                                 "progress": job["progress"]})
        self._save(job)

    def _save(self, job):
        """寫回作業，但**不可以把對方的「取消中」蓋掉**。

        store 每次回傳的是資料庫的新副本：引擎手上那份的 status 一直是 running，
        而對方呼叫 cancel 時寫進去的是 cancelling。引擎寫進度時若直接整份蓋回去，
        對方查詢會看到「取消了卻還在跑」。進度改成每秒寫一次之後這個窗口變得很大，
        所以寫之前先看資料庫裡是不是已經被改成 cancelling。
        """
        cur = self.store.get_job(job["job_id"])
        if cur and cur.get("status") == "cancelling" and job.get("status") == "running":
            job["status"] = "cancelling"
        self.store.put_job(job)

    def _progress(self, job, force=False, **fields):
        """更新目前階段的進度。

        api_revision 2.2（2026-09-23，JTDT 要求）：以前 asr／diarization／correction 三個階段
        從開始到結束進度完全不動，對方分不出「在 GPU 排隊」「正在算」「卡住了」。
        現在排隊時填 `waiting`、辨識中更新 `processed_audio_ms`、校正中更新 `items_done`。

        寫資料庫最多每秒一次；`waiting` 有變化（開始排隊／輪到了）時立刻寫，
        那是對方判斷要不要計時的依據，不能晚一秒。
        """
        p = job.get("progress")
        if not p:
            return
        changed_waiting = "waiting" in fields and \
            bool(fields["waiting"]) != bool(p.get("waiting"))
        p.update(fields)
        now = time.monotonic()
        if force or changed_waiting or now - job.get("_progress_saved", 0.0) >= 1.0:
            job["_progress_saved"] = now
            job["updated_at"] = now_iso()     # 對方看得到「還在動」
            self._save(job)

    def _gpu_event_handler(self, job):
        """把 GPU 伺服器的串流事件轉成 progress：queued → waiting；segment → 進度往前"""
        state = {"since": None}

        def on_event(ev):
            t = ev.get("type")
            if t == "queued":
                state["since"] = state["since"] or now_iso()
                self._progress(job, waiting={"reason": "gpu_queue",
                                             "ahead": max(int(ev.get("ahead") or 0), 0),
                                             "since": state["since"]})
            elif t == "segment":
                end_ms = int(round(float(ev.get("end") or 0) * 1000))
                total = job.get("_total_audio_ms")
                if total:
                    end_ms = min(end_ms, total)
                self._progress(job, waiting=None, processed_audio_ms=end_ms)
            elif t == "heartbeat":
                fields = {"waiting": None}
                if ev.get("current") is not None:     # openai-whisper 後端的心跳帶位置
                    fields["processed_audio_ms"] = int(float(ev["current"]) * 1000)
                self._progress(job, **fields)
        return on_event

    def _fetch(self, job):
        """拉音訊（與詞彙庫），驗 sha256 與大小"""
        src = job["_source"]
        if self.settings.fake_engine:
            return "(fake)"
        if src.get("type") == "upload":
            # 上傳的檔案已經在這台（送件時就交給作業了）；處理完會刪掉，所以重啟後可能已經不在
            up = self.store.get_upload(src.get("upload_id") or "")
            if up and os.path.isfile(up["path"]):
                return up["path"]
            raise EngineError("source_unreachable", "fetch",
                              {"reason": "upload_consumed", "hint": "請重新上傳後再送一件"})
        dest = os.path.join(self.settings.work_dir, f"{job['job_id']}.src")
        self._download(src["url"], dest, expect_sha=src.get("sha256"),
                       expect_size=src.get("size_bytes"), stage="fetch")
        if job.get("_glossary_url"):
            g = job["_glossary_url"]
            gp = os.path.join(self.settings.work_dir, f"{job['job_id']}.glossary.json")
            try:
                self._download(g["url"], gp, expect_sha=g.get("sha256"),
                               expect_size=g.get("size_bytes"), stage="fetch")
            except EngineError:
                raise EngineError("glossary_unreachable", "fetch")
            job["_glossary_terms"], job["_glossary_keep"] = self._load_glossary_file(gp)
            job["glossary"] = self._glossary_summary(job)
        return dest

    def _download(self, url, dest, expect_sha=None, expect_size=None, stage="fetch"):
        """下載（支援續傳），核對 sha256 與大小"""
        headers = {}
        token = config.current_source_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        got = 0
        if os.path.exists(dest):
            got = os.path.getsize(dest)
            if got:
                headers["Range"] = f"bytes={got}-"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as resp, \
                    open(dest, "ab" if got and resp.status == 206 else "wb") as f:
                shutil.copyfileobj(resp, f, 1024 * 256)
        except urllib.error.HTTPError as e:
            code = "source_auth_failed" if e.code in (401, 403) else "source_unreachable"
            raise EngineError(code, stage, {"http_status": e.code})
        except (urllib.error.URLError, OSError) as e:
            raise EngineError("source_unreachable", stage, {"reason": str(e)[:200]})

        size = os.path.getsize(dest)
        if expect_size and size != expect_size:
            raise EngineError("source_checksum_mismatch", stage,
                              {"expected_size": expect_size, "actual_size": size})
        if expect_sha:
            h = hashlib.sha256()
            with open(dest, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 256), b""):
                    h.update(chunk)
            if h.hexdigest() != expect_sha:
                raise EngineError("source_checksum_mismatch", stage, {"field": "sha256"})
        return dest

    def _load_glossary_file(self, path):
        import json
        try:
            with open(path, encoding="utf-8") as f:
                entries = json.load(f)
        except (OSError, json.JSONDecodeError):
            raise EngineError("glossary_unreachable", "fetch", {"reason": "invalid_json"})
        if not isinstance(entries, list) or len(entries) > self.settings.limits["max_glossary_entries"]:
            raise EngineError("glossary_too_large", "fetch",
                              {"max_entries": self.settings.limits["max_glossary_entries"]})
        terms = [e.get("source", "") for e in entries if isinstance(e, dict) and e.get("source")]
        keep = [e["source"] for e in entries
                if isinstance(e, dict) and e.get("mode") == "keep" and e.get("source")]
        return terms, keep

    def _glossary_summary(self, job):
        terms = job.get("_glossary_terms") or []
        if not terms:
            return None
        # 遠端辨識（GPU 伺服器）目前無法傳遞 hotwords，只有本機辨識用得到
        bias = 0 if job.get("_use_remote_asr") else min(len(terms), config.ASR_BIAS_MAX_TERMS)
        return {"entries": len(terms), "keep_terms": len(job.get("_glossary_keep") or []),
                "asr_bias_terms": bias}

    def _normalize(self, job):
        """轉 16 kHz 單聲道，取得總長度"""
        if self.settings.fake_engine:
            return None, 15 * 60 * 1000
        src = job["_source_path"]
        probe = tm._ffprobe_info(src)
        if not probe or not probe[0]:
            raise EngineError("unsupported_media", "normalize")
        duration_ms = int(probe[0] * 1000)
        if duration_ms > self.settings.limits["max_duration_ms"]:
            raise EngineError("audio_too_long", "normalize",
                              {"duration_ms": duration_ms,
                               "max_duration_ms": self.settings.limits["max_duration_ms"]})
        if probe[2] and probe[2] < self.settings.limits["min_sample_rate_hz"]:
            raise EngineError("unsupported_media", "normalize", {"sample_rate_hz": probe[2]})
        wav, _is_tmp = tm._convert_to_wav(src, source_label="來源")
        if not wav or not os.path.isfile(wav):
            raise EngineError("unsupported_media", "normalize")
        return wav, duration_ms

    def _asr(self, job, wav_path, cancelled):
        """辨識並分批送出 raw 段落"""
        segments = (self._fake_segments(job) if self.settings.fake_engine
                    else self._real_asr(job, wav_path))
        batch, last_flush = [], time.time()
        seq = 0
        langs = []
        for seg in segments:
            if cancelled():
                break
            seq += 1
            item = {
                "seq": seq,
                "start_ms": int(round(seg["start"] * 1000)),
                "end_ms": int(round(seg["end"] * 1000)),
                "text": _to_tw(seg.get("text") or ""),
                "language": seg.get("language") or job.get("_detected_language") or "und",
                "confidence": seg.get("confidence"),
            }
            langs.append(item["language"])
            batch.append(item)
            job["_processed_ms"] = item["end_ms"]
            if len(batch) >= config.SEGMENT_BATCH or \
                    time.time() - last_flush >= config.SEGMENT_BATCH_SECONDS:
                self._flush_raw(job, batch)
                batch, last_flush = [], time.time()
        if batch:
            self._flush_raw(job, batch)
        job["segment_count"] = seq
        job["_languages"] = list(dict.fromkeys(langs))
        job["tasks"]["transcribe"] = "succeeded"
        self.bus.emit(job, "task.completed", {"task": "transcribe", "status": "succeeded"})
        self.store.put_job(job)

    def _flush_raw(self, job, batch):
        self.store.add_segments(job["job_id"], "raw", batch)
        self.bus.emit(job, "segments.appended",
                      {"first_seq": batch[0]["seq"], "last_seq": batch[-1]["seq"],
                       "segments": batch})

    def _real_asr(self, job, wav_path):
        """遠端 GPU 伺服器優先，失敗降級本機（台語模式另走 _taiwanese_asr）"""
        if job.get("profile_id") == TAIWANESE_PROFILE:
            return self._taiwanese_asr(job, wav_path)
        model = job.get("_asr_model") or self.settings.asr_model
        lang = None if job.get("_language", "auto") == "auto" else job["_language"].split("-")[0]
        rw = self.settings.remote_whisper
        if rw and rw.get("host"):
            try:
                # 沒指定語言時先本機偵測一次，伺服器端不接受 "auto"
                detect_lang = lang or self._detect_language(wav_path)
                segs, _dur, _pt, _dev = tm._remote_whisper_transcribe(
                    rw, wav_path, model, detect_lang, on_event=self._gpu_event_handler(job))
                job["_use_remote_asr"] = True
                # 伺服器端不回語言，用送件指定或本機偵測到的語言標記每一段
                job["_detected_language"] = _to_bcp47(detect_lang)
                return [{"start": s["start"], "end": s["end"], "text": s["text"],
                         "confidence": s.get("confidence"), "language": s.get("language")}
                        for s in segs]
            except Exception as e:
                job.setdefault("warnings", []).append(
                    {"code": "asr_fallback_local", "message": f"GPU 伺服器不可用，改用本機辨識: {e}"[:200]})
        return self._local_asr(job, wav_path, model, lang)

    def _taiwanese_asr(self, job, wav_path):
        """台語模式（profile transcribe.taiwanese）：GPU 伺服器上的 Breeze-ASR-26（v2.25.2 起）。

        2.4 以前這個 profile 雖然列在 /profiles，程式卻從來沒有換模型：language=nan-Hant 送到伺服器
        直接失敗（'nan' is not a valid language code），zh-Hant 則照一般模型辨識成諧音的華語。
        **不退回本機**：只有 CPU 的 API 主機跑一小時的台語會議要約 4 小時，一般模型則會把台語
        辨識成諧音的華語——兩種「成功」都比明講失敗更糟。時間戳是伺服器依語音活動切的 ≤28 秒視窗"""
        rw = self.settings.remote_whisper
        if not (rw and rw.get("host")):
            raise EngineError("asr_failed", "asr", {
                "reason": "台語辨識需要 GPU 伺服器（config.json 的 remote_whisper），API 主機本身不跑台語模型"})
        try:
            segs, _dur, _pt, _dev = tm._remote_whisper_transcribe(
                rw, wav_path, tm.BREEZE_MODEL, "nan", on_event=self._gpu_event_handler(job))
        except Exception as e:
            msg = str(e)
            details = {"reason": msg[:200]}
            if "not a valid language code" in msg or "Breeze" in msg or "400" in msg:
                details["hint"] = "GPU 伺服器需 v2.25.2 以上才支援台語"
            raise EngineError("asr_failed", "asr", details)
        job["_use_remote_asr"] = True
        job["_detected_language"] = "nan-Hant"
        return [{"start": s["start"], "end": s["end"], "text": s["text"], "confidence": None,
                 "language": "nan-Hant"} for s in segs]

    def _detect_language(self, wav_path):
        """用最小的模型偵測語言（只讀前 30 秒），失敗時回退英文"""
        try:
            from faster_whisper import WhisperModel
            m = WhisperModel("base", **tm._fw_device_kwargs())
            _segs, info = m.transcribe(wav_path, vad_filter=True, without_timestamps=True)
            lang = getattr(info, "language", None) or "en"
            del m
            tm._release_gpu_resources()
            return lang
        except Exception:
            return "en"

    def _local_asr(self, job, wav_path, model, lang):
        from faster_whisper import WhisperModel
        kw = dict(tm._FW_OFFLINE_KW)
        terms = job.get("_glossary_terms") or []
        if terms:
            kw["hotwords"] = " ".join(terms[:config.ASR_BIAS_MAX_TERMS])
        try:
            m = WhisperModel(model, **tm._fw_device_kwargs())
            seg_iter, info = m.transcribe(wav_path, language=lang, **kw)
            out = []
            for s in seg_iter:
                conf = None
                if getattr(s, "avg_logprob", None) is not None:
                    conf = round(min(1.0, max(0.0, 2.718281828 ** s.avg_logprob)), 4)
                out.append({"start": s.start, "end": s.end, "text": s.text,
                            "confidence": conf,
                            "language": tm._bcp47(info.language) if hasattr(tm, "_bcp47") else None})
            job["_detected_language"] = _to_bcp47(lang or getattr(info, "language", None))
            del seg_iter, m
            tm._release_gpu_resources()
            return out
        except Exception as e:
            raise EngineError("asr_failed", "asr", {"reason": str(e)[:200]})

    def _diarize(self, job, wav_path):
        raw = self.store.all_segments(job["job_id"], "raw")
        if not raw:
            job["tasks"]["diarize"] = "failed"
            return
        hints = job.get("_hints") or {}
        # api_revision 2.5：呼叫端用 hints.diarize_engine 選（legacy／auto）；沒送＝現行方法（JTDT v2.17 要求，預設不變）
        requested = hints.get("diarize_engine") or config.DIARIZE_ENGINE_DEFAULT
        info = {}
        if self.settings.fake_engine:
            labels = [0 if i % 3 else 1 for i in range(len(raw))]
            info = {"engine": "nemotron" if requested == "auto" else "legacy", "note": None, "reason": None,
                    "saturated": False}
        else:
            segs = [{"start": r["start_ms"] / 1000, "end": r["end_ms"] / 1000, "text": r["text"]}
                    for r in raw]
            try:
                rw = self.settings.remote_whisper
                if rw and rw.get("host") and job.get("_use_remote_asr"):
                    labels, _proc = tm._remote_diarize(rw, wav_path, segs,
                                                       num_speakers=hints.get("num_speakers"),
                                                       on_event=self._gpu_event_handler(job),
                                                       engine=requested, info=info)
                else:
                    labels = tm._diarize_segments(wav_path, segs,
                                                  num_speakers=hints.get("num_speakers"),
                                                  engine=requested, info=info)
            except Exception:
                labels = None
        # 回報實際用了哪個方法（Result.diarization）：auto 在 >8 人、8 人全滿而且現行方法分出超過 8 人（v2.26.4）、或伺服器沒有 Nemotron 時會退回現行方法
        job["_diarization"] = {"requested": requested, "engine": info.get("engine") if labels is not None else None,
                               "note": info.get("note") if labels is not None else None,
                               # api_revision 2.6：退回現行方法的代碼（JTDT 要翻成英文／日文）；note 照舊給人看
                               "reason": info.get("reason") if labels is not None else None,
                               # api_revision 2.7（JTDT 要求）：Nemotron 8 位全滿。照用 Nemotron 時 reason 是 null，
                               # 只看 engine／reason 分不出來；JTDT 拿它提醒「實際發言者更多時請填人數」
                               "saturated": bool(info.get("saturated")) if labels is not None else False}
        degraded = labels is None
        if degraded:
            labels = [0] * len(raw)
            job.setdefault("warnings", []).append(
                {"code": "diarization_degraded", "message": "語者分離失敗，已退回單一語者"})
        if len(labels) != len(raw):
            labels = (list(labels) + [0] * len(raw))[:len(raw)]
        items = [{"seq": r["seq"], "speaker_id": f"S{int(l) + 1}"} for r, l in zip(raw, labels)]
        self.store.add_segments(job["job_id"], "speakers", items)
        job["tasks"]["diarize"] = "degraded" if degraded else "succeeded"
        self.bus.emit(job, "speakers.assigned",
                      {"speaker_ids": sorted({i["speaker_id"] for i in items}),
                       "assignments": items, "degraded": degraded})
        self.bus.emit(job, "task.completed",
                      {"task": "diarize", "status": job["tasks"]["diarize"]})
        self.store.put_job(job)

    def _correct(self, job):
        raw = self.store.all_segments(job["job_id"], "raw")
        if not raw:
            job["tasks"]["correct"] = "failed"
            return
        level = job.get("correction_level") or "standard"
        if self.settings.fake_engine:
            finals = [{"seq": r["seq"], "text": r["text"].replace(" ,", ","), "edited": r["seq"] % 3 == 0}
                      for r in raw]
            rejected = 0
        else:
            finals, rejected = self._llm_correct(job, raw, level)
        self.store.replace_layer(job["job_id"], "final", finals)
        edited = sum(1 for f in finals if f["edited"])
        # 一併回報實際使用的校正模型：校正品質與模型高度相關
        # （實測同一批語料換模型，英文的語意損壞率相差數倍），
        # 下游要能回頭查「這份結果是在什麼條件下產生的」。
        job["_correction"] = {"edited_segments": edited,
                              "unchanged_segments": len(finals) - edited,
                              "kept_original_segments": rejected,
                              "model": self.settings.correction_model}
        job["tasks"]["correct"] = "succeeded"
        self.bus.emit(job, "final_segments.appended",
                      {"first_seq": finals[0]["seq"], "last_seq": finals[-1]["seq"],
                       "segments": finals})
        self.bus.emit(job, "task.completed", {"task": "correct", "status": "succeeded"})
        self.store.put_job(job)

    def _llm_correct(self, job, raw, level):
        host, port = self.settings.llm_host, self.settings.llm_port
        if not host:
            raise EngineError("llm_unavailable", "correction", {"reason": "未設定 LLM 伺服器"})
        server_type = tm._detect_llm_server(host, port)
        if not server_type:
            raise EngineError("llm_unavailable", "correction", {"host": f"{host}:{port}"})
        # 借用既有的逐行校正（含把關）：包成它要的 segments_data 結構
        data = [{"start": r["start_ms"] / 1000, "end": r["end_ms"] / 1000, "speaker": None,
                 "lines": [{"label": "EN", "text": r["text"]}]} for r in raw]
        before = [r["text"] for r in raw]
        keep_terms = job.get("_glossary_keep") or []
        topic = " / ".join(job.get("_glossary_terms", [])[:20]) or None
        try:
            tm._correct_segments_with_llm(
                data, self.settings.correction_model, host, port,
                server_type=server_type, topic=topic,
                on_progress=lambda done, total: self._progress(
                    job, items_done=done, items_total=total, force=(done == total)))
        except Exception as e:
            raise EngineError("llm_failed", "correction", {"reason": str(e)[:200]})
        finals, rejected = [], 0
        for r, orig, seg in zip(raw, before, data):
            text = seg["lines"][0]["text"] if seg.get("lines") else orig
            if text != orig and level == "punctuation_only" and \
                    not punctuation_only_ok(orig, text, keep_terms + (job.get("_glossary_terms") or [])):
                text = orig            # 保守模式：超出標點與詞彙庫範圍的修改一律退回
                rejected += 1
            finals.append({"seq": r["seq"], "text": text, "edited": text != orig})
        return finals, rejected

    def _fake_segments(self, job):
        """測試用假逐字稿：中英混雜、confidence 有高有低"""
        lines = [
            ("zh-Hant", "各位早，今天的會議先從上週的進度開始。", 0.93),
            ("zh-Hant", "我們把儲存叢集從三台擴充到五台，延遲下降了大約四成。", 0.88),
            ("en", "The migration window is next Tuesday, from 10 p.m. to 2 a.m.", 0.91),
            ("zh-Hant", "備援的部分，王經理說會在週五前確認。", 0.85),
            ("en", "Let's keep the rollback plan simple.", None),
            ("zh-Hant", "簽呈已經送出去了，等核准就可以採購。", 0.9),
        ]
        out, t = [], 0.0
        for i in range(158):
            lang, text, conf = lines[i % len(lines)]
            dur = 4.0 + (i % 5)
            out.append({"start": t, "end": t + dur, "text": text, "language": lang,
                        "confidence": conf})
            t += dur + 0.4
        return out

    # ── 會議摘要（api_revision 2.4）───────────────────────────
    def summarize(self, job, cancelled):
        """逐字稿 → 會議摘要（jt-doc-tools 的會議分析，與 jtlw 命令列同一套）。
        存成 artifacts：summary.json（結構化）與 summary.md。回傳 "succeeded"／"failed"／"cancelled"。
        失敗只讓 summarize 這個任務失敗，逐字稿照樣交付"""
        jid = job["job_id"]
        self._set_stage(job, "summary")
        try:
            doc, md = self._build_summary(job, cancelled)
        except EngineError as e:
            if cancelled():
                return "cancelled"
            jlog.error("task.failed", job_id=jid, task="summarize", code=e.code, details=e.details)
            job["tasks"]["summarize"] = "failed"
            meta = {"language_not_supported": ("media", False)}.get(e.code, ("dependency", True))
            job.setdefault("errors", []).append(
                {"code": e.code, "category": meta[0], "retryable": meta[1], "message": e.code,
                 "stage": "summary", "task": "summarize", "details": e.details})
            self.bus.emit(job, "task.completed",
                          {"task": "summarize", "status": "failed",
                           "error": {"code": e.code, "category": meta[0], "retryable": meta[1],
                                     "message": e.code}})
            self._save(job)
            return "failed"
        if cancelled():
            return "cancelled"
        self.store.put_artifact(jid, "summary.json", "application/json",
                                json.dumps(doc, ensure_ascii=False))
        self.store.put_artifact(jid, "summary.md", "text/markdown; charset=utf-8", md)
        job["tasks"]["summarize"] = "succeeded"
        job["_summary_calls"] = doc.get("llm_calls")
        self.bus.emit(job, "task.completed", {"task": "summarize", "status": "succeeded"})
        self._save(job)
        return "succeeded"

    def _summary_segments(self, job):
        """API 的段落 → 會議分析的段落。文字優先用 final（校正成功時），時間取 raw。
        照 JTDT 的規則合併同一講者的短句（≤400 字、間隔 ≤3 秒）、切開超長段落——JTDT 自己接 jtlw 時也是這樣做的；
        合併後的段號跟 API 的 seq 不同，所以每一段另外記下它涵蓋的 API seq（source_seqs）"""
        jid = job["job_id"]
        raw = self.store.all_segments(jid, "raw")
        finals = ({f["seq"]: f["text"] for f in self.store.all_segments(jid, "final")}
                  if job["tasks"].get("correct") == "succeeded" else {})
        spk = {a["seq"]: a["speaker_id"] for a in self.store.all_segments(jid, "speakers")}
        base = []
        for r in raw:
            text = (finals.get(r["seq"]) or r["text"] or "").strip()
            if not text:
                continue
            seg = {"text": text, "start_ms": r["start_ms"], "end_ms": r["end_ms"], "_seq": r["seq"]}
            if r["seq"] in spk:
                seg["speaker"] = spk[r["seq"]]
            base.append(seg)
        mods = tm._meeting_modules()
        if not mods:
            raise EngineError("internal_error", "summary", {"reason": "jtdt_meeting 不在（安裝不完整）"})
        merged = mods[2]._merge([{k: v for k, v in b.items() if k != "_seq"} for b in base])
        for m in merged:
            a, e = m.get("start_ms"), m.get("end_ms", m.get("start_ms"))
            m["source_seqs"] = [b["_seq"] for b in base
                                if b.get("speaker") == m.get("speaker")
                                and b["start_ms"] < (e if e is not None else a) + 1 and b["end_ms"] > a - 1]
        return merged

    def _summary_context(self, job):
        """hints.meeting → 會議分析的背景資料（只拿來讀懂逐字稿，不會變成項目；JTDT 有防抄機制）"""
        m = (job.get("_hints") or {}).get("meeting") or {}
        lines = []
        if m.get("title"):
            lines.append(f"會議主題：{m['title']}")
        if m.get("room"):
            lines.append(f"會議室：{m['room']}")
        if m.get("host"):
            lines.append(f"主持人：{m['host']}")
        if m.get("started_at") or m.get("ended_at"):
            lines.append(f"時間：{m.get('started_at') or '?'} ～ {m.get('ended_at') or '?'}")
        names = [p.get("name") for p in (m.get("participants") or []) if isinstance(p, dict) and p.get("name")]
        if names:
            lines.append("與會者：" + "、".join(dict.fromkeys(names)))
        if m.get("notes"):
            lines.append(str(m["notes"]))
        return "\n".join(lines) or None

    def _build_summary(self, job, cancelled):
        jid = job["job_id"]
        segs = self._summary_segments(job)
        if len(segs) < 1:
            raise EngineError("language_not_supported", "summary", {"reason": "逐字稿沒有內容"})
        ok, why = tm._meeting_language_ok(segs)
        if not ok:
            raise EngineError("language_not_supported", "summary", {"reason": why})
        if self.settings.fake_engine:
            # 測試模式：不叫 LLM，但結構照真的來（引用第一段與最後一段、一個涵蓋整場的章節），
            # 引用對回 API seq（source_seqs）與講者統計才測得到
            first, last = segs[0], segs[-1]
            items = {k: [] for k in ("impacts", "decisions", "actions", "risks", "questions")}
            items["decisions"].append({"text": "（測試模式）決議", "segment_ids": [first["seq"]]})
            items["actions"].append({"text": "（測試模式）待辦", "owner": "（測試）", "due_text": None,
                                     "segment_ids": [last["seq"]]})
            span = (last.get("end_ms") or 0) - (first.get("start_ms") or 0)
            pub = {"summary": {"text": "（測試模式）會議摘要", "grounded": True, "unsupported": []},
                   "items": items,
                   "chapters": [{"title": "（測試模式）整場", "start_seq": first["seq"], "end_seq": last["seq"],
                                 "start_ms": first.get("start_ms"), "end_ms": last.get("end_ms"),
                                 "duration_ms": span, "percentage": 100.0}],
                   "mindmap": [], "charts": [], "dropped_count": 0, "llm_calls": 0,
                   "speaker_stats": tm._meeting_modules()[0].speaker_stats(segs)}
        else:
            host, port = self.settings.llm_host, self.settings.llm_port
            if not host:
                raise EngineError("llm_unavailable", "summary", {"reason": "未設定 LLM 伺服器"})
            server_type = tm._detect_llm_server(host, port)
            if not server_type:
                raise EngineError("llm_unavailable", "summary", {"host": f"{host}:{port}"})

            def prog(frac, msg):
                self._progress(job, detail=msg,
                               percent=round(100 * (len(job["_stages"]) - 1 + frac) / len(job["_stages"]), 1))
            try:
                pub = tm.meeting_analysis(segs, self.settings.summary_model, host, port, server_type,
                                          context=self._summary_context(job), on_progress=prog,
                                          cancelled=cancelled)
            except Exception as e:
                raise EngineError("llm_failed", "summary", {"reason": str(e)[:200]})
            if (pub.get("llm_calls") or 0) > 0 and not pub["summary"].get("text") \
                    and not any(pub["items"].values()) and not pub.get("chapters"):
                # 每一次呼叫都失敗（LLM 掛了）：不要交出一份「什麼都沒有」的摘要當成功
                raise EngineError("llm_failed", "summary", {"reason": "會議分析的每一次 LLM 呼叫都失敗"})
        by_seq = {s["seq"]: s for s in segs}

        def cite(ids):
            out = []
            for i in ids or []:
                s = by_seq.get(i)
                if s:
                    out.append({"seq": i, "start_ms": s.get("start_ms"), "end_ms": s.get("end_ms"),
                                "speaker_id": s.get("speaker"), "source_seqs": s.get("source_seqs") or []})
            return out
        items = {}
        for kind, rows in (pub.get("items") or {}).items():
            items[kind] = []
            for it in rows:
                one = {"text": it.get("text", ""), "citations": cite(it.get("segment_ids"))}
                if kind == "actions":
                    one["owner"] = it.get("owner")
                    one["due_text"] = it.get("due_text")
                items[kind].append(one)
        chapters = [{"title": c.get("title"), "start_ms": c.get("start_ms"), "end_ms": c.get("end_ms"),
                     "duration_ms": c.get("duration_ms"), "percentage": c.get("percentage"),
                     "first_seq": c.get("start_seq"), "last_seq": c.get("end_seq")}
                    for c in pub.get("chapters") or []]
        speakers = []
        for sid, st in (pub.get("speaker_stats") or {}).items():
            speakers.append({"speaker_id": None if sid == "unknown" else sid,
                             "turn_count": st.get("turn_count"), "chars": st.get("chars"),
                             "char_pct": st.get("char_pct"), "speaking_ms": st.get("speaking_ms"),
                             "percentage": st.get("percentage")})
        summ = pub.get("summary") or {}
        doc = {"summary_schema_version": SUMMARY_SCHEMA_VERSION, "job_id": jid,
               "language": "zh-Hant", "model": self.settings.summary_model,
               "generated_at": now_iso(), "llm_calls": pub.get("llm_calls", 0),
               "summary": {"text": summ.get("text") or "", "grounded": summ.get("grounded", True),
                           "unsupported": summ.get("unsupported") or []},
               "items": items, "chapters": chapters, "speakers": speakers,
               "segments": [{"seq": s["seq"], "speaker_id": s.get("speaker"), "start_ms": s.get("start_ms"),
                             "end_ms": s.get("end_ms"), "text": s.get("text", ""),
                             "source_seqs": s.get("source_seqs") or []} for s in segs],
               "dropped_count": pub.get("dropped_count", 0)}
        md = tm.meeting_summary_markdown(pub, segs, measured=True)
        return doc, md

    def _cleanup(self, job, wav_path):
        # 上傳的來源檔不在這裡刪：逐字稿失敗、之後可能 retry 時要留著（由 app.py 的 _release_upload 決定）
        upload = job.get("_source_path") if (job.get("_source") or {}).get("type") == "upload" else None
        for p in (job.get("_source_path"), wav_path):
            if p and p != upload and p != "(fake)" and os.path.isfile(p) and p.startswith(self.settings.work_dir):
                try:
                    os.unlink(p)
                except OSError:
                    pass
        # _convert_to_wav 產生的暫存檔在 recordings/ 底下
        if wav_path and wav_path != upload and os.path.isfile(wav_path) and "tmp_" in os.path.basename(wav_path):
            try:
                os.unlink(wav_path)
            except OSError:
                pass


def _to_bcp47(lang):
    """Whisper 的語言代碼轉 BCP-47（對應介面允許的值）"""
    return {"zh": "zh-Hant", "yue": "zh-Hant", "nan": "nan-Hant",
            "en": "en", "ja": "ja", "ko": "ko"}.get(lang, "und" if not lang else lang)
