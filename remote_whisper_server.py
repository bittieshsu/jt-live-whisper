#!/usr/bin/env python3
"""
jt-live-whisper 伺服器 Whisper ASR 伺服器
部署到 GPU 伺服器，提供 REST API 讓本機上傳音訊檔進行語音辨識。

後端引擎自動偵測：
  1. faster-whisper (CTranslate2 CUDA) — x86_64 GPU，速度最快
  2. openai-whisper (PyTorch CUDA) — aarch64 GPU（如 DGX Spark），也能 GPU 加速
  3. faster-whisper (CPU) — 無 GPU 降級

依賴：faster-whisper, fastapi, uvicorn, python-multipart
      （aarch64 無 CTranslate2 CUDA 時額外需要 openai-whisper）
      （講者辨識需額外安裝 resemblyzer, spectralcluster）
啟動：python3 server.py [--port 8978] [--host 0.0.0.0]

Author: Jason Cheng (Jason Tools)
"""

import argparse
import asyncio
import json
import math
import os
import queue
import re
import shutil
import sys
import tempfile
import threading
import time

# ── Qwen3-ASR worker（v2.23.0，實驗）──────────────────────────────
# vLLM 0.14 鎖 torch 2.9.1，這支服務的 venv 是 torch 2.10 → Qwen 必須在**獨立 venv 的子行程**跑。
# 伺服器自動更新只推 server.py 一個檔案，所以 worker 也寫在這裡，以 `--qwen-worker <port>` 啟動；
# 放在所有第三方 import 之前，worker 的 venv 不需要有這支服務的其他套件。
# 實測（2026-09-25，tools/asr_bench/）：中文 20 場會議 CER 28.78% → 15.75%；中英夾雜少數語言召回 2~3 倍
QWEN_ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"
QWEN_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
QWEN_GPU_MEM = 0.06     # E6：0.06 可跑（行程 5.4 GB），0.04 以下起不來；共用機不要給多


def _qwen_worker_main():
    """只聽 127.0.0.1。POST /transcribe {path, windows, language} → {texts, stamps}（每窗文字＋對齊器逐字時間）"""
    import http.server
    import signal
    port = int(sys.argv[sys.argv.index("--qwen-worker") + 1])
    parent = os.getppid()

    def _die():
        try:
            os.killpg(0, signal.SIGKILL)       # 連 vLLM 的 EngineCore 子行程一起收掉，不留孤兒佔 GPU
        finally:
            os._exit(0)

    def _watch():
        while True:
            time.sleep(5)
            if os.getppid() != parent:          # 主服務結束了
                _die()
    threading.Thread(target=_watch, daemon=True).start()
    state = {"ready": False}
    lock = threading.Lock()

    def work(req):
        wav, _ = librosa.load(req["path"], sr=16000, mono=True)
        chunks = [wav[max(0, int(a * 16000)):int(b * 16000)] for a, b in req["windows"]]
        lang = req["language"]
        texts = [""] * len(chunks)
        # 極短的窗（<0.2 秒）不送模型：沒有內容可辨識，還可能讓前處理出錯
        live = [k for k, c in enumerate(chunks) if len(c) >= 3200]
        for k0 in range(0, len(live), 32):
            ks = live[k0:k0 + 32]
            r = asr.transcribe(audio=[(chunks[k], 16000) for k in ks], language=[lang] * len(ks))
            for k, x in zip(ks, r):
                texts[k] = x.text
        stamps = [[] for _ in texts]
        idx = [k for k, t in enumerate(texts) if t.strip()]
        align_failed = 0
        for k0 in range(0, len(idx), 8):
            ks = idx[k0:k0 + 8]
            try:
                r = fa.align(audio=[(chunks[k], 16000) for k in ks], text=[texts[k] for k in ks],
                             language=[lang] * len(ks))
            except Exception as e:           # 對齊失敗：這幾窗沒有逐字時間，文字照樣回（主服務切句時不丟字）
                align_failed += len(ks)
                print(f"[qwen-worker] 對齊失敗 {len(ks)} 窗：{type(e).__name__}: {e}", flush=True)
                continue
            for k, xs in zip(ks, r):
                stamps[k] = [[x.text, float(x.start_time), float(x.end_time)] for x in xs]
        return {"texts": texts, "stamps": stamps, "align_failed": align_failed}

    class _H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            b = json.dumps(obj, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path != "/health":
                return self._send(404, {"error": "not found"})
            # 載入中回 503：主服務只把 200 當成就緒
            self._send(200, {"ok": True, "model": QWEN_ASR_MODEL}) if state["ready"] \
                else self._send(503, {"ok": False, "loading": True})

        def do_POST(self):
            if self.path != "/transcribe":
                return self._send(404, {"error": "not found"})
            if not state["ready"]:
                return self._send(503, {"error": "Qwen3-ASR worker 載入中"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                with lock:                      # 一次一件（主服務本來就排隊，這裡是保險）
                    out = work(req)
            except Exception as e:
                return self._send(500, {"error": f"{type(e).__name__}: {e}"})
            self._send(200, out)

    # **先綁埠號再載入模型**（約 7 GB、1~3 分鐘）：同一個埠已有 worker 時這裡立刻失敗退出，
    # 不會白白載一份模型；主服務在載入期間也看得出埠被佔（2026-09-26 實測：原本載完才綁，兩個 worker 同時載入）
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    import librosa
    import torch as _torch
    from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
    asr = Qwen3ASRModel.LLM(model=QWEN_ASR_MODEL, gpu_memory_utilization=QWEN_GPU_MEM, max_model_len=4096,
                            max_inference_batch_size=32, max_new_tokens=512)
    fa = Qwen3ForcedAligner.from_pretrained(QWEN_ALIGNER_MODEL, dtype=_torch.bfloat16, device_map="cuda")
    state["ready"] = True
    print(f"[qwen-worker] 就緒 127.0.0.1:{port}（{QWEN_ASR_MODEL}）", flush=True)
    threading.Event().wait()


if __name__ == "__main__" and "--qwen-worker" in sys.argv:
    _qwen_worker_main()
    sys.exit(0)

# 原始碼編譯的 CTranslate2 將 libctranslate2.so 安裝到 /usr/local/lib
# 需在 import ctranslate2 前確保 LD_LIBRARY_PATH 包含此路徑
if "/usr/local/lib" not in os.environ.get("LD_LIBRARY_PATH", ""):
    os.environ["LD_LIBRARY_PATH"] = f"/usr/local/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

# ── 版本 ──
# **必須與 translate_meeting.py 的 APP_VERSION 同步**（版本號同步清單第 9 處）。
# 2026-09-21 之前伺服器完全沒有版本號，用戶端也不檢查——GPU 上的服務缺了
# v2.20.0 的講者辨識時間軸修正，而它是預設路徑，三天沒有人發現。
SERVER_VERSION = "2.23.0"

# 講者辨識：只有 >= 這個秒數的段落才進分群（1.6s = resemblyzer partial 長度，
# 短於它的聲紋是補零算出來的）。與 translate_meeting.py 必須一致。
_DIAR_MIN_CLUSTER_SEC = 1.6
_DIAR_MIN_CLUSTER_UNITS = 8

# 講者辨識：判斷「現場幾個人」的門檻。
# 做法是數「正規化 Laplacian 的特徵值低於這個值的個數」——近似連通塊數。
# 取代原本的 eigengap（相鄰特徵值差最大處），因為 eigengap 取的是**全域最大**
# 間隙，而前面幾個間隙天生就比較大（2 群 vs 3 群的差異本來就比 5 群 vs 6 群明顯），
# 於是系統性地低估。2026-09-22 在 AMI 保留集 16 場實測：
#   eigengap NormalizedDiff  混 13.99%（5/16 場判太少）
#   特徵值 < 0.5             混 11.81%
# 0.45 / 0.5 / 0.55 是平滑的平台不是尖峰（dev 10.06 / 8.57 / 8.30、
# test 11.62 / 11.81 / 12.00），取中間值。
#
# **這個規則只有在短段落被排除之後才成立**：同一個方法在含短段落的聲紋上
# 反而把中文那場從 30.51% 惡化到 48.10%（見 _diar_cluster_floor）。
_DIAR_EIGENVALUE_TAU = 0.5


def _diar_estimate_speakers(embeddings, refinement_opts, laplacian_type,
                            lo=2, hi=8):
    """估計講者人數：數 Laplacian 特徵值低於門檻的個數。

    這裡要自己把 affinity → refinement → Laplacian → 特徵值再算一次，
    因為 spectralcluster 沒有提供這個規則（它只支援 eigengap 的兩種變體），
    而它內部那段是私有的。算兩次的成本是一次特徵分解，可接受。
    失敗時回傳 None，呼叫端退回函式庫自己的估計。
    """
    try:
        from spectralcluster import laplacian as _lap
        from spectralcluster import utils as _u
        import numpy as _np
        aff = _u.compute_affinity_matrix(embeddings)
        for name in (refinement_opts.refinement_sequence or []):
            aff = refinement_opts.get_refinement_operator(name).refine(aff)
        lap = _lap.compute_laplacian(aff, laplacian_type=laplacian_type)
        ev, _vec = _u.compute_sorted_eigenvectors(lap, descend=False)
        n = int(_np.sum(_np.asarray(ev) < _DIAR_EIGENVALUE_TAU))
        return int(max(lo, min(hi, n)))
    except Exception:
        return None



def _diar_cluster_floor(segments):
    """講者辨識：決定多長的段落才進分群，回傳秒數門檻。

    抽成獨立函式是為了測得到——判斷埋在 _diarize_segments 裡面時只能比對
    原始碼字串，而那種斷言會被註解騙過（2026-09-21 踩過，見 test_refinement_alive）。

    < 1.6s 的段落聲紋是 resemblyzer 補零算出來的，實測標錯率 63~67%；
    1.6~4.0s 只有 0~15%。但夠長的段落太少時（短訪談、幾句話的錄音）
    一律套門檻會變成沒東西可以分群，那時寧可全收。
    """
    long_n = sum(1 for s in segments
                 if s["end"] - s["start"] >= _DIAR_MIN_CLUSTER_SEC)
    return _DIAR_MIN_CLUSTER_SEC if long_n >= _DIAR_MIN_CLUSTER_UNITS else 0.3


# 更新用的共用密鑰。**沒有設定就完全不開放更新端點**（預設關閉不是預設開啟）：
# 這個端點本質上是遠端程式碼執行，內網不是空的。
UPDATE_TOKEN = os.environ.get("JT_WHISPER_UPDATE_TOKEN", "").strip()
# 更新用的 body 上限。`await request.body()` 會把整包讀進記憶體，
# 沒有上限的話一個大 POST 就能把這台機器的記憶體吃光。
UPDATE_MAX_BYTES = 8 * 1024 * 1024
UPDATE_KEEP_BACKUPS = 5
# 有作業時不拒絕更新，而是**排定**：驗證完先存起來，等作業做完才換（v2.21.8）。
# 排定期間離線線不收新作業（否則一直有人送就永遠換不了）；等太久就放棄這次更新、重新開門，
# 以免一件卡死的作業讓整台伺服器永遠不收離線作業。
UPDATE_DRAIN_TIMEOUT = int(os.environ.get("JT_WHISPER_UPDATE_DRAIN_SEC", 30 * 60))   # 環境變數只給測試用
UPDATE_RETRY_AFTER = 10
_UPDATE_LOCK = threading.Lock()
_UPDATE_PENDING = None   # {"to", "from", "since", "client_ip", "path"}


def _version_tuple(v):
    """'2.21.1' → (2, 21, 1)；解析不動的部分當 0，未知版本視為最舊"""
    out = []
    for part in str(v or "0").split("."):
        digits = "".join(c for c in part if c.isdigit())
        out.append(int(digits) if digits else 0)
    while len(out) < 3:
        out.append(0)
    return tuple(out[:3])


def _parse_server_version(path):
    """從一份 server.py 原始碼裡讀出 SERVER_VERSION"""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.startswith("SERVER_VERSION"):
                    return line.split("=", 1)[1].strip().strip('"\'')
    except Exception:
        pass
    return "0"


def _prune_backups(me, keep=UPDATE_KEEP_BACKUPS):
    """只留最近幾份備份，避免長年累積把磁碟吃掉"""
    import glob
    try:
        old = sorted(glob.glob(me + ".bak-*"))
        for f in old[:-keep]:
            os.remove(f)
    except OSError:
        pass


app = FastAPI(title="jt-whisper-server")

# ── 作業排隊 ──
# **GPU 一次只跑一件，其餘排隊（先到先做）。** 先前是「來幾件就同時跑幾件」，
# 多個用戶端同時送離線檔時全部擠在 GPU 上：每一件都變慢、顯示記憶體可能不夠，
# 而 `/v1/status` 只有一個欄位，後到的作業會把先到的蓋掉（busy 的判斷跟著錯）。
#
# 分兩條線，**各自一次一件、彼此不互等**（2026-09-23 使用者指定：
# 「即時的另開一條，跟辨識分開」）：
#   batch     離線辨識（串流）、講者辨識、大檔的非串流辨識
#   realtime  即時字幕送來的幾秒短音訊（非串流、小檔）
# 不分開的話，即時字幕要等一場一小時的離線檔跑完才出得了字。
#
# 判斷哪條線沿用既有協定，**不需要用戶端改版**：即時路徑本來就是非串流、
# 每次約 160KB；離線路徑本來就是 stream=true。

_REALTIME_MAX_BYTES = 8 * 1024 * 1024   # 非串流且不超過這個大小 → 即時線


class _Ticket:
    """隊伍裡的一件作業。比對用物件身分（不要改成 dict：dict 會以內容比對，
    兩件參數相同的作業會被當成同一件）。"""
    __slots__ = ("type", "model", "language", "client_ip", "enqueued", "started")

    def __init__(self, task_type, model, language, client_ip):
        self.type = task_type
        self.model = model
        self.language = language
        self.client_ip = client_ip
        self.enqueued = time.time()
        self.started = None


class _Lane:
    """先到先做的單線隊伍。`_items[0]` 是正在跑的那件，其餘在等。"""

    def __init__(self, name):
        self.name = name
        self._cv = threading.Condition()
        self._items = []
        self.closed = False   # 更新排定時關門：已在隊伍裡的照樣做完，新的不收

    def enter(self, task_type, model, language, client_ip=""):
        """排進隊伍，回傳 ticket；**關門中回傳 None**（呼叫端要回 503）。
        「檢查有沒有關門」與「排進去」在同一把鎖裡，不然更新可能在兩者之間換掉程式。"""
        t = _Ticket(task_type, model, language, client_ip)
        with self._cv:
            if self.closed:
                return None
            self._items.append(t)
            if self._items[0] is t:
                t.started = time.time()
        return t

    def close(self):
        with self._cv:
            self.closed = True

    def reopen(self):
        with self._cv:
            self.closed = False
            self._cv.notify_all()

    def wait_empty(self, timeout):
        """等隊伍清空（含正在跑的那件），最多 timeout 秒；回傳是否清空"""
        with self._cv:
            return self._cv.wait_for(lambda: not self._items, max(timeout, 0))

    def position(self, t):
        """0＝輪到了；n＝前面還有 n 件；-1＝已不在隊伍裡"""
        with self._cv:
            for i, x in enumerate(self._items):
                if x is t:
                    return i
            return -1

    def wait(self, t, timeout):
        """阻塞等候輪到自己，最多 timeout 秒；回傳 position()（執行緒內使用）"""
        with self._cv:
            self._cv.wait_for(lambda: not self._items or self._items[0] is t
                              or all(x is not t for x in self._items), timeout)
        return self.position(t)

    def leave(self, t):
        """做完、出錯或用戶端放棄排隊時都要呼叫；重複呼叫無害。
        **漏掉一次，後面的人就永遠等不到。**"""
        with self._cv:
            self._items = [x for x in self._items if x is not t]
            if self._items and self._items[0].started is None:
                self._items[0].started = time.time()
            self._cv.notify_all()

    def snapshot(self):
        now = time.time()
        with self._cv:
            items = list(self._items)

        def _d(x, running):
            d = {"type": x.type, "model": x.model, "language": x.language,
                 "client_ip": x.client_ip}
            if running:
                d["elapsed"] = round(now - (x.started or now), 1)
            else:
                d["waited"] = round(now - x.enqueued, 1)
            return d
        return {"running": _d(items[0], True) if items else None,
                "waiting": [_d(x, False) for x in items[1:]]}

    def busy(self):
        with self._cv:
            return bool(self._items)

    def count(self):
        with self._cv:
            return len(self._items)


_LANES = {"batch": _Lane("batch"), "realtime": _Lane("realtime")}


def _any_busy():
    return any(l.busy() for l in _LANES.values())


def _update_pending_info():
    with _UPDATE_LOCK:
        p = dict(_UPDATE_PENDING) if _UPDATE_PENDING else None
    if not p:
        return None
    return {"to": p["to"], "waited": round(time.time() - p["since"], 1),
            "waiting_jobs": _LANES["batch"].count()}


def _updating_response():
    """更新排定中、這條線已關門時的回應。用戶端（v2.21.8 起）看到會等更新完再重送；
    舊版用戶端會當成伺服器錯誤、改用本機辨識——兩者都不會拿到殘缺的結果。"""
    with _UPDATE_LOCK:
        to = (_UPDATE_PENDING or {}).get("to")
    return JSONResponse(
        status_code=503, headers={"Retry-After": str(UPDATE_RETRY_AFTER)},
        content={"error": "updating", "retry_after": UPDATE_RETRY_AFTER, "to": to,
                 "detail": f"伺服器即將更新到 v{to}，正在等目前的作業做完"})


async def _wait_turn_async(lane, t, request):
    """在 async 端點裡等輪到自己。用戶端斷線就退出隊伍，回傳 False。"""
    while True:
        pos = lane.position(t)
        if pos == 0:
            return True
        if pos < 0:
            return False
        if await request.is_disconnected():
            lane.leave(t)
            print(f"[排隊] {t.client_ip} 在{lane.name}隊伍中斷線，已移出", flush=True)
            return False
        await asyncio.sleep(0.3)


# ── 偵測最佳後端引擎 ──
_models: dict = {}
_backend = "faster-whisper"  # "faster-whisper" 或 "openai-whisper"
_device = "cpu"
_compute_type = "int8"
_torch_device = "cpu"

if torch.cuda.is_available():
    _torch_device = "cuda"
    # 嘗試 CTranslate2 CUDA（faster-whisper 用）
    try:
        import ctranslate2
        cuda_types = ctranslate2.get_supported_compute_types("cuda")
        if cuda_types:
            _device = "cuda"
            _compute_type = "float16"
            _backend = "faster-whisper"
            print("[引擎] faster-whisper (CTranslate2 CUDA)")
        else:
            raise RuntimeError("CTranslate2 無 CUDA")
    except Exception:
        # CTranslate2 沒 CUDA，嘗試 openai-whisper（PyTorch CUDA）
        try:
            import whisper as openai_whisper  # noqa: F401
            _backend = "openai-whisper"
            _device = "cuda"
            print("[引擎] openai-whisper (PyTorch CUDA)")
        except ImportError:
            print("[警告] CTranslate2 無 CUDA 且 openai-whisper 未安裝，改用 CPU")
            _backend = "faster-whisper"
else:
    print("[引擎] faster-whisper (CPU)")

# ── 偵測 diarization 套件 ──
_HAS_DIARIZE = False
try:
    import warnings as _w
    with _w.catch_warnings():
        _w.filterwarnings("ignore", message="pkg_resources is deprecated")
        from resemblyzer import VoiceEncoder, preprocess_wav  # noqa: F401
    from spectralcluster import SpectralClusterer  # noqa: F401
    from spectralcluster import refinement, laplacian  # noqa: F401
    _HAS_DIARIZE = True
    print(f"[講者辨識] resemblyzer + spectralcluster 可用 (device={_torch_device})")
except ImportError:
    print("[講者辨識] resemblyzer/spectralcluster 未安裝")

# Nemotron 3 Diarization：transformers 內建 nemotron3_diarization（5.18 起）才有
_HAS_NEMO = False
try:
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as _CMN
    _HAS_NEMO = "nemotron3_diarization" in _CMN
    print(f"[講者辨識] Nemotron {'可用' if _HAS_NEMO else '不可用（transformers 版本太舊，需要 5.18 以上）'}")
except Exception:
    print("[講者辨識] Nemotron 不可用（未安裝 transformers）")
if not (_HAS_DIARIZE or _HAS_NEMO):
    print("[講者辨識] 沒有可用的方法，diarize API 停用")


# ── Diarization 核心函式 ──

# ── Qwen3-ASR：主服務這一側（切窗、呼叫 worker、切句）──
# 切窗與用戶端 _nan_vad_windows **是同一支**（台語已在用，≤28 秒）；tools/test_qwen_server.py 逐一比對
_NAN_WINDOW_SEC = 28.0
_NAN_VAD_SILENCE_MS = 500
_QWEN_MODELS = ("qwen3-asr-0.6b",)
_QWEN_LANG = {"zh": "Chinese", "en": "English", "ko": "Korean"}     # 日文實測長檔較差（E3），先不開
_QWEN_SENT_END = "。？！?!"
_QWEN_FILLERS = set("嗯啊呃唔哦喔欸誒呀哈") | {"um", "uh", "mm", "hmm", "mhm"}
_QWEN = {"proc": None, "port": None, "ready": False, "error": "", "restarts": 0, "stopping": False}
_QWEN_MAX_RESTARTS = 3          # 一小時內最多自動重啟幾次（起不來時不要無限重試、一直佔 GPU 載入）


def _nan_vad_windows(audio, samplerate=16000):
    """（與 translate_meeting._nan_vad_windows 相同）依語音活動切成 ≤28 秒視窗，回傳 [(起, 迄)] 秒"""
    total = len(audio) / float(samplerate)
    try:
        from faster_whisper.vad import get_speech_timestamps, VadOptions
        regions = get_speech_timestamps(
            audio, VadOptions(min_silence_duration_ms=_NAN_VAD_SILENCE_MS),
            sampling_rate=samplerate)
    except Exception:
        regions = []

    if not regions:
        out, t = [], 0.0
        while t < total:
            out.append((t, min(t + _NAN_WINDOW_SEC, total)))
            t += _NAN_WINDOW_SEC
        return out or [(0.0, total)]

    windows = []
    cur_start = cur_end = None
    for r in regions:
        rs, re_ = r["start"] / float(samplerate), r["end"] / float(samplerate)
        if cur_start is None:
            cur_start, cur_end = rs, re_
        elif re_ - cur_start <= _NAN_WINDOW_SEC:
            cur_end = re_
        else:
            windows.append((cur_start, cur_end))
            cur_start, cur_end = rs, re_
        while cur_end - cur_start > _NAN_WINDOW_SEC:
            windows.append((cur_start, cur_start + _NAN_WINDOW_SEC))
            cur_start += _NAN_WINDOW_SEC
    if cur_start is not None:
        windows.append((cur_start, cur_end))
    return windows


def _qwen_core(s):
    return re.sub(r"[\W_]+", "", s)


def _qwen_filler_only(text):
    """一窗只有語氣詞（嗯／啊／um…）→ 丟掉。E3：60 秒靜音、雜訊、和弦、嗡嗡聲 Qwen 會吐「嗯。」"""
    words = re.findall(r"[a-z]+", text.lower())
    cjk = [c for c in _qwen_core(text) if not ("a" <= c.lower() <= "z")]
    return bool(words or cjk) and all(w in _QWEN_FILLERS for w in words) and all(c in _QWEN_FILLERS for c in cjk)


def _qwen_sentences(text, stamps, off, win_end):
    """一窗的文字依句末標點切句，用對齊器的逐字時間定起訖（E2 方案 A）。
    對齊器的 token 沒有標點 → 用「去掉標點後的字數」對回去。**不丟字**：
    對齊結果不夠時，剩下的文字照樣成一段（時間用到窗尾）；只有標點的尾巴接回前一句"""
    tc = [(s, e) for tk, s, e in stamps for _ in _qwen_core(tk)]
    out, buf, n, pos = [], "", 0, 0

    def flush():
        nonlocal buf, n, pos
        if n:
            if pos < len(tc):
                s0, e0 = off + tc[pos][0], off + tc[min(pos + n, len(tc)) - 1][1]
            else:
                s0, e0 = (out[-1]["end"] if out else off), win_end
            out.append({"start": round(s0, 3), "end": round(max(e0, s0), 3), "text": buf.strip()})
        elif buf.strip() and out:
            out[-1]["text"] += buf.strip()
        pos += n
        buf, n = "", 0

    for i, ch in enumerate(text):
        buf += ch
        if _qwen_core(ch):
            n += 1
        # 英文／韓文的句點：後面是空白或結尾、前一個字不是數字（「3.5」不切）才算句末
        if ch in _QWEN_SENT_END or (ch == "." and (i + 1 == len(text) or text[i + 1].isspace())
                                    and not (i and text[i - 1].isdigit())):
            flush()
    flush()
    return out


def _qwen_python():
    p = os.environ.get("JT_QWEN_PYTHON") or os.path.expanduser("~/jt-whisper-server/venv-qwen/bin/python")
    return p if os.path.exists(p) else None


def _qwen_start(port):
    """有 Qwen 的 venv 才啟動 worker（背景載入，約 1~3 分鐘；就緒前 /health 不列 Qwen）。
    worker 意外結束時自動重啟（一小時內最多 _QWEN_MAX_RESTARTS 次）"""
    import signal
    import subprocess
    import urllib.request
    py = _qwen_python()
    if not py or not torch.cuda.is_available():
        return
    if _qwen_port_busy(port):
        _QWEN.update(proc=None, port=port, ready=False,
                     error=f"埠號 {port} 已被佔用（可能是上一次沒收乾淨的 worker），Qwen3-ASR 停用；設 JT_QWEN_PORT 換一個")
        print(f"[Qwen3-ASR] {_QWEN['error']}")
        return
    env = dict(os.environ)
    if os.path.exists("/usr/local/cuda/bin/ptxas"):
        env.setdefault("TRITON_PTXAS_PATH", "/usr/local/cuda/bin/ptxas")   # GB10（sm_121a）Triton 內建的不認得
    log = open(os.path.join(tempfile.gettempdir(), f"jt-qwen-worker-{port}.log"), "ab")
    proc = subprocess.Popen([py, os.path.abspath(__file__), "--qwen-worker", str(port)],
                            stdin=subprocess.DEVNULL, stdout=log, stderr=log, env=env,
                            start_new_session=True)
    _QWEN.update(proc=proc, port=port, ready=False, error="")
    print(f"[Qwen3-ASR] worker 啟動中（pid {proc.pid}，127.0.0.1:{port}）")

    def _wait():
        t0 = time.monotonic()
        while proc.poll() is None:                 # 不設死線：第一次啟動可能在下載模型（約 4 GB）
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3):
                    _QWEN.update(ready=True, error="")
                    print(f"[Qwen3-ASR] 就緒（{time.monotonic() - t0:.0f}s）")
                    break
            except Exception:
                if time.monotonic() - t0 > 900 and not _QWEN["error"]:
                    _QWEN["error"] = "載入超過 15 分鐘（第一次啟動可能在下載模型），仍在等待"
                time.sleep(3)
        while proc.poll() is None:                 # 就緒後守著：意外結束就重啟
            time.sleep(5)
        _QWEN["ready"] = False
        # worker 意外結束（被 kill -9、當掉）時，它底下 vLLM 的 EngineCore **不會跟著走**，
        # 會變成孤兒繼續佔約 5 GB 顯示記憶體（2026-09-26 實測，重啟兩次就疊到 10 GB）。
        # 它還留在 worker 的行程群組裡，整個群組一起收
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except Exception:
            pass
        if _QWEN["stopping"]:
            return
        now = time.time()
        recent = [t for t in _QWEN.setdefault("restart_times", []) if now - t < 3600]
        _QWEN["restart_times"] = recent
        if len(recent) >= _QWEN_MAX_RESTARTS:
            _QWEN["error"] = (f"worker 一小時內結束 {len(recent) + 1} 次，不再自動重啟；"
                              f"見 {log.name}，排除後重啟服務")
            print(f"[Qwen3-ASR] {_QWEN['error']}")
            return
        _QWEN["error"] = f"worker 結束（代碼 {proc.returncode}），30 秒後重啟；見 {log.name}"
        print(f"[Qwen3-ASR] {_QWEN['error']}")
        time.sleep(30)
        if _QWEN["stopping"]:
            return
        _QWEN["restart_times"].append(time.time())
        _QWEN["restarts"] += 1
        _qwen_start(port)
    threading.Thread(target=_wait, daemon=True).start()


def _qwen_port_busy(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as so:
        so.settimeout(1)
        return so.connect_ex(("127.0.0.1", port)) == 0


def _qwen_stop(wait=0.0):
    """收掉 worker 整個行程群組（含 vLLM 的 EngineCore）。wait>0 時等它真的結束，逾時就 SIGKILL"""
    import signal
    _QWEN["stopping"] = True
    p = _QWEN.get("proc")
    if p is None or p.poll() is not None:
        return
    try:
        os.killpg(p.pid, signal.SIGTERM)
    except Exception:
        pass
    t0 = time.monotonic()
    while wait and p.poll() is None and time.monotonic() - t0 < wait:
        time.sleep(0.2)
    if wait and p.poll() is None:
        try:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait(5)
        except Exception:
            pass


def _qwen_ready():
    p = _QWEN.get("proc")
    return bool(_QWEN.get("ready") and p is not None and p.poll() is None)


def _transcribe_qwen(wav_path, language):
    """切窗 → worker 辨識＋對齊 → 切句。回傳 (segments, duration, proc_time)"""
    import librosa
    import urllib.error
    import urllib.request
    t0 = time.monotonic()
    wav, _ = librosa.load(wav_path, sr=16000, mono=True)
    windows = _nan_vad_windows(wav, 16000)
    body = json.dumps({"path": wav_path, "windows": windows, "language": _QWEN_LANG[language]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{_QWEN['port']}/transcribe", data=body,
                                 headers={"Content-Type": "application/json"})
    # 逾時依音訊長度（vLLM 約 50 倍即時，這裡給到 1 倍即時＋5 分鐘，只擋真的卡死）
    try:
        with urllib.request.urlopen(req, timeout=300 + len(wav) / 16000) as r:
            res = json.loads(r.read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Qwen3-ASR worker 回報錯誤：{e.read().decode(errors='replace')[:200]}") from e
    except Exception as e:
        # 原始訊息（例：Remote end closed connection）轉給用戶端會看起來像主服務斷線，講清楚是 worker
        raise RuntimeError(f"Qwen3-ASR worker 中途沒有回應（{type(e).__name__}），它會自動重啟") from e
    if res.get("align_failed"):
        print(f"[Qwen3-ASR] {res['align_failed']} 窗對齊失敗，這些段落的時間以整窗估計")
    segs = []
    for (ws, we), txt, st in zip(windows, res["texts"], res["stamps"]):
        if not txt.strip() or _qwen_filler_only(txt):
            continue
        segs += _qwen_sentences(txt, st, ws, we)
    return segs, len(wav) / 16000, round(time.monotonic() - t0, 1)


# ── Nemotron 3 Diarization ──
# **與 translate_meeting.py 的同名函式是同一套邏輯**（這支伺服器自動更新只推單一檔案，不能共用模組）。
# tools/test_diarizer.py 逐一比對兩邊輸出；改一邊一定要改另一邊。實測數據見那邊的註解。
NEMO_DIAR_MODEL = "nvidia/Nemotron-3-Diarization"
_NEMO_FRAME = 0.01
_NEMO_CHANNELS = 8
_NEMO_CACHE = {}


def _recommended_diarizer(num_speakers=None, engine="auto"):
    """回傳 (engine, 原因)，engine 為 "nemotron" 或 "legacy"（伺服器版：不必判斷 Intel Mac）"""
    if engine == "legacy":
        return "legacy", "指定使用現行方法"
    if num_speakers and num_speakers > _NEMO_CHANNELS:
        return "legacy", f"指定 {num_speakers} 人，超過 Nemotron 上限 {_NEMO_CHANNELS} 人"
    if not _HAS_NEMO:
        return "legacy", "伺服器的 transformers 不支援 Nemotron"
    return "nemotron", ""


def _nemo_span(probs_len, seg):
    a = int(seg["start"] / _NEMO_FRAME)
    b = max(a + 1, int(seg["end"] / _NEMO_FRAME))
    b = min(b, probs_len)
    a = min(a, b - 1)
    return max(a, 0), max(b, 1)


def _nemo_segment_labels(probs, segments):
    import numpy as np
    out = []
    for s in segments:
        a, b = _nemo_span(len(probs), s)
        out.append(int(np.asarray(probs[a:b], dtype="float32").sum(axis=0).argmax()))
    return out


def _nemo_saturated(segments, labels):
    used = {l for s, l in zip(segments, labels) if s["end"] - s["start"] >= _DIAR_MIN_CLUSTER_SEC}
    return len(used) >= _NEMO_CHANNELS


def _nemo_limit_speakers(probs, segments, labels, k):
    import numpy as np
    sec = {}
    for s, l in zip(segments, labels):
        sec[l] = sec.get(l, 0.0) + (s["end"] - s["start"])
    if len(sec) <= k:
        return list(labels)
    keep = sorted(sec, key=lambda l: (-sec[l], l))[:k]
    out = []
    for s, l in zip(segments, labels):
        if l in keep:
            out.append(l)
            continue
        a, b = _nemo_span(len(probs), s)
        tot = np.asarray(probs[a:b], dtype="float32").sum(axis=0)
        out.append(max(keep, key=lambda c: tot[c]))
    return out


def _renumber_first_seen(labels):
    m = {}
    return [m.setdefault(l, len(m)) for l in labels]


def _nemo_probs(wav_path):
    import librosa
    import numpy as np
    from transformers import AutoModelForAudioFrameClassification, AutoProcessor
    if "model" not in _NEMO_CACHE:
        proc = AutoProcessor.from_pretrained(NEMO_DIAR_MODEL)
        model = AutoModelForAudioFrameClassification.from_pretrained(NEMO_DIAR_MODEL).to(_torch_device).eval()
        _NEMO_CACHE.update(proc=proc, model=model)
    proc, model = _NEMO_CACHE["proc"], _NEMO_CACHE["model"]
    wav, _ = librosa.load(wav_path, sr=16000, mono=True)
    inp = {k: (v.to(_torch_device) if hasattr(v, "to") else v) for k, v in proc(wav, sampling_rate=16000).items()}
    with torch.inference_mode():
        lg = model(**inp).logits[0].float().cpu().numpy()
    return lg if (lg.min() >= 0 and lg.max() <= 1) else 1 / (1 + np.exp(-lg))


def _nemotron_diarize(wav_path, segments, num_speakers=None):
    try:
        probs = _nemo_probs(wav_path)
    except Exception as e:
        return None, f"Nemotron 執行失敗（{type(e).__name__}: {e}）"
    labels = _nemo_segment_labels(probs, segments)
    if not num_speakers and _nemo_saturated(segments, labels):
        return None, f"{_NEMO_CHANNELS} 位講者全部用滿，可能超過 Nemotron 上限"
    if num_speakers:
        labels = _nemo_limit_speakers(probs, segments, labels, num_speakers)
    return _renumber_first_seen(labels), ""


def _diarize(wav_path, segments, num_speakers=None, engine="auto"):
    """講者辨識入口，回傳 (labels, 實際用的方法, 說明)。labels 失敗為 None"""
    choice, why = _recommended_diarizer(num_speakers, engine)
    if choice == "nemotron":
        labels, why = _nemotron_diarize(wav_path, segments, num_speakers)
        if labels is not None:
            print(f"[diarize] Nemotron（{_torch_device}）{len(set(labels))} 位講者")
            return labels, "nemotron", ""
        print(f"[diarize] 改用現行方法：{why}")
    if not _HAS_DIARIZE:
        return None, "legacy", why or "resemblyzer/spectralcluster 未安裝"
    note = why if (engine == "nemotron" or choice == "nemotron"
                   or (num_speakers and num_speakers > _NEMO_CHANNELS)) else ""
    return _diarize_legacy(wav_path, segments, num_speakers=num_speakers), "legacy", note


def _diarize_legacy(wav_path, segments, num_speakers=None):
    """用 resemblyzer + spectralcluster 辨識講者。
    segments: list of dict，每個含 start, end, text
    回傳: list of int（講者編號 0-based），失敗回傳 None
    """
    from resemblyzer import VoiceEncoder, preprocess_wav
    from spectralcluster import SpectralClusterer, refinement, laplacian
    from spectralcluster import utils as sc_utils

    if not segments:
        return None

    # 載入音訊。
    # **不可以用 preprocess_wav(wav_path) 整檔載入**：它除了重取樣還會做 VAD 靜音
    # 修剪並「刪掉」那些樣本（安裝的版本連 trim_silence 參數都沒有，一定會修），
    # 但下面是用**原始時間軸**去切段落——實測 AMI 一場 18.5 分鐘的會議被刪掉
    # 31.9%，檔尾偏移將近 6 分鐘，等於拿會議別處的聲音去比對。
    # 用戶端 v2.20.0 已修，但伺服器這份是獨立副本，2026-09-21 才發現沒跟到
    # （有 GPU 伺服器時預設就走這條路徑，等於多數使用者一直拿到壞的結果）。
    sr = 16000
    try:
        import librosa
        wav, _ = librosa.load(wav_path, sr=sr, mono=True)
        _per_segment_trim = True
    except Exception:
        wav = preprocess_wav(wav_path)      # 退而求其次，維持舊行為
        _per_segment_trim = False

    # 初始化聲紋編碼器（有 GPU 就用 GPU）
    encoder = VoiceEncoder(_torch_device)
    print(f"[diarize] 提取聲紋（{len(segments)} 段, device={_torch_device}）")

    # ── 只有夠長的段落才進分群 ──
    # 1.6 秒是 resemblyzer 的 partial utterance 長度：短於它時 embed_utterance
    # 會把音訊補零到 1.6s 再算，那個聲紋不可靠。2026-09-22 用有標準答案的中文
    # 會議量到——標錯率 <1.0s 63.6%、1.0~1.6s 66.9%，而 1.6~2.5s 只有 14.8%、
    # 2.5~4.0s 是 0.0%。短段落只佔 24% 的秒數卻貢獻 64% 的「講者搞錯」，
    # 而且它們一起進 affinity 矩陣，把長段落的分群也一起帶壞。
    # 原本的兩個補救（<0.5s 撐成 0.5s 視窗、連續 <0.8s 合併共用一個 embedding）
    # 方向是反的：合併等於強迫相鄰短段落同一個講者，搶話時它們多半不是。
    cluster_floor = _diar_cluster_floor(segments)

    # 逐段提取聲紋
    embeddings = []
    valid_indices = []

    for i, seg in enumerate(segments):
        duration = seg["end"] - seg["start"]
        if duration < cluster_floor:
            embeddings.append(None)
            continue

        audio_slice = wav[int(seg["start"] * sr):int(seg["end"] * sr)]

        if len(audio_slice) < int(0.3 * sr):
            embeddings.append(None)
            continue
        if _per_segment_trim:
            audio_slice = preprocess_wav(audio_slice, source_sr=sr)
            if len(audio_slice) < int(0.3 * sr):
                embeddings.append(None)
                continue

        try:
            if duration >= 1.6:
                emb, partials, _ = encoder.embed_utterance(
                    audio_slice, return_partials=True, rate=1.6, min_coverage=0.75
                )
                emb = np.median(partials, axis=0)
                emb = emb / np.linalg.norm(emb)
            else:
                emb = encoder.embed_utterance(audio_slice)
            embeddings.append(emb)
            valid_indices.append(i)
        except Exception:
            embeddings.append(None)

    if not valid_indices:
        print("[diarize] 無法提取任何有效聲紋")
        return None

    print(f"[diarize] 分群辨識（{len(valid_indices)} 有效段落）")

    # 組合有效 embedding 矩陣
    valid_embeddings = np.array([embeddings[i] for i in valid_indices])

    # SpectralClusterer 分群
    min_clusters = 2 if num_speakers is None else num_speakers
    max_clusters = 8 if num_speakers is None else num_speakers

    refinement_opts = refinement.RefinementOptions(
        # gaussian_blur_sigma=0：不要模糊。高斯模糊假設相鄰列是時間上連續的等寬
        # 視窗，但我們送的是「已合併的講者連續發言」，模糊會抹掉講者交界。
        # 18 場 AMI 實測：blur=1 → DER 43.60%、blur=0 → 16.25%
        gaussian_blur_sigma=0,
        p_percentile=0.98,
        thresholding_soft_multiplier=0.01,
        thresholding_type=refinement.ThresholdType.RowMax,
        symmetrize_type=refinement.SymmetrizeType.Max,
        # 沒有這個參數，上面五個全是死的：預設 None 時整組 refinement 一步都不跑
        refinement_sequence=[
            refinement.RefinementName.CropDiagonal,
            refinement.RefinementName.GaussianBlur,
            refinement.RefinementName.RowWiseThreshold,
            refinement.RefinementName.Symmetrize,
            refinement.RefinementName.Diffuse,
            refinement.RefinementName.RowWiseNormalize,
        ],
    )

    # 使用者沒指定人數時，用特徵值門檻自己估一個（見 _diar_estimate_speakers）。
    # 估不出來就把 min/max 交給函式庫自己的 eigengap，行為與先前相同。
    if num_speakers is None:
        _est = _diar_estimate_speakers(valid_embeddings, refinement_opts,
                                       laplacian.LaplacianType.GraphCut)
        if _est:
            min_clusters = max_clusters = _est

    try:
        clusterer = SpectralClusterer(
            min_clusters=min_clusters,
            max_clusters=max_clusters,
            refinement_options=refinement_opts,
            # 不指定時用 affinity 直接分解，特徵值間隙幾乎總是落在 k=2
            laplacian_type=laplacian.LaplacianType.GraphCut,
            # NormalizedDiff：預設的 Ratio 是「後一個特徵值 / 前一個」，分母很靠近
            # 0 時比值爆大，於是永遠挑最小的 k。會議越長段落越多、譜越平滑，偏誤
            # 越嚴重——中文 37 分鐘那場 1043 段一律吐 k=2（實際 7 人），混淆 40.20%；
            # 改用 NormalizedDiff 後判 3 群、22.90%。
            # **兩個改動必須一起上**（見上面 cluster_floor）。真實 ASR 切段實測：
            # 只換 eigengap 幾乎沒有作用；只換門檻會讓英文 ES2011a 的混淆率
            # 由 12.44% 惡化到 23.29%。一起上才是 12.44% → 9.47%。
            eigengap_type=sc_utils.EigenGapType.NormalizedDiff,
        )
        cluster_labels = clusterer.predict(valid_embeddings)
    except Exception as e:
        print(f"[diarize] 分群失敗: {e}，所有段落標記為 Speaker 1")
        return [0] * len(segments)

    # ── 餘弦相似度二次校正 ──
    unique_labels = sorted(set(cluster_labels))
    if len(unique_labels) > 1:
        centroids = {}
        for label in unique_labels:
            mask = [i for i, l in enumerate(cluster_labels) if l == label]
            centroids[label] = np.mean(valid_embeddings[mask], axis=0)
        reassigned = 0
        for idx in range(len(cluster_labels)):
            emb = valid_embeddings[idx]
            assigned = cluster_labels[idx]
            assigned_sim = float(np.dot(emb, centroids[assigned]))
            best_label, best_sim = assigned, assigned_sim
            for label, centroid in centroids.items():
                sim = float(np.dot(emb, centroid))
                if sim > best_sim:
                    best_label, best_sim = label, sim
            if best_label != assigned and (best_sim - assigned_sim) > 0.1:
                cluster_labels[idx] = best_label
                reassigned += 1
        if reassigned > 0:
            print(f"[diarize] 餘弦校正 {reassigned} 段")

    # 映射回所有段落
    speaker_labels = [None] * len(segments)
    for idx, valid_idx in enumerate(valid_indices):
        speaker_labels[valid_idx] = int(cluster_labels[idx])

    # 填補跳過的段落
    last_valid = 0
    for i in range(len(speaker_labels)):
        if speaker_labels[i] is not None:
            last_valid = speaker_labels[i]
        else:
            speaker_labels[i] = last_valid

    # 多數決平滑已移除（2026-09-21）。
    # 它強制每段採用前後窗口內的多數講者，是當年分群壞掉（未知人數時一律吐 2 群）
    # 時加的補丁。分群修好之後，它變成純粹的傷害，而且**窗口越大越差**：
    #   真實 ASR 段落、AMI 3 場平均 DER —— 不平滑 16.76%、窗口3 25.05%、
    #   窗口5（原設定）30.38%、窗口7 35.12%
    # 單調惡化代表問題出在這個啟發式本身，不是窗口大小沒調好。

    # 按首次出現順序重新編號
    seen = {}
    renumber_map = {}
    counter = 0
    for label in speaker_labels:
        if label not in seen:
            seen[label] = True
            renumber_map[label] = counter
            counter += 1
    speaker_labels = [renumber_map[l] for l in speaker_labels]

    n_speakers = len(set(speaker_labels))
    print(f"[diarize] 完成（{n_speakers} 位講者）")

    return speaker_labels


# ── 模型載入 ──

def _get_model_faster(model_size: str):
    """faster-whisper 模型"""
    from faster_whisper import WhisperModel
    key = f"fw:{model_size}"
    if key not in _models:
        print(f"[載入模型] {model_size} (faster-whisper, device={_device}, compute={_compute_type})")
        _models[key] = WhisperModel(model_size, device=_device, compute_type=_compute_type)
        print(f"[模型就緒] {model_size}")
    return _models[key]


def _get_model_openai(model_size: str):
    """openai-whisper 模型"""
    import whisper as openai_whisper
    # openai-whisper 模型名稱對應：large-v3-turbo → turbo, large-v3 → large
    name_map = {
        "large-v3-turbo": "turbo",
        "large-v3": "large-v3",
        "medium.en": "medium.en",
        "small.en": "small.en",
        "base.en": "base.en",
    }
    ow_name = name_map.get(model_size, model_size)
    key = f"ow:{ow_name}"
    if key not in _models:
        print(f"[載入模型] {ow_name} (openai-whisper, device={_torch_device})")
        _models[key] = openai_whisper.load_model(ow_name, device=_torch_device)
        print(f"[模型就緒] {ow_name}")
    return _models[key], ow_name


# ── 辨識函式 ──

# faster-whisper 離線辨識參數（含長音檔幻覺防護，與用戶端 _FW_OFFLINE_KW 一致）
# - condition_on_previous_text=False：切斷上一段 prompt 傳染
# - hallucination_silence_threshold=2.0：偵測到幻覺時跳過 ≥2s 靜音（需 word_timestamps=True）
# - repetition_penalty=1.05：抑制連續重複片段
_FW_KW = dict(
    beam_size=5,
    vad_filter=True,
    vad_parameters={"min_silence_duration_ms": 500},
    condition_on_previous_text=False,
    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    compression_ratio_threshold=2.4,
    log_prob_threshold=-1.0,
    no_speech_threshold=0.6,
    repetition_penalty=1.05,
    word_timestamps=True,
    hallucination_silence_threshold=2.0,
)

# 寬鬆模式參數（用戶端帶 noisy=1 時切換，對應低音量/監視器/行車紀錄類音源）
# 用戶端在上傳前已做音量增益，伺服器只需放寬 VAD / no_speech / log_prob
_FW_KW_LOOSE = dict(
    beam_size=5,
    vad_filter=False,
    condition_on_previous_text=False,
    temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
    compression_ratio_threshold=2.4,
    # **這兩個一定要一起看**（2026-09-22 修）：faster-whisper 的判斷是
    #     should_skip = no_speech_prob > no_speech_threshold
    #     if log_prob_threshold is not None and avg_logprob > log_prob_threshold:
    #         should_skip = False        ← 唯一的救援
    #     if should_skip: 整個 30 秒視窗直接丟掉
    # **門檻調低是「更容易跳過」，方向與「寬鬆」相反**；原本又把 log_prob_threshold
    # 設成 None 關掉救援，於是它變成唯一且嚴苛的閘門。large-v3 因此吐 0 段
    # （turbo 的 no_speech_prob 剛好低一點才躲過，所以 v2.16.3 至今沒被發現）。
    # 實測同一份低音量中文會議：0.3/None → 0 段、0.6/-2.0 → 100 段；
    # turbo 兩者皆 92 段（無回歸）。-2.0 比嚴格模式的 -1.0 寬，低信心的字仍留得住。
    log_prob_threshold=-2.0,
    no_speech_threshold=0.6,
    repetition_penalty=1.05,
    word_timestamps=False,
)


def _transcribe_faster(wav_path, model_size, language, noisy=False):
    """faster-whisper 辨識"""
    m = _get_model_faster(model_size)
    t0 = time.monotonic()
    kw = _FW_KW_LOOSE if noisy else _FW_KW
    segments_iter, info = m.transcribe(wav_path, language=language, **kw)
    segments = []
    full_text = []
    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            segments.append({"start": round(seg.start, 3), "end": round(seg.end, 3), "text": text})
            full_text.append(text)
    return segments, full_text, round(info.duration, 1), round(time.monotonic() - t0, 1)


def _transcribe_faster_stream(wav_path, model_size, language, noisy=False):
    """faster-whisper 串流版，yield (segment_dict, duration) per segment"""
    m = _get_model_faster(model_size)
    kw = _FW_KW_LOOSE if noisy else _FW_KW
    segments_iter, info = m.transcribe(wav_path, language=language, **kw)
    for seg in segments_iter:
        text = seg.text.strip()
        if text:
            out = {"start": round(seg.start, 3), "end": round(seg.end, 3), "text": text,
                   "language": info.language}
            # 平均對數機率換算成 0～1，供用戶端（REST API 的 confidence）相對比較用
            if getattr(seg, "avg_logprob", None) is not None:
                out["confidence"] = round(min(1.0, max(0.0, math.exp(seg.avg_logprob))), 4)
            yield out, info.duration


class _ProgressCapture:
    """攔截 stdout，解析 openai-whisper verbose 輸出追蹤辨識進度。
    whisper verbose=True 每段輸出格式: [00:00.000 --> 00:30.000]  text..."""

    _TS_RE = re.compile(r'\[[\d:.]+\s*-->\s*([\d:.]+)\]')

    def __init__(self, original, progress_q, audio_duration):
        self._orig = original
        self._q = progress_q
        self._duration = audio_duration

    def write(self, text):
        self._orig.write(text)
        m = self._TS_RE.search(text)
        if m and self._duration > 0:
            secs = self._parse_ts(m.group(1))
            if secs is not None:
                pct = min(secs / self._duration, 1.0)
                self._q.put(("progress", secs, self._duration, pct))
        return len(text) if text else 0

    @staticmethod
    def _parse_ts(ts_str):
        parts = ts_str.split(':')
        try:
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        except ValueError:
            pass
        return None

    def flush(self):
        self._orig.flush()


def _transcribe_openai(wav_path, model_size, language, progress_q=None, noisy=False):
    """openai-whisper 辨識。progress_q: Queue，用於回報辨識進度。
    noisy=True：低音量音源切換寬鬆參數（no_speech 0.3、停用 logprob 過濾）。"""
    m, ow_name = _get_model_openai(model_size)

    # 取得音訊時長
    audio_duration = 0
    if progress_q is not None:
        try:
            import whisper as _ow
            audio = _ow.load_audio(wav_path)
            audio_duration = len(audio) / 16000
            progress_q.put(("duration", audio_duration))
        except Exception:
            pass

    t0 = time.monotonic()

    # openai-whisper 防幻覺參數（API 與 faster-whisper 略有不同）
    if noisy:
        _ow_kw = dict(
            beam_size=5,
            condition_on_previous_text=False,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            compression_ratio_threshold=2.4,
            # 與 _FW_KW_LOOSE 同一個修正：門檻調低＝更容易整段跳過，
            # 而 logprob_threshold=None 會關掉唯一的救援（openai-whisper 同邏輯）
            logprob_threshold=-2.0,
            no_speech_threshold=0.6,
        )
    else:
        _ow_kw = dict(
            beam_size=5,
            condition_on_previous_text=False,
            temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
        )

    # 有 progress_q 時用 verbose=True + stdout 攔截追蹤進度
    if progress_q is not None and audio_duration > 0:
        old_stdout = sys.stdout
        sys.stdout = _ProgressCapture(old_stdout, progress_q, audio_duration)
        try:
            result = m.transcribe(wav_path, language=language, verbose=True, **_ow_kw)
        finally:
            sys.stdout = old_stdout
    else:
        result = m.transcribe(wav_path, language=language, **_ow_kw)

    segments = []
    full_text = []
    for seg in result.get("segments", []):
        text = seg["text"].strip()
        if text:
            segments.append({"start": round(seg["start"], 3), "end": round(seg["end"], 3), "text": text})
            full_text.append(text)
    # openai-whisper 不直接回傳 duration，從最後一段取
    duration = round(segments[-1]["end"], 1) if segments else 0
    return segments, full_text, duration, round(time.monotonic() - t0, 1)


# ── API ──

@app.get("/health")
def health():
    """健康檢查"""
    return {
        "status": "ok",
        "version": SERVER_VERSION,
        "gpu": _device == "cuda",
        "device": _device,
        "backend": _backend,
        "diarize": _HAS_DIARIZE or _HAS_NEMO,
        # Qwen3-ASR（實驗）：null＝這台沒裝；ready=false＝載入中或啟動失敗（見 error）
        "qwen": ({"ready": _qwen_ready(), "model": "qwen3-asr-0.6b", "languages": list(_QWEN_LANG),
                  "error": _QWEN["error"], "restarts": _QWEN["restarts"]}
                 if (_QWEN["proc"] is not None or _QWEN["error"]) else None),
        # 講者辨識可用的方法；auto 時優先 nemotron
        "diar_engines": [e for e, ok in (("nemotron", _HAS_NEMO), ("legacy", _HAS_DIARIZE)) if ok],
        # 用戶端用這個判斷「能不能自動更新」，不必試了才知道
        "can_update": bool(UPDATE_TOKEN),
        # v2.21.7 起一次一件、其餘排隊；用戶端據此決定要不要問「等候／改用本機」
        "queue": True,
        # v2.21.8：有排定的更新時用戶端先等它換完再送件（null＝沒有）
        "update_pending": _update_pending_info(),
    }


@app.post("/v1/admin/update")
async def admin_update(request: Request):
    """用新版的 server.py 取代自己，驗證通過後重啟。

    **這個端點會執行對方送來的程式碼。** 下面的把關分成兩類，不要混為一談：

      「誰可以更新」——只有簽章這一關。沒有密鑰就完全不開放。
      「更新的東西會不會把服務弄死」——校驗碼、語法、selftest、忙碌檢查。
        這些擋的是**壞掉的**更新，**擋不住惡意的**更新（selftest 本身就會
        執行上傳的程式碼）。密鑰是唯一的安全邊界，請當成密碼保管。

    簽章用 HMAC 而不是直接送密鑰：這條連線是 HTTP 不是 HTTPS，
    直接送 Bearer token 的話，任何能側錄封包的人都拿得到可重複使用的憑證，
    等於拿到這台機器的任意程式碼執行權。HMAC 讓側錄者只能重放
    「同一份內容」（無害——那就是同一支程式），無法偽造新的 payload。
    做法與 jtlw_api 的 webhook 簽章一致：HMAC-SHA256 over "{timestamp}.{body}"。
    """
    global _UPDATE_PENDING
    import hashlib
    import hmac as _hmac
    import subprocess

    client_ip = request.client.host if request.client else "?"

    def _deny(code, status, **extra):
        print(f"[更新] 拒絕 {code} 來自 {client_ip}", flush=True)
        return JSONResponse({"error": code, **extra}, status_code=status)

    if not UPDATE_TOKEN:
        return _deny("update_disabled", 403,
                     detail="伺服器未設定 JT_WHISPER_UPDATE_TOKEN")

    # 先看 Content-Length 再決定要不要讀。await request.body() 會把整包讀進
    # 記憶體，沒有上限的話一個大 POST 就能把這台機器的記憶體吃光。
    try:
        clen = int(request.headers.get("content-length") or 0)
    except ValueError:
        clen = 0
    if clen > UPDATE_MAX_BYTES:
        return _deny("payload_too_large", 413,
                     detail=f"上限 {UPDATE_MAX_BYTES} bytes，收到 {clen}")

    ts = (request.headers.get("x-jtw-timestamp") or "").strip()
    sig = (request.headers.get("x-jtw-signature") or "").strip()
    if not ts or not sig:
        return _deny("unauthorized", 401, detail="缺少簽章標頭")
    try:
        skew = abs(time.time() - float(ts))
    except ValueError:
        return _deny("unauthorized", 401, detail="時間戳格式錯誤")
    if skew > 300:
        # 限制重放窗口；兩邊時鐘差太多也會落在這裡
        return _deny("unauthorized", 401, detail=f"時間戳超出容許範圍（差 {int(skew)}s）")

    body = await request.body()
    if len(body) > UPDATE_MAX_BYTES:
        return _deny("payload_too_large", 413, detail=f"上限 {UPDATE_MAX_BYTES} bytes")

    expect = "v1=" + _hmac.new(UPDATE_TOKEN.encode("utf-8"),
                               f"{ts}.".encode("utf-8") + body,
                               hashlib.sha256).hexdigest()
    if not _hmac.compare_digest(sig, expect):
        return _deny("unauthorized", 401, detail="簽章不符")

    want_sha = (request.headers.get("x-content-sha256") or "").strip().lower()
    got_sha = hashlib.sha256(body).hexdigest()
    if want_sha and want_sha != got_sha:
        return _deny("checksum_mismatch", 400, expected=want_sha, actual=got_sha)

    # **有作業在跑時不拒絕**（v2.21.8）：先把驗證做完、排定，等作業做完才換。
    # 以前是回 409 busy 然後放棄——伺服器一直有人在用就永遠更新不了。

    me = os.path.abspath(__file__)
    # 每個請求用自己的暫存檔：兩個用戶端同時推更新時不會互相蓋掉對方正在驗證的檔案
    new_path = f"{me}.new-{os.getpid()}-{threading.get_ident()}-{int(time.time() * 1000)}"
    with open(new_path, "wb") as f:
        f.write(body)

    def _cleanup():
        try:
            os.remove(new_path)
        except OSError:
            pass

    # 驗證一：語法
    try:
        import ast
        ast.parse(open(new_path, encoding="utf-8").read())
    except SyntaxError as e:
        _cleanup()
        return _deny("invalid_syntax", 400, detail=str(e)[:200])

    new_ver = _parse_server_version(new_path)

    # **不接受降版**。多個用戶端共用同一台伺服器是常見情況；只比對「版本不同」
    # 的話，舊用戶端會把伺服器降回舊版，接著新用戶端又推回去——兩邊無限來回，
    # 而每次重啟都會中斷別人正在跑的辨識。
    if _version_tuple(new_ver) < _version_tuple(SERVER_VERSION):
        _cleanup()
        return _deny("downgrade_refused", 409,
                     detail=f"伺服器 {SERVER_VERSION} 比送來的 {new_ver} 新",
                     server_version=SERVER_VERSION, offered=new_ver)
    if _version_tuple(new_ver) == _version_tuple(SERVER_VERSION):
        _cleanup()
        return JSONResponse({"status": "already_current", "version": SERVER_VERSION},
                            status_code=200)
    with _UPDATE_LOCK:
        pend = dict(_UPDATE_PENDING) if _UPDATE_PENDING else None
    if pend and _version_tuple(new_ver) <= _version_tuple(pend["to"]):
        # 已經排定同版或更新的版本：不必再驗一次，告訴對方目前的排定狀態
        _cleanup()
        return JSONResponse(_update_state_body("scheduled"), status_code=202)

    # 驗證二：真的能啟動（import 得起來、設定沒寫壞）
    try:
        r = subprocess.run([sys.executable, new_path, "--selftest"],
                           capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            _cleanup()
            return _deny("selftest_failed", 400, detail=(r.stderr or r.stdout)[-500:])
    except subprocess.TimeoutExpired:
        _cleanup()
        return _deny("selftest_timeout", 400)

    # 排定。**排定之後離線線立刻關門**：已經在跑、在排隊的照樣做完，新的回 503。
    # 即時線要到換檔前一刻才關，讓即時字幕只斷幾秒。
    staged = me + ".pending"
    with _UPDATE_LOCK:
        if _UPDATE_PENDING and _version_tuple(new_ver) <= _version_tuple(_UPDATE_PENDING["to"]):
            _cleanup()   # selftest 期間別人排定了同版或更新的版本
            return JSONResponse(_update_state_body("scheduled"), status_code=202)
        os.replace(new_path, staged)
        first = _UPDATE_PENDING is None
        _UPDATE_PENDING = {"to": new_ver, "from": SERVER_VERSION, "since": time.time(),
                           "client_ip": client_ip, "path": staged}
        _LANES["batch"].close()
    if first:
        threading.Thread(target=_update_worker, daemon=True).start()
    waiting = _LANES["batch"].count()
    print(f"[更新] 排定 {SERVER_VERSION} → {new_ver}，來自 {client_ip}，"
          f"{'立即換版' if waiting == 0 else f'等 {waiting} 件作業做完'}", flush=True)
    if waiting == 0:
        # 舊版用戶端只認得這個格式（看到就開始輪詢版本號）
        return {"status": "updating", "from": SERVER_VERSION, "to": new_ver,
                "restart_in_sec": 1}
    return JSONResponse(_update_state_body("scheduled"), status_code=202)


def _update_state_body(status):
    with _UPDATE_LOCK:
        pend = dict(_UPDATE_PENDING) if _UPDATE_PENDING else {}
    return {"status": status, "from": SERVER_VERSION, "to": pend.get("to"),
            "waiting": _LANES["batch"].count(),
            "since": round(pend["since"], 1) if pend.get("since") else None}


def _update_worker():
    """等作業做完再換版。

    順序很重要：**先關門、再等清空、最後換檔**。先等清空再關門的話，
    「清空」與「關門」之間進來的作業會在跑到一半時被換掉（v2.21.7 以前的 1 秒空窗
    就是這樣砍掉作業的，而用戶端還把它當成 0 段、成功）。
    """
    global _UPDATE_PENDING
    me = os.path.abspath(__file__)
    batch, rt = _LANES["batch"], _LANES["realtime"]
    deadline = time.time() + UPDATE_DRAIN_TIMEOUT

    def _abort(why):
        global _UPDATE_PENDING
        with _UPDATE_LOCK:
            pend = _UPDATE_PENDING
            _UPDATE_PENDING = None
        rt.reopen()
        batch.reopen()
        try:
            os.remove((pend or {}).get("path") or me + ".pending")
        except OSError:
            pass
        print(f"[更新] 放棄這次更新（{why}），恢復收件", flush=True)

    # 離線線在排定時就關了；這裡等它清空
    if not batch.wait_empty(deadline - time.time()):
        return _abort(f"等了 {UPDATE_DRAIN_TIMEOUT} 秒作業仍未做完")
    # 最後一刻才關即時線，並等正在辨識的那一小段做完（通常不到一秒）
    rt.close()
    if not rt.wait_empty(30):
        return _abort("即時辨識 30 秒內沒有做完")

    with _UPDATE_LOCK:
        pend = dict(_UPDATE_PENDING)
    backup = f"{me}.bak-{time.strftime('%Y%m%d-%H%M%S')}"
    try:
        shutil.copy2(me, backup)
        os.replace(pend["path"], me)
        _prune_backups(me)
    except Exception as e:
        return _abort(f"換檔失敗：{e}")
    print(f"[更新] {SERVER_VERSION} → {pend['to']}，來自 {pend['client_ip']}，"
          f"備份 {os.path.basename(backup)}，重啟", flush=True)
    # 讓「立即換版」那次請求的回應先送出去。兩條線都關著、都是空的，
    # 這段時間進來的請求只會拿到 503，不會有作業被砍到一半。
    time.sleep(1.0)
    # **先收掉 Qwen worker 再換**：execv 保留同一個 PID，atexit 不會跑、worker 的看門狗也看不出主服務換了，
    # 不收的話舊 worker 會一直佔著埠號與約 7 GB 顯示記憶體，新服務的 worker 綁不到埠（2026-09-26 審查時發現）
    _qwen_stop(wait=20)
    # os.execv 直接替換行程映像，保留同一個 PID——systemd 看不出差別
    os.execv(sys.executable, [sys.executable] + sys.argv)


@app.get("/v1/status")
def status():
    """伺服器狀態：忙碌、排隊狀況、磁碟空間。

    `busy` / `task` 只看離線線（batch），維持舊用戶端的意思——舊用戶端看到
    busy 會問使用者要不要等；即時線的幾秒短音訊不該觸發那個提示。
    新用戶端看 `queue`：有這個欄位就代表伺服器會自己排隊，直接送出即可。
    """
    batch = _LANES["batch"].snapshot()
    running = batch["running"]

    # /tmp 磁碟空間（暫存檔寫入處）
    disk = shutil.disk_usage(tempfile.gettempdir())
    result = {
        "busy": running is not None,
        "disk_free_gb": round(disk.free / (1024 ** 3), 1),
        "disk_total_gb": round(disk.total / (1024 ** 3), 1),
        "queue": {name: lane.snapshot() for name, lane in _LANES.items()},
        "update_pending": _update_pending_info(),
    }
    if running is not None:
        result["task"] = running
    return result


@app.get("/models")
def list_models():
    """列出已快取的模型"""
    cached = set()
    cached.update(k.split(":", 1)[1] for k in _models.keys())
    # 掃描 HuggingFace cache
    try:
        from huggingface_hub import scan_cache_dir
        cache_info = scan_cache_dir()
        for repo in cache_info.repos:
            name = repo.repo_id
            if name.startswith("Systran/faster-whisper-"):
                cached.add(name.replace("Systran/faster-whisper-", ""))
            elif name.startswith("guillaumekln/faster-whisper-"):
                cached.add(name.replace("guillaumekln/faster-whisper-", ""))
    except Exception:
        pass
    # openai-whisper 模型放在 ~/.cache/whisper/
    whisper_cache = os.path.expanduser("~/.cache/whisper")
    if os.path.isdir(whisper_cache):
        # 檔名格式: large-v3-turbo.pt, medium.en.pt 等
        for f in os.listdir(whisper_cache):
            if f.endswith(".pt"):
                cached.add(f[:-3])
    if _qwen_ready():
        cached.update(_QWEN_MODELS)
    return {"models": sorted(cached)}


def _queued_event(lane, t):
    """排隊中的 NDJSON 事件。舊用戶端不認得 type=queued，會直接略過（if/elif 沒有 else）。"""
    pos = lane.position(t)
    return json.dumps({"type": "queued", "position": pos, "ahead": pos,
                       "waited": round(time.time() - t.enqueued, 1)}) + "\n"


@app.post("/v1/audio/transcriptions")
async def transcribe(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form("large-v3-turbo"),
    language: str = Form("en"),
    stream: str = Form("false"),
    noisy: str = Form("false"),
):
    """接收音訊檔，回傳辨識結果（stream=true 時串流 NDJSON）。
    noisy=1/true：用戶端音源分析判定為低音量錄音，套用寬鬆參數。

    排隊：串流（離線）走 batch 線，在串流裡先送 `{"type":"queued"}` 事件
    直到輪到自己；非串流小檔（即時字幕）走 realtime 線。"""
    client_ip = request.client.host if request.client else ""
    is_noisy = str(noisy).lower() in ("1", "true", "yes")
    is_stream = stream.lower() == "true"

    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()
    except Exception:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise

    is_qwen = model in _QWEN_MODELS
    if is_qwen:
        err = None
        if not is_stream:
            err = (400, "Qwen3-ASR 只支援離線辨識（stream=true）")
        elif not _qwen_ready():
            err = (503, f"Qwen3-ASR 尚未就緒（{_QWEN['error'] or ('載入中' if _QWEN['proc'] else '這台沒有安裝')}）")
        elif language not in _QWEN_LANG:
            err = (400, f"Qwen3-ASR 不支援 language={language}（支援 {'／'.join(_QWEN_LANG)}）")
        if err:
            try:
                os.unlink(tmp.name)
            except OSError:
                pass
            return JSONResponse(status_code=err[0], content={"error": err[1]})

    lane = _LANES["batch"] if (is_stream or len(content) > _REALTIME_MAX_BYTES) \
        else _LANES["realtime"]
    ticket = lane.enter("transcribe", model, language, client_ip)
    if ticket is None:   # 更新排定中，這條線已關門
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        return _updating_response()
    stream_handed_off = False   # 串流回應交出後，清理改由 background 負責
    if is_noisy:
        print(f"[{client_ip}] noisy=1 → 寬鬆參數")
    ahead = lane.position(ticket)
    if ahead > 0:
        print(f"[排隊] {client_ip} 的辨識排入{lane.name}隊伍，前面 {ahead} 件", flush=True)

    try:
        # 串流模式（NDJSON）
        if is_stream:
            tmp_path = tmp.name

            def _wait_turn():
                """generator 開頭：還沒輪到就每 2 秒送一次排隊事件。
                用戶端斷線時 yield 會丟 GeneratorExit，由外層 finally 移出隊伍。"""
                while True:
                    pos = lane.position(ticket)
                    if pos <= 0:
                        return
                    yield _queued_event(lane, ticket)
                    lane.wait(ticket, 2.0)

            if is_qwen:
                # Qwen3-ASR：worker 做完整件才回（切窗、辨識、對齊），期間每 2 秒心跳
                def generate():
                    import concurrent.futures
                    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                    cancelled = False
                    try:
                        yield from _wait_turn()
                        t0 = time.monotonic()
                        future = pool.submit(_transcribe_qwen, tmp_path, language)
                        try:
                            while not future.done():
                                yield json.dumps({"type": "heartbeat",
                                                  "elapsed": round(time.monotonic() - t0, 1)}) + "\n"
                                concurrent.futures.wait([future], timeout=2)
                            segments, duration, proc_time = future.result()
                            for i, seg in enumerate(segments):
                                yield json.dumps({"type": "segment", "index": i, "start": seg["start"],
                                                  "end": seg["end"], "text": seg["text"],
                                                  "duration": round(duration, 1)}, ensure_ascii=False) + "\n"
                            yield json.dumps({"type": "done", "total_segments": len(segments),
                                              "duration": round(duration, 1), "processing_time": proc_time,
                                              "device": "cuda", "engine": "qwen3-asr"}) + "\n"
                        except GeneratorExit:
                            cancelled = True
                            print("[取消] 客戶端中斷連線，等 Qwen3-ASR 這件做完才讓出隊伍...")
                            pool.shutdown(wait=True)
                            return
                        except Exception as e:
                            print(f"[錯誤] Qwen3-ASR 失敗（{client_ip}）：{e}", flush=True)
                            yield json.dumps({"type": "error", "detail": str(e)}, ensure_ascii=False) + "\n"
                    finally:
                        if not cancelled:
                            pool.shutdown(wait=False)
                        lane.leave(ticket)
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
            elif _backend == "faster-whisper":
                def generate():
                    try:
                        yield from _wait_turn()
                        t0 = time.monotonic()
                        count = 0
                        dur = 0
                        try:
                            for seg, dur in _transcribe_faster_stream(tmp_path, model, language, noisy=is_noisy):
                                count += 1
                                yield json.dumps({
                                    "type": "segment", "index": count - 1,
                                    "start": seg["start"], "end": seg["end"],
                                    "text": seg["text"], "duration": round(dur, 1),
                                    "confidence": seg.get("confidence"),
                                    "language": seg.get("language"),
                                }) + "\n"
                            proc_time = round(time.monotonic() - t0, 1)
                            yield json.dumps({
                                "type": "done", "total_segments": count,
                                "duration": round(dur, 1), "processing_time": proc_time,
                                "device": _device,
                            }) + "\n"
                        except GeneratorExit:
                            elapsed = round(time.monotonic() - t0, 1)
                            print(f"[取消] 客戶端中斷連線（{elapsed:.1f}s），faster-whisper 辨識已停止")
                            return
                        except Exception as e:
                            yield json.dumps({"type": "error", "detail": str(e)}) + "\n"
                    finally:
                        lane.leave(ticket)
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass
            else:
                # openai-whisper：辨識中發心跳（含進度），完成後逐段回傳
                def generate():
                    import concurrent.futures
                    pool = None
                    cancelled = False
                    try:
                        yield from _wait_turn()
                        t0 = time.monotonic()
                        progress_q = queue.Queue()
                        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                        future = pool.submit(_transcribe_openai, tmp_path, model, language,
                                             progress_q=progress_q, noisy=is_noisy)
                        audio_dur = 0
                        last_pct = 0
                        last_pos = 0
                        try:
                            while not future.done():
                                # 讀取 progress queue 中的最新進度
                                while not progress_q.empty():
                                    try:
                                        msg = progress_q.get_nowait()
                                        if msg[0] == "duration":
                                            audio_dur = msg[1]
                                        elif msg[0] == "progress":
                                            last_pos = msg[1]
                                            last_pct = msg[3]
                                    except queue.Empty:
                                        break
                                elapsed = round(time.monotonic() - t0, 1)
                                hb = {"type": "heartbeat", "elapsed": elapsed}
                                if audio_dur > 0:
                                    hb["progress"] = round(last_pct, 3)
                                    hb["current"] = round(last_pos, 1)
                                    hb["duration"] = round(audio_dur, 1)
                                yield json.dumps(hb) + "\n"
                                time.sleep(2)
                            segments, full_text, duration, proc_time = future.result()
                            for i, seg in enumerate(segments):
                                yield json.dumps({
                                    "type": "segment", "index": i,
                                    "start": seg["start"], "end": seg["end"],
                                    "text": seg["text"], "duration": round(duration, 1),
                                }) + "\n"
                            yield json.dumps({
                                "type": "done", "total_segments": len(segments),
                                "duration": round(duration, 1), "processing_time": proc_time,
                                "device": _device,
                            }) + "\n"
                        except GeneratorExit:
                            cancelled = True
                            future.cancel()
                            elapsed = round(time.monotonic() - t0, 1)
                            print(f"[取消] 客戶端中斷連線（{elapsed:.1f}s），等待 openai-whisper 辨識執行緒結束...")
                            # 等 transcribe thread 真正結束再清理（GPU 仍在跑）。
                            # **也要等它結束才讓出隊伍**，否則下一件會跟它同時跑。
                            pool.shutdown(wait=True)
                            print(f"[取消] openai-whisper 執行緒已結束")
                            return
                        except Exception as e:
                            yield json.dumps({"type": "error", "detail": str(e)}) + "\n"
                    finally:
                        if pool is not None and not cancelled:
                            pool.shutdown(wait=False)
                        lane.leave(ticket)
                        try:
                            os.unlink(tmp_path)
                        except OSError:
                            pass

            # 串流模式由 generator 負責刪除暫存檔與讓出隊伍，不走 finally。
            # 用戶端中途斷線時，Starlette 只取消外層迭代、不會關閉這個同步 generator，
            # generator 的 finally 就永遠不會執行 → 隊伍卡住、後面的人永遠等不到、暫存檔殘留。
            # 回應結束（含斷線）後一定會跑 background，由它關閉 generator 並補做清理。
            # （generator 還沒開始跑就斷線時 close() 不會進 finally，所以這裡也要 leave。）
            gen = generate()

            def _cleanup_stream():
                try:
                    gen.close()   # 未執行完時觸發 GeneratorExit，走 generator 自己的取消流程
                except Exception as e:
                    print(f"[警告] 關閉辨識串流失敗: {e}")
                lane.leave(ticket)
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

            async def _cleanup_stream_async():
                await run_in_threadpool(_cleanup_stream)   # openai-whisper 取消時會等執行緒結束，不可卡住 event loop

            stream_handed_off = True
            return StreamingResponse(gen, media_type="text/x-ndjson",
                                     background=BackgroundTask(_cleanup_stream_async))

        # 非串流模式：先等輪到自己（即時線通常只等前一段幾百毫秒）
        if not await _wait_turn_async(lane, ticket, request):
            return JSONResponse(status_code=499, content={"error": "client_disconnected"})

        # 用 asyncio.to_thread 避免阻塞 event loop
        try:
            if _backend == "openai-whisper":
                segments, full_text, duration, proc_time = await asyncio.to_thread(
                    _transcribe_openai, tmp.name, model, language, None, is_noisy)
            else:
                segments, full_text, duration, proc_time = await asyncio.to_thread(
                    _transcribe_faster, tmp.name, model, language, is_noisy)
        except Exception as e:
            print(f"[錯誤] 辨識失敗: {model} — {e}")
            return JSONResponse(
                status_code=500,
                content={"error": f"辨識失敗: {model}", "detail": str(e)},
            )

        return {
            "text": " ".join(full_text),
            "segments": segments,
            "language": language,
            "model": model,
            "duration": duration,
            "processing_time": proc_time,
            "device": _device,
            "backend": _backend,
        }
    finally:
        # 非串流模式，或串流回應交出前就出錯時在這裡清理（交出後由 background 清理）
        if not stream_handed_off:
            lane.leave(ticket)
            try:
                os.unlink(tmp.name)
            except OSError:
                pass


@app.post("/v1/audio/diarize")
async def diarize(
    request: Request,
    file: UploadFile = File(...),
    segments: str = Form(...),
    num_speakers: int = Form(0),
    stream: str = Form("false"),
    engine: str = Form("auto"),
):
    """接收音訊檔 + segments JSON，回傳講者辨識結果。
    engine：auto（能用 Nemotron 就用）／nemotron／legacy（resemblyzer）。回應的 engine 是實際用的方法

    走 batch 線排隊。**回應是「前導空白 + JSON」的串流**：排隊與計算期間每 5 秒
    送一個空白字元保持連線（用戶端的讀取逾時是 300 秒，排在一場長會議後面
    一定會超過），最後才送 JSON 本體。JSON 允許前導空白，舊用戶端的
    `json.loads(resp.read())` 照樣解得開。代價是狀態碼一開始就得定成 200，
    排隊之後才發生的錯誤改放在 JSON 的 `error` 欄位。"""
    from fastapi.responses import JSONResponse

    if not (_HAS_DIARIZE or _HAS_NEMO):
        return JSONResponse(
            status_code=500,
            content={"error": "沒有可用的講者辨識方法（resemblyzer 與 Nemotron 都不可用）"},
        )
    if engine not in ("auto", "nemotron", "legacy"):
        return JSONResponse(status_code=400, content={"error": f"engine 必須是 auto／nemotron／legacy，收到 {engine!r}"})

    # 解析 segments JSON
    try:
        seg_list = json.loads(segments)
        if not isinstance(seg_list, list):
            raise ValueError("segments 必須是 list")
        for s in seg_list:
            if not all(k in s for k in ("start", "end", "text")):
                raise ValueError("每個 segment 必須含 start, end, text")
    except (json.JSONDecodeError, ValueError) as e:
        return JSONResponse(
            status_code=400,
            content={"error": f"segments JSON 格式錯誤: {e}"},
        )

    client_ip = request.client.host if request.client else ""
    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        content = await file.read()
        tmp.write(content)
        tmp.close()
    except Exception:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        raise

    lane = _LANES["batch"]
    ticket = lane.enter("diarize", _recommended_diarizer(num_speakers or None, engine)[0], "", client_ip)
    if ticket is None:   # 更新排定中，這條線已關門
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
        return _updating_response()
    ahead = lane.position(ticket)
    if ahead > 0:
        print(f"[排隊] {client_ip} 的講者辨識排入隊伍，前面 {ahead} 件", flush=True)
    ns = num_speakers if num_speakers > 0 else None

    def _release(_=None):
        lane.leave(ticket)
        try:
            os.unlink(tmp.name)
        except OSError:
            pass

    # stream=true（v2.21.9 起的用戶端）：改回 NDJSON，排隊時每 2 秒一個 queued 事件、
    # 計算中每 5 秒一個 heartbeat，最後一行 type=result。呼叫端（v3 API）要靠
    # queued 分辨「在排隊」與「卡住」——空白保活只能保住連線，說不出在等什麼。
    ndjson = str(stream).lower() in ("1", "true", "yes")

    def _line(obj):
        return (json.dumps(obj) + "\n").encode()

    async def body():
        work = None
        try:
            last = 0.0 if ndjson else time.monotonic()
            while lane.position(ticket) > 0:
                if time.monotonic() - last >= (2 if ndjson else 5):
                    yield (_line({"type": "queued", "ahead": lane.position(ticket),
                                  "waited": round(time.time() - ticket.enqueued, 1)})
                           if ndjson else b" ")
                    last = time.monotonic()
                await asyncio.sleep(0.3)
            t0 = time.monotonic()
            work = asyncio.ensure_future(
                asyncio.to_thread(_diarize, tmp.name, seg_list, num_speakers=ns, engine=engine))
            while not work.done():
                await asyncio.wait({work}, timeout=5)
                if not work.done():
                    yield (_line({"type": "heartbeat", "elapsed": round(time.monotonic() - t0, 1)})
                           if ndjson else b" ")
            try:
                speaker_labels, used, note = work.result()
            except Exception as e:
                print(f"[錯誤] diarize 失敗: {e}")
                err = {"error": f"講者辨識失敗: {e}"}
                yield _line({"type": "error", **err}) if ndjson else json.dumps(err).encode()
                return
            if speaker_labels is None:
                # 無法提取聲紋，降級全部 Speaker 0
                speaker_labels = [0] * len(seg_list)
            res = {
                "speaker_labels": speaker_labels,
                "num_speakers": len(set(speaker_labels)),
                "processing_time": round(time.monotonic() - t0, 2),
                "device": _torch_device,
                "engine": used,
                "note": note,
            }
            yield _line({"type": "result", **res}) if ndjson else json.dumps(res).encode()
        finally:
            if work is not None and not work.done():
                # 用戶端斷線了但執行緒還在算：**等它算完才讓出隊伍**，
                # 否則下一件會跟它同時佔用 GPU
                work.add_done_callback(_release)
            else:
                _release()

    return StreamingResponse(body(), media_type="application/x-ndjson" if ndjson else "application/json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="jt-whisper-server")
    parser.add_argument("--port", type=int, default=8978)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--selftest", action="store_true",
                        help="只檢查這份程式能不能正常啟動，然後離開（自動更新前的把關）")
    args = parser.parse_args()

    if args.selftest:
        # 走到這裡代表模組層級的 import 與後端偵測都已經跑完沒有出錯。
        # 再確認幾個實際會被呼叫到的東西存在，避免「import 得起來但端點壞掉」。
        missing = [n for n in ("health", "status", "admin_update", "_diarize", "_diarize_legacy",
                               "_nemotron_diarize", "_transcribe_qwen", "_qwen_worker_main",
                               "_update_worker", "_update_pending_info")
                   if n not in globals()]
        if missing:
            print(f"[selftest] 失敗：缺少 {missing}", file=sys.stderr)
            sys.exit(1)
        if _HAS_DIARIZE:
            # 講者辨識的設定最容易在改版時寫壞，直接把它建起來看看
            from spectralcluster import refinement as _r, laplacian as _l
            _r.RefinementOptions(
                gaussian_blur_sigma=0, p_percentile=0.98,
                thresholding_soft_multiplier=0.01,
                thresholding_type=_r.ThresholdType.RowMax,
                symmetrize_type=_r.SymmetrizeType.Max,
                refinement_sequence=[_r.RefinementName.CropDiagonal])
            _ = _l.LaplacianType.GraphCut
        print(f"[selftest] OK version={SERVER_VERSION} backend={_backend} device={_device}")
        sys.exit(0)

    print(f"[jt-whisper-server] v{SERVER_VERSION} 啟動 {args.host}:{args.port} "
          f"(backend={_backend}, device={_device}"
          f"{', 可遠端更新' if UPDATE_TOKEN else ''})")
    import atexit
    _qwen_start(int(os.environ.get("JT_QWEN_PORT") or args.port + 11))
    atexit.register(_qwen_stop)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
