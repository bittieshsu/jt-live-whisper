"""jtlw REST API 設定

來源（後者覆蓋前者）：
1. 專案根目錄的 config.json 的 "api" 區塊
2. 環境變數 JTLW_API_*

API Key 存的是雜湊，不是明文；用 `python -m jtlw_api.keys add jtdt` 產生。
"""
import hashlib
import json
import os
import secrets
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import translate_meeting as tm   # noqa: E402  取 SUMMARY_DEFAULT_MODEL，避免兩處各寫一份預設

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(ROOT, "config.json")

API_VERSION = "v1"
API_REVISION = "2.6"                   # 2.5（2026-10-01）：hints.diarize_engine、Result.diarization；2.6（同日，JTDT 要求）：diarization.reason
# 講者辨識方法：API 一律**明確送出**，預設維持現行方法（resemblyzer）。
# 2026-09-28 JTDT v2.17 要求、使用者同意：Nemotron 由 JTDT 自己送參數切換（api_revision 2.5 的 hints.diarize_engine），
# 不可以因為 GPU 伺服器裝了 transformers 5.18 就悄悄換掉他們的逐字稿。
# 不送的話用戶端會送 "auto"，GPU 伺服器能用 Nemotron 時就會改用它
DIARIZE_ENGINE_DEFAULT = "legacy"
RESULT_SCHEMA_VERSION = "2.3"          # 2.1（api_revision 2.4）：新增選填的 summary_url；2.2（2.5）：diarization；2.3（2.6）：diarization.reason

# 上限（會回在 /capabilities.limits）
DEFAULT_LIMITS = {
    "max_duration_ms": 6 * 3600 * 1000,      # 6 小時
    "max_source_bytes": 4 * 1024 ** 3,       # 4 GB
    "min_sample_rate_hz": 8000,
    "max_glossary_inline_entries": 500,
    "max_glossary_entries": 10000,
    "max_segments_page": 1000,
}
SEGMENT_PAGE_DEFAULT = 500
ASR_BIAS_MAX_TERMS = 50            # ASR 提示長度有限，只取前面的詞
SEGMENT_BATCH = 20                 # 累積幾段送一次 segments.appended
SEGMENT_BATCH_SECONDS = 5.0        # 或最多間隔幾秒
UNACKED_TTL_SEC = 7 * 24 * 3600    # 未 ACK 的終態作業保留內容
RECORD_TTL_SEC = 7 * 24 * 3600     # ACK / 過期後，不含內容的紀錄再留多久
MAX_QUEUE_DEPTH = 100
# 預抓音檔時要保留的磁碟空間。預抓等於把還沒輪到的作業的檔案先堆在磁碟上，
# 低於這個水位就不預抓（改回輪到時才抓），避免把整台機器寫滿。
FETCH_DISK_RESERVE_BYTES = int(os.environ.get("JTLW_API_FETCH_RESERVE_GB", "5")) * 1024 ** 3
# 上傳的來源檔（api_revision 2.4）：沒有被送件用掉的，放多久之後刪掉（這台磁碟不大，錄影一小時 0.5~1 GB）
UPLOAD_TTL_SEC = 24 * 3600
WEBHOOK_MAX_ATTEMPTS = 5
WEBHOOK_REPLAY_WINDOW_SEC = 300

SCOPES = ("jobs:write", "jobs:read", "jobs:cancel", "profiles:read", "admin")


def current_source_token():
    """拉音檔要帶的 Bearer token，**每次現讀**。

    Settings 只在啟動時讀一次設定，走 self.source_token 的話對方換 token
    就得重啟服務。拉檔本來就要讀網路，多讀一次小檔的成本可以忽略。
    """
    try:
        return ((_load_config().get("api") or {}).get("source_token")
                or os.environ.get("JTLW_API_SOURCE_TOKEN", ""))
    except Exception:
        return os.environ.get("JTLW_API_SOURCE_TOKEN", "")


def _load_config():
    if not os.path.isfile(CONFIG_FILE):
        return {}
    try:
        with open(CONFIG_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def hash_key(raw_key):
    """API Key 只存雜湊；格式 jtlw_<id>_<secret>"""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def new_key(client_id):
    """產生一組新的 API Key，回傳 (明文, 雜湊)"""
    raw = f"jtlw_{client_id}_{secrets.token_urlsafe(32)}"
    return raw, hash_key(raw)


class Settings:
    """一次讀好設定，之後不再讀檔（重新載入請重建物件）"""

    def __init__(self, overrides=None):
        cfg = _load_config()
        api = dict(cfg.get("api") or {})
        api.update(overrides or {})

        self.host = os.environ.get("JTLW_API_HOST", api.get("host", "0.0.0.0"))
        self.port = int(os.environ.get("JTLW_API_PORT", api.get("port", 8080)))
        self.data_dir = os.environ.get(
            "JTLW_API_DATA_DIR", api.get("data_dir", os.path.join(ROOT, "api_data")))
        self.db_path = os.path.join(self.data_dir, "jobs.db")
        self.work_dir = os.path.join(self.data_dir, "work")

        # {金鑰雜湊: {"client": 名稱, "scopes": set}}
        self.api_keys = {}
        for item in api.get("api_keys", []):
            if isinstance(item, dict) and item.get("key_sha256"):
                self.api_keys[item["key_sha256"]] = {
                    "client": item.get("client", "unknown"),
                    "scopes": set(item.get("scopes") or SCOPES),
                }
        # 開發用：環境變數直接給明文（正式部署請用 config.json 的雜湊）
        for raw in filter(None, os.environ.get("JTLW_API_DEV_KEY", "").split(",")):
            self.api_keys[hash_key(raw.strip())] = {"client": "dev", "scopes": set(SCOPES)}

        # 只允許從這些主機拉音訊與詞彙庫（防 SSRF）
        env_hosts = os.environ.get("JTLW_API_ALLOWED_HOSTS", "")
        self.allowed_hosts = {h.strip() for h in
                              (env_hosts.split(",") if env_hosts else api.get("allowed_hosts", []))
                              if h and h.strip()}

        self.limits = dict(DEFAULT_LIMITS, **(api.get("limits") or {}))
        self.max_queue_depth = int(api.get("max_queue_depth", MAX_QUEUE_DEPTH))
        self.workers = int(os.environ.get("JTLW_API_WORKERS", api.get("workers", 1)))

        # 引擎設定（沿用 config.json 既有欄位，API 專屬可在 api 區塊覆蓋）
        self.remote_whisper = api.get("remote_whisper", cfg.get("remote_whisper"))
        self.llm_host = api.get("llm_host", cfg.get("llm_host", ""))
        self.llm_port = int(api.get("llm_port", cfg.get("llm_port", 11434)))
        # 預設跟著 translate_meeting 的 SUMMARY_DEFAULT_MODEL 走，不要另外寫死一個舊模型。
        # 2026-09-18 用 882 段有標準答案的語料實測：gpt-oss:120b 的校正會讓英文 CER
        # 從 9.67% 惡化到 10.81%，qwen3.8:27b 則降到 9.48%。給 JTDT 的風險數字是用
        # qwen3.8:27b 量的，API 預設若停在舊模型，他們拿到的品質會與數字不符。
        # 仍可用 config.json 的 api.correction_model 明確指定（覆蓋預設）。
        self.correction_model = api.get("correction_model") or tm.SUMMARY_DEFAULT_MODEL
        # 會議摘要（api_revision 2.4）用的模型：有專用設定（api.summary_model）就用專用的，
        # 沒有才繼承 API 的模型設定（校正模型）。JTDT 的「每個工具可以自訂模型、沒設才用全域」也是這個規則
        self.summary_model = api.get("summary_model") or self.correction_model
        self.asr_model = api.get("asr_model", "large-v3-turbo")
        # 拉音檔時要帶的服務 Token（JTDT 2026-09-20 確認他們的音檔網址需要認證）。
        # 這裡只是讓 /capabilities 之類的地方看得到有沒有設；實際拉檔走
        # current_source_token()，換 token 不必重啟服務。
        self.source_token = api.get("source_token", "")
        # ── TLS ──
        # 預設開啟並自簽（這支 API 會帶著 API Key 與客戶的會議逐字稿在區網上跑，
        # 區網不是安全網路）。管理者要換成正式憑證時，把 tls_cert / tls_key
        # 指到自己的檔案即可——**已存在的憑證不會被覆蓋**。
        self.tls_enabled = bool(api.get("tls", True))
        if os.environ.get("JTLW_API_TLS", "") in ("0", "off", "false"):
            self.tls_enabled = False
        self.tls_cert = api.get("tls_cert") or os.path.join(self.data_dir, "tls", "server.crt")
        self.tls_key = api.get("tls_key") or os.path.join(self.data_dir, "tls", "server.key")
        # 自簽憑證要寫進 subjectAltName 的位址；沒設就用本機能查到的 IP
        self.tls_hosts = api.get("tls_hosts") or []

        # 測試用：不做真正的語音處理，改產生假逐字稿
        self.fake_engine = os.environ.get("JTLW_API_FAKE_ENGINE", "") == "1"
        self.speed = float(os.environ.get("JTLW_API_SPEED", "1"))

    def key_info(self, raw_key):
        return self.api_keys.get(hash_key(raw_key))

    def host_allowed(self, url):
        """回傳 (是否允許, 主機名)。沒設定清單時一律不允許（避免誤開 SSRF）"""
        import urllib.parse
        host = urllib.parse.urlparse(url or "").hostname or ""
        if not self.allowed_hosts:
            return (self.fake_engine, host)   # 只有測試模式可以不設清單
        return (host in self.allowed_hosts, host)
