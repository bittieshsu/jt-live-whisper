"""轉呼叫專案根目錄的 `jtlw_tls`。

**不要在這裡再寫一份。** 這個專案已經因為「同一段邏輯有兩份副本、
只改了一邊」吃過好幾次虧（GPU 伺服器的講者辨識漏掉時間軸修正、
`webui.html` 寫死後端預設值）。TLS 這種只會寫一次、
出錯又很難發現的東西更不該有兩份。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jtlw_tls import (SELF_SIGNED_DAYS, ensure_self_signed,  # noqa: F401,E402
                      fingerprint, not_after, san_hosts)


def _api_self_signed(cert_path, key_path, hosts):
    """API 用的憑證，主體名稱維持原本的 CN（憑證已交付給 JTDT，不要動）"""
    return ensure_self_signed(cert_path, key_path, hosts,
                              subject="/CN=jt-live-whisper API")
