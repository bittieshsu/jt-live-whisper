"""啟動 API 伺服器：venv/bin/python -m jtlw_api [--host H] [--port P]

另外兩個不啟動伺服器、只給管理者用的模式：
    --info              印出金鑰、憑證、允許來源等設定在哪裡（回答「要去哪看」）
    --new-key <名稱>    產生一組新金鑰，印出明文一次與要貼進 config.json 的片段（不改設定檔）

要直接寫進 config.json 的話用 `python -m jtlw_api.keys add <名稱>`（列出：`keys list`、撤銷：`keys revoke <sha256 前綴>`）
"""
import argparse
import json
import os
import socket
import sys

# 缺套件時講清楚缺什麼、怎麼補（v2.25.1 起公開：照手冊安裝的機器若沒跑伺服器模式，
# 或是舊版升上來還沒補裝，直接 import 會噴一長串 ImportError，看不出該做什麼）
_missing = []
for _mod, _pkg in (("fastapi", "fastapi"), ("uvicorn", "uvicorn"), ("jsonschema", "jsonschema"),
                   ("multipart", "python-multipart")):
    try:
        __import__(_mod)
    except ImportError:
        _missing.append(_pkg)
if _missing:
    sys.stderr.write(
        f"  [錯誤] REST API 缺少套件：{'、'.join(_missing)}\n"
        f"  補裝：./install.sh --server（Linux 伺服器版會自動安裝），"
        f"或 venv/bin/pip install {' '.join(_missing)}\n")
    sys.exit(1)

import uvicorn  # noqa: E402

from . import config  # noqa: E402
from .keys import CLIENT_SCOPES  # noqa: E402
from . import tls as tlsmod
from .app import ApiState, app
from . import app as app_module


def _local_addresses():
    """本機能對外的位址，寫進自簽憑證的 subjectAltName。

    少了這些，對方用 IP 連進來會驗不過憑證（憑證裡沒有那個 IP），
    症狀是「連得上但一直說憑證無效」。
    """
    hosts = {"localhost", "127.0.0.1"}
    try:
        hosts.add(socket.gethostname())
    except Exception:
        pass
    try:
        # 連一個不會真的送封包的位址，問核心「出去會用哪個 IP」
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("192.0.2.1", 1))
        hosts.add(s.getsockname()[0])
        s.close()
    except Exception:
        pass
    return sorted(hosts)


def _print_cert(st):
    """憑證資訊。管理者最常要的就是指紋（對方要釘選）與到期日。"""
    if not os.path.exists(st.tls_cert):
        print(f"  憑證：尚未產生（第一次啟動時會自簽到 {st.tls_cert}）")
        return
    print(f"  憑證：{st.tls_cert}")
    print(f"    私鑰：{st.tls_key}（權限應為 600，不可外流）")
    print(f"    有效期限：{tlsmod.not_after(st.tls_cert)}")
    print(f"    SHA-256 指紋：{tlsmod.fingerprint(st.tls_cert)}")
    hosts = tlsmod.san_hosts(st.tls_cert)
    if hosts:
        print(f"    憑證中的位址：{', '.join(hosts)}")
        print("    （只能用上面這些位址連，換成別的會驗不過；"
              "對方要釘選憑證時把指紋給他核對）")


def _print_keys(st):
    """金鑰只存雜湊，這裡能給的就是「有幾組、誰的、什麼權限」。"""
    if not st.api_keys:
        print("  API Key：未設定（除 /health 外都會回 401）")
        return
    print(f"  API Key：{len(st.api_keys)} 組（**只存 sha256 雜湊，無法從這裡還原明文**）")
    for h, info in sorted(st.api_keys.items(), key=lambda kv: kv[1]["client"]):
        print(f"    - {info['client']:<12} sha256={h[:16]}…  "
              f"scopes={','.join(sorted(info['scopes']))}")
    print("    金鑰遺失時無法救回，只能重新產生（python -m jtlw_api.keys add <名稱>）並同步給對方；")
    print("    舊的用 python -m jtlw_api.keys revoke <sha256 前綴> 撤銷（寫名稱會撤掉同名的全部）")


def do_info(host=None, port=None):
    """回答「設定在哪裡看」，不啟動伺服器"""
    st = config.Settings()
    print(f"  jt-live-whisper API（api_revision {config.API_REVISION}）")
    print(f"  設定檔：{config.CONFIG_FILE} 的 \"api\" 區塊")
    # **設定檔的值不一定是實際在跑的值**：systemd 的 ExecStart 可以用
    # --host / --port 覆蓋。只印設定檔的數字會把管理者導到錯的埠
    # （223 上設定是 8080、實際跑 8790，2026-09-22 踩到）。
    eff_host, eff_port = host or st.host, port or st.port
    note = "" if (host or port) else "　← 設定檔的值；systemd 可用 --port 覆蓋"
    print(f"  監聽：{'https' if st.tls_enabled else 'http'}://{eff_host}:{eff_port}/api/v1{note}")
    if not (host or port):
        print("    實際在跑的埠：systemctl cat jtlw-api.service | grep ExecStart")
    print(f"  資料目錄：{st.data_dir}")
    _print_keys(st)
    _print_cert(st)
    print(f"  允許的來源主機：{', '.join(sorted(st.allowed_hosts)) or '未設定（用網址送件會被拒；上傳的不受影響）'}")
    print(f"  辨識模型：{st.asr_model}　校正模型：{st.correction_model}")
    sys.stdout.flush()


def do_new_key(client_id):
    """產生金鑰。**不自動寫進 config.json**——讓管理者看清楚要加什麼，
    也避免在不該改的機器上誤改設定。明文只印這一次。"""
    raw, h = config.new_key(client_id)
    print(f"  新金鑰（{client_id}）**只會出現這一次，請立刻交付給對方**：")
    print(f"\n    {raw}\n")
    # **一定要寫 scopes**：沒寫的金鑰會拿到全部權限（含 admin，見 config.Settings），
    # 而 admin 刻意不發給外部系統（keys.py 的 CLIENT_SCOPES）。2026-09-28 前這段沒寫 scopes
    entry = {"client": client_id, "key_sha256": h, "scopes": list(CLIENT_SCOPES)}
    print(f"  把下面這段加進 {config.CONFIG_FILE} 的 api.api_keys 陣列，然後重啟服務：")
    print(f"\n    {json.dumps(entry, ensure_ascii=False)}\n")
    print("  （不想手動貼：改用 python -m jtlw_api.keys add <名稱>，會直接寫進設定檔）")
    print("  對方填的是**金鑰本身**，送出時才組成 Authorization: Bearer <金鑰>；")
    print("  連前綴一起貼會得到 401 details.reason=key_has_bearer_prefix。")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser(description="jt-live-whisper REST API")
    ap.add_argument("--host", default=None)
    ap.add_argument("--port", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None, help="同時處理幾件作業")
    ap.add_argument("--no-tls", action="store_true", help="關閉 TLS（只建議在本機測試用）")
    ap.add_argument("--info", action="store_true",
                    help="印出金鑰、憑證、允許來源等設定在哪裡，不啟動伺服器")
    ap.add_argument("--new-key", metavar="名稱",
                    help="產生一組新的 API Key（明文只印一次），不啟動伺服器")
    args = ap.parse_args()

    if args.info:
        return do_info(args.host, args.port)
    if args.new_key:
        return do_new_key(args.new_key)

    state = ApiState()
    if args.workers:
        state.settings.workers = args.workers
    app_module.STATE = state
    host = args.host or state.settings.host
    port = args.port or state.settings.port
    st = state.settings

    ssl_kw = {}
    scheme = "http"
    if st.tls_enabled and not args.no_tls:
        hosts = list(st.tls_hosts) or _local_addresses()
        try:
            created = tlsmod.ensure_self_signed(st.tls_cert, st.tls_key, hosts)
        except Exception as e:
            print(f"  [TLS] 產生自簽憑證失敗，改用 HTTP：{e}")
        else:
            ssl_kw = {"ssl_certfile": st.tls_cert, "ssl_keyfile": st.tls_key}
            scheme = "https"
            kind = "自簽（本次新產生）" if created else "沿用既有憑證"
            print(f"  TLS：{kind}")
            _print_cert(st)

    print(f"  jt-live-whisper API（api_revision {config.API_REVISION}）："
          f"{scheme}://{host}:{port}/api/v1/health")
    print(f"  資料目錄：{state.settings.data_dir}")
    _print_keys(st)
    print(f"  允許的來源主機：{', '.join(sorted(st.allowed_hosts)) or '未設定（用網址送件會被拒；上傳的不受影響）'}")
    # **一定要 flush**：systemd 下 stdout 是區塊緩衝，不 flush 的話這段啟動資訊
    # 會卡在緩衝區裡，要等之後的請求日誌把緩衝填滿才一起吐出來。
    # 管理者重啟後馬上看 journalctl 會看到「什麼都沒有」，而憑證指紋正是
    # 那時候最需要的東西（2026-09-22 在伺服器版實機上實際踩到）。
    sys.stdout.flush()
    uvicorn.run(app, host=host, port=port, log_level="warning", **ssl_kw)


if __name__ == "__main__":
    main()
