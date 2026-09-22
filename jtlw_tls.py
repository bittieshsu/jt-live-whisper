"""TLS 憑證：預設自簽，管理者可以換成自己的。

**這支同時被 `webui.py` 與 `jtlw_api/` 使用**，所以放在專案根目錄而不是
`jtlw_api/` 裡面——`jtlw_api/` 不在 GitHub repo 裡，`webui.py` 不能依賴它。
（`jtlw_api/tls.py` 現在只是轉呼叫這裡。）

為什麼預設自簽而不是不加密：這支 API 會帶著 API Key 與客戶的會議逐字稿
在區域網路上跑。區網不是安全網路——同一天我們才因為「更新端點的密鑰走
明文 HTTP」而改用 HMAC 簽章。自簽憑證擋不住主動的中間人，但擋得住被動側錄，
而且**換成正式憑證只要改兩行設定**。

用 `openssl` CLI 而不是 `cryptography` 套件：兩台機器上 openssl 本來就有，
不必為了產一張憑證多裝一個相依。
"""
import hashlib
import ipaddress
import os
import subprocess

# 憑證有效期。自簽憑證不該給太長——過期會提醒你「這東西本來就該換掉」。
SELF_SIGNED_DAYS = 825


def _san_entries(hosts):
    """把主機清單轉成 openssl 的 subjectAltName。

    **IP 必須寫成 `IP:`**，寫成 `DNS:` 的話用 IP 連線時驗不過——
    這是自簽憑證最常見的坑（連得上但對方一直說憑證無效）。
    """
    out = []
    for h in hosts:
        h = (h or "").strip()
        if not h:
            continue
        try:
            ipaddress.ip_address(h)
            out.append(f"IP:{h}")
        except ValueError:
            out.append(f"DNS:{h}")
    return out or ["DNS:localhost", "IP:127.0.0.1"]


def fingerprint(cert_path):
    """憑證的 SHA-256 指紋（冒號分隔大寫），給對方釘選用"""
    try:
        r = subprocess.run(["openssl", "x509", "-in", cert_path, "-noout",
                            "-fingerprint", "-sha256"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and "=" in r.stdout:
            return r.stdout.strip().split("=", 1)[1]
    except Exception:
        pass
    # openssl 不在時自己算（DER 的 SHA-256 才是指紋，不能對 PEM 直接雜湊）
    try:
        import ssl
        der = ssl.PEM_cert_to_DER_cert(open(cert_path, encoding="utf-8").read())
        h = hashlib.sha256(der).hexdigest().upper()
        return ":".join(h[i:i + 2] for i in range(0, len(h), 2))
    except Exception:
        return "?"


def not_after(cert_path):
    try:
        r = subprocess.run(["openssl", "x509", "-in", cert_path, "-noout", "-enddate"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0 and "=" in r.stdout:
            return r.stdout.strip().split("=", 1)[1]
    except Exception:
        pass
    return "?"


def san_hosts(cert_path):
    """讀出憑證裡的 subjectAltName，回傳位址清單。

    給 --info 用：管理者要回答「這張憑證能用哪個位址連」時，
    不該叫他自己去背 openssl 指令。
    """
    try:
        r = subprocess.run(["openssl", "x509", "-in", cert_path, "-noout",
                            "-ext", "subjectAltName"],
                           capture_output=True, text=True, timeout=10)
        if r.returncode == 0:
            out = []
            for part in r.stdout.replace("\n", ",").split(","):
                part = part.strip()
                for pre in ("IP Address:", "DNS:", "IP:"):
                    if part.startswith(pre):
                        out.append(part[len(pre):])
            return out
    except Exception:
        pass
    return []


def ensure_self_signed(cert_path, key_path, hosts,
                       subject="/CN=jt-live-whisper"):
    """憑證不存在時產一張自簽的；已存在就原樣沿用（**不覆蓋**）。

    不覆蓋是刻意的：管理者可能把正式憑證直接放在同一個路徑，
    每次啟動就蓋掉的話，換憑證這件事會變成「換完重啟就沒了」。
    回傳 (是否是這次新產生的)。
    """
    if os.path.isfile(cert_path) and os.path.isfile(key_path):
        return False
    os.makedirs(os.path.dirname(cert_path) or ".", exist_ok=True)
    san = ",".join(_san_entries(hosts))
    cmd = [
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
        "-keyout", key_path, "-out", cert_path,
        "-days", str(SELF_SIGNED_DAYS), "-sha256",
        "-subj", subject,
        "-addext", f"subjectAltName={san}",
        "-addext", "keyUsage=critical,digitalSignature,keyEncipherment",
        "-addext", "extendedKeyUsage=serverAuth",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"產生自簽憑證失敗：{(r.stderr or r.stdout)[-300:]}")
    os.chmod(key_path, 0o600)      # 私鑰不可以讓別人讀
    return True
