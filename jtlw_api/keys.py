"""API Key 管理

  venv/bin/python -m jtlw_api.keys add <客戶名稱> [scope ...]
  venv/bin/python -m jtlw_api.keys list
  venv/bin/python -m jtlw_api.keys revoke <客戶名稱|金鑰前綴>

revoke 寫**客戶名稱**會撤掉同名的**全部**金鑰；只撤一把要寫 `keys list` 顯示的 sha256 前綴。
新增或撤銷之後要重啟服務（systemctl restart jtlw-api）才生效。

金鑰只顯示一次，config.json 內只存 sha256。

**不指定 scope 時給的是一般客戶權限，不含 admin**——`admin` 會繞過所有權限檢查
（見 app.py 的 auth()），要給必須明確寫出來。
"""
import json
import os
import sys

from .config import CONFIG_FILE, SCOPES, new_key

# 一般客戶（例如 JTDT）需要的權限：送件、查狀態、取消、讀 profiles。
# admin 刻意排除——它是萬用權限，只給自己人維運用。
CLIENT_SCOPES = tuple(s for s in SCOPES if s != "admin")


def _load():
    if os.path.isfile(CONFIG_FILE):
        with open(CONFIG_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save(cfg):
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")


def cmd_add(argv):
    client = argv[0]
    scopes = argv[1:] or list(CLIENT_SCOPES)
    bad = [s for s in scopes if s not in SCOPES]
    if bad:
        print(f"不認得的權限: {', '.join(bad)}；可用: {', '.join(SCOPES)}")
        sys.exit(1)
    if "admin" in scopes:
        print("注意：admin 會繞過所有權限檢查，不要發給外部合作方。")

    cfg = _load()
    raw, digest = new_key(client)
    cfg.setdefault("api", {}).setdefault("api_keys", []).append(
        {"client": client, "key_sha256": digest, "scopes": scopes})
    _save(cfg)
    print(f"客戶：{client}")
    print(f"權限：{', '.join(scopes)}")
    print(f"API Key（只會顯示這一次）：\n  {raw}")


def cmd_list(_argv):
    keys = (_load().get("api") or {}).get("api_keys") or []
    if not keys:
        print("（沒有任何 API Key）")
        return
    print(f"{'客戶':<16}{'sha256 前 12 碼':<16}權限")
    for k in keys:
        # 沒寫 scopes 的金鑰在服務裡拿到的是**全部權限（含 admin）**（config.Settings），
        # 原本這裡印空白，看起來像沒有權限——剛好相反
        scopes = ", ".join(k.get("scopes") or []) or "（未寫＝全部權限，含 admin；建議補上或重建）"
        print(f"{k.get('client', '?'):<16}{(k.get('key_sha256') or '')[:12]:<16}{scopes}")


def cmd_revoke(argv):
    """依客戶名稱或 sha256 前綴移除；同名有多把時全部移除並回報數量"""
    target = argv[0]
    cfg = _load()
    api = cfg.setdefault("api", {})
    keys = api.get("api_keys") or []
    keep, gone = [], []
    for k in keys:
        if k.get("client") == target or (k.get("key_sha256") or "").startswith(target):
            gone.append(k)
        else:
            keep.append(k)
    if not gone:
        print(f"找不到符合「{target}」的金鑰")
        sys.exit(1)
    api["api_keys"] = keep
    _save(cfg)
    for k in gone:
        print(f"已撤銷：{k.get('client')} ({(k.get('key_sha256') or '')[:12]}…) "
              f"權限 {', '.join(k.get('scopes') or [])}")
    print(f"共撤銷 {len(gone)} 把，剩餘 {len(keep)} 把")


def main():
    cmds = {"add": (cmd_add, 1), "list": (cmd_list, 0), "revoke": (cmd_revoke, 1)}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__)
        sys.exit(1)
    fn, need = cmds[sys.argv[1]]
    args = sys.argv[2:]
    if len(args) < need:
        print(__doc__)
        sys.exit(1)
    fn(args)


if __name__ == "__main__":
    main()
