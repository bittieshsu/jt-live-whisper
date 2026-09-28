"""zip 炸彈防護 —— **全站唯一一份判斷**。

## 為什麼要集中

辦公文件（.docx / .odt / .xlsx…）、設定備份、資產匯入、工作區收檔 —— 全都是
zip。判斷寫在各處一定會漂（這個專案在「同一份清單寫兩個地方」上吃過很多次虧），
而且漏掉的那一處**完全看不出來**：使用者上傳一個 40 KB 的檔案，伺服器安靜地
展開成幾十 GB。

## 判準

* **解開後的總量**超過 1 GB → 拒絕。正常的辦公文件極少接近，
  而炸彈動輒幾十 GB。
* **壓縮比**超過 200 倍**而且**解開後超過 64 MB → 拒絕。
  只看比例會誤傷「很多重複內容」的正常文件（純文字的 .odt 壓縮比可以很高
  但總量很小），所以兩個條件要同時成立。
* **只看檔頭宣告的大小，不真的解壓** —— 毫秒級，而且不會為了檢查先把炸彈
  展開一次（那就本末倒置了）。

**例外**：設定備份的匯入（`settings_export.import_from_zip`）**有自己更嚴格的
一套**（總量 2 GiB、單檔 512 MiB，外加 zip-slip 的路徑白名單），因為它會把內容
寫進 `data/` 底下 —— 那條路的風險不只是資源耗盡。**不要為了「統一」把它換成
這裡的通用判斷，那是弱化。**

> 檔頭宣告的大小理論上可以造假，但 zipfile 解壓時會以實際資料為準；
> 這一層擋的是「宣稱很大」的那一類，配合上層的上傳大小上限已經足夠。
"""
from __future__ import annotations

import os
import zipfile
from typing import Union

#: 解開後的總量上限。
MAX_UNCOMPRESSED = 1024 * 1024 * 1024
#: 壓縮比上限（要與 MIN_SIZE_FOR_RATIO 同時成立才判定）。
MAX_RATIO = 200
#: 低於這個大小不看壓縮比 —— 小檔案的高壓縮比很常見且無害。
MIN_SIZE_FOR_RATIO = 64 * 1024 * 1024


class ZipBombError(ValueError):
    """這份 zip 解開後大得離譜，拒絕處理。"""


def check(zf: zipfile.ZipFile, *,
          max_uncompressed: int = MAX_UNCOMPRESSED,
          max_ratio: int = MAX_RATIO) -> None:
    """檢查一個**已開啟**的 ZipFile。超標就丟 `ZipBombError`。

    上限可以依用途調整，但**實作只有這一份** —— 例如統編資料庫的匯入
    （170 萬筆）解開後本來就好幾百 MB，套一般文件的門檻會擋掉合法操作；
    那種地方要**明確傳一個更大的值並寫明理由**，而不是各自再寫一套判斷。
    """
    total = comp = 0
    for info in zf.infolist():
        total += info.file_size
        comp += info.compress_size
    if total > max_uncompressed:
        raise ZipBombError(
            "這份檔案解開後異常龐大，為了避免耗盡伺服器資源而拒絕處理。")
    if (comp > 0 and total > MIN_SIZE_FOR_RATIO
            and total // max(comp, 1) > max_ratio):
        raise ZipBombError(
            "這份檔案的壓縮比異常（解開後遠大於檔案本身），"
            "為了避免耗盡伺服器資源而拒絕處理。")


def check_path(path: Union[str, "os.PathLike"]) -> None:
    """檢查一個 zip 檔（不是 zip 就直接放行，交給呼叫端自己判斷格式）。"""
    try:
        with zipfile.ZipFile(path) as zf:
            check(zf)
    except zipfile.BadZipFile:
        return
