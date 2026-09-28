"""把各種逐字稿檔案讀成 `meeting_insight` 吃的段落。

輸出的每一段是 `{"seq", "text"}`，有的話再加 `speaker` / `start_ms` / `end_ms`。
**時間與發言者是選用的** —— 沒有的話發言者佔比與章節時間軸自動不出現
（`meeting_insight.suitable_charts` 用資料判斷），而不是畫一張空的圖。

支援：WebVTT、SRT、JSON（語音服務常見的段落陣列）、純文字、Word / ODF。

**為什麼要自己合併相鄰的字幕**：VTT / SRT 的每一句常常只有一兩秒、半句話，
一場會議會切出好幾千段。那樣不是跑不動（視窗是按字數切的），而是
**引用會指到半句話** —— 而「每一條都要指得回逐字稿」正是這個功能的賣點。
所以同一個發言者連續的段落會合併到一個上限，時間取「第一段的開始、
最後一段的結束」。
"""
from __future__ import annotations

import io
import json
import re
import zipfile
from collections import Counter
from typing import Optional

#: 合併相鄰字幕時，一段最多這麼長。**不是效能上限，是可讀性上限** ——
#: 引用要能放進畫面上一張卡片裡。
MERGE_CHARS = 400

#: 相鄰字幕之間隔超過這麼久就不合併（換話題了）。
MERGE_GAP_MS = 3000

#: 發言者名稱的長度上限，**分中日韓與拉丁兩套** ——
#: 中文名字是 2~4 個字（加頭銜像「王經理」「主席」也在 6 個字以內），
#: 而拉丁名字光是 `Dr. Jennifer Rodriguez` 就 22 個字元。
#: 用同一個數字的話，不是放掉「這是一段非常長的開場白」那種句子，
#: 就是把正常的英文名字判掉。
MAX_SPEAKER_CJK = 8
MAX_SPEAKER_LATIN = 24


class TranscriptError(ValueError):
    """檔案讀不出可用的逐字稿。

    **訊息要說得出「你可以怎麼辦」** —— 使用者看到「解析失敗」只會再試一次
    同一個檔案。
    """


# ------------------------------------------------------------------ 時間

_TS = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")
_TS_SHORT = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})$")


def _ms(text: str) -> Optional[int]:
    """`01:02:03.456` / `02:03,456` / `02:03` → 毫秒。"""
    m = _TS.match(text.strip())
    if m:
        h, mi, s, frac = m.groups()
        ms = int(frac.ljust(3, "0"))
        return ((int(h or 0) * 60 + int(mi)) * 60 + int(s)) * 1000 + ms
    m = _TS_SHORT.match(text.strip())
    if m:
        h, mi, s = m.groups()
        return ((int(h or 0) * 60 + int(mi)) * 60 + int(s)) * 1000
    return None


#: WebVTT / SRT 的時間軸箭頭（`00:00:01.000 --> 00:00:06.000`）。
#: **用字串切，不用正規式** —— 寫成正規式的話 CodeQL 會把它當成「過濾 HTML 註解」
#: 的寫法報成 High（`-->` 剛好也是 HTML 註解的結尾）。這支剖析器一個 HTML 都不碰。
_ARROW = "-->"


def _cue_times(line: str) -> Optional[tuple[int, int]]:
    head, arrow, tail = line.strip().partition(_ARROW)
    if not arrow:
        return None
    parts = [head.strip(), tail.strip()]
    a = _ms(parts[0])
    # 結束時間後面可能跟著 VTT 的排版設定（`align:start position:10%`）
    b = _ms(parts[1].split()[0]) if parts[1].split() else None
    if a is None or b is None:
        return None
    return a, b


# ------------------------------------------------------------------ 發言者

_VOICE = re.compile(r"^<v\s+([^>]{1,40})>(.*?)(?:</v>)?$", re.S)
#: `王小明：` / `Alice:` / `[王小明]`
_PREFIX = re.compile(r"^\s*(?:\[([^\]]{1,24})\]|([^：:\[\]]{1,24})\s*[：:])\s*(.*)$", re.S)
#: 名字裡不會出現的東西 —— 有這些就代表那個冒號是句子的一部分
_NOT_A_NAME = re.compile(r"[。，！？；、,.!?;…「」『』（）()\n]")
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]")
#: 名字裡的點號**只允許縮寫**（`Dr.` `Ms.` `J.R.R.`）——
#: 句號前面接的是一個完整的字（`…a sentence.`），縮寫前面只有一兩個字母。
#: 一律禁掉點號的話，`Dr. Jennifer Rodriguez：` 這種正式記錄一個發言者都認不出來；
#: 一律放行的話，半句話會被當成名字。
_ABBREV_DOT = re.compile(r"(?<![A-Za-z])[A-Za-z]{1,4}\.")


def _dots_are_all_abbreviations(name: str) -> bool:
    return len(_ABBREV_DOT.findall(name)) == name.count(".")


def _name_too_long(name: str) -> bool:
    cap = MAX_SPEAKER_CJK if _CJK.search(name) else MAX_SPEAKER_LATIN
    return len(name) > cap


def _strip_voice(text: str) -> tuple[Optional[str], str]:
    m = _VOICE.match(text.strip())
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return None, text


def _split_prefix(line: str) -> tuple[Optional[str], str]:
    """把 `王小明：內容` 拆開。拆不出來就回 `(None, 整行)`。"""
    m = _PREFIX.match(line)
    if not m:
        return None, line
    name = (m.group(1) or m.group(2) or "").strip()
    rest = m.group(3).strip()
    if not name or not rest or _name_too_long(name):
        return None, line
    bad = _NOT_A_NAME.search(name)
    if bad and not (bad.group() == "." and _dots_are_all_abbreviations(name)):
        return None, line
    return name, rest


def _confirm_speakers(rows: list[tuple[Optional[str], str]]
                      ) -> list[tuple[Optional[str], str]]:
    """哪些前綴真的是發言者。

    `我們下週要做三件事：A、B、C` 的前半段也符合「冒號前面是一小段字」，
    **單看一行分不出來**。兩層判準：

    1. **重複出現的名字一定是發言者** —— 真正的發言者在一場會議裡不會只出現一次，
       而且句子裡的冒號不會用同一段字重複。
    2. **只出現一次的，拿「已確認的名字有多長」當尺** —— 這場逐字稿自己
       用的名字是 3 個字，那 4 個字的「注意事項」就不是名字。
       **這把尺是從文件自己量出來的，不是寫死的常數**（寫死的話，
       名字比較長的那些逐字稿會整批被判掉）。

    完全沒有名字重複時**一個都不採信** —— 那代表這份檔案根本沒有在用
    「發言者：內容」的格式，那些冒號都是句子的一部分。

    判不是發言者的**要把那段字還原回內文**，不可以吃掉。
    """
    seen: dict[str, int] = {}
    for name, _ in rows:
        if name:
            seen[name] = seen.get(name, 0) + 1
    confirmed = {n for n, c in seen.items() if c >= 2}
    # 已確認的名字有多長 —— 只出現一次的名字要在這個範圍內才採信
    cap = max((len(n) for n in confirmed), default=0)
    out = []
    for name, text in rows:
        ok = bool(name) and (name in confirmed or (cap and len(name) <= cap))
        if ok:
            out.append((name, text))
        elif name:
            out.append((None, f"{name}：{text}"))
        else:
            out.append((None, text))
    return out


# ------------------------------------------------------------------ 合併

#: 一段最長多少字。**超過就切開。**
#:
#: 引用是「第 N 段」—— 如果那一段有兩萬字，「指得回逐字稿」就是一句空話：
#: 使用者點過去看到的是一大片文字，找不到那句話在哪裡。而引用驗證也會失準
#: （拿一句話去比對兩萬字，幾乎什麼都比得上）。
#:
#: 2026-09-18 實測：一份 26,907 字的公聽會紀錄只切出 **4 段**，
#: 每一條引用都指向「第 1 段」，而且 30 條因為比對不上被丟掉。
#:
#: 跟 `MERGE_CHARS` 一樣是 400 —— 合併的上限與切分的上限一致，
#: 不然切完又被合併回去。
MAX_SEG_CHARS = 400

#: 切超長段落時優先在這些地方斷（句子邊界），斷不開才硬切。
_SPLIT_AT = "。！？!?；;"


def _split_long(segs: list[dict]) -> list[dict]:
    """把超長的段落切成讀得完的幾段。**在解析時就切**，不是等到要送模型才切
    —— 引用指的是這裡的段號，晚切的話引用仍然指向那一大塊。"""
    out: list[dict] = []
    for seg in segs:
        text = str(seg.get("text") or "")
        if len(text) <= MAX_SEG_CHARS:
            out.append(seg)
            continue
        # 時間平均分配到切出來的幾段上（沒有時間就不給）
        a, b = seg.get("start_ms"), seg.get("end_ms")
        pieces: list[str] = []
        i = 0
        while i < len(text):
            j = min(i + MAX_SEG_CHARS, len(text))
            if j < len(text):
                cut = max((text.rfind(ch, i + MAX_SEG_CHARS // 2, j)
                           for ch in _SPLIT_AT), default=-1)
                if cut > i:
                    j = cut + 1
            pieces.append(text[i:j])
            i = j
        for k, piece in enumerate(pieces):
            one = {kk: vv for kk, vv in seg.items() if kk not in ("start_ms", "end_ms")}
            one["text"] = piece
            if a is not None and b is not None and b > a:
                step = (b - a) / len(pieces)
                one["start_ms"] = int(a + step * k)
                one["end_ms"] = int(a + step * (k + 1))
            elif a is not None:
                one["start_ms"] = a
            out.append(one)
    return out


def _merge(segs: list[dict]) -> list[dict]:
    """同一個發言者連續的段落合併起來，重新編號。"""
    out: list[dict] = []
    for s in segs:
        text = (s.get("text") or "").strip()
        if not text:
            continue
        prev = out[-1] if out else None
        joinable = (
            prev is not None
            and prev.get("speaker") == s.get("speaker")
            and len(prev["text"]) + len(text) + 1 <= MERGE_CHARS
        )
        if joinable and prev.get("end_ms") is not None and s.get("start_ms") is not None:
            joinable = (s["start_ms"] - prev["end_ms"]) <= MERGE_GAP_MS
        if joinable:
            sep = "" if prev["text"][-1] in "，。！？、；,.!?;" else " "
            prev["text"] = f"{prev['text']}{sep}{text}".strip()
            if s.get("end_ms") is not None:
                prev["end_ms"] = s["end_ms"]
        else:
            out.append({k: v for k, v in s.items() if v is not None} | {"text": text})
    # **切在編號之前** —— 編號之後再切的話會有兩段共用同一個段號，
    # 而引用是靠段號指回去的。
    out = _split_long(out)
    for i, s in enumerate(out, 1):
        s["seq"] = i
    return out


# ------------------------------------------------------------------ 各格式

def parse_cues(text: str) -> list[dict]:
    """WebVTT / SRT —— 兩種的差別只有小數點與那個序號行，一起處理。"""
    segs: list[dict] = []
    cur: Optional[dict] = None
    for raw in text.splitlines():
        line = raw.rstrip()
        times = _cue_times(line)
        if times:
            cur = {"start_ms": times[0], "end_ms": times[1], "text": ""}
            segs.append(cur)
            continue
        if cur is None or not line.strip():
            continue
        if line.strip().upper().startswith("WEBVTT") or line.strip().isdigit():
            continue
        speaker, body = _strip_voice(line)
        if speaker:
            cur["speaker"] = speaker
        cur["text"] = (cur["text"] + " " + body).strip()
    # VTT 沒有 `<v>` 時，發言者常寫成 `王小明：內容`
    rows = _confirm_speakers([(None, s["text"]) if s.get("speaker")
                              else _split_prefix(s["text"]) for s in segs])
    for s, (name, body) in zip(segs, rows):
        if name:
            s["speaker"] = name
            s["text"] = body
        elif not s.get("speaker"):
            s["text"] = body
    return _merge(segs)


def parse_json(data: bytes) -> list[dict]:
    """語音轉寫服務常見的段落陣列 `[{text, speaker, start_ms, end_ms}, …]`。"""
    try:
        obj = json.loads(data.decode("utf-8", "replace"))
    except ValueError as e:
        raise TranscriptError(f"JSON 讀不進來：{e}") from e
    # **發言者改名要跟著走**（v1.16.10，使用者回報）：「會議錄音轉逐字稿」存的 JSON 帶著
    # `speaker_names`（S1 → 陳協理，整位改）與 `speaker_overrides`（第 N 段 → 名字，
    # 只改那一段）。原本這裡只讀每一段的 `speaker`，於是在轉逐字稿改好的名字，
    # 轉送到會議摘要之後全部變回 S1、S2 —— 而轉送那側的說明還寫著「走的是同一份資料」。
    names: dict = {}
    overrides: dict = {}
    if isinstance(obj, dict):
        if isinstance(obj.get("speaker_names"), dict):
            names = {str(k): str(v).strip() for k, v in obj["speaker_names"].items()
                     if str(v or "").strip()}
        if isinstance(obj.get("speaker_overrides"), dict):
            overrides = {str(k): str(v).strip() for k, v in obj["speaker_overrides"].items()
                         if str(v or "").strip()}
    if isinstance(obj, dict):
        for key in ("segments", "final_segments", "raw_segments", "data"):
            if isinstance(obj.get(key), list):
                obj = obj[key]
                break
        else:
            raise TranscriptError(
                "這份 JSON 裡找不到段落清單（要有 segments 或直接是一個陣列）。")
    if not isinstance(obj, list):
        raise TranscriptError("這份 JSON 不是段落清單。")
    segs = []
    for row in obj:
        if not isinstance(row, dict):
            continue
        text = str(row.get("text") or row.get("content") or "").strip()
        if not text:
            continue
        seg: dict = {"text": text}
        sp = row.get("speaker") or row.get("speaker_id")
        # 單段的改名優先，其次整位的改名（跟轉逐字稿畫面上的 `speakerName()` 同一個順序）
        named = overrides.get(str(row.get("seq"))) if row.get("seq") is not None else None
        if not named and sp:
            named = names.get(str(sp))
        if named or sp:
            seg["speaker"] = str(named or sp)
        for src, dst in (("start_ms", "start_ms"), ("end_ms", "end_ms"),
                         ("start", "start_ms"), ("end", "end_ms")):
            if dst in seg or row.get(src) is None:
                continue
            try:
                seg[dst] = int(row[src])
            except (TypeError, ValueError):
                pass
        segs.append(seg)
    return _merge(segs)


_LEADING_TIME = re.compile(r"^\s*[\[(]?((?:\d+:)?\d{1,2}:\d{2}(?:[.,]\d{1,3})?)[\])]?\s*[-–]?\s*")


# ------------------------------------------------------------------ 純文字

#: 行首（或句首）的時間標記：`[00:12]`、`(1:02:03)`、`00:12 -`、`【00:12】`
_TIME_AT_START = re.compile(
    r"^[\s\[\(【]*((?:\d+:)?\d{1,2}:\d{2}(?:[.,]\d{1,3})?)[\]\)】]?\s*[-–—]?\s*")

#: 發言者標頭**單獨一行**（冒號後面什麼都沒有）：
#:
#:     [00:00] PM - 雅婷：
#:     好，那我們時間到了就直接開始。
#:
#: **這種行沒有歧義** —— 句子裡的冒號後面一定有字，冒號後面空著的只可能是標頭。
#: 所以這裡的名字可以放寬到 40 個字（`Backend Lead - 凱文` 有 17 個），
#: 不必擔心把半句話當成發言者。
_HEADER_ONLY = re.compile(r"^([^：:\n]{1,40}?)\s*[：:]\s*$")

#: 發言者寫在括號裡的時間前面：`雅婷 (00:12)：內容`
_NAME_THEN_TIME = re.compile(
    r"^([^：:\n\(\[]{1,24}?)\s*[\(\[]((?:\d+:)?\d{1,2}:\d{2})[\)\]]\s*[：:]\s*(.*)$", re.S)

#: 清單符號開頭（`- 雅婷：…`、`* 雅婷：…`、`1. 雅婷：…`）
_BULLET = re.compile(r"^\s*(?:[-*•>]|\d{1,3}[.)])\s+")

#: Markdown 標題／分隔線／清單那幾行不是發言 —— 逐字稿檔案前面常有一段
#: 會議資訊（主題、時間、與會人員）。把它們當成發言的話，`## 會議主題`
#: 會變成一位「發言者」（2026-09-18 使用者實際遇到）。
_NOT_AN_UTTERANCE = re.compile(r"^\s*([-=_*]{3,}\s*$|\|)")
#: Markdown 的井字號標題：**內容要留著**（會議主題是有用的脈絡），
#: 只是它不是「發言」，所以把井字號去掉、當成前言的一部分。
_HEADING = re.compile(r"^\s*#{1,6}\s*")


def _strip_time(line: str) -> tuple[Optional[int], str]:
    m = _TIME_AT_START.match(line)
    if not m:
        return None, line
    return _ms(m.group(1)), line[m.end():]


def _plain_rows(text: str, shape: str = "auto"
                ) -> tuple[list[tuple[Optional[int], Optional[str], str, bool]], str]:
    """把純文字拆成 `(時間, 發言者, 內容)`。**認得出好幾種常見的排法。**

    使用者的逐字稿來源很多（會議軟體匯出、語音服務、人工打字），
    格式各家不同 —— 少認一種，那一份的發言者就整個不見，
    而畫面上只會顯示「沒有認出任何發言者」，看不出是格式沒支援。
    """
    raw = [ln.rstrip() for ln in text.splitlines()]
    # 第四個欄位 `certain`：這個發言者是不是**來自沒有歧義的形狀**（獨立的標頭行）。
    # 那種不必再過 `_confirm_speakers` 的「名字長度尺」—— 過了反而會把
    # `UI/UX Designer - 萱萱` 這種只出現一次的長名字判掉。
    rows: list[tuple[Optional[int], Optional[str], str, bool]] = []

    # ── 形狀一：標頭自己一行，內容在後面幾行 ──
    heads = []
    for i, ln in enumerate(raw):
        if _NOT_AN_UTTERANCE.match(ln) or _HEADING.match(ln):
            continue
        body = _BULLET.sub("", ln)
        ts, rest = _strip_time(body)
        m = _HEADER_ONLY.match(rest)
        if m and not _NOT_A_NAME.search(m.group(1)):
            heads.append((i, ts, m.group(1).strip()))
    # **哪些標頭是真的發言者？用位置判，不要用出現次數。**
    #
    # 逐字稿前面那段會議資訊長得跟標頭一樣（`* 與會人員：`、`## 會議主題：`），
    # 不濾掉的話它會變成一位「發言者」，而後面那一大串參加者名單會變成它的
    # 「發言」（2026-09-18 使用者實際遇到）。
    #
    # 但**不能用「出現次數」濾** —— 只講過一次話的人也是發言者
    # （`李美華：` 只出現一次就被判掉的話，她的話會併到上一個人身上）。
    #
    # 判準：**只要有任何一個標頭帶時間戳記，那第一個帶時間的就是逐字稿的開始**，
    # 它之前的都是會議資訊。完全沒有時間戳記時就全部保留
    # （那是「標頭自己一行、沒有時間」的排法，沒有別的訊號可用）。
    if heads:
        first_timed = next((k for k, h in enumerate(heads) if h[1] is not None), None)
        if first_timed is not None:
            heads = heads[first_timed:]
        # 清單符號開頭的**獨立標頭**幾乎一定是會議資訊 —— 逐字稿不會把
        # 每一句發言都寫成清單項目（`- 王小明：各位早` 那種是**同一行有內容**
        # 的形狀，不會走到這裡）。
        heads = [h for h in heads if not _BULLET.match(raw[h[0]])]
    if shape in ("auto", "header") and len(heads) >= 2:
        idx = {i for i, _, _ in heads}
        for k, (i, ts, name) in enumerate(heads):
            stop = heads[k + 1][0] if k + 1 < len(heads) else len(raw)
            body = " ".join(ln.strip() for ln in raw[i + 1:stop] if ln.strip())
            if body:
                rows.append((ts, name or None, body, True))
        # **標頭之前的前言不是發言** —— 逐字稿檔案前面常有一段會議資訊
        # （主題、時間、與會人員）。把它收成**一段沒有發言者的文字**，
        # 這樣模型看得到（誰參加、談什麼），但不會變成一位「發言者」，
        # 也不會污染發言者佔比（2026-09-18 使用者實際遇到：
        # `## 會議主題`、`* 會議時間` 都被當成發言者）。
        pre = [_HEADING.sub("", _BULLET.sub("", ln)).strip()
               for ln in raw[:heads[0][0]]
               if ln.strip() and not _NOT_AN_UTTERANCE.match(ln)]
        if pre:
            rows.insert(0, (None, None, " ".join(pre), True))
        return rows, "header"

    # ── 形狀二：一行一句（可能帶時間、可能帶發言者） ──
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) <= 1:
        blocks = [ln.strip() for ln in raw if ln.strip()]
    for b in blocks:
        if _NOT_AN_UTTERANCE.match(b):
            continue
        b = _HEADING.sub("", _BULLET.sub("", b))
        m = _NAME_THEN_TIME.match(b)          # `雅婷 (00:12)：內容`
        if m and not _NOT_A_NAME.search(m.group(1)) and m.group(3).strip():
            rows.append((_ms(m.group(2)), m.group(1).strip(), m.group(3).strip(), False))
            continue
        ts, rest = _strip_time(b)
        name, body = _split_prefix(rest)
        if body.strip():
            # `plain` 是使用者明講「這份沒有發言者」—— 不要去猜
            rows.append((ts, None if shape == "plain" else name, body.strip(), False))
    return rows, ("plain" if shape == "plain" else "inline")


def parse_plain(text: str, shape: str = "auto") -> tuple[list[dict], str]:
    """純文字逐字稿 → `(段落, 用了哪一種排法)`。**支援好幾種排法**，見 `_plain_rows`。"""
    rows, used = _plain_rows(text, shape)
    # **只有「有歧義」的那些才要過名字長度尺**（`_confirm_speakers`）——
    # 來自獨立標頭行的名字是確定的（句子裡的冒號後面一定有字），
    # 再質疑一次會把只出現一次的長名字（`UI/UX Designer - 萱萱`）判掉。
    judged = _confirm_speakers([(None if certain else name, body)
                                for _, name, body, certain in rows])
    segs: list[dict] = []
    for (ts, name, body, certain), (judged_name, judged_body) in zip(rows, judged):
        if not certain:
            name, body = judged_name, judged_body
        if not body:
            continue
        seg: dict = {"text": body}
        if name:
            seg["speaker"] = name
        if ts is not None:
            seg["start_ms"] = ts
        segs.append(seg)
    # 行首時間只給得出「開始」—— 下一段的開始就是這一段的結束
    for a, b in zip(segs, segs[1:]):
        if a.get("start_ms") is not None and b.get("start_ms") is not None:
            a["end_ms"] = b["start_ms"]
    return _merge(segs), used


def _office_text(data: bytes, ext: str) -> str:
    """從 .docx / .odt 抽段落。**只抽文字，段落之間留空行**（純文字那條路
    才分得出段落）。"""
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise TranscriptError("這個檔案讀不開 —— 可能不是 Word / ODF 檔，或已經毀損。") from e
    # **zip 炸彈**：這是使用者上傳的檔案，`.docx` / `.odt` 就是 zip。
    # 判斷集中在 `zip_guard`，全站一份。
    from .zip_guard import ZipBombError, check as _zip_check
    try:
        _zip_check(zf)
    except ZipBombError as e:
        raise TranscriptError(f"這個檔案被拒絕：{e}") from e
    inner = "word/document.xml" if ext == ".docx" else "content.xml"
    names = zf.namelist()
    if inner not in names:
        # Word 有時候寫成 document2.xml
        cand = [n for n in names if n.startswith("word/document") and n.endswith(".xml")]
        if not cand:
            raise TranscriptError("這個檔案裡找不到文件內容。")
        inner = cand[0]
    from defusedxml import ElementTree as DET
    root = DET.fromstring(zf.read(inner).decode("utf-8", "replace"))
    para_tags = {
        "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p",
        "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}p",
        "{urn:oasis:names:tc:opendocument:xmlns:text:1.0}h",
    }
    out = []
    for node in root.iter():
        if node.tag in para_tags:
            txt = "".join(node.itertext()).strip()
            if txt:
                out.append(txt)
    return "\n\n".join(out)


#: 純文字逐字稿的排法。**預設自動判斷，但要讓使用者可以指定** ——
#: 自動判斷一定有猜錯的時候，而猜錯的症狀是「發言者全不見」或「多出奇怪的發言者」，
#: 使用者看得出來不對卻沒有任何辦法（2026-09-18 使用者要求）。
SHAPES: dict[str, str] = {
    "auto":   "自動判斷",
    "header": "發言者標頭自己一行（下一行才是內容）",
    "inline": "一行一句（發言者：內容）",
    "plain":  "純文字，沒有發言者",
}


#: 收得進來的副檔名。**明確列出來**，讓畫面上的 `accept` 與這裡是同一份。
SUPPORTED = (".vtt", ".srt", ".json", ".txt", ".md", ".docx", ".odt")


def parse(data: bytes, filename: str, shape: str = "auto"
          ) -> tuple[list[dict], str]:
    """依副檔名挑解析方式，回 `(段落, 用了哪一種排法)`。

    `shape` 只對純文字那條路有意義（字幕檔與 JSON 的結構是確定的）。
    **預設 `auto`**；自動判斷錯的時候使用者可以指定（見 `SHAPES`）。

    讀不出東西一律丟 `TranscriptError` —— **不可以回空清單**，
    那會讓後面的分析「成功」產出一份空摘要，而使用者看到的是「這個工具沒作用」。
    """
    if shape not in SHAPES:
        shape = "auto"
    ext = ("." + filename.rsplit(".", 1)[-1].lower()) if "." in filename else ""
    if ext not in SUPPORTED:
        raise TranscriptError(
            f"不支援的檔案類型（{ext or '沒有副檔名'}）。"
            f"可以用：{'、'.join(SUPPORTED)}")
    if ext == ".json":
        segs, used = parse_json(data), "json"
    elif ext in (".docx", ".odt"):
        segs, used = parse_plain(_office_text(data, ext), shape)
    else:
        text = data.decode("utf-8-sig", "replace")
        if "-->" in text:
            segs, used = parse_cues(text), "cues"
        else:
            segs, used = parse_plain(text, shape)
    if not segs:
        raise TranscriptError("這個檔案裡沒有讀到任何文字。")
    return segs, used
