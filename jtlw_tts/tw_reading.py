"""台灣念法與切句（Mac 本機合成用；GPU 伺服器那份在 remote_whisper_server.py 的 worker 區）

**與 remote_whisper_server.py 的同名函式逐字相同**：伺服器自動更新只推單一檔案，不能共用模組。
tools/test_tts_tw_reading.py 比對兩邊；改一邊一定要改另一邊。規則見 specs/2026-10-08_TTS開發規格_v2.md 第七節：
自訂（和 → ㄏㄢˋ）＞教育部《重編國語辭典修訂本》的詞（多讀音依 g2pW 挑）＞g2pW；與模型預設（pypinyin）不同才換成 {拼音}。
"""
import json
import os
import re

_TTS_HAN = re.compile(r"[㐀-鿿]+")
_TTS_TONE = {"ˊ": "2", "ˇ": "3", "ˋ": "4"}
_TTS_NO_HINT = frozenset("一不")
# **念法以台灣日常說法為準**（2026-10-09 使用者：「萌典不要以他為準，請以台灣日常為準」）：教育部辭典只是基礎，
# 跟台灣日常說法不同的字，不管在哪個詞裡都改用日常說法（自訂發音照樣優先）。值：(要換掉的念法，None＝一律換, 換成)。
# 液、亞、俄：教育部 ㄧㄝˋ／ㄧㄚˋ／ㄜˊ 跟大陸相同，台灣多念 ㄧˋ／ㄧㄚˇ／ㄜˋ；黑：ㄏㄜˋ 是讀音，辭典 265 個含黑的詞只有 5 個用它；
# 熟（成熟、熟悉）ㄕㄨˊ→ㄕㄡˊ、癌 ㄧㄢˊ→ㄞˊ、它（它們）ㄊㄨㄛ→ㄊㄚ、洽（接洽）ㄒㄧㄚˊ→ㄑㄧㄚˋ、燥（肉燥）ㄙㄠˋ→ㄗㄠˋ、
# 括（包括）ㄎㄨㄛˋ→ㄍㄨㄚ、魄（落魄）ㄊㄨㄛˋ→ㄆㄛˋ；
# 語料裡「因辭典而指定、跟模型預設不同」的 220 種逐一檢查、使用者試聽後再加（2026-10-09）：場（市場、現場、一場）ㄔㄤˊ→ㄔㄤˇ、
# 妨（無妨）ㄈㄤ→ㄈㄤˊ、縱（縱貫、縱谷）ㄗㄨㄥ→ㄗㄨㄥˋ、多（多麼）ㄉㄨㄛˊ→ㄉㄨㄛ、擷（擷取）ㄐㄧㄝˊ→ㄒㄧㄝˊ、
# 伐（步伐）ㄈㄚ→ㄈㄚˊ、玩（把玩）ㄨㄢˋ→ㄨㄢˊ、署（簽署、部署）ㄕㄨˋ→ㄕㄨˇ。
# 使用者試聽決定照教育部的（不要改）：蝸牛 ㄍㄨㄚ、優酪乳 ㄌㄨㄛˋ、從容 ㄘㄨㄥ、剝皮 ㄅㄛ、曝光 ㄆㄨˋ、說服 ㄕㄨㄟˋ、寂寞 ㄐㄧˊ、艘 ㄙㄠ、
# 盡快／盡量 ㄐㄧㄣˋ、言行 ㄒㄧㄥˋ
_TTS_TW_COMMON = {"液": (None, "ㄧ4"), "亞": (None, "ㄧㄚ3"), "俄": (None, "ㄜ4"), "黑": ("ㄏㄜ4", "ㄏㄟ1"),
                  "熟": ("ㄕㄨ2", "ㄕㄡ2"), "癌": ("ㄧㄢ2", "ㄞ2"), "它": ("ㄊㄨㄛ1", "ㄊㄚ1"), "洽": ("ㄒㄧㄚ2", "ㄑㄧㄚ4"),
                  "燥": ("ㄙㄠ4", "ㄗㄠ4"), "括": ("ㄎㄨㄛ4", "ㄍㄨㄚ1"), "魄": ("ㄊㄨㄛ4", "ㄆㄛ4"),
                  "場": (None, "ㄔㄤ3"), "妨": (None, "ㄈㄤ2"), "縱": (None, "ㄗㄨㄥ4"), "多": ("ㄉㄨㄛ2", "ㄉㄨㄛ1"),
                  "擷": (None, "ㄒㄧㄝ2"), "伐": ("ㄈㄚ1", "ㄈㄚ2"), "玩": ("ㄨㄢ4", "ㄨㄢ2"), "署": (None, "ㄕㄨ3")}
# 台灣日常念法（詞）：比辭典優先、自訂發音照樣更優先。角色（教育部主音 ㄐㄩㄝˊ）、暖暖（基隆的暖暖區；教育部 ㄒㄩㄢ）、
# 著急（教育部 ㄓㄠ）、裝載（教育部 ㄗㄞˋ，使用者：要念三聲）、兒子（教育部 ㄗˇ，日常輕聲）、
# 強制（教育部 ㄑㄧㄤˇ）、牛仔（教育部 ㄗˇ）、折返（教育部 ㄓㄜ）、胜肽（教育部 ㄒㄧㄥ）、
# 參與（教育部 ㄩˋ）、罪行（教育部 ㄒㄧㄥˋ）、記載（教育部 ㄗㄞˋ）：使用者試聽決定（言行照教育部 ㄒㄧㄥˋ）、
# 丁丁（教育部是伐木聲 ㄓㄥ）、家樂福、麥當當、亂數；挑戰、慎重：辭典由左往右會切出「大挑」「重考」
# （「強行」不加：會把「加強行員」切成強行）
_TTS_TW_WORDS = {"角色": ["ㄐㄧㄠ3", "ㄙㄜ4"], "主角": ["ㄓㄨ3", "ㄐㄧㄠ3"], "配角": ["ㄆㄟ4", "ㄐㄧㄠ3"],
                 "暖暖": ["ㄋㄨㄢ3", "ㄋㄨㄢ3"], "著急": ["ㄓㄠ2", "ㄐㄧ2"], "裝載": ["ㄓㄨㄤ1", "ㄗㄞ3"], "兒子": ["ㄦ2", "ㄗ5"],
                 "目的事業": ["ㄇㄨ4", "ㄉㄧ4", "ㄕ4", "ㄧㄝ4"],
                 "強制": ["ㄑㄧㄤ2", "ㄓ4"], "牛仔": ["ㄋㄧㄡ2", "ㄗㄞ3"], "折返": ["ㄓㄜ2", "ㄈㄢ3"],
                 "參與": ["ㄘㄢ1", "ㄩ3"], "罪行": ["ㄗㄨㄟ4", "ㄒㄧㄥ2"], "記載": ["ㄐㄧ4", "ㄗㄞ3"],
                 "胜肽": ["ㄕㄥ4", "ㄊㄞ4"], "丁丁": ["ㄉㄧㄥ1", "ㄉㄧㄥ1"], "家樂福": ["ㄐㄧㄚ1", "ㄌㄜ4", "ㄈㄨ2"],
                 "麥當當": ["ㄇㄞ4", "ㄉㄤ1", "ㄉㄤ1"], "亂數": ["ㄌㄨㄢ4", "ㄕㄨ4"], "挑戰": ["ㄊㄧㄠ3", "ㄓㄢ4"],
                 "慎重": ["ㄕㄣ4", "ㄓㄨㄥ4"]}
# 不在辭典詞裡的字（念法是 g2pW 猜的）改用這個念法，值：(要換掉的念法，None＝一律換, 換成)（2026-10-09 自動偵測）。
# 蘋：蘋概股、蘋粉的蘋都是蘋果的蘋（g2pW 猜 ㄆㄧㄣˊ；辭典裡念 ㄆㄧㄣˊ 的白蘋、蘋婆照辭典）。
# 差：g2pW 把很差、太差、變差都判成 ㄔㄚ；教育部「不好、欠缺」念 ㄔㄚˋ，ㄔㄚˋ 也是 ㄔㄚ 的語音（差別、差距、誤差是辭典詞，照辭典）
# 兒：g2pW 把兒化（那兒、鳥兒、好玩兒）標成 ㄦ 一聲，教育部是輕聲 ˙ㄦ（輕聲不加提示，模型自己念兒化）
_TTS_TW_SINGLE = {"蘋": (None, "ㄆㄧㄥ2"), "差": ("ㄔㄚ1", "ㄔㄚ4"), "兒": ("ㄦ1", "ㄦ5")}
# 異體字：辭典查不到時換成辭典用的字再查（沈積→沉積 ㄔㄣˊ，g2pW 判成姓氏的 ㄕㄣˇ；什麽→什麼）。
# 姓氏的沈（沈約、沈括）辭典本來就查得到，不換
_TTS_VARIANT = str.maketrans("沈麽", "沉麼")
# 台灣念法跟模型預設一樣、模型卻還是會念錯的字：一律加念法提示（2026-10-09 試聽：命脈的脈、協會的協念錯；
# 自動偵測：阿嬤的嬤念成ㄇㄛˊ；一曝十寒的曝念成大陸「曝光」的ㄅㄠˋ，教育部只有ㄆㄨˋ）
_TTS_ALWAYS_HINT = frozenset("脈協嬤曝")
# 繁體一個字、簡體依念法分成兩個字，OpenCC 分不出來的：念法不是第一個就寫成第二個字再送進模型
# （扮演著：簡體的「著」只念ㄓㄨˋ，模型照著念；助詞與著急、著陸在簡體寫「着」，2026-10-09 自動偵測）
_TTS_SIMP_BY_READING = {"著": ("ㄓㄨ4", "着")}
# 辭典由左往右找最長的詞會切錯：「扮演著重要」切出「著重」（ㄓㄨㄛˊ）、「組中的字」切出「中的」（射中靶心 ㄓㄨㄥˋ ㄉㄧˋ）、
# 「他的是不是」切出文言「的是」（ㄉㄧˊ）、「都會忘記」切出「都會」（都市）、「環境和文化」切出「和文」、「生存沒有」切出「存沒」。
# 這些常用字在辭典詞裡的念法跟 g2pW（看上下文）不同時，那個詞多半是切錯的 → 不採用，改試短一點的詞或照 g2pW
# （2026-10-09 用 Common Voice 4,636 句比對：改到 111 處，只有公文的「目的事業」改壞，另列在 _TTS_TW_WORDS）。
# 再加種分間得要當重（一種生物≠種生、多分布≠多分、之間有≠間有、找得到≠得到、要不要≠不要、當晚餐≠當晚、很多重要≠多重）：
# 改到 34 處、改壞 5 處（慎重另列在 _TTS_TW_WORDS；鹽分、當名嘴、當日、才會得是 g2pW 判錯）
_TTS_G2P_FIRST = frozenset("的了著都和沒給從參覺會種分間得要當重")
# 送進模型前把沒加提示的字轉成簡體（見 _tts_spoken）
_TTS_SIMPLIFIED = True
_TTS_MAX_WORD = 8
_TTS_SENT_END = "。！？!?；;\n"


_TTS_EN_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                "sixteen seventeen eighteen nineteen").split()
_TTS_EN_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_TTS_EN_MONEY = re.compile(r"(?<![A-Za-z])(NT|US)?\$\s?(\d[\d,]*(?:\.\d+)?)")
_TTS_EN_DOTTED = re.compile(r"(?<![\w.])([vV]?)(\d+(?:\.\d+){2,})(?![\w]|\.\d)")
_TTS_EN_COMMA = re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+)(?![\d,]|\.\d)")


def _tts_en_int(n):
    """英文的整數念法（0～999,999,999,999）"""
    if n < 20:
        return _TTS_EN_ONES[n]
    if n < 100:
        return _TTS_EN_TENS[n // 10] + ("-" + _TTS_EN_ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return _TTS_EN_ONES[n // 100] + " hundred" + (" " + _tts_en_int(n % 100) if n % 100 else "")
    for div, name in ((10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")):
        if n >= div:
            return _tts_en_int(n // div) + " " + name + (" " + _tts_en_int(n % div) if n % div else "")


def _tts_en_text(text):
    """英文句子送進模型前（v2.28.0 雙向口譯）：不套台灣念法、不把數字換成中文（_tts_numbers 會把 1,250,000 換成一百二十五萬）。
    2026-10-10 GPU 實測三種聲音各 4 句：1,250,000 被念成 150,000、IP 與版本號偶爾念錯 →
    千分位的數字寫成英文、金額改成「數字＋幣別」、IP／版本號一段一段用 dot 連起來；其他照原文（模型念得對）"""
    t = " ".join(str(text).split())
    t = _TTS_EN_MONEY.sub(lambda m: f"{m.group(2)} " + {"NT": "NT dollars", "US": "US dollars"}.get(m.group(1) or "", "dollars"), t)
    t = _TTS_EN_DOTTED.sub(lambda m: ("version " if m.group(1) else "") + " dot ".join(m.group(2).split(".")), t)

    def comma(m):
        n = int(m.group(1).replace(",", ""))
        return _tts_en_int(n) if n < 10 ** 12 else m.group(1)
    return _TTS_EN_COMMA.sub(comma, t)


def _tts_syl(b):
    """教育部注音（ㄌㄜˋ、˙ㄇㄣ、ㄒㄧ）→ 注音＋聲調數字（ㄌㄜ4、ㄇㄣ5、ㄒㄧ1），與 g2pW 的格式相同"""
    if b.startswith("˙"):
        return b[1:] + "5"
    if b and b[-1] in _TTS_TONE:
        return b[:-1] + _TTS_TONE[b[-1]]
    return b + "1"


def _tts_moe_load(lines):
    words = {}
    for ln in lines:
        t, _, rs = ln.rstrip("\n").partition("\t")
        if t and rs:
            words[t] = [r.split() for r in rs.split("|")]
    return words


def _tts_custom(entries):
    """自訂字典 {"和": "ㄏㄢˋ", "垃圾": "ㄌㄜˋ ㄙㄜˋ"} → {詞: [注音＋數字…]}；字數與讀音數不合的回傳在 bad"""
    good, bad = {}, []
    for w, v in (entries or {}).items():
        syl = [x for x in re.split(r"[\s　]+", str(v).strip()) if x]
        if w and _TTS_HAN.fullmatch(w) and len(syl) == len(w):
            good[w] = [_tts_syl(x) for x in syl]
        else:
            bad.append(w)
    return good, bad


def _tts_score(cand, ctx):
    """注音＋聲調都對 2 分、只有注音對 1 分：字音（便 ㄆㄧㄢ／ㄅㄧㄢ）比輕聲與否重要"""
    return sum(2 if a == b else (1 if b and a[:-1] == b[:-1] else 0) for a, b in zip(cand, ctx))


def _tts_overlay(text, tw, moe, custom):
    """每個字的台灣念法：自訂（詞或單字）＞台灣常用念法（_TTS_TW_COMMON，詞裡也換）＞教育部辭典的詞（最長比對；多個讀音挑跟 g2pW 最接近的，
    一樣接近取辭典的第一個）＞g2pW（tw 傳入的就是 g2pW 的結果）。單字的自訂只用在沒被詞涵蓋的字（和平的和照辭典）。
    全由數字字組成的辭典詞不比對：那些是專名或成語（「五百」是古代職官，念ㄨˇ ㄅㄛˊ），數字照 g2pW（2026-10-09 實測三千五百元被念成五{bo2}）"""
    tw = list(tw)
    g2p = list(tw)
    fixed = set()                                    # 自訂發音給的位置：辭典詞、台灣常用念法都不蓋掉它
    words = {**_TTS_TW_WORDS, **{w: r for w, r in custom.items() if len(w) > 1}}
    # 先套自訂的詞：自訂比辭典優先，就算辭典有更長的詞（自訂「液化」、辭典有「液化石油氣」，2026-10-09）
    for m in _TTS_HAN.finditer(text):
        i, end = m.start(), m.end()
        while i < end:
            for n in range(min(max([0] + [len(w) for w in words]), end - i), 1, -1):
                if text[i:i + n] in words:
                    tw[i:i + n] = words[text[i:i + n]]
                    fixed.update(range(i, i + n))
                    i += n
                    break
            else:
                i += 1
    for m in _TTS_HAN.finditer(text):
        i, end = m.start(), m.end()
        while i < end:
            if i in fixed:
                i += 1
                continue
            for n in range(min(_TTS_MAX_WORD, end - i), 1, -1):
                w = text[i:i + n]
                key = w if w in moe else w.translate(_TTS_VARIANT)
                if key in moe and w.strip(_TTS_NUM_HAN) and not fixed.intersection(range(i, i + n)):
                    cands = moe[key]
                    ctx = tw[i:i + n]
                    best = max(cands, key=lambda c: (_tts_score(c, ctx), -cands.index(c)))
                    if any(g2p[i + k] and text[i + k] in _TTS_G2P_FIRST and best[k] != g2p[i + k] for k in range(n)):
                        continue                         # 切錯了：試短一點的詞
                    tw[i:i + n] = best
                    i += n
                    break
            else:
                if text[i] in custom:
                    tw[i] = custom[text[i]][0]
                    fixed.add(i)
                elif text[i] in _TTS_TW_SINGLE and tw[i] and _TTS_TW_SINGLE[text[i]][0] in (None, tw[i]):
                    tw[i] = _TTS_TW_SINGLE[text[i]][1]
                i += 1
    for k, ch in enumerate(text):
        rule = _TTS_TW_COMMON.get(ch)
        if rule and k not in fixed and tw[k] and (rule[0] is None or tw[k] == rule[0]):
            tw[k] = rule[1]
    return tw


def _tts_hint(text, tw, cn, to_pinyin, force=()):
    """台灣念法（tw）與模型預設會念的（cn，pypinyin 的大陸念法）不同的字換成 {拼音}；
    一、不、台灣念輕聲的不換。to_pinyin：注音＋數字 → 拼音＋數字（如 ㄌㄜ4 → le4），換不出來回 None。
    force：一定要換的位置（模型詞彙裡沒有的字，見 _tts_vocab_chars）；g2pW 沒給念法時用 pypinyin 的"""
    out = []
    for k, (ch, t, c) in enumerate(zip(text, tw, cn)):
        if k in force and not t:
            t = c
        if t and (c and (t != c or ch in _TTS_ALWAYS_HINT) or k in force) and ch not in _TTS_NO_HINT and not t.endswith("5"):
            py = to_pinyin(t)
            if py:
                out.append("{" + py + "}")
                continue
        out.append(ch)
    return "".join(out)


_TTS_CNUM = "零一二三四五六七八九"
_TTS_NUM_HAN = "零〇一二三四五六七八九十百千萬億兆兩"
_TTS_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
# 模型自己念數字（文字正規化關閉），2026-10-09 GPU 實測各 3 次：千分位逗號 3/3 亂念、負號 3/3 被吞掉、
# NT$ 念成美元；日期、時間、IP、版本號、電話、小數、百分比、分數都對 → 只改念錯的這三種，其他不動
_TTS_MONEY = (
    (re.compile(r"(?:NT|NTD)\$\s*(" + _TTS_NUM + r")(?:\s*元)?"), r"新台幣\1元"),
    (re.compile(r"(?:US|USD)\$\s*(" + _TTS_NUM + r")"), r"\1美元"),
    (re.compile(r"(?<![A-Za-z])\$\s*(" + _TTS_NUM + r")"), r"\1美元"),
    (re.compile(r"€\s*(" + _TTS_NUM + r")"), r"\1歐元"),
    (re.compile(r"£\s*(" + _TTS_NUM + r")"), r"\1英鎊"),
)
_TTS_TEMP = (
    (re.compile(r"(?<![A-Za-z0-9_.])[-−](\d+(?:\.\d+)?)\s*(?:°C|℃)"), r"零下\1度"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:°C|℃)"), r"\1度"),
    (re.compile(r"(?<![A-Za-z0-9_.])[-−](\d+(?:\.\d+)?)\s*(?:°F|℉)"), r"華氏零下\1度"),
    (re.compile(r"(\d+(?:\.\d+)?)\s*(?:°F|℉)"), r"華氏\1度"),
)
# 前面是英數字、小數點、斜線、冒號等就不是負號（2026-10-09、02-2345-6789、A-1、3-5 天）
_TTS_NEG = re.compile(r"(?<![A-Za-z0-9_.,/:\-−+])[-−](?=\d)")
_TTS_COMMA_NUM = re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+)(\.\d+)?(?!\d|,\d)")


def _tts_cn_sec(x, leading):
    """1～9999 → 國字；2 在千、百前念「兩」；開頭的十幾不說「一十」"""
    out, zero, started = "", False, False
    for d, u in zip((x // 1000, x // 100 % 10, x // 10 % 10, x % 10), ("千", "百", "十", "")):
        if d == 0:
            zero = zero or started
            continue
        if zero:
            out += "零"
            zero = False
        ch = "兩" if d == 2 and u in ("千", "百") else _TTS_CNUM[d]
        if d == 1 and u == "十" and not started and leading:
            ch = ""
        out += ch + u
        started = True
    return out


def _tts_cn_int(n):
    """整數 → 國字（台灣說法）：1250000 → 一百二十五萬、10005 → 一萬零五、20000 → 兩萬"""
    if n == 0:
        return "零"
    secs = []
    while n:
        secs.append(n % 10000)
        n //= 10000
    out, gap = "", False
    for i in range(len(secs) - 1, -1, -1):
        sec = secs[i]
        if sec == 0:
            gap = gap or bool(out)
            continue
        if out and (gap or sec < 1000):
            out += "零"
        out += ("兩" if sec == 2 and i else _tts_cn_sec(sec, not out)) + ("", "萬", "億", "兆")[i]
        gap = False
    return out


def _tts_numbers(text):
    """模型念錯的數字寫法先換成念得對的：金額符號 → 幣別、溫度、負號 → 負、千分位逗號 → 國字"""
    for rx, rep in _TTS_MONEY + _TTS_TEMP:
        text = rx.sub(rep, text)
    text = _TTS_NEG.sub("負", text)
    return _TTS_COMMA_NUM.sub(lambda m: _tts_cn_int(int(m.group(1).replace(",", "")))
                              + ("點" + "".join(_TTS_CNUM[int(c)] for c in m.group(2)[1:]) if m.group(2) else ""), text)


def _tts_glued(s, k):
    """在 s[k] 之後切會不會切斷一個詞：英數字中間、數字裡的逗號／小數點／冒號（1,250,000、0.5、3:30）"""
    a, b = s[k], s[k + 1] if k + 1 < len(s) else ""
    if a.isascii() and a.isalnum() and b.isascii() and b.isalnum():
        return True
    if a in ",.:" and k and s[k - 1].isdigit() and b.isdigit():
        return True
    return b in ",.:" and a.isdigit() and k + 2 < len(s) and s[k + 2].isdigit()


def _tts_split(text, limit=80):
    """切句：句末標點與換行；一句超過 limit 字再依逗號切，仍太長就硬切（不切在英數字、1,250,000、3:30 中間）。
    只有標點、沒有字的片段丟掉（送進模型會產生雜音）"""
    sents, buf = [], ""
    for i, ch in enumerate(text):
        buf += ch
        if ch in _TTS_SENT_END or (ch == "." and (i + 1 == len(text) or text[i + 1].isspace())
                                   and not (i and text[i - 1].isdigit())):
            sents.append(buf)
            buf = ""
    sents.append(buf)
    out = []
    for s in sents:
        s = s.strip()
        while len(s) > limit:
            cut = max((k for k, c in enumerate(s[:limit]) if c in "，,、：:" and not _tts_glued(s, k)), default=-1)
            if cut < limit // 3:
                cut = limit - 1
                while cut > limit // 2 and _tts_glued(s, cut):
                    cut -= 1
            out.append(s[:cut + 1].strip())
            s = s[cut + 1:].strip()
        out.append(s)
    return [s for s in out if re.search(r"\w", s)]


def _tts_load_text(tts_dir):
    """台灣念法要用的資源：教育部辭典對照檔、g2pW、pypinyin（模型預設念法的近似）、OpenCC 繁轉簡。
    Mac 本機合成也用同一份（jtlw_tts/tw_reading.py）"""
    import opencc
    from g2pw import G2PWConverter
    from pypinyin import Style, lazy_pinyin
    from pypinyin.contrib.tone_convert import to_tone3
    from pypinyin.pinyin_dict import pinyin_dict
    from pypinyin.style.bopomofo import BopomofoConverter
    moe_path = os.path.join(tts_dir, "moe_words.tsv")
    if not os.path.exists(moe_path):
        raise FileNotFoundError(f"找不到教育部辭典對照檔 {moe_path}（安裝程式會下載並轉檔）")
    with open(moe_path, encoding="utf-8") as f:
        moe = _tts_moe_load(f)
    bc = BopomofoConverter()

    def base_bopo(base):   # 沒有聲調的拼音 pypinyin 會當輕聲加「˙」，拿掉才能跟 g2pW 的格式比（第一版因此一個字都沒換）
        return bc.to_bopomofo(base.replace("v", "ü")).replace("˙", "")

    def split_tone(p):
        m = re.match(r"([a-zü]+)([1-5])$", p.replace("ü", "v"))
        return (m.group(1), m.group(2)) if m else (p, "")

    bopo2base = {}
    for readings in pinyin_dict.values():
        for r in readings.split(","):
            base, _ = split_tone(to_tone3(r, neutral_tone_with_five=True))
            bopo2base.setdefault(base_bopo(base), base)

    def py2bopo(p):
        base, d = split_tone(p)
        return base_bopo(base) + d if d else None

    def bopo2py(b):
        base = bopo2base.get(b[:-1]) if b and b[-1].isdigit() else None
        return base + b[-1] if base else None

    g2p = G2PWConverter(model_dir=os.path.join(tts_dir, "G2PWModel") + "/", style="bopomofo",
                        model_source=os.path.join(tts_dir, "bert-base-chinese"))
    g2p.num_workers = 0          # 預設開子行程：macOS／Windows 用 spawn 會卡死；建構時傳 0 會被當成沒指定
    return {"moe": moe, "g2p": g2p, "lazy_pinyin": lazy_pinyin, "TONE3": Style.TONE3,
            "t2s": opencc.OpenCC("t2s"), "py2bopo": py2bopo, "bopo2py": bopo2py}


def _tts_readings(R, text, custom=None):
    """每個字的 (台灣念法, 模型預設念法)，都是注音＋數字；非漢字為 None"""
    good, _ = _tts_custom(custom)
    tw = _tts_overlay(text, [g if g else None for g in R["g2p"](text)[0]], R["moe"], good)
    src = _tts_simp_src(text, tw)                    # 模型預設念法拿「送進模型的字」算：着急的着、著作的著念法不同
    cn = [None] * len(text)
    for m in _TTS_HAN.finditer(src):
        run = m.group(0)
        simp = R["t2s"].convert(run)
        if len(simp) != len(run):
            continue
        for k, p in enumerate(R["lazy_pinyin"](simp, style=R["TONE3"], neutral_tone_with_five=True)):
            cn[m.start() + k] = R["py2bopo"](p)
    # 刻意指定念法的詞（台灣日常念法、自訂發音）：pypinyin 可能跟前後文切成別的詞（「萬人參與」切出「人參」，「與」算成單字的ㄩˇ，
    # 跟台灣念法一樣就不加提示，模型卻照「參與」念ㄩˋ），所以這個詞再單獨算一次，兩種算法有一種跟台灣念法不同就加提示（2026-10-09）
    for w, r in {**_TTS_TW_WORDS, **{w: r for w, r in good.items() if len(w) > 1}}.items():
        i = text.find(w)
        while i >= 0:
            simp = R["t2s"].convert(src[i:i + len(w)])
            if tw[i:i + len(w)] == r and len(simp) == len(w):
                for k, p in enumerate(R["lazy_pinyin"](simp, style=R["TONE3"], neutral_tone_with_five=True)):
                    b = R["py2bopo"](p)
                    if b and tw[i + k] == cn[i + k] and b != tw[i + k]:
                        cn[i + k] = b
            i = text.find(w, i + 1)
    return tw, cn


def _tts_vocab_chars(path):
    """模型詞彙裡的中文單字（模型資料夾的 tokenizer.json）。不在裡面的字只能拆成位元組送進模型，模型不知道怎麼念，
    一律加念法提示（2026-10-09 自動偵測：人名用字的昀、婞，蚵仔煎的蚵都念錯，三個字都不在詞彙裡）。讀不到回 None（不套這條）"""
    try:
        with open(path, encoding="utf-8") as f:
            vocab = json.load(f)["model"]["vocab"]
        return frozenset(k for k in vocab if len(k) == 1 and _TTS_HAN.fullmatch(k))
    except Exception:
        return None


def _tts_simp_src(text, tw):
    """送進模型前要寫成的字（_TTS_SIMP_BY_READING）：扮演著 → 扮演着、著急 → 着急；著作照舊"""
    if not _TTS_SIMPLIFIED:
        return text
    return "".join(_TTS_SIMP_BY_READING[ch][1] if ch in _TTS_SIMP_BY_READING and t and t != _TTS_SIMP_BY_READING[ch][0]
                   else ch for ch, t in zip(text, tw))


def _tts_spoken(R, text, custom=None):
    """原文 → 送進模型的文字（念錯的數字寫法先換掉；台灣念法與模型預設不同的字換成 {拼音}）。
    沒加提示的字轉成簡體再送：模型幾乎只學過簡體，繁體字會念錯（2026-10-09 實測 協、脈、漲、頒、衝 等）；
    「模型預設念法」本來就是拿簡體算的（_tts_readings 的 cn），轉了之後模型念的正好就是比對時假設的"""
    text = _tts_numbers(text)
    tw, cn = _tts_readings(R, text, custom)
    text = _tts_simp_src(text, tw)
    sent = R["t2s"].convert(text) if _TTS_SIMPLIFIED else text      # 沒加提示時送進模型的字
    vocab = R.get("vocab")
    force = {k for k, ch in enumerate(sent) if _TTS_HAN.fullmatch(ch) and ch not in vocab} if vocab and len(sent) == len(text) else ()
    hinted = _tts_hint(text, tw, cn, R["bopo2py"], force)
    if not _TTS_SIMPLIFIED:
        return hinted
    return "".join(p if p.startswith("{") else R["t2s"].convert(p) for p in re.split(r"(\{[a-z]+[1-5]\})", hinted))
