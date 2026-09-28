"""會議摘要的圖：**在伺服器端產生 SVG，一份用在三個地方。**

* 網頁上直接嵌（SVG 向量、放大不糊、字可以選取）
* 匯出的 Markdown 內嵌（`![](…png)`，交給別的工具排版時圖還在）
* 匯出成 PNG / PDF

**不要在前端與後端各畫一份** —— 那兩份一定會漂，而且匯出的圖會跟畫面上看到的
不一樣（本專案在「同一份清單寫在兩個地方」上踩過很多次）。

**SVG → PNG / PDF 走 PyMuPDF**（MuPDF 本來就讀得懂 SVG），不需要新相依。
中文字形驗過：不同的字畫出不同的圖，不是缺字方框。
"""
from __future__ import annotations

import html
from typing import Optional, Sequence

#: 圖的配色。**同一個發言者在每張圖上要是同一個顏色** —— 不然兩張圖並排時
#: 讀的人會以為是不同的人。
PALETTE = ("#6366f1", "#10b981", "#f59e0b", "#ef4444", "#0ea5e9",
           "#8b5cf6", "#14b8a6", "#f97316", "#84cc16", "#ec4899",
           "#06b6d4", "#a855f7")

#: 節點類別 → 顏色與中文標籤。
KIND_STYLE = {
    "topic":    ("#4338ca", "主題"),
    # 事件與影響：已經發生的事。**顏色要跟風險（紅）分得開** ——
    # 兩者的界線是「發生了沒有」，用相近的顏色會讓人以為是同一類。
    "impact":   ("#0f766e", "事件"),
    "decision": ("#047857", "決議"),
    "action":   ("#1d4ed8", "待辦"),
    "risk":     ("#b91c1c", "風險"),
    "question": ("#b45309", "未決"),
}

_FONT = ("Noto Sans CJK TC, Noto Sans TC, PingFang TC, "
         "Microsoft JhengHei, sans-serif")


def _esc(s: object) -> str:
    return html.escape(str(s if s is not None else ""), quote=True)


def _wrap(text: str, per_line: int) -> list[str]:
    """把一段字折成幾行。

    **中文沒有詞界，按字數折就好**；拉丁字母按字數折會切在單字中間，
    所以遇到空白時優先在空白處斷。
    """
    text = " ".join(str(text or "").split())
    if not text:
        return []
    out, cur = [], ""
    for ch in text:
        cur += ch
        if len(cur) >= per_line:
            cut = cur.rfind(" ")
            if cut > per_line * 0.5:
                out.append(cur[:cut])
                cur = cur[cut + 1:]
            else:
                out.append(cur)
                cur = ""
    if cur:
        out.append(cur)
    return out


def _mmss(ms: Optional[int]) -> str:
    if ms is None:
        return ""
    t = int(ms) // 1000
    h, r = divmod(t, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _svg(w: int, h: int, body: str, title: str = "") -> str:
    t = f"<title>{_esc(title)}</title>" if title else ""
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" font-family="{_FONT}" role="img">'
            f'{t}<rect width="{w}" height="{h}" fill="#ffffff"/>{body}</svg>')


# ------------------------------------------------------------------ 發言者佔比

#: `unknown` 是分析時塞的代號不是名字 —— **資料裡留著**（統計、引用、
#: 匯出都靠它對應），**只在顯示的時候換掉**。
#: 這跟「欄位同義詞字典不可以翻」是一體兩面：資料不翻，顯示要翻。
UNLABELLED = "未標示發言者"


def _spk(name: object) -> str:
    s = str(name or "")
    return UNLABELLED if (not s or s == "unknown") else s


def speaker_timeline(segments: Sequence[dict], stats: dict, *,
                     width: int = 760) -> Optional[str]:
    """誰在什麼時候講話 —— **跟畫面上那一欄同一件事**。

    ## 為什麼不是長條圖

    「誰講最多」那份數字**匯出的表格裡已經有了**，再畫一次長條只是換個形狀。
    這張畫的是每個人在會議的**哪些時候**講話：一人一列，講話的地方就有一格。
    誰主導（墨水量）、節奏（穿插還是集中）、誰在哪一段沒出現（空白處），
    表格一件都說不出來。

    ## ⚠ 這一支存在的理由：畫面改了，匯出沒跟著改

    2026-09-19 前端把發言者長條圖併進表格、換成發言分布，**但匯出用的是
    伺服器畫的這一份** —— 於是網頁上是分布、PDF 裡還是舊的長條圖，
    連 `unknown` 都還沒換成中文。使用者一眼就看到了。
    **「同一份東西寫在兩個地方一定會漂」在這裡又應驗一次。**

    沒有時間戳記也畫得出來：橫軸改用「逐字稿的第幾段」，一樣是會議的進程。
    """
    if not segments or len(stats or {}) < 2:
        return None
    use_time = all(s.get("start_ms") is not None for s in segments)

    def at(s: dict) -> int:
        return int(s["start_ms"] if use_time else s.get("seq") or 0)

    def dur(s: dict) -> int:
        if not use_time:
            return 0
        a, b = s.get("start_ms"), s.get("end_ms")
        return int(b - a) if (b is not None and a is not None and b > a) else 0

    lo = min(at(s) for s in segments)
    hi = max(at(s) + dur(s) for s in segments)
    span = max(1, hi - lo)

    by_time = all(v.get("speaking_ms") is not None for v in stats.values())
    rows = sorted(
        ((n, v) for n, v in stats.items() if v.get("turn_count") or v.get("speaking_ms")),
        key=lambda kv: -(kv[1].get("speaking_ms") if by_time else (kv[1].get("chars") or 0)),
    )
    if len(rows) < 2:
        return None
    total = sum((v.get("speaking_ms") if by_time else (v.get("chars") or 0))
                for _n, v in rows) or 1

    mine: dict[str, list[dict]] = {}
    for s in segments:
        mine.setdefault(str(s.get("speaker") or "unknown"), []).append(s)

    name_w, gap, tail_w = 100, 10, 140
    track_x = name_w + gap
    track_w = max(120, width - track_x - tail_w)
    row_h, mark_h, top = 26, 13, 46
    h = top + row_h * len(rows) + 16

    unit_note = ("橫軸是會議時間，色塊是那個人在講話" if use_time
                 else "這份逐字稿沒有時間戳記，橫軸改用逐字稿的順序")
    body = [f'<text x="0" y="20" font-size="14" font-weight="700" fill="#0f172a">'
            f'誰在什麼時候講話</text>',
            f'<text x="0" y="36" font-size="11" fill="#64748b">{unit_note}</text>']

    for i, (name, v) in enumerate(rows):
        y = top + i * row_h
        colour = PALETTE[i % len(PALETTE)]
        body.append(
            f'<rect x="{track_x}" y="{y + (row_h - mark_h) // 2}" width="{track_w}" '
            f'height="{mark_h}" rx="3" fill="#f1f5f9"/>'
            f'<text x="{name_w}" y="{y + row_h // 2 + 4}" font-size="12.5" '
            f'text-anchor="end" fill="#334155">{_esc(_spk(name))}</text>')
        # **相鄰的段落要合併** —— 兩小時的會議上千段，逐段畫會糊成一片。
        marks: list[list[float]] = []
        for s in sorted(mine.get(name, []), key=at):
            x0 = track_x + (at(s) - lo) / span * track_w
            w0 = max(2.0, dur(s) / span * track_w)
            if marks and x0 - (marks[-1][0] + marks[-1][1]) < 1.5:
                marks[-1][1] = max(marks[-1][1], x0 + w0 - marks[-1][0])
            else:
                marks.append([x0, w0])
        for x0, w0 in marks:
            body.append(
                f'<rect x="{x0:.1f}" y="{y + (row_h - mark_h) // 2}" '
                f'width="{min(w0, track_x + track_w - x0):.1f}" height="{mark_h}" '
                f'rx="2.5" fill="{colour}"/>')
        val = v.get("speaking_ms") if by_time else (v.get("chars") or 0)
        shown = _mmss(val) if by_time else f"{val:,} 字"
        # **數字分欄靠右** —— 串成一句的話前面那欄會被後面的長度推著跑。
        for x, s in ((width - 96, f"{val * 100 / total:.1f}%"),
                     (width - 46, shown),
                     (width, f"{v.get('turn_count') or 0} 次")):
            body.append(f'<text x="{x}" y="{y + row_h // 2 + 4}" font-size="11.5" '
                        f'text-anchor="end" fill="#64748b">{_esc(s)}</text>')
    return _svg(width, h, "".join(body), "誰在什麼時候講話")


def speaker_share(stats: dict, *, width: int = 760) -> Optional[str]:
    """誰講了多少（長條圖）。

    **只在沒有逐字稿可用時才畫** —— 有逐字稿的話 `speaker_timeline()` 說得
    更多（同樣看得出佔比，還看得出什麼時候講）。公開 API 只給分析結果、
    沒有逐段資料，那時才退到這一張。
    """
    rows = []
    use_time = any(v.get("speaking_ms") for v in stats.values())
    for name, v in stats.items():
        val = v.get("speaking_ms") if use_time else v.get("chars")
        if not val:
            continue
        rows.append((name, int(val), v.get("turn_count") or 0))
    if len(rows) < 2:
        return None
    rows.sort(key=lambda r: -r[1])
    total = sum(r[1] for r in rows) or 1

    row_h, pad_t, label_w = 30, 44, 150
    h = pad_t + row_h * len(rows) + 16
    bar_x = label_w + 12
    bar_w = width - bar_x - 96
    unit = "發言時間" if use_time else "發言字數"
    body = [f'<text x="0" y="20" font-size="14" font-weight="700" fill="#0f172a">'
            f'各發言者佔多少（依{unit}）</text>',
            f'<text x="0" y="36" font-size="11" fill="#64748b">'
            f'{"重疊的插話只算一次" if use_time else "這份逐字稿沒有時間戳記，改用字數"}</text>']
    for i, (name, val, turns) in enumerate(rows):
        y = pad_t + i * row_h
        pct = val * 100 / total
        w = max(2, int(bar_w * val / total))
        colour = PALETTE[i % len(PALETTE)]
        shown = _mmss(val) if use_time else f"{val:,} 字"
        body.append(
            f'<text x="{label_w}" y="{y + 15}" font-size="12.5" text-anchor="end" '
            f'fill="#334155">{_esc(_spk(name))}</text>'
            f'<rect x="{bar_x}" y="{y + 4}" width="{w}" height="14" rx="3" fill="{colour}"/>'
            f'<text x="{bar_x + w + 8}" y="{y + 15}" font-size="11.5" fill="#475569">'
            f'{pct:.1f}%　{_esc(shown)}　{turns} 次</text>')
    return _svg(width, h, "".join(body), f"各發言者佔多少（依{unit}）")


# ------------------------------------------------------------------ 章節時間軸

def chapter_timeline(chapters: Sequence[dict], *, width: int = 760) -> Optional[str]:
    """章節長度條。**沒有時間就按段落數畫** —— 那仍然是真的比例，
    只是單位不同；圖上要寫出來。"""
    if len(chapters) < 2:
        return None
    # **要用 `is not None` 不可以用真假值** —— 一章裡只有一個時間標記時
    # 長度就是 0，而 `0` 是 falsy，整張圖會退回用段數。
    use_time = all(c.get("duration_ms") is not None for c in chapters)
    vals = [int(c["duration_ms"]) if use_time else len(c.get("segment_ids") or [])
            for c in chapters]
    total = sum(vals) or 1

    bar_y, bar_h, pad_t = 44, 26, 0
    legend_h = 22 * len(chapters)
    h = bar_y + bar_h + 18 + legend_h
    unit = "時間" if use_time else "發言段數"
    body = [f'<text x="0" y="20" font-size="14" font-weight="700" fill="#0f172a">'
            f'各議題佔多少{unit}</text>']
    if not use_time:
        body.append('<text x="0" y="36" font-size="11" fill="#64748b">'
                    '這份逐字稿沒有時間戳記，改用發言段數</text>')
    x = 0
    for i, (c, v) in enumerate(zip(chapters, vals)):
        w = max(2, int(width * v / total))
        colour = PALETTE[i % len(PALETTE)]
        body.append(f'<rect x="{x}" y="{bar_y}" width="{w}" height="{bar_h}" fill="{colour}"/>')
        x += w
    for i, (c, v) in enumerate(zip(chapters, vals)):
        y = bar_y + bar_h + 22 + i * 22
        colour = PALETTE[i % len(PALETTE)]
        extra = _mmss(c.get("start_ms")) if use_time else f"第 {(c.get('segment_ids') or [0])[0]} 段起"
        body.append(
            f'<rect x="0" y="{y - 9}" width="10" height="10" rx="2" fill="{colour}"/>'
            f'<text x="18" y="{y}" font-size="12" fill="#334155">{_esc(c.get("title"))}</text>'
            f'<text x="{width}" y="{y}" font-size="11.5" text-anchor="end" fill="#64748b">'
            f'{v * 100 / total:.1f}%　{_esc(extra)}</text>')
    return _svg(width, h, "".join(body), f"各議題佔多少{unit}")


# ------------------------------------------------------------------ 心智圖

#: 節點文字每行幾個字。太長會讓圖變得非常寬，太短會折成一條細柱。
_NODE_CHARS = 14
_LINE_H = 16


def mindmap(nodes: Sequence[dict], *, width: int = 980) -> Optional[str]:
    """討論結構：左邊主題、右邊掛決議／待辦／風險／未決問題。

    **是「組」出來的不是「生成」的** —— 每個節點都來自已經通過引用驗證的項目，
    所以圖上的每一格都指得回逐字稿（規格明訂：禁止只生成不可追溯的圖）。

    版面用**水平樹**不用放射狀：中文節點是長條形，放射狀會互相重疊，
    而且節點一多就完全看不懂。

    **⚠ 沒有掛東西的章節不可以也占一整列的寬度**（使用者 2026-09-19 回報
    「畫面右邊太空了」）。真實的會議裡決議與待辦幾乎都集中在後段，所以
    十個章節常常只有三個有子節點 —— 兩欄版面就變成右半邊有七列是空的，
    看起來像圖畫壞了。那些章節**確實沒有產出**，是真的資料不是缺陷，
    所以做法不是隱藏也不是編一個節點出來，而是**換一種畫法**：
    畫成一條橫跨整個寬度的扁條，右邊註明「無決議／待辦」。
    讀的人一眼就分得出「談過但沒有結論」與「談出東西來了」。
    """
    if len(nodes) < 2:
        return None
    kids: dict[Optional[str], list[dict]] = {}
    for n in nodes:
        kids.setdefault(n.get("parent_id"), []).append(n)
    roots = kids.get(None) or []
    if not roots:
        return None

    col1_w, gap, pad = 230, 46, 14
    col2_x = pad + col1_w + gap
    col2_w = width - col2_x - pad
    full_w = width - pad * 2

    def box_h(label: str, w_chars: int) -> int:
        return max(34, len(_wrap(label, w_chars)) * _LINE_H + 16)

    body, y = [], pad + 26
    body.append('<text x="0" y="18" font-size="14" font-weight="700" fill="#0f172a">'
                '討論結構</text>')
    for root in roots:
        children = kids.get(root.get("node_id")) or []
        rc, _ = KIND_STYLE.get(root.get("type") or "topic", ("#4338ca", ""))
        label = root.get("label") or ""

        # --- 沒有子節點：畫成整寬的扁條，不要留一片空白 ---------------------
        if not children:
            note = "無決議／待辦"
            # 扣掉右邊那行註記的位置再折行，字才不會壓到它。
            rh = box_h(label, 46)
            body.append(
                f'<rect x="{pad}" y="{y}" width="{full_w}" height="{rh}" rx="8" '
                f'fill="{rc}" opacity="0.06"/>'
                f'<rect x="{pad}" y="{y}" width="4" height="{rh}" rx="2" '
                f'fill="{rc}" opacity="0.55"/>')
            for li, line in enumerate(_wrap(label, 46)):
                body.append(f'<text x="{pad + 14}" y="{y + 21 + li * _LINE_H}" '
                            f'font-size="13" font-weight="600" fill="{rc}">'
                            f'{_esc(line)}</text>')
            body.append(f'<text x="{width - pad - 12}" y="{y + rh / 2 + 4:.0f}" '
                        f'font-size="11" text-anchor="end" fill="#94a3b8">'
                        f'{_esc(note)}</text>')
            y += rh + 10
            continue

        # --- 有子節點：兩欄水平樹 -------------------------------------------
        rh = box_h(label, 12)
        child_h = [box_h(c.get("label") or "", 34) for c in children]
        block_h = max(rh, sum(child_h) + max(0, len(children) - 1) * 8)
        ry = y + (block_h - rh) // 2

        body.append(
            f'<rect x="{pad}" y="{ry}" width="{col1_w}" height="{rh}" rx="8" '
            f'fill="{rc}" opacity="0.10"/>'
            f'<rect x="{pad}" y="{ry}" width="4" height="{rh}" rx="2" fill="{rc}"/>')
        for li, line in enumerate(_wrap(label, 12)):
            body.append(f'<text x="{pad + 14}" y="{ry + 21 + li * _LINE_H}" font-size="13" '
                        f'font-weight="600" fill="{rc}">{_esc(line)}</text>')

        cy = y
        for c, ch in zip(children, child_h):
            cc, klabel = KIND_STYLE.get(c.get("type") or "topic", ("#475569", ""))
            mid_r, mid_c = ry + rh / 2, cy + ch / 2
            body.append(
                f'<path d="M{pad + col1_w} {mid_r:.0f} C{pad + col1_w + gap / 2:.0f} {mid_r:.0f},'
                f' {col2_x - gap / 2:.0f} {mid_c:.0f}, {col2_x} {mid_c:.0f}" '
                f'fill="none" stroke="{cc}" stroke-width="1.4" opacity="0.45"/>')
            body.append(
                f'<rect x="{col2_x}" y="{cy}" width="{col2_w}" height="{ch}" rx="7" '
                f'fill="#ffffff" stroke="{cc}" stroke-width="1" opacity="0.95"/>'
                f'<rect x="{col2_x}" y="{cy}" width="3" height="{ch}" rx="1.5" fill="{cc}"/>')
            seq = (c.get("segment_ids") or [None])[0]
            tag = f'{klabel}' + (f'・第 {seq} 段' if seq is not None else "")
            body.append(f'<text x="{col2_x + 12}" y="{cy + 14}" font-size="10" fill="{cc}">'
                        f'{_esc(tag)}</text>')
            for li, line in enumerate(_wrap(c.get("label") or "", 34)):
                body.append(f'<text x="{col2_x + 12}" y="{cy + 29 + li * _LINE_H}" '
                            f'font-size="12.5" fill="#1e293b">{_esc(line)}</text>')
            cy += ch + 8
        y += block_h + 18
    return _svg(width, y + 4, "".join(body), "討論結構")


# ------------------------------------------------------------------ 轉檔

def to_png(svg: str, *, dpi: int = 144) -> bytes:
    """SVG → PNG。**走 MuPDF，不需要新相依。**"""
    import fitz
    doc = fitz.open("svg", svg.encode("utf-8"))
    try:
        return doc[0].get_pixmap(dpi=dpi).tobytes("png")
    finally:
        doc.close()


def to_pdf_pages(svgs: Sequence[str]) -> bytes:
    """幾張 SVG → 一份多頁 PDF。"""
    import fitz
    out = fitz.open()
    try:
        for s in svgs:
            d = fitz.open("svg", s.encode("utf-8"))
            try:
                out.insert_pdf(fitz.open("pdf", d.convert_to_pdf()))
            finally:
                d.close()
        return out.tobytes()
    finally:
        out.close()


def build_all(analysis: dict, segments: Optional[Sequence[dict]] = None) -> dict:
    """一次產出這場會議適合的每一張圖：`{名稱: svg}`。

    **哪幾張畫得出來由資料決定**（`meeting_insight.suitable_charts` 同一條規則）
    —— 一場只有一個主題的會議畫章節佔比沒有意義，而沒有內容的圖會讓人以為
    功能壞了。
    """
    out: dict[str, str] = {}
    mm = mindmap(analysis.get("mindmap") or [])
    if mm:
        out["mindmap"] = mm
    # **有逐段資料就畫「誰在什麼時候講話」** —— 跟畫面上那一欄同一件事。
    # 沒有（公開 API 只給分析結果）才退到長條圖。
    stats = analysis.get("speaker_stats") or {}
    sp = speaker_timeline(segments, stats) if segments else None
    if sp is None:
        sp = speaker_share(stats)
    if sp:
        out["speaker_share"] = sp
    tl = chapter_timeline(analysis.get("chapters") or [])
    if tl:
        out["timeline"] = tl
    return out
