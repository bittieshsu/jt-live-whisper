"""命令列：python -m jtlw_tts "要念的文字" -o 輸出.wav [--voice 聲音ID] [--file 文字檔]

用 config.json 的設定（GPU 伺服器或 Apple Silicon 本機、預設聲音、自訂讀音），逐句合成後合併成一個 WAV。"""
import argparse
import json
import os
import sys
import time
import wave

from . import TTSError, default_voice_id, get_voice, list_voices, pick_provider, settings
from .tw_reading import _tts_split


def main():
    ap = argparse.ArgumentParser(prog="python -m jtlw_tts", description="台灣華語文字轉語音（輸出 WAV）")
    ap.add_argument("text", nargs="?", help="要念的文字（或用 --file）")
    ap.add_argument("--file", help="從 UTF-8 文字檔讀入")
    ap.add_argument("-o", "--output", default="tts_output.wav")
    ap.add_argument("--voice", help="聲音 ID（預設用設定裡的聲音）")
    ap.add_argument("--list-voices", action="store_true")
    a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    try:
        with open(os.path.join(root, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
    except (OSError, ValueError):
        cfg = {}
    if a.list_voices:
        for v in list_voices():
            print(f"{v['id']}  {v['name']}（{v['duration']} 秒，來源：{v['source']}）")
        return 0
    text = open(a.file, encoding="utf-8").read() if a.file else (a.text or "")
    s = settings(cfg)
    try:
        voice = get_voice(a.voice or default_voice_id(cfg))
        prov, why = pick_provider(cfg)
        if not prov:
            raise TTSError("tts_unavailable", why)
        chunks = _tts_split(text, int(s["chunk_chars"]))
        if not chunks:
            raise TTSError("empty_text", "沒有要朗讀的文字")
        print(f"[文字轉語音] {prov.label}，聲音「{voice['name']}」，共 {len(chunks)} 段")
        frames, params, t0 = [], None, time.time()
        for i, c in enumerate(chunks, 1):
            data, _ = prov.synth(c, voice, s["custom"])
            import io
            with wave.open(io.BytesIO(data), "rb") as r:
                params = params or (r.getnchannels(), r.getsampwidth(), r.getframerate())
                frames.append(r.readframes(r.getnframes()))
            print(f"  {i}/{len(chunks)}  {c[:40]}")
        with wave.open(a.output, "wb") as w:
            w.setnchannels(params[0])
            w.setsampwidth(params[1])
            w.setframerate(params[2])
            for fr in frames:
                w.writeframes(fr)
        print(f"[完成] {a.output}（{time.time() - t0:.1f} 秒）")
        return 0
    except TTSError as e:
        print(f"[錯誤] {e.message}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
