from pathlib import Path

import chardet

paths = ["docs/余华 活着.txt"]
for p in paths:
    raw = Path(p).read_bytes()
    print(f"{p}: raw bytes = {len(raw)}")
    guess = chardet.detect(raw)
    print(f"chardet guess: {guess}")
    # 试着用 utf-8 和 gb18030 解码
    for enc in ["utf-8", "gb18030"]:
        try:
            txt = raw.decode(enc, errors="ignore")
            print(f"  decode({enc}, ignore) length={len(txt)} first50={txt[:50]!r}")
        except Exception as e:
            print(f"  decode({enc}) failed: {e}")
