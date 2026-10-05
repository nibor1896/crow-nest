#!/usr/bin/env python3
"""The README's front image, both themes: docs/images/readme/crow-nest-dark.svg (Crow's crow
theme, for GitHub dark) and crow-nest-light.svg (for GitHub light). README.md shows the one that
matches the viewer through <picture> and prefers-color-scheme. Same layout as Crow's
tools/readme_image.py.

Every figure carries its date in the image (the rule of tools/check_readme_dates.py). The
version pill reads the newest released heading of CHANGELOG.md, so a release re-runs this and
commits the two files. The mark is docs/images/readme/logo-nest.svg.
Usage:  python3 tools/readme_image.py"""
import os, pathlib, re, subprocess, sys
from xml.sax.saxutils import escape

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "images" / "readme"

C = dict(page="#0b0e17", panel="#0e1220", raised="#131829", line="#1c2438", text="#e8eef8",
         soft="#cfdaea", faint="#9fb0c9", dim="#6d7b95", ok="#4ec98f", gold="#e5c04b",
         sub="#39c6d8", mark="#7eb0f8", bevel="#2c5bac", bad="#f0655a", skip="#b392f0",
         term="#080b13")
VARIANT = os.environ.get("VARIANT")  # dark = crow theme, light = GitHub light
MOBILE = os.environ.get("LAYOUT") == "mobile"  # one column at 440 px for phones (crow-nest-mobile-*.svg)
if VARIANT is None:  # one call writes all four files: this module draws into globals, so each is its own run
    for layout in ("desktop", "mobile"):
        for v in ("dark", "light"):
            subprocess.run([sys.executable, __file__], env={**os.environ, "VARIANT": v, "LAYOUT": layout}, check=True)
    sys.exit(0)
C = dict(page="#ffffff", panel="#ffffff", raised="#f0f1f3", line="#e4e4e7", text="#0f1114",
         soft="#3f4550", faint="#6b7280", dim="#6b7280", ok="#12855a", gold="#8a6400",
         sub="#0e7a8a", mark="#2c5bac", bevel="#2c5bac", bad="#c0362b", skip="#6f42c1",
         term="#f6f8fa") if VARIANT == "light" else C
UI = "-apple-system,BlinkMacSystemFont,'Segoe UI','Noto Sans',Helvetica,Arial,sans-serif"  # GitHub's own stack
MONO = "ui-monospace,SFMono-Regular,'SF Mono',Menlo,Consolas,'Liberation Mono',monospace"  # GitHub's code stack
W, X0, X1 = 880, 40, 840
o = []


def t(x, y, s, size=13, fill=C["faint"], font=UI, weight=400, anchor="start", ls=0):
    o.append(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" font-weight="{weight}" '
             f'fill="{fill}" text-anchor="{anchor}" letter-spacing="{ls}">{s}</text>')


def card(x, y, w, h, fill=C["panel"], stroke=C["line"], r=10):
    o.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}"/>')


def section(y, title, acc, sub=""):
    o.append(f'<rect x="{X0}" y="{y}" width="4" height="22" rx="2" fill="{C[acc]}"/>')
    t(56, y + 17, title, 20, C["text"], weight=600)
    lx = 80 + len(title) * 10.4
    if sub:
        t(lx, y + 16, sub, 12.5, C["dim"])
        lx += len(sub) * 6.4 + 16
    o.append(f'<line x1="{lx:.0f}" y1="{y+11}" x2="{X1}" y2="{y+11}" stroke="{C["line"]}"/>')
    return y + 40


def pill(x, y, label, col, font=MONO, size=12.5, fill="none"):
    w = 22 + len(label) * 7.6
    o.append(f'<rect x="{x}" y="{y}" width="{w:.0f}" height="26" rx="13" fill="{fill}" stroke="{col}"/>')
    t(x + w / 2, y + 17.5, escape(label), size, col, font, anchor="middle")
    return x + w + 10


def mark():
    s = (OUT / "logo-nest.svg").read_text()
    inner = s[s.index(">") + 1:s.rindex("</svg>")]
    inner = inner.replace("#64faf2", C["mark"])
    if VARIANT == "light":  # the same shade mapping mark-on-light.svg uses for the crow
        for a, b in (("#6789b2", "#283647"), ("#90a7c2", "#576575"), ("#acbece", "#75838b"), ("#b9c0c4", "#7f8c98")):
            inner = inner.replace(a, b)
    return inner


# ---------------------------------------------------------------- hero
y = 32
card(X0, y, 800, 300, r=14)
o.append(f'<svg x="48" y="44" width="280" height="280" viewBox="0 0 1024 1024">{mark()}</svg>')
t(340, 146, "CROW-NEST", 50, C["text"], weight=300, ls=12)
t(344, 182, "INFERENCE ENGINE", 13, C["faint"], ls=5)
o.append(f'<text x="342" y="226" font-family="{UI}" font-size="20" fill="{C["soft"]}">One GPU, its own quant, two model families.'
         f'<tspan fill="{C["mark"]}">▍<animate attributeName="opacity" values="1;1;0;0" keyTimes="0;.5;.5;1" '
         f'dur="1.1s" repeatCount="indefinite"/></tspan></text>')
x = 342
VERSION = re.search(r"^## .*? — v(\d+\.\d+\.\d+)", (ROOT / "CHANGELOG.md").read_text(), re.M).group(1)
for label, acc in (("v" + VERSION, "mark"), ("Apache-2.0", "faint"), ("Linux · Windows · sm_120", "sub")):
    x = pill(x, 258, label, C[acc])

# ---------------------------------------------------------------- intro text
y = 364
for i, line in enumerate((
        "An inference engine for one GPU: its own quantization, its own container, thin CUDA",
        "kernels in Rust. OpenAI-compatible HTTP, and the engine behind Crow. Two models from the",
        "original safetensors: Qwen3.8-Flash-Next as CNQ4.5-M and the dense Qwen3.8-27B as CNQ4.5.")):
    t(W / 2, y + i * 24, line, 15.5, C["soft"], anchor="middle")

# ---------------------------------------------------------------- stats
y = 452
STATS = [("35.8", "tok/s decode, Flash-Next, 122k"), ("771", "tok/s prefill, 16k prompt"), ("104.7 GB", "Flash-Next container file"),
         ("4.5 bpw", "NVFP4, CNQ4.5-M"), ("200k", "context, one slot"), ("1.7 GiB", "VRAM lent to Crow while idle"),
         ("120", "tok/s decode, 27B with MTP"), ("17.8 GB", "27B container file"), ("8.48 GiB", "free beside the 27B, 64k context")]
STATS_NOTE = ("Flash-Next, one RTX 5090: decode Linux 2026-09-24, after the PLE fix; prefill Windows 2026-09-13/14, before it; VRAM loan #117.",
              "27B: one RTX 5090, Linux, CROW_MMA=1 CROW_GRAPH=1, BF16 KV, 2026-09-27 (#95, #122). Conditions: docs/measurements.md, CHANGELOG.md")
for i, (v, l) in enumerate(STATS):
    sx, sy = X0 + (i % 3) * 270, y + (i // 3) * 86
    card(sx, sy, 260, 76, C["raised"], C["raised"])
    t(sx + 18, sy + 36, v, 26, C["text"], MONO, 600)
    t(sx + 18, sy + 59, l, 12.5, C["faint"])
SROWS = len(STATS) // 3
for k, line in enumerate(STATS_NOTE):
    t(W / 2, y + SROWS * 86 + 18 + k * 16, line, 11, C["dim"], anchor="middle")

# ---------------------------------------------------------------- features
y = section(y + SROWS * 86 + 62, "Features", "ok")
FEATURES = [
 ("Own quantization", "CNQ4.5: NVFP4 at 4.5 bits per weight", "ok"),
 ("Own container", "one .cnq file with a verification sidecar", "gold"),
 ("Rust + CUDA", "thin kernels via NVRTC, Blackwell sm_120", "sub"),
 ("OpenAI-compatible", "HTTP API, Crow is the client", "ok"),
 ("Two model families", "MoE Qwen3.8-Flash-Next, dense Qwen3.8-27B", "gold"),
 ("MTP on the 27B", "speculative decode, output identical to plain", "sub"),
 ("Vision", "both models, 1,024 to 1,280 tokens per image", "ok"),
 ("Hot set", "Flash-Next, calibrated on Crow traffic (#106)", "gold"),
 ("Prefix cache", "a side request parks the main conversation", "sub"),
 ("GPU sharing", "lends VRAM to Crow's renderer while idle", "ok"),
 ("Image generation", "the 27B at 64k leaves room for Qwen-Image 2.1", "gold"),
 ("Sampling", "the model card's row, never greedy (#111)", "sub"),
 ("Output integrity", "#91 fixed 2026-09-23: 60 to 0 corrupt tokens", "ok"),
 ("Linux + Windows", "both models run on both; 27B speed measured on Linux", "gold"),
]
for i, (ti, d, acc) in enumerate(FEATURES):
    fx, fy = X0 + (i % 2) * 405, y + (i // 2) * 80
    card(fx, fy, 395, 70)
    o.append(f'<circle cx="{fx+22}" cy="{fy+26}" r="4" fill="{C[acc]}"/>')
    t(fx + 36, fy + 31, ti, 16, C["text"], weight=600)
    t(fx + 36, fy + 53, escape(d), 12.5, C["faint"])
y += (len(FEATURES) + 1) // 2 * 80 + 20

# ---------------------------------------------------------------- measured
y = section(y, "Measured, not claimed.", "gold")
MEAS = [("60 → 0", C["ok"], "corrupt tokens, live agent|run 2026-09-23 (#91)"),
        ("4/23", C["gold"], "corruption set 2026-09-23,|was 15/23; llama.cpp 4/23"),
        ("45.1 / 44.9", C["sub"], "tok/s decode vs llama.cpp,|before the PLE fix, 09-13/14"),
        ("bit-identical", C["mark"], "greedy ids run to run,|before the PLE fix, 09-13/14"),
        ("0.290 → 0.223", C["ok"], "27B KLD vs BF16, calibrated|scales, 2026-09-26"),
        ("120 / 123.05", C["gold"], "27B MTP decode vs llama.cpp|MTP, 2026-09-27 (#95)"),
        ("byte-identical", C["sub"], "27B MTP output against|plain decode, 2026-09-27"),
        ("11 / 11", C["mark"], "27B colour probe, own F16|projector, 2026-09-27 (#122)")]
MROWS = (len(MEAS) + 3) // 4
card(X0, y, 800, 60 + MROWS * 100)
for i, (v, col, d) in enumerate(MEAS):
    mx, my = X0 + 24 + (i % 4) * 194, y + (i // 4) * 100
    t(mx, my + 46, v, 19, col, MONO, 600)
    for k, part in enumerate(d.split("|")):
        t(mx, my + 74 + k * 18, escape(part), 12.5, C["faint"])
t(X0 + 24, y + MROWS * 100 + 28, "Every figure has an issue or a release note behind it. Full table: docs/status.md", 13, C["soft"])
y += MROWS * 100 + 100

# ---------------------------------------------------------------- requirements
y = section(y, "Requirements", "sub", "one GPU, either model")
REQ = [("GPU", "NVIDIA Blackwell sm_120, RTX 5090 32 GB"),
       ("Host RAM", "64 GB"),
       ("CUDA", "13.3 runtime (NVRTC)"),
       ("Rust", "stable"),
       ("OS", "Linux, Windows; 27B speed measured on Linux"),
       ("Container", "Qwen3.8-Flash-Next-CNQ4.5-M.cnq, 104.7 GB, Hugging Face"),
       ("27B container", "Qwen3.8-27B-CNQ4.5.cnq, 17.8 GB, Hugging Face"),
       ("27B vision", "models/Qwen3.8-27B/mmproj-F16.gguf, the F16 projector")]
card(X0, y, 800, 24 + len(REQ) * 34)
for i, (g, v) in enumerate(REQ):
    ry = y + 20 + i * 34
    if i:
        o.append(f'<line x1="{X0+20}" y1="{ry-12}" x2="{X1-20}" y2="{ry-12}" stroke="{C["line"]}" stroke-dasharray="2 4"/>')
    t(X0 + 24, ry + 10, g, 13.5, C["text"], weight=600)
    t(X0 + 140, ry + 10, escape(v), 13, C["sub"], MONO)
y += 24 + len(REQ) * 34 + 40

# ---------------------------------------------------------------- against llama.cpp
y = section(y, "Against llama.cpp", "mark", "same card, same models")
OPS = [("Flash-Next decode, Windows", "45.1†", "44.9", "tok/s, same prompt"),
       ("Flash-Next prefill, Windows", "771†", "922.5", "tok/s, 16k reference prompt"),
       ("Flash-Next prefill, Linux", "968†", "", "tok/s, cold, same 16k prompt"),
       ("Flash-Next decode, Linux", "35.8", "", "tok/s at 122k, after the fix"),
       ("27B decode, Linux", "72.3", "66.5", "tok/s, plain"),
       ("27B decode + MTP, Linux", "120.2", "123.05", "tok/s, speculative"),
       ("27B decode at 30k, Linux", "58.9", "", "tok/s plain; 72.1 with MTP")]
OPS_NOTE = ("† before the PLE fix (2026-09-23), not re-measured: Windows 2026-09-13/14 (#62, #10), Linux 2026-09-17. 35.8: 2026-09-24.",
            "27B: 2026-09-27 (#95), median of 3, 512 tokens; llama.cpp = Crow's 27B UD-Q4_K_XL point on its own Crow turn (Crow #118).")
card(X0, y, 800, 40 + len(OPS) * 44 + 60)
for hx, h in ((64, "measure"), (300, "crow-nest"), (430, "llama.cpp"), (560, "unit")):
    t(hx, y + 26, h, 11.5, C["dim"], ls=1)
for i, (a, cn, lc, u) in enumerate(OPS):
    ry = y + 40 + i * 44
    o.append(f'<line x1="{X0+20}" y1="{ry}" x2="{X1-20}" y2="{ry}" stroke="{C["line"]}"/>')
    t(64, ry + 28, a, 14, C["text"])
    t(300, ry + 28, cn, 15, C["mark"], MONO, 600)
    t(430, ry + 28, lc or "—", 15, C["soft"], MONO, 600)
    t(560, ry + 28, u, 13, C["faint"])
for k, line in enumerate(OPS_NOTE):
    t(X0 + 24, y + 40 + len(OPS) * 44 + 26 + k * 16, line, 11.5, C["dim"])
y += 40 + len(OPS) * 44 + 60 + 40

# ---------------------------------------------------------------- run pointer
y = section(y, "Build and run", "ok")
card(X0, y, 800, 140, C["term"], C["bevel"], 12)
t(X0 + 24, y + 38, "Build, run and ask. Copy it right below this picture.", 16, C["text"], weight=600)
t(X0 + 24, y + 66, "cargo build, then serve on port 8099. Crow connects with --base-url.", 13, C["faint"])
t(X0 + 24, y + 90, "The 104.7 GB Flash-Next container downloads from Hugging Face.", 13, C["faint"])
t(X0 + 24, y + 114, "The 17.8 GB 27B container downloads from Hugging Face too.", 13, C["faint"])
t(X1 - 30, y + 66, "↓", 40, C["ok"], anchor="end")
y += 180


def wrap(text, n):
    lines, cur = [], ""
    for w in text.split():
        if cur and len(cur) + 1 + len(w) > n:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    return lines + ([cur] if cur else [])


def mobile():
    """One column at 440 px, drawn from the same lists as the desktop image above."""
    global W, X0, X1
    W, X0, X1 = 440, 10, 430
    CW = X1 - X0
    o.clear()

    def msection(y, title, acc):
        o.append(f'<rect x="{X0}" y="{y}" width="4" height="22" rx="2" fill="{C[acc]}"/>')
        t(X0 + 14, y + 17, escape(title), 19, C["text"], weight=600)
        return y + 38

    y = 10
    card(X0, y, CW, 350, r=14)
    o.append(f'<svg x="{W/2 - 110:.0f}" y="{y - 14}" width="220" height="220" viewBox="0 0 1024 1024">{mark()}</svg>')
    t(W / 2, y + 222, "CROW-NEST", 40, C["text"], weight=300, anchor="middle", ls=8)
    t(W / 2, y + 248, "INFERENCE ENGINE", 12.5, C["faint"], anchor="middle", ls=5)
    o.append(f'<text x="{W/2}" y="{y + 284}" text-anchor="middle" font-family="{UI}" font-size="18" fill="{C["soft"]}">'
             f'One GPU, its own quant, two model families.<tspan fill="{C["mark"]}">▍<animate attributeName="opacity" values="1;1;0;0" '
             f'keyTimes="0;.5;.5;1" dur="1.1s" repeatCount="indefinite"/></tspan></text>')
    pills = (("v" + VERSION, "mark"), ("Apache-2.0", "faint"), ("Linux · Windows · sm_120", "sub"))
    tot = sum(22 + len(l) * 7.6 for l, _ in pills) + 10 * (len(pills) - 1)
    x = W / 2 - tot / 2
    for label, acc in pills:
        x = pill(x, y + 304, label, C[acc])
    y += 350 + 26

    intro = ("An inference engine for one GPU: its own quantization, its own container, thin CUDA kernels in "
             "Rust. OpenAI-compatible HTTP, and the engine behind Crow. Two models from the original "
             "safetensors: Qwen3.8-Flash-Next as CNQ4.5-M and the dense Qwen3.8-27B as CNQ4.5.")
    for i, line in enumerate(wrap(intro, 46)):
        t(W / 2, y + i * 23, line, 16, C["soft"], anchor="middle")
    y += len(wrap(intro, 46)) * 23 + 14

    for i, (v, l) in enumerate(STATS):
        sx, sy = X0 + (i % 2) * 214, y + (i // 2) * 84
        last = i == len(STATS) - 1 and i % 2 == 0  # an odd card out spans both columns
        card(sx, sy, CW if last else 206, 76, C["raised"], C["raised"])
        t(sx + 16, sy + 36, v, 24, C["text"], MONO, 600)
        t(sx + 16, sy + 59, escape(l), 13, C["faint"])
    y += (len(STATS) + 1) // 2 * 84 + 8
    note = " ".join(STATS_NOTE)
    for i, line in enumerate(wrap(note, 62)):
        t(W / 2, y + i * 16, line, 11.5, C["dim"], anchor="middle")
    y += len(wrap(note, 62)) * 16 + 30

    y = msection(y, "Features", "ok")
    for i, (ti, d, acc) in enumerate(FEATURES):
        fy = y + i * 74
        card(X0, fy, CW, 66)
        o.append(f'<circle cx="{X0+20}" cy="{fy+25}" r="4.5" fill="{C[acc]}"/>')
        t(X0 + 34, fy + 30, ti, 17, C["text"], weight=600)
        t(X0 + 34, fy + 52, escape(d), 14, C["faint"])
    y += len(FEATURES) * 74 + 26

    y = msection(y, "Measured, not claimed.", "gold")
    mh = (len(MEAS) + 1) // 2 * 104
    card(X0, y, CW, mh + 54)
    for i, (v, col, d) in enumerate(MEAS):
        mx, my = X0 + 18 + (i % 2) * 206, y + 16 + (i // 2) * 104
        t(mx, my + 24, v, 18, col, MONO, 600)
        for k, part in enumerate(d.split("|")):
            t(mx, my + 50 + k * 19, escape(part), 13, C["faint"])
    for k, line in enumerate(wrap("Every figure has an issue or a release note behind it. Full table: docs/status.md", 52)):
        t(X0 + 18, y + mh + 22 + k * 19, line, 13.5, C["soft"])
    y += mh + 54 + 30

    y = msection(y, "Requirements", "sub")
    lines = []
    for g, v in REQ:
        lines.append(("g", g))
        lines += [("n", part) for part in wrap(v, 40)]
    h = 16 + sum(28 if k == "g" else 22 for k, _ in lines) + 12
    card(X0, y, CW, h)
    ly = y + 16
    for k, v in lines:
        if k == "g":
            if ly > y + 20:
                o.append(f'<line x1="{X0+16}" y1="{ly-2}" x2="{X1-16}" y2="{ly-2}" stroke="{C["line"]}" stroke-dasharray="2 4"/>')
            t(X0 + 18, ly + 20, v, 15, C["text"], weight=600); ly += 28
        else:
            t(X0 + 18, ly + 16, escape(v), 14, C["sub"], MONO); ly += 22
    y += h + 30

    y = msection(y, "Against llama.cpp", "mark")
    nl = [l for n in OPS_NOTE for l in wrap(n, 60)]
    card(X0, y, CW, 20 + len(OPS) * 62 + 16 + len(nl) * 16)
    for i, (a, cn, lc, u) in enumerate(OPS):
        ry = y + 14 + i * 62
        if i:
            o.append(f'<line x1="{X0+16}" y1="{ry}" x2="{X1-16}" y2="{ry}" stroke="{C["line"]}"/>')
        t(X0 + 18, ry + 26, a, 15.5, C["text"])
        t(X1 - 18, ry + 26, cn + (" vs " + lc if lc else ""), 15.5, C["mark"], MONO, 600, anchor="end")
        t(X0 + 18, ry + 49, u, 13.5, C["faint"])
        t(X1 - 18, ry + 49, "crow-nest" + (" vs llama.cpp" if lc else ""), 13, C["dim"], anchor="end")
    ny = y + 20 + len(OPS) * 62 + 8
    for k, line in enumerate(nl):
        t(X0 + 18, ny + k * 16, line, 11.5, C["dim"])
    y += 20 + len(OPS) * 62 + 16 + len(nl) * 16 + 30

    y = msection(y, "Build and run", "ok")
    body = wrap("cargo build, then serve on port 8099. Crow connects with --base-url. The 104.7 GB Flash-Next container "
                "downloads from Hugging Face; the 17.8 GB 27B container downloads from Hugging Face too.", 50)
    ih = 58 + len(body) * 20 + 14
    card(X0, y, CW, ih, C["term"], C["bevel"], 12)
    t(X0 + 18, y + 34, "Copy it right below this picture.", 17, C["text"], weight=600)
    for k, line in enumerate(body):
        t(X0 + 18, y + 62 + k * 20, line, 14, C["faint"])
    t(X1 - 20, y + 36, "↓", 30, C["ok"], anchor="end")
    return y + ih + 12


if MOBILE:
    y = mobile()
H = y - 20

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
       f'aria-label="crow-nest: one GPU, its own quant, two model families. Features, measurements, requirements.">'
       + "".join(o) + "</svg>\n")
(OUT / ("crow-nest-" + ("mobile-" if MOBILE else "") + VARIANT + ".svg")).write_text(svg)
