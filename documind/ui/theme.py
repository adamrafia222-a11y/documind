"""The look of the Streamlit front end: one CSS (colours, cards, buttons, forms, tabs, animations) and small HTML pieces (logo, avatar, stat cards, ring, scan animation).

Pure strings, no Streamlit import. Everything the person typed (names, file names, texts) goes through html.escape before it is put in a piece of HTML.
Animations are CSS only (no script, no external font, no CDN): they work offline, and `prefers-reduced-motion` switches them off.
"""
from __future__ import annotations

import hashlib
from html import escape

TYPE_COLOR = {"invoice": "#60a5fa", "receipt": "#fbbf24", "cheque": "#34d399", "contract": "#a78bfa"}
TYPE_ICON = {"invoice": "🧾", "receipt": "🛒", "cheque": "🏦", "contract": "📑"}
TYPE_LABEL = {"invoice": "Invoice", "receipt": "Receipt", "cheque": "Cheque", "contract": "Contract"}

CSS = """
<style>
:root{
  --bg:#080d1c; --surface:#101831; --surface2:#16203f; --line:rgba(255,255,255,.09); --text:#e8ecff; --muted:#9aa6cf;
  --a1:#6366f1; --a2:#8b5cf6; --a3:#22d3ee; --ok:#34d399; --warn:#fbbf24; --bad:#f87171; --radius:18px;
  --grad:linear-gradient(135deg,#6366f1 0%,#8b5cf6 55%,#22d3ee 130%);
}
@property --n{syntax:'<integer>';inherits:false;initial-value:0}
@property --p{syntax:'<number>';inherits:false;initial-value:0}

/* ---------- page ---------- */
.stApp{
  background:
    radial-gradient(900px 500px at 8% -10%, rgba(99,102,241,.28), transparent 60%),
    radial-gradient(800px 480px at 100% 0%, rgba(34,211,238,.16), transparent 55%),
    radial-gradient(700px 600px at 50% 120%, rgba(139,92,246,.20), transparent 60%),
    var(--bg);
  background-attachment:fixed;
  color:var(--text);
}
header[data-testid="stHeader"]{background:transparent}
#MainMenu,footer{visibility:hidden}
.block-container{padding-top:2.2rem;padding-bottom:4rem;max-width:1220px}
h1,h2,h3,h4{letter-spacing:-.02em}
a{color:var(--a3)}
hr{border-color:var(--line)}

/* ---------- animations ---------- */
@keyframes dm-up{from{opacity:0;transform:translateY(18px)}to{opacity:1;transform:none}}
@keyframes dm-in{from{opacity:0}to{opacity:1}}
@keyframes dm-float{0%,100%{transform:translateY(0)}50%{transform:translateY(-10px)}}
@keyframes dm-pulse{0%,100%{box-shadow:0 0 0 0 rgba(99,102,241,.45)}60%{box-shadow:0 0 0 14px rgba(99,102,241,0)}}
@keyframes dm-scan{0%{top:6%}50%{top:90%}100%{top:6%}}
@keyframes dm-shimmer{0%{background-position:-400px 0}100%{background-position:400px 0}}
@keyframes dm-spin{to{transform:rotate(360deg)}}
@keyframes dm-pop{0%{transform:scale(.6);opacity:0}70%{transform:scale(1.08)}100%{transform:scale(1);opacity:1}}
@keyframes dm-draw{to{stroke-dashoffset:0}}
@keyframes dm-count{from{--n:0}to{--n:var(--to)}}
@keyframes dm-ring{from{--p:0}to{--p:var(--to)}}
@keyframes dm-box{0%,12%{opacity:0;transform:scale(.9)}22%,88%{opacity:1;transform:none}100%{opacity:0}}
@keyframes dm-dots{0%{content:''}25%{content:'.'}50%{content:'..'}75%{content:'...'}}
@keyframes dm-cyc{0%,24%{color:#fff;background:var(--grad);border-color:transparent}30%,100%{color:var(--muted);background:transparent;border-color:var(--line)}}
@keyframes dm-bar{0%{transform:translateX(-100%)}100%{transform:translateX(250%)}}

.dm-up{animation:dm-up .6s cubic-bezier(.2,.7,.2,1) both}
.dm-d1{animation-delay:.08s}.dm-d2{animation-delay:.16s}.dm-d3{animation-delay:.24s}.dm-d4{animation-delay:.32s}
[data-testid="stVerticalBlock"]>[data-testid="stElementContainer"]{animation:dm-in .35s ease both}

/* ---------- cards ---------- */
.dm-card{background:linear-gradient(180deg,rgba(255,255,255,.055),rgba(255,255,255,.02));border:1px solid var(--line);border-radius:var(--radius);
  padding:1.1rem 1.25rem;backdrop-filter:blur(8px);transition:transform .25s,border-color .25s,box-shadow .25s}
.dm-card:hover{transform:translateY(-3px);border-color:rgba(139,92,246,.45);box-shadow:0 14px 40px -18px rgba(99,102,241,.55)}
.dm-title{font-size:2.05rem;font-weight:800;letter-spacing:-.03em;margin:0;background:var(--grad);-webkit-background-clip:text;background-clip:text;color:transparent}
.dm-sub{color:var(--muted);margin:.25rem 0 1.2rem}
.dm-muted{color:var(--muted)}
.dm-row{display:flex;gap:.8rem;flex-wrap:wrap}
.dm-grid{display:grid;gap:.9rem;grid-template-columns:repeat(auto-fit,minmax(190px,1fr))}

/* ---------- stat cards, ring, badges ---------- */
.dm-stat{display:flex;gap:.8rem;align-items:center}
.dm-stat .ico{font-size:1.5rem;width:46px;height:46px;border-radius:14px;display:grid;place-items:center;background:rgba(99,102,241,.16)}
.dm-stat .lab{font-size:.78rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
.dm-stat .val{font-size:1.65rem;font-weight:800;line-height:1.1}
.dm-count{animation:dm-count 1.2s ease-out both;counter-reset:dmn var(--n)}
.dm-count::before{content:counter(dmn)}
.dm-count>i{display:none}
.dm-ring{--c:var(--ok);width:74px;height:74px;border-radius:50%;display:grid;place-items:center;animation:dm-ring 1.1s ease-out both;
  background:conic-gradient(var(--c) calc(var(--p)*1%),rgba(255,255,255,.1) 0)}
.dm-ring::after{content:attr(data-label);width:58px;height:58px;border-radius:50%;background:var(--surface);display:grid;place-items:center;font-weight:700;font-size:.95rem}
.dm-badge{display:inline-flex;align-items:center;gap:.4rem;padding:.2rem .7rem;border-radius:999px;font-size:.8rem;font-weight:700;color:#0b1020;animation:dm-pop .45s both}
.dm-chip{display:inline-flex;align-items:center;gap:.35rem;padding:.2rem .7rem;border-radius:999px;font-size:.8rem;border:1px solid var(--line);background:rgba(255,255,255,.04);color:var(--muted)}
.dm-chip.ok{color:var(--ok);border-color:rgba(52,211,153,.4)}.dm-chip.warn{color:var(--warn);border-color:rgba(251,191,36,.45)}.dm-chip.bad{color:var(--bad);border-color:rgba(248,113,113,.45)}
.dm-avatar{border-radius:50%;display:grid;place-items:center;font-weight:800;color:white;flex:none;animation:dm-pop .5s both}
.dm-brand{display:flex;align-items:center;gap:.7rem;margin-bottom:.4rem}
.dm-brand svg{animation:dm-float 4s ease-in-out infinite;filter:drop-shadow(0 6px 14px rgba(99,102,241,.55))}
.dm-brand b{font-size:1.45rem;letter-spacing:-.03em;background:var(--grad);-webkit-background-clip:text;background-clip:text;color:transparent}
.dm-brand small{display:block;color:var(--muted);font-size:.72rem;margin-top:-.15rem}

/* ---------- hero (sign-in page) and the animated document ---------- */
.dm-hero h1{font-size:2.9rem;line-height:1.05;font-weight:850;letter-spacing:-.04em;margin:.6rem 0 .9rem}
.dm-hero h1 .g{background:var(--grad);-webkit-background-clip:text;background-clip:text;color:transparent}
.dm-hero p{color:var(--muted);font-size:1.05rem;max-width:34rem}
.dm-doc{position:relative;width:min(340px,100%);aspect-ratio:3/4;margin:1.4rem 0 .4rem;border-radius:20px;padding:20px;overflow:hidden;
  background:linear-gradient(180deg,#f7f8ff,#dfe4ff);box-shadow:0 30px 60px -25px rgba(0,0,0,.7),0 0 0 1px rgba(255,255,255,.15);animation:dm-float 6s ease-in-out infinite}
.dm-doc .ln{height:9px;border-radius:5px;background:#c3cbf2;margin:11px 0}
.dm-doc .ln.s{width:55%}.dm-doc .ln.m{width:78%}.dm-doc .ln.t{height:16px;width:62%;background:#8c97d8}
.dm-doc .bx{position:absolute;border:2px solid;border-radius:6px;font-size:10px;font-weight:700;color:white;padding:0 5px;animation:dm-box 6s infinite}
.dm-doc .scan{position:absolute;left:0;right:0;height:46px;top:6%;background:linear-gradient(180deg,rgba(99,102,241,0),rgba(99,102,241,.55),rgba(34,211,238,.0));
  border-bottom:2px solid #22d3ee;animation:dm-scan 3.4s ease-in-out infinite}
.dm-feat{display:flex;gap:.6rem;flex-wrap:wrap;margin-top:1rem}

/* ---------- reading animation ---------- */
.dm-read{display:flex;gap:1.1rem;align-items:center;padding:1.1rem 1.3rem}
.dm-spinner{width:44px;height:44px;border-radius:50%;border:4px solid rgba(255,255,255,.12);border-top-color:var(--a3);animation:dm-spin .9s linear infinite;flex:none}
.dm-steps{display:flex;gap:.5rem;flex-wrap:wrap;margin-top:.5rem}
.dm-step{font-size:.78rem;padding:.15rem .6rem;border-radius:999px;border:1px solid var(--line);color:var(--muted)}
.dm-step.cyc{animation:dm-cyc 4.8s infinite}
.dm-step.on{color:white;border-color:transparent;background:var(--grad);animation:dm-pulse 1.5s infinite}
.dm-step.done{color:var(--ok);border-color:rgba(52,211,153,.4)}
.dm-progress{height:6px;border-radius:99px;background:rgba(255,255,255,.08);overflow:hidden;margin-top:.7rem;width:100%}
.dm-progress>i{display:block;height:100%;width:40%;border-radius:99px;background:var(--grad);animation:dm-bar 1.3s ease-in-out infinite}
.dm-skel{height:14px;border-radius:8px;margin:.5rem 0;background:linear-gradient(90deg,rgba(255,255,255,.05) 25%,rgba(255,255,255,.14) 50%,rgba(255,255,255,.05) 75%);background-size:800px 100%;animation:dm-shimmer 1.4s infinite}
.dm-check{width:46px;height:46px;flex:none}
.dm-check circle{fill:rgba(52,211,153,.14);stroke:var(--ok);stroke-width:2}
.dm-check path{fill:none;stroke:var(--ok);stroke-width:3;stroke-linecap:round;stroke-linejoin:round;stroke-dasharray:40;stroke-dashoffset:40;animation:dm-draw .6s .15s ease forwards}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0d1430,#0a1024);border-right:1px solid var(--line)}
section[data-testid="stSidebar"] .stButton>button{width:100%;justify-content:flex-start;border:1px solid transparent;background:transparent;color:var(--muted);
  border-radius:12px;padding:.55rem .8rem;font-weight:600;transition:all .2s}
section[data-testid="stSidebar"] .stButton>button:hover{background:rgba(255,255,255,.06);color:var(--text);transform:translateX(4px)}
section[data-testid="stSidebar"] .stButton>button[kind="primary"],section[data-testid="stSidebar"] .stButton>button[data-testid="stBaseButton-primary"]{
  background:var(--grad);color:white;box-shadow:0 10px 24px -12px rgba(99,102,241,.9)}
.dm-user{display:flex;gap:.7rem;align-items:center;padding:.7rem;border-radius:14px;background:rgba(255,255,255,.05);border:1px solid var(--line);margin:.6rem 0 1rem}
.dm-user b{display:block;font-size:.92rem}.dm-user small{color:var(--muted);font-size:.74rem;word-break:break-all}

/* ---------- widgets ---------- */
.stButton>button,.stDownloadButton>button,[data-testid="stFormSubmitButton"]>button{border-radius:12px;font-weight:700;border:1px solid var(--line);transition:transform .2s,box-shadow .2s,filter .2s}
.stButton>button:hover,.stDownloadButton>button:hover,[data-testid="stFormSubmitButton"]>button:hover{transform:translateY(-2px);box-shadow:0 12px 26px -14px rgba(99,102,241,.9);border-color:rgba(139,92,246,.6)}
.stButton>button[kind="primary"],[data-testid="stBaseButton-primary"],[data-testid="stBaseButton-primaryFormSubmit"],.stFormSubmitButton>button[kind="primaryFormSubmit"]{
  background:var(--grad)!important;color:white!important;border:0!important}
.stButton>button[kind="primary"]:hover,[data-testid="stBaseButton-primary"]:hover{filter:brightness(1.1)}
.stTextInput input,.stTextArea textarea,.stNumberInput input,.stDateInput input,[data-baseweb="select"]>div{background:rgba(255,255,255,.05)!important;border-radius:12px!important;
  border:1px solid var(--line)!important;color:var(--text)!important}
.stTextInput input:focus,.stTextArea textarea:focus{border-color:var(--a2)!important;box-shadow:0 0 0 3px rgba(139,92,246,.25)!important}
[data-testid="stForm"]{background:linear-gradient(180deg,rgba(255,255,255,.06),rgba(255,255,255,.02));border:1px solid var(--line);border-radius:22px;padding:1.4rem 1.4rem .8rem;
  backdrop-filter:blur(10px);box-shadow:0 30px 60px -35px rgba(0,0,0,.8);animation:dm-up .6s both}
[data-testid="stFormSubmitButton"]>button{width:100%}
.stTabs [data-baseweb="tab-list"]{gap:.3rem;border-bottom:1px solid var(--line)}
.stTabs [data-baseweb="tab"]{border-radius:10px 10px 0 0;padding:.5rem 1rem;transition:background .2s}
.stTabs [data-baseweb="tab"]:hover{background:rgba(255,255,255,.05)}
.stTabs [aria-selected="true"]{color:var(--a3)!important}
[data-testid="stMetric"]{background:rgba(255,255,255,.04);border:1px solid var(--line);border-radius:16px;padding:.8rem 1rem;transition:transform .25s}
[data-testid="stMetric"]:hover{transform:translateY(-3px)}
[data-testid="stFileUploaderDropzone"]{border:2px dashed rgba(139,92,246,.55);border-radius:20px;background:rgba(99,102,241,.06);padding:1.6rem;transition:all .25s}
[data-testid="stFileUploaderDropzone"]:hover{background:rgba(99,102,241,.14);border-color:var(--a3);animation:dm-pulse 1.8s infinite}
[data-testid="stAlert"]{border-radius:14px;animation:dm-up .4s both}
[data-testid="stExpander"]{border:1px solid var(--line);border-radius:14px;background:rgba(255,255,255,.03)}
[data-testid="stDataFrame"]{border-radius:14px;overflow:hidden;border:1px solid var(--line)}
[data-testid="stImage"] img{border-radius:14px;box-shadow:0 18px 40px -22px rgba(0,0,0,.9)}
.stProgress>div>div>div>div{background:var(--grad)}

/* ---------- charts (documind.ui.charts) ---------- */
@keyframes dm-draw1{to{stroke-dashoffset:0}}
@keyframes dm-sweep{from{stroke-dasharray:0 var(--c)}to{stroke-dasharray:var(--len) var(--c)}}
@keyframes dm-grow{from{width:0}to{width:calc(var(--w)*1%)}}
@keyframes dm-rise{from{transform:scaleY(0)}to{transform:scaleY(1)}}
@keyframes dm-cell{from{opacity:0;transform:scale(.3)}to{opacity:1;transform:none}}
@keyframes dm-slide{from{opacity:0;transform:translateX(-26px)}to{opacity:1;transform:none}}
@keyframes dm-drift{0%,100%{transform:translate(0,0) scale(1)}33%{transform:translate(40px,-30px) scale(1.08)}66%{transform:translate(-30px,25px) scale(.94)}}
@keyframes dm-textshine{to{background-position:200% center}}
@keyframes dm-rot{0%,18%{transform:translateY(0)}25%,43%{transform:translateY(-20%)}50%,68%{transform:translateY(-40%)}75%,93%{transform:translateY(-60%)}100%{transform:translateY(-80%)}}
@keyframes dm-confetti{0%{opacity:1;transform:translate(0,0) rotate(0)}100%{opacity:0;transform:translate(var(--x),var(--y)) rotate(540deg)}}
@keyframes dm-sheen{0%{left:-60%}60%,100%{left:130%}}

.dm-chart h4{margin:0 0 .15rem;font-size:1.02rem}.dm-chart .body{margin-top:.8rem}
.dm-spark{overflow:visible;flex:none}
.dm-spark .line{stroke-dasharray:1;stroke-dashoffset:1;animation:dm-draw1 1.4s ease forwards}
.dm-spark .area{animation:dm-in 1.2s .4s both}
.dm-spark .dot{transform-box:fill-box;transform-origin:center;animation:dm-pop .4s 1.3s both}
.dm-kpi .top{display:flex;gap:.6rem;align-items:center}
.dm-kpi .ico{width:38px;height:38px;border-radius:12px;display:grid;place-items:center;background:rgba(99,102,241,.16);font-size:1.2rem;transition:transform .3s}
.dm-kpi:hover .ico{transform:rotate(-8deg) scale(1.14)}
.dm-kpi .lab{font-size:.76rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted)}
.dm-kpi .val{font-size:2.15rem;font-weight:850;line-height:1.1;margin:.4rem 0 .15rem}
.dm-kpi .foot{display:flex;justify-content:space-between;align-items:flex-end;gap:.5rem}
.dm-delta{display:inline-block;font-size:.74rem;font-weight:700;padding:.1rem .5rem;border-radius:99px;margin-right:.4rem;background:rgba(255,255,255,.08);color:var(--muted)}
.dm-delta.up{background:rgba(52,211,153,.16);color:var(--ok)}.dm-delta.down{background:rgba(248,113,113,.16);color:var(--bad)}
.dm-donut{display:flex;gap:1.3rem;align-items:center;flex-wrap:wrap}
.dm-donut .ring{position:relative;flex:none}
.dm-donut .seg{animation:dm-sweep 1.1s cubic-bezier(.3,.7,.2,1) both}
.dm-donut .mid{position:absolute;inset:0;display:grid;place-content:center;text-align:center}
.dm-donut .mid b{font-size:1.7rem;font-weight:850;line-height:1}.dm-donut .mid small{color:var(--muted)}
.dm-legend{flex:1;min-width:150px}
.dm-legend .row{display:grid;grid-template-columns:12px 1fr auto 46px;gap:.55rem;align-items:center;margin:.35rem 0;animation:dm-slide .5s both}
.dm-legend .row i{width:12px;height:12px;border-radius:4px}.dm-legend .row small{color:var(--muted);text-align:right}
.dm-hbars{display:flex;flex-direction:column}
.dm-hbar{display:grid;grid-template-columns:minmax(90px,40%) 1fr auto;gap:.65rem;align-items:center;margin:.42rem 0}
.dm-hbar .l{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:.86rem;color:var(--muted)}
.dm-hbar .t{height:10px;border-radius:99px;background:rgba(255,255,255,.07);overflow:hidden}
.dm-hbar .t i{display:block;height:100%;width:calc(var(--w)*1%);border-radius:99px;animation:dm-grow 1s cubic-bezier(.3,.7,.2,1) both}
.dm-hbar b{font-size:.86rem;min-width:2.4rem;text-align:right}
.dm-hist{display:flex;gap:6px}
.dm-hist .col{flex:1;min-width:0;display:flex;flex-direction:column;align-items:center;gap:4px}
.dm-hist .col small{height:1rem;color:var(--muted);font-size:.72rem}
.dm-hist .bar{flex:1;width:100%;display:flex;align-items:flex-end}
.dm-hist .bar i{display:block;width:100%;min-height:3px;border-radius:8px 8px 3px 3px;transform-origin:bottom;animation:dm-rise .9s cubic-bezier(.3,.7,.2,1) both}
.dm-hist .col span{font-size:.7rem;color:var(--muted)}
.dm-heat{display:grid;grid-template-columns:38px repeat(24,minmax(0,1fr));gap:3px;align-items:center}
.dm-heat .c{aspect-ratio:1;border-radius:4px;background:rgba(139,92,246,var(--a));animation:dm-cell .5s both}
.dm-heat .c:hover{outline:2px solid var(--a3)}
.dm-heat .d,.dm-heat .h{font-size:.68rem;color:var(--muted)}.dm-heat .h{text-align:center}
.dm-feed .item{display:flex;gap:.8rem;align-items:center;padding:.55rem .2rem;border-bottom:1px solid var(--line);animation:dm-slide .5s both}
.dm-feed .item:last-child{border-bottom:0}
.dm-feed .nm{flex:1;min-width:0}.dm-feed .nm b{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.dm-feed .nm small{color:var(--muted)}
.dm-feed .cf{font-weight:800;color:var(--ok)}

/* ---------- more motion: drifting light, shimmering titles, sheen on buttons, rotating word, confetti, live dot ---------- */
.stApp{isolation:isolate}
.dm-orb{position:fixed;border-radius:50%;filter:blur(70px);opacity:.33;pointer-events:none;z-index:-1;animation:dm-drift 22s ease-in-out infinite}
.dm-orb.o1{width:380px;height:380px;left:-90px;top:-70px;background:#6366f1}
.dm-orb.o2{width:440px;height:440px;right:-120px;bottom:-110px;background:#22d3ee;animation-delay:-7s}
.dm-orb.o3{width:300px;height:300px;left:42%;top:38%;background:#8b5cf6;animation-delay:-14s;opacity:.2}
.dm-title,.dm-hero h1 .g,.rot b{background-image:linear-gradient(90deg,#818cf8,#a78bfa,#22d3ee,#818cf8);background-size:200% auto;animation:dm-textshine 6s linear infinite}
.stButton>button[kind="primary"],[data-testid="stBaseButton-primary"],[data-testid="stBaseButton-primaryFormSubmit"]{position:relative;overflow:hidden}
.stButton>button[kind="primary"]::after,[data-testid="stBaseButton-primary"]::after,[data-testid="stBaseButton-primaryFormSubmit"]::after{content:"";position:absolute;top:0;bottom:0;left:-60%;width:40%;
  background:linear-gradient(100deg,transparent,rgba(255,255,255,.35),transparent);transform:skewX(-20deg);animation:dm-sheen 3.4s infinite;pointer-events:none}
.rot{display:inline-block;height:1.1em;overflow:hidden;vertical-align:bottom}
.rot .rin{display:block;animation:dm-rot 10s cubic-bezier(.7,0,.3,1) infinite}
.rot b{display:block;height:1.1em;line-height:1.1em;font-weight:inherit;-webkit-background-clip:text;background-clip:text;color:transparent}
.dm-confetti{position:relative}
.dm-confetti .cf{position:absolute;left:34px;top:50%;width:7px;height:11px;border-radius:2px;background:var(--c);animation:dm-confetti 1.3s ease-out both;animation-delay:var(--d);pointer-events:none}
.dm-live{display:inline-block;width:8px;height:8px;border-radius:50%;background:var(--ok);margin-right:.45rem;animation:dm-pulse 1.8s infinite}

@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}.dm-count>i{display:inline}.dm-count::before{content:none}}
@media (max-width:760px){.dm-hero h1{font-size:2.1rem}.dm-doc{display:none}}
</style>
"""

AUTH_CSS = """<style>section[data-testid="stSidebar"],[data-testid="stSidebarCollapsedControl"],[data-testid="collapsedControl"]{display:none}</style>"""

LOGO_SVG = """<svg width="40" height="40" viewBox="0 0 48 48" aria-hidden="true"><defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1">
<stop offset="0" stop-color="#6366f1"/><stop offset=".6" stop-color="#8b5cf6"/><stop offset="1" stop-color="#22d3ee"/></linearGradient></defs>
<rect x="8" y="4" width="28" height="38" rx="6" fill="url(#g)"/><rect x="14" y="13" width="16" height="3" rx="1.5" fill="#fff" opacity=".9"/>
<rect x="14" y="20" width="12" height="3" rx="1.5" fill="#fff" opacity=".7"/><rect x="14" y="27" width="16" height="3" rx="1.5" fill="#fff" opacity=".7"/>
<circle cx="35" cy="35" r="9" fill="#0a1024" stroke="#22d3ee" stroke-width="3"/><path d="M31 35l3 3 5-6" fill="none" stroke="#22d3ee" stroke-width="2.6" stroke-linecap="round" stroke-linejoin="round"/></svg>"""


def brand_html(tagline: str = "Read. Check. Export.") -> str:
    return f'<div class="dm-brand">{LOGO_SVG}<div><b>DocuMind</b><small>{escape(tagline)}</small></div></div>'


def initials(name: str) -> str:
    parts = [p for p in (name or "?").replace("@", " ").split() if p]
    return ((parts[0][0] + (parts[1][0] if len(parts) > 1 else "")) if parts else "?").upper()


def avatar_html(name: str, size: int = 44) -> str:
    h = int(hashlib.md5((name or "?").encode()).hexdigest()[:6], 16) % 360
    return (f'<div class="dm-avatar" style="width:{size}px;height:{size}px;font-size:{size * 0.38:.0f}px;'
            f'background:linear-gradient(135deg,hsl({h} 75% 55%),hsl({(h + 55) % 360} 80% 50%))">{escape(initials(name))}</div>')


def user_chip_html(name: str, email: str) -> str:
    return f'<div class="dm-user">{avatar_html(name, 40)}<div><b>{escape(name)}</b><small>{escape(email)}</small></div></div>'


def page_header_html(title: str, subtitle: str = "", icon: str = "") -> str:
    return (f'<div class="dm-up"><h1 class="dm-title">{escape(icon + " " if icon else "")}{escape(title)}</h1>'
            f'<p class="dm-sub">{escape(subtitle)}</p></div>')


def type_badge_html(doc_type: str) -> str:
    c = TYPE_COLOR.get(doc_type, "#9aa6cf")
    return f'<span class="dm-badge" style="background:{c}">{TYPE_ICON.get(doc_type, "📄")} {escape(TYPE_LABEL.get(doc_type, doc_type))}</span>'


def chip_html(text: str, kind: str = "") -> str:
    return f'<span class="dm-chip {kind}">{escape(text)}</span>'


def count_html(value: int) -> str:
    """A number that counts up from 0 (CSS @property animation). The real number stays in the text for browsers / readers that do not animate."""
    v = int(value)
    return f'<span class="dm-count" style="--to:{v}"><i>{v}</i></span>'


def stat_card_html(icon: str, label: str, value: str | int, sub: str = "", delay: int = 0) -> str:
    val = count_html(value) if isinstance(value, int) else escape(str(value))
    return (f'<div class="dm-card dm-up{f" dm-d{min(delay, 4)}" if delay > 0 else ""}"><div class="dm-stat"><div class="ico">{escape(icon)}</div><div><div class="lab">{escape(label)}</div>'
            f'<div class="val">{val}</div>{f"<small class=dm-muted>{escape(sub)}</small>" if sub else ""}</div></div></div>')


def stat_grid_html(cards: list[tuple]) -> str:
    return '<div class="dm-grid">' + "".join(stat_card_html(*c, delay=i + 1) for i, c in enumerate(cards)) + "</div>"


def ring_html(percent: float, label: str | None = None) -> str:
    """A circle that fills to `percent` (0-100); green above 85, amber above 60, red below."""
    p = max(0.0, min(100.0, float(percent)))
    color = "var(--ok)" if p >= 85 else "var(--warn)" if p >= 60 else "var(--bad)"
    return f'<div class="dm-ring" style="--to:{p:.0f};--c:{color}" data-label="{escape(label if label is not None else f"{p:.0f}%")}"></div>'


ORBS_HTML = '<div class="dm-orb o1"></div><div class="dm-orb o2"></div><div class="dm-orb o3"></div>'


def live_html(text: str) -> str:
    return f'<span class="dm-live"></span><small class="dm-muted">{escape(text)}</small>'


def hero_html() -> str:
    """Left side of the sign-in page: headline + an animated document being scanned, its fields lighting up one after the other."""
    boxes = [("12%", "9%", "58%", "7%", "#6366f1", "TITLE", "0s"), ("12%", "27%", "40%", "5%", "#22d3ee", "DATE", "1.1s"),
             ("12%", "44%", "64%", "5%", "#34d399", "AMOUNT", "2.2s"), ("12%", "61%", "50%", "5%", "#fbbf24", "IBAN", "3.3s"), ("12%", "78%", "36%", "5%", "#f472b6", "SIGNATURE", "4.4s")]
    bx = "".join(f'<div class="bx" style="left:{l};top:{t};width:{w};height:{h};border-color:{c};background:{c}33;animation-delay:{d}"><span style="background:{c};position:absolute;top:-15px;left:-2px;padding:0 5px;border-radius:4px">{n}</span></div>'
                 for l, t, w, h, c, n, d in boxes)
    lines = '<div class="ln t"></div><div class="ln m"></div><div class="ln s"></div>' * 1 + '<div class="ln m"></div><div class="ln"></div><div class="ln s"></div>' * 3
    return (f'<div class="dm-hero dm-up">{brand_html()}<h1>Read any <span class="rot"><span class="rin"><b>invoice</b><b>receipt</b><b>cheque</b><b>contract</b><b>invoice</b></span></span>.<br><span class="g">Check it. Export it.</span></h1>'
            '<p>Invoices, receipts, cheques and loan contracts: DocuMind finds every field, checks the numbers against each other and gives you a clean CSV for each type.</p>'
            f'<div class="dm-doc">{lines}{bx}<div class="scan"></div></div>'
            '<div class="dm-feat">' + "".join(chip_html(t) for t in ("🧾 Invoices", "🛒 Receipts", "🏦 Cheques", "📑 Contracts", "✅ Automatic checks", "📊 One CSV per type")) + '</div></div>')


def reading_html(filename: str, total: int = 1, index: int = 1) -> str:
    """Shown while a document is being read: spinner, the file name, the four stages (they light up one after the other, in CSS), a moving bar."""
    stages = ["Finding the type", "Locating the fields", "Reading the text", "Checking the values"]
    steps = "".join(f'<span class="dm-step cyc" style="animation-delay:{i * 1.2:.1f}s">{escape(s)}</span>' for i, s in enumerate(stages))
    return (f'<div class="dm-card dm-read dm-up"><div class="dm-spinner"></div><div style="flex:1"><b>Reading {escape(filename)}</b> '
            f'<span class="dm-muted">({index}/{total})</span><div class="dm-steps">{steps}</div><div class="dm-progress"><i></i></div></div></div>')


def done_html(message: str, confetti: bool = True) -> str:
    colors = ("#6366f1", "#8b5cf6", "#22d3ee", "#34d399", "#fbbf24", "#f472b6")
    bits = "".join(f'<i class="cf" style="--x:{(18 + (i * 37) % 120) * (1 if i % 2 else 0.6):.0f}px;--y:{-30 - (i * 53) % 70}px;--c:{colors[i % 6]};--d:{i * 0.04:.2f}s"></i>' for i in range(14)) if confetti else ""
    return (f'<div class="dm-card dm-read dm-up dm-confetti"><svg class="dm-check" viewBox="0 0 48 48"><circle cx="24" cy="24" r="22"/><path d="M14 25l7 7 13-15"/></svg>'
            f'<div><b>{escape(message)}</b></div>{bits}</div>')


def skeleton_html(rows: int = 4) -> str:
    return "".join(f'<div class="dm-skel" style="width:{w}%"></div>' for w in (92, 78, 85, 60, 70, 88)[:rows])


def empty_html(icon: str, title: str, text: str) -> str:
    return (f'<div class="dm-card dm-up" style="text-align:center;padding:2.2rem"><div style="font-size:2.6rem;animation:dm-float 4s ease-in-out infinite">{escape(icon)}</div>'
            f'<h3 style="margin:.4rem 0">{escape(title)}</h3><p class="dm-muted">{escape(text)}</p></div>')
