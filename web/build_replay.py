"""Build a self-contained side-by-side replay page (video + animated board).

Reads the reconstructed game (per-frame FEN + moves) and the clean video, and
writes web/replay_<game>.html that plays the real video on the left and drives
an animated digital board on the right, synced by time, with a move list and a
material score. No server required — just open the HTML.

Usage:
    python web/build_replay.py game1
"""

import json
import os
import sys

import chess

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

LABELS_DIR = "data/labels"
HOLD_SECONDS = 1.0  # must match dedupe_frames/make_video --hold used for the video


def load_game(game):
    path = os.path.join(LABELS_DIR, f"{game}_states_yolo.json")
    if not os.path.exists(path):
        sys.exit(f"❌ {path} not found. Run inference/reconstruct_game_yolo.py first.")
    return json.load(open(path))


def ensure_browser_video(game):
    """Return a browser-playable (H.264) video path, transcoding if needed.

    OpenCV writes mp4v, which browsers' <video> can't play; H.264 (yuv420p) can.
    """
    # If a browser-ready H.264 already exists, just use it.
    existing_h264 = os.path.join("data/videos", f"{game}_clean_h264.mp4")
    if os.path.exists(existing_h264):
        return existing_h264

    src = os.path.join("data/videos", f"{game}_clean.mp4")
    if not os.path.exists(src):
        src = os.path.join("data/videos", f"{game}.mp4")
    if not os.path.exists(src):
        sys.exit(f"❌ No video found for {game} in data/videos/.")
    h264 = os.path.splitext(src)[0] + "_h264.mp4"
    if not os.path.exists(h264) or os.path.getmtime(h264) < os.path.getmtime(src):
        import subprocess
        import imageio_ffmpeg
        ff = imageio_ffmpeg.get_ffmpeg_exe()
        print("🎞️  Transcoding video to H.264 for the browser ...")
        subprocess.run([ff, "-y", "-i", src, "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-movflags", "+faststart", h264],
                       capture_output=True)
    return h264


def compute_san(fens):
    """SAN for each transition fens[i-1] -> fens[i] (best-effort)."""
    san = [""]
    for i in range(1, len(fens)):
        prev = chess.Board(fens[i - 1])
        target = chess.Board(fens[i])
        move_san = ""
        for m in prev.legal_moves:
            prev.push(m)
            if prev.board_fen() == target.board_fen():
                prev.pop()
                move_san = prev.san(m)
                break
            prev.pop()
        san.append(move_san)
    return san


def main():
    game = sys.argv[1] if len(sys.argv) > 1 else "game1"
    data = load_game(game)
    fens = [f["fen"] for f in data["frames"]]
    san = compute_san(fens)

    h264 = ensure_browser_video(game)
    video_rel = "../" + h264  # relative to web/
    payload = json.dumps({"fens": fens, "san": san, "hold": HOLD_SECONDS,
                          "video": video_rel, "game": game})

    html = TEMPLATE.replace("__PAYLOAD__", payload)
    os.makedirs("web", exist_ok=True)
    out = os.path.join("web", f"replay_{game}.html")
    with open(out, "w") as f:
        f.write(html)
    print(f"✅ Wrote {out}  ({len(fens)} positions)")
    print(f"👉 Open it:  open {out}")
    if sys.platform == "darwin":
        os.system(f'open "{out}"')


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Chess Real-to-Animated — Replay</title>
<style>
  body { margin:0; font-family:-apple-system,Segoe UI,Roboto,sans-serif;
         background:#1e1f22; color:#e6e6e6; }
  header { padding:12px 18px; font-size:18px; font-weight:600;
           background:#2b2d31; border-bottom:1px solid #000; }
  .wrap { display:flex; gap:18px; padding:18px; align-items:flex-start;
          flex-wrap:wrap; }
  .panel { background:#2b2d31; border-radius:10px; padding:14px; }
  video { width:460px; max-width:46vw; border-radius:8px; background:#000; display:block; }
  .board { width:460px; height:460px; display:grid;
           grid-template-columns:repeat(8,1fr); grid-template-rows:repeat(8,1fr);
           border-radius:8px; overflow:hidden; }
  .sq { display:flex; align-items:center; justify-content:center;
        font-size:46px; line-height:1; user-select:none; }
  .light { background:#f0d9b5; } .dark { background:#b58863; }
  .sq.wp { color:#ffffff; -webkit-text-stroke:1.6px #2b2b2b; text-shadow:0 1px 1px rgba(0,0,0,.35); }
  .sq.bp { color:#141414; -webkit-text-stroke:1.6px #f0f0f0; text-shadow:0 1px 1px rgba(255,255,255,.25); }
  .sq.hl { box-shadow: inset 0 0 0 4px rgba(255,235,59,0.8); }
  .side { min-width:230px; }
  .score { font-size:15px; margin:6px 0 12px; }
  .bar { height:14px; background:#444; border-radius:7px; overflow:hidden; }
  .bar > div { height:100%; background:#e6e6e6; width:50%; transition:width .2s; }
  .moves { max-height:430px; overflow:auto; font-size:14px; line-height:1.7;
           font-variant-numeric:tabular-nums; }
  .moves span { cursor:pointer; padding:1px 4px; border-radius:4px; }
  .moves span.cur { background:#5865f2; color:#fff; }
  .controls { margin-top:10px; font-size:13px; color:#aaa; }
  .status { margin-top:6px; font-size:13px; color:#9ad; }
</style>
</head>
<body>
<header>♟️ Chess Real-to-Animated — real video (left) vs detected game (right)</header>
<div class="wrap">
  <div class="panel">
    <video id="vid" src="" controls autoplay muted></video>
    <div class="controls">Play the video — the board follows automatically.</div>
  </div>
  <div class="panel">
    <div id="board" class="board"></div>
    <div class="status" id="status"></div>
  </div>
  <div class="panel side">
    <div class="score">Material: <span id="scoreTxt">even</span>
      <div class="bar"><div id="scoreBar"></div></div>
    </div>
    <div class="moves" id="moves"></div>
  </div>
</div>
<script>
const DATA = __PAYLOAD__;
// Use the solid glyph shapes for BOTH colors; distinguish white/black via CSS.
const GLYPH = {p:'\u265F',n:'\u265E',b:'\u265D',r:'\u265C',q:'\u265B',k:'\u265A',
               P:'\u265F',N:'\u265E',B:'\u265D',R:'\u265C',Q:'\u265B',K:'\u265A'};
const VAL = {p:1,n:3,b:3,r:5,q:9,k:0};

document.getElementById('vid').src = DATA.video;

function parseFEN(fen){
  const rows = fen.split(' ')[0].split('/');
  const g = [];
  for(const row of rows){
    const r=[];
    for(const ch of row){
      if(/\d/.test(ch)){ for(let i=0;i<+ch;i++) r.push(''); }
      else r.push(ch);
    }
    g.push(r);
  }
  return g; // g[0] = rank8 (top)
}

const boardEl = document.getElementById('board');
const cells = [];
for(let i=0;i<64;i++){
  const d=document.createElement('div');
  const r=Math.floor(i/8), c=i%8;
  d.className='sq '+((r+c)%2===0?'light':'dark');
  boardEl.appendChild(d); cells.push(d);
}

function render(idx){
  const fen = DATA.fens[idx];
  const g = parseFEN(fen);
  let mat=0;
  for(let r=0;r<8;r++) for(let c=0;c<8;c++){
    const piece=g[r][c];
    const cell=cells[r*8+c];
    cell.textContent = piece?GLYPH[piece]:'';
    cell.classList.remove('hl','wp','bp');
    if(piece){ cell.classList.add(piece===piece.toUpperCase()?'wp':'bp'); }
    if(piece){ const v=VAL[piece.toLowerCase()]; mat += (piece===piece.toUpperCase()? v : -v); }
  }
  // score bar (white advantage -> bar wider)
  const pct = Math.max(5, Math.min(95, 50 + mat*4));
  document.getElementById('scoreBar').style.width = pct+'%';
  document.getElementById('scoreTxt').textContent =
     mat===0? 'even' : (mat>0? ('White +'+mat) : ('Black +'+(-mat)));
  document.getElementById('status').textContent =
     'Position '+(idx+1)+' / '+DATA.fens.length;
  // highlight move in list
  document.querySelectorAll('.moves span').forEach((s,i)=>{
     s.classList.toggle('cur', i===idx);
  });
  const cur=document.querySelector('.moves span.cur');
  if(cur) cur.scrollIntoView({block:'nearest'});
}

// build move list
const movesEl=document.getElementById('moves');
let html='';
for(let i=1;i<DATA.fens.length;i++){
  if(i%2===1) html += '<b>'+Math.ceil(i/2)+'.</b> ';
  html += '<span data-idx="'+i+'">'+(DATA.san[i]||'?')+'</span> ';
}
movesEl.innerHTML=html;
movesEl.querySelectorAll('span').forEach(s=>{
  s.onclick=()=>{ const idx=+s.dataset.idx; document.getElementById('vid').currentTime=idx*DATA.hold+0.05; render(idx); };
});

const vid=document.getElementById('vid');
function currentIndex(){ return Math.max(0, Math.min(DATA.fens.length-1, Math.floor(vid.currentTime/DATA.hold))); }
vid.addEventListener('timeupdate', ()=>render(currentIndex()));
render(0);
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
