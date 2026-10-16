"""Local review UI: show each image beside its OCR text, edit, and save.

Saving writes `corrected_text` and review_status="corrected" back into the
result JSON, giving you ground-truth pairs (image + corrected_text).
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import json
import webbrowser
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]

PAGE = """<!doctype html><meta charset="utf-8"><title>OCR review</title>
<style>
body{font-family:system-ui,sans-serif;margin:0;display:flex;height:100vh}
#list{width:260px;overflow:auto;border-right:1px solid #ccc;font-size:13px}
#list div{padding:6px 8px;cursor:pointer;border-bottom:1px solid #eee}
#list div.sel{background:#dde8ff}#list .corrected::before{content:"\\2713 ";color:green}
#main{flex:1;display:flex;flex-direction:column;min-width:0}
#panes{flex:1;display:flex;min-height:0}
#img{flex:1;overflow:auto;padding:8px}#img img{max-width:100%}
#edit{flex:1;display:flex;flex-direction:column;padding:8px}
textarea{flex:1;font:16px/1.5 'Leelawadee UI',Tahoma,sans-serif}
#bar{padding:8px;border-top:1px solid #ccc;display:flex;gap:8px;align-items:center}
</style>
<div id="list"></div>
<div id="main"><div id="panes"><div id="img"></div>
<div id="edit"><textarea id="t"></textarea></div></div>
<div id="bar"><button id="save">Save &amp; next (Ctrl+Enter)</button>
<button id="reset">Reset to model text</button><span id="msg"></span></div></div>
<script>
let items=[],cur=0;
const $=id=>document.getElementById(id);
async function load(){items=await (await fetch('/api/items')).json();render();show(0)}
function render(){$('list').innerHTML='';items.forEach((it,i)=>{const d=document.createElement('div');
d.textContent=it.name;d.className=(it.status=='corrected'?'corrected ':'')+(i==cur?'sel':'');
d.onclick=()=>show(i);$('list').appendChild(d)})}
function show(i){cur=i;const it=items[i];$('img').innerHTML='<img src="/image/'+encodeURIComponent(it.name)+'">';
$('t').value=it.corrected_text??it.text;$('msg').textContent='';render()}
async function save(){const it=items[cur];
await fetch('/api/save/'+encodeURIComponent(it.name),{method:'POST',body:$('t').value});
it.corrected_text=$('t').value;it.status='corrected';$('msg').textContent='saved';
if(cur<items.length-1)show(cur+1);else render()}
$('save').onclick=save;$('reset').onclick=()=>{$('t').value=items[cur].text};
document.addEventListener('keydown',e=>{if(e.ctrlKey&&e.key=='Enter')save()});
load();
</script>"""


def make_handler(result_dir: Path):
    def read(name: str) -> dict:
        return json.loads((result_dir / name).read_text(encoding="utf-8"))

    def image_path(data: dict) -> Path:
        p = Path(data["image"])
        return p if p.is_absolute() else ROOT / p

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, body: bytes, ctype: str, code: int = 200):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/":
                self.send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            elif self.path == "/api/items":
                items = []
                for f in sorted(result_dir.glob("*.json")):
                    d = read(f.name)
                    items.append({
                        "name": f.name,
                        "text": d["text"],
                        "corrected_text": d.get("corrected_text"),
                        "status": d.get("review_status"),
                    })
                self.send(json.dumps(items, ensure_ascii=False).encode("utf-8"),
                          "application/json")
            elif self.path.startswith("/image/"):
                img = image_path(read(unquote(self.path[7:])))
                ctype = "image/png" if img.suffix.lower() == ".png" else "image/jpeg"
                self.send(img.read_bytes(), ctype)
            else:
                self.send(b"not found", "text/plain", 404)

        def do_POST(self):
            if not self.path.startswith("/api/save/"):
                return self.send(b"not found", "text/plain", 404)
            name = Path(unquote(self.path[10:])).name  # no path traversal
            length = int(self.headers.get("Content-Length", 0))
            text = self.rfile.read(length).decode("utf-8")
            path = result_dir / name
            data = read(name)
            data["corrected_text"] = text.strip()
            data["review_status"] = "corrected"
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                            encoding="utf-8")
            self.send(b"ok", "text/plain")

    return Handler


def main():
    parser = argparse.ArgumentParser(description="Review and correct OCR results.")
    parser.add_argument("--dir", default="outputs/vision",
                        help="Folder of result JSON files to review.")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    result_dir = Path(args.dir)
    if not result_dir.is_absolute():
        result_dir = ROOT / result_dir
    if not any(result_dir.glob("*.json")):
        raise FileNotFoundError(f"No result JSON files in {result_dir}")

    url = f"http://127.0.0.1:{args.port}/"
    print(f"Reviewing {result_dir}\nOpen {url}  (Ctrl+C to stop)")
    webbrowser.open(url)
    ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(result_dir)).serve_forever()


if __name__ == "__main__":
    main()
