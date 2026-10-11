# 대시보드/capture_screenshot.py — Streamlit 대시보드를 headless Chrome(DevTools 프로토콜)으로 렌더 후 캡처(2026-09-27)
# Chrome --screenshot 한 방으로는 Streamlit이 웹소켓으로 그리기 전 스켈레톤만 찍혀서, (d) 패널 h3와 plotly 그래프 3개가
# 뜰 때까지 기다린 뒤 전체 페이지와 (d) 패널 영역을 저장한다. 산출물은 대시보드/screenshots/(gitignore) 권장.
# 사용: streamlit run <대시보드 스크립트> --server.port 8599 --server.headless true 실행 후
#       python 대시보드/capture_screenshot.py <URL> <출력.png> [기다릴 h3 글자, 기본 "(d)"] [최소 plotly 수, 기본 3]
#   옛 대시보드: … "http://127.0.0.1:8599/?date=2026-09-21" 대시보드/screenshots/dash.png
#   새 대시보드: … "http://127.0.0.1:8599/" 대시보드/screenshots/new.png "최근 14일" 1

import asyncio, base64, json, os, subprocess, sys, time, urllib.request
import websockets
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
url, out, port = sys.argv[1], sys.argv[2], 9333
wait_text = sys.argv[3] if len(sys.argv) > 3 else "(d)"
min_plots = int(sys.argv[4]) if len(sys.argv) > 4 else 3
prof = os.path.abspath(out) + "_prof"  # 상대 경로면 Chrome 디버그 포트가 안 열려 실패했다(2026-10-11)
proc = subprocess.Popen([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--remote-debugging-port={port}",
                         "--window-size=1500,4600", f"--user-data-dir={prof}", url])
try:
    for _ in range(50):
        try:
            tabs = json.load(urllib.request.urlopen(f"http://localhost:{port}/json/list")); break
        except Exception: time.sleep(0.3)
    ws_url = [t for t in tabs if t["type"] == "page"][0]["webSocketDebuggerUrl"]
    async def main():
        async with websockets.connect(ws_url, max_size=2**28) as ws:
            n = 0
            async def call(method, **params):
                nonlocal n; n += 1; my = n
                await ws.send(json.dumps({"id": my, "method": method, "params": params}))
                while True:
                    m = json.loads(await ws.recv())
                    if m.get("id") == my: return m.get("result", {})
            js = """(() => { const h=[...document.querySelectorAll('h3')].find(e=>e.innerText.includes(%s));
              const plots=document.querySelectorAll('.js-plotly-plot').length;
              if(!h) return JSON.stringify({ready:false, plots});
              const r=h.getBoundingClientRect(); return JSON.stringify({ready:true, plots, top:r.top+window.scrollY}); })()""" % json.dumps(wait_text)
            info = {}
            for _ in range(120):
                r = await call("Runtime.evaluate", expression=js, returnByValue=True)
                info = json.loads(r["result"]["value"])
                if info.get("ready") and info["plots"] >= min_plots: break
                await asyncio.sleep(1)
            await asyncio.sleep(4)
            print("render:", info)
            full = await call("Page.captureScreenshot", format="png", captureBeyondViewport=True)
            open(out, "wb").write(base64.b64decode(full["data"]))
            top = max(0, info.get("top", 0) - 20)
            clip = await call("Page.captureScreenshot", format="png", captureBeyondViewport=True,
                              clip={"x": 0, "y": top, "width": 1500, "height": 1250, "scale": 1})
            open(out.replace(".png", "_d.png"), "wb").write(base64.b64decode(clip["data"]))
    asyncio.run(main())
finally:
    proc.terminate()
