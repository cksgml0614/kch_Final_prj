# 대시보드/capture_screenshot.py — 띄워 둔 Streamlit 화면을 headless Chrome(DevTools 프로토콜)으로 렌더가 끝난 뒤 캡처하는 점검 도구.
# Chrome --screenshot 한 번으로는 웹소켓으로 그리기 전 화면이 찍혀서, 지정한 h3 글자와 plotly 그래프 수가 나타날 때까지 기다린다.
# 저장: 전체 페이지 <출력.png>와 그 h3부터 1250px 영역 <출력>_d.png. 산출물은 대시보드/screenshots/(gitignore) 권장.
# 사용: streamlit run 대시보드/운영_대시보드.py --server.port 8599 --server.headless true 실행 후
#       python 대시보드/capture_screenshot.py "http://127.0.0.1:8599/" 대시보드/screenshots/new.png "최근 14일" 1
# ⚠️ 3·4번째 인자의 기본값("(d)", 3)은 지금은 없는 옛 대시보드 기준이라 운영 대시보드에는 위처럼 직접 넘긴다.

import asyncio, base64, json, os, subprocess, sys, time, urllib.request
import websockets
CHROME = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
url, out, port = sys.argv[1], sys.argv[2], 9333
wait_text = sys.argv[3] if len(sys.argv) > 3 else "(d)"
min_plots = int(sys.argv[4]) if len(sys.argv) > 4 else 3
prof = os.path.abspath(out) + "_prof"  # 프로필 경로가 상대 경로면 Chrome 디버그 포트가 열리지 않는다
proc = subprocess.Popen([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars", f"--remote-debugging-port={port}",
                         "--window-size=1500,4600", f"--user-data-dir={prof}", url])
try:
    for _ in range(50):
        try:
            tabs = json.load(urllib.request.urlopen(f"http://localhost:{port}/json/list")); break
        except Exception: time.sleep(0.3)
    ws_url = [t for t in tabs if t["type"] == "page"][0]["webSocketDebuggerUrl"]
    async def main():
        """
        설명: DevTools 웹소켓으로 렌더 완료(h3 글자 + plotly 수)를 최대 120초 기다린 뒤 전체·영역 스크린샷 두 장을 저장한다.
        주의 사항: 끝까지 준비되지 않아도 그 시점 화면을 찍는다(출력 "render:"의 ready·plots로 확인).
        입력 타입->반환 타입: () -> None
        """
        async with websockets.connect(ws_url, max_size=2**28) as ws:
            n = 0
            async def call(method, **params):
                """
                설명: DevTools 명령 하나를 보내고 같은 id의 응답 result를 돌려준다.
                주의 사항: 다른 id의 메시지(이벤트 등)는 버린다.
                입력 타입->반환 타입: (str, **object) -> dict
                """
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
