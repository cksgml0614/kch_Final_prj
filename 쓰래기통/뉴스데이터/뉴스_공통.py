# 뉴스_공통.py
# NaverSearchBackfill.py에서 분리(2026-08-23). search.naver.com 날짜범위 크롤링에 필요한
# 공용 상수·함수 — 뉴스_최초적재.py와 뉴스_일일수집.py가 공유한다.
#
# clean_title/clean_press/upsert_articles는 원래 NaverFinanceNews.py(폐기됨)에 있었는데,
# 살아있는 코드가 폐기된 파일에 의존하는 역방향 구조였다. 이 파일로 옮겨 그 역전을 바로잡았다.
#
# ⚠️ 알려진 근본적 한계: 검색 결과 목록 페이지는 발행 "시각"을 주지 않고 날짜만 준다(연-월-일 단위).
# 언론사마다 원문 페이지의 시각 표기 위치가 달라 일괄 파싱이 불안정하므로 시각을 수집하지 않는다
# — published_at은 해당 날짜 00:00으로 채운다. 이 때문에 Task D(윈도우 재정렬)는 "안 B(완화):
# D-1 09:00~D 09:00 24시간 윈도우"로 확정됐다(2026-08-23, 사람 결정). 시각 정보가 필요한
# "안 A(엄격)"는 이 소스로는 원천적으로 적용 불가.
#
# ⚠️ 페이지네이션 반복 방어 필요: 하루치 결과를 다 소진하면 이후 start= 값에서 앞서 나온 기사를
# 그대로 재반환한다(빈 결과나 안내 메시지가 아님). "이번 페이지에서 신규(미확인) 기사가 0건"을
# 종료 조건으로 삼는다.
#
# ⚠️ 동시 실행 금지: 이 크롤러(또는 다른 search.naver.com 크롤러)를 두 개 이상 동시에 돌리면
# 요청 빈도 제한(403)이 유발된다(2026-08-23 실측). 반드시 한 번에 하나의 프로세스만 실행할 것.

import random
import re
import time
from datetime import datetime

import requests
from bs4 import BeautifulSoup

from constants import STOCK_NAMES

SOURCE_NAME = "search_backfill"
# 2026-08-23 실측: 0.7~2.0s 간격에서도 요청이 누적되면 일정 개수를 넘는 시점부터 403이 시작되고,
# 개별 요청은 그 와중에도 성공한다(짧은 창 기반 요청 빈도 제한으로 추정). 간격을 늘려 트리거 빈도를 낮춘다.
REQUEST_DELAY_RANGE = (2.0, 4.0)
BACKOFF_SECONDS = 60  # 연속 실패 감지 시 한 번 길게 쉬고 재시도(짧은 창이면 이 사이 리셋될 수 있음)
START_STEP = 10
MAX_START_PER_DAY = 300  # 안전장치. 하루 300건 이상은 사실상 없음(관측상 실제 소진은 40~90건대)

SEARCH_URL = "https://search.naver.com/search.naver"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
}

# 종목코드 -> 검색어(회사명)는 constants.STOCK_NAMES로 중앙화(2026-08-30) — 활성 여부는
# constants.ACTIVE_TICKERS 참고. 현재 활성 종목은 005930뿐이다(사람 결정, 2026-08-23).

# 언론사 화이트리스트 — 경제/증권 전문지 위주. NaverFinanceNews.py 스모크 테스트와 본 모듈 조사에서
# 실제로 관측된 언론사를 기준으로 시작한 목록이며, 커버리지에 직접 영향을 주므로 필요시 사람이 검토·확장할 것.
PRESS_WHITELIST = {
    "연합뉴스", "연합뉴스TV", "한국경제", "한경비즈니스", "매일경제", "매경이코노미",
    "머니투데이", "이데일리", "조선비즈", "조선일보", "서울경제", "헤럴드경제",
    "파이낸셜뉴스", "아시아경제", "뉴시스", "전자신문", "이투데이", "비즈니스포스트",
    "뉴스1", "디지털타임스", "이코노미스트",
    "SBS Biz", "YTN",  # 2026-08-23 사람 승인으로 추가 — 방송계열 경제/증권 채널
}

# 하루 안에서 연속 실패 시 그 날은 포기하고 다음 날로 (2026-08-23 실측: 동시에 두 번째 크롤러
# 프로세스를 같이 돌렸더니 요청 중간부터 전 페이지가 403으로 막혔고, 회로차단기 없이는 하루당
# 최대 시도(30회)를 끝까지 채우며 계속 헛수고를 반복한다 — 동시 실행 금지 + 조기 감지 필요)
CONSECUTIVE_FAILURE_LIMIT = 8
CONSECUTIVE_BLOCKED_DAYS_LIMIT = 2  # 연속 며칠이 통째로 차단되면 전체 실행을 중단

# 2026-08-23 속도 개선: 필터 통과 없이 연속 이만큼 페이지/후보가 쌓이면 그 날짜를 조기 종료한다.
EARLY_STOP_PAGES = 5
EARLY_STOP_CANDIDATES = 50

# 2026-08-29 예방적 쿨다운: 위 CONSECUTIVE_FAILURE_LIMIT/BACKOFF_SECONDS는 이미 403이 시작된
# "뒤"에 반응하는 회로차단기다. 이건 그 전에 미리 한 번씩 쉬어서 차단 자체를 덜 유발하려는
# 예방 조치이며 기존 로직을 대체하지 않고 그대로 얹는다. run_backfill_resumable처럼 여러
# 날짜에 걸쳐 도는 실행 전체 기간 동안 누적되어야 의미가 있으므로 하루 단위가 아니라 모듈
# 전역 카운터(_fetch_call_count)로 추적한다.
PREVENTIVE_COOLDOWN_EVERY = 80
PREVENTIVE_COOLDOWN_RANGE = (90.0, 120.0)

_fetch_call_count = 0

_PATH_IDS_RE = re.compile(r"/mnews/article/(\d+)/(\d+)")
_UI_LABEL_CLASS = "fender-ui_0cb57fb2"  # "새 창 열림" 같은 접근성 라벨용 span의 클래스


def clean_title(text):
    """제목 공백 정규화. BeautifulSoup.get_text()가 HTML 엔티티는 이미 unescape 처리함."""
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def clean_press(text):
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _canonical_article_url(naver_url, original_url):
    """
    네이버뉴스 배지 링크가 있으면 정규 URL 형태로 통일한다(같은 실제 기사면 다른 크롤러가 만든
    article_url과도 겹칠 수 있어야 소스 간 비교가 가능하다). 배지가 없는 기사(네이버 미제휴
    언론사)는 원문 URL을 그대로 쓴다.
    """
    if naver_url:
        m = _PATH_IDS_RE.search(naver_url)
        if m:
            office_id, article_id = m.groups()
            return f"https://n.news.naver.com/mnews/article/{office_id}/{article_id}"
    return original_url


def fetch_search_page(stock_name, target_date, start):
    """
    하루(target_date) 안에서 start번째부터 결과를 가져온다.
    반환: list[dict(title, press, article_url)]
    """
    params = {
        "where": "news",
        "query": stock_name,
        "pd": "3",
        "ds": target_date.strftime("%Y.%m.%d"),
        "de": target_date.strftime("%Y.%m.%d"),
        "start": start,
    }
    resp = requests.get(SEARCH_URL, params=params, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    items = []
    for title_a in soup.find_all("a", attrs={"data-heatmap-target": ".tit"}):
        title_spans = [
            sp for sp in title_a.find_all("span", recursive=False)
            if _UI_LABEL_CLASS not in (sp.get("class") or [])
        ]
        if not title_spans:
            continue
        title = clean_title(title_spans[0].get_text())
        original_url = title_a.get("href")

        container = title_a.parent
        press_span = container.select_one(".sds-comps-profile-info-title-text")
        press = clean_press(press_span.get_text()) if press_span else None

        naver_url = None
        for a2 in container.find_all("a"):
            span = a2.find("span")
            if span and span.get_text(strip=True) == "네이버뉴스":
                naver_url = a2.get("href")
                break

        article_url = _canonical_article_url(naver_url, original_url)
        items.append({"title": title, "press": press, "article_url": article_url})

    return items


def _track_fetch_call():
    """
    fetch_search_page 호출 직전에 불러 누적 카운터를 올리고, PREVENTIVE_COOLDOWN_EVERY회에
    도달하면 한 번 길게 쉬고 카운터를 리셋한다. 모듈 전역 상태라 run_backfill_resumable이
    날짜를 넘어가며 crawl_day를 반복 호출해도 하루 단위로 끊기지 않고 누적된다.
    """
    global _fetch_call_count
    _fetch_call_count += 1
    if _fetch_call_count >= PREVENTIVE_COOLDOWN_EVERY:
        cooldown = random.uniform(*PREVENTIVE_COOLDOWN_RANGE)
        print(
            f"      🧊 예방적 쿨다운: 누적 요청 {_fetch_call_count}회 — {cooldown:.1f}초 대기 후 재개",
            flush=True,
        )
        time.sleep(cooldown)
        _fetch_call_count = 0


def passes_quality_filter(title, press, stock_name):
    """(a) 종목명이 제목에 실제 포함, (b) 언론사 화이트리스트. 둘 다 통과해야 채택."""
    if stock_name not in title:
        return False, "종목명_제목_미포함"
    if press not in PRESS_WHITELIST:
        return False, "언론사_화이트리스트_제외"
    return True, None


def crawl_day(ticker, target_date):
    """
    하루치를 start=1,11,21,...로 순회하며 신규 기사가 0건인 페이지가 나오면 종료한다
    (사이트가 결과 소진 후 이전 기사를 반복 반환하는 것에 대한 방어).

    연속 CONSECUTIVE_FAILURE_LIMIT회 요청 실패 시 한 번 BACKOFF_SECONDS 길게 쉬고 재시도한다
    (짧은 창 기반 요청빈도 제한이라면 이 사이 리셋될 수 있다 — 2026-08-23 실측상 개별 요청은
    간헐적으로 성공하는 것으로 보아 완전 차단이 아니라 이런 유형일 가능성이 있음).
    백오프 후에도 다시 CONSECUTIVE_FAILURE_LIMIT회 연속 실패하면 그제서야 차단 의심으로
    그 날을 조기 포기한다(blocked=True) — 회로차단기 없이는 막힌 채로 나머지 시도를 전부 낭비한다.

    ⚠️ 2026-08-23 실측으로 드러난 또 다른 패턴: 회로차단기가 트리거될 만큼 연속 실패하지는
    않았지만 그 날 안에 403이 섞여 있었던 날은, 후보(candidates_total)는 정상적으로 수백 건씩
    잡히는데 필터 통과(kept)가 0건으로 나오는 경우가 있었다 — 같은 날짜를 나중에 깨끗하게(요청
    사이 지연 없이 단발로) 다시 긁어보면 정상적으로 여러 건이 필터를 통과했다. 즉 완전 차단은
    아니지만 요청 빈도 제한이 걸린 상태의 응답이 구조는 파싱되지만 내용이 정상이 아니었던 것으로
    보인다. 이런 날은 "실제로 관련 기사가 없었다"고 신뢰할 수 없으므로 degraded=True로 표시하고
    체크포인트에 기록하지 않아 다음 실행 때 재시도되게 한다.

    ⚠️ 조기 종료 휴리스틱(2026-08-23, 속도 개선): 필터를 통과한 기사 없이 연속
    EARLY_STOP_PAGES회(또는 누적 EARLY_STOP_CANDIDATES건) 후보만 쌓이면 그 날짜를 조기 종료한다
    (early_stopped=True). 날짜가 실제로 소진돼서가 아니라 "이 뒤로도 관련 기사가 나올 가능성이
    낮다"는 판단으로 시간을 아끼는 것 — 트레이드오프로 아주 드물게 뒤쪽 페이지에 있는 관련
    기사를 놓칠 수 있다. degraded(응답 이상 의심)와는 무관하다(page_errors 없이도 트리거될 수
    있음) — degraded는 반드시 page_errors가 함께 있어야 트리거되도록 구분했다(아래).

    반환: dict(kept, filtered_out, page_errors, pages_fetched, candidates_total, blocked, degraded,
               early_stopped)
    """
    stock_name = STOCK_NAMES[ticker]
    seen_urls = set()
    kept = []
    filtered_out = {}
    page_errors = []
    start = 1
    pages_fetched = 0
    consecutive_failures = 0
    backoff_used = False
    blocked = False
    early_stopped = False
    pages_since_keep = 0
    candidates_since_keep = 0

    while start <= MAX_START_PER_DAY:
        pages_fetched += 1
        try:
            _track_fetch_call()
            items = fetch_search_page(stock_name, target_date, start)
        except Exception as e:
            page_errors.append({"start": start, "error": str(e)})
            print(f"      ⚠️ [{ticker}] {target_date} start={start} 요청/파싱 실패: {e} (건너뛰고 계속)")
            consecutive_failures += 1
            if consecutive_failures >= CONSECUTIVE_FAILURE_LIMIT:
                if not backoff_used:
                    backoff_used = True
                    consecutive_failures = 0
                    print(f"      ⏸️ [{ticker}] {target_date} 연속 실패 {CONSECUTIVE_FAILURE_LIMIT}회 — {BACKOFF_SECONDS}초 대기 후 한 번 재시도")
                    time.sleep(BACKOFF_SECONDS)
                    continue
                blocked = True
                print(f"      🛑 [{ticker}] {target_date} 백오프 후에도 연속 실패 — 차단 확정, 이 날짜는 포기")
                break
            time.sleep(random.uniform(*REQUEST_DELAY_RANGE))
            start += START_STEP
            continue

        consecutive_failures = 0
        new_items = [it for it in items if it["article_url"] not in seen_urls]
        if not new_items:
            break
        for it in new_items:
            seen_urls.add(it["article_url"])

        page_kept_count = 0
        for it in new_items:
            ok, reason = passes_quality_filter(it["title"], it["press"], stock_name)
            if ok:
                kept.append(it)
                page_kept_count += 1
            else:
                filtered_out[reason] = filtered_out.get(reason, 0) + 1

        if page_kept_count > 0:
            pages_since_keep = 0
            candidates_since_keep = 0
        else:
            pages_since_keep += 1
            candidates_since_keep += len(new_items)
            if pages_since_keep >= EARLY_STOP_PAGES or candidates_since_keep >= EARLY_STOP_CANDIDATES:
                early_stopped = True
                print(
                    f"      ⏭️ [{ticker}] {target_date} 연속 {pages_since_keep}페이지"
                    f"({candidates_since_keep}건 후보) 필터통과 0건 — 조기 종료"
                )
                break

        start += START_STEP
        time.sleep(random.uniform(*REQUEST_DELAY_RANGE))
    else:
        page_errors.append({"start": start, "error": "MAX_START_PER_DAY 한도 도달"})

    # degraded(응답 이상 의심)는 반드시 page_errors가 있어야 한다 — 그냥 관련 기사가 없어서
    # kept=0인 정상 케이스(조기종료 포함)까지 degraded로 잘못 묶으면 최초적재가 그 날짜를
    # 영원히 재시도하게 된다.
    degraded = bool(page_errors) and not kept and len(seen_urls) > 0

    return {
        "kept": kept,
        "filtered_out": filtered_out,
        "page_errors": page_errors,
        "pages_fetched": pages_fetched,
        "candidates_total": len(seen_urls),
        "blocked": blocked,
        "early_stopped": early_stopped,
        "degraded": degraded,
    }


def upsert_articles(conn, ticker, articles, source):
    """
    article_url 기준 UPSERT. 단, 기존 daily_news에는 (ticker, date, title, summary) 복합
    UNIQUE 제약도 걸려 있고, 목록 페이지에 별도 요약이 없어 summary=title로 채우다 보니
    레거시 수집분과 실제로 동일한 기사면 이 복합 제약에도 걸릴 수 있다. 두 제약 중 어느
    쪽이든 걸리면 "이미 있는 기사"로 봐야 하므로 conflict target을 지정하지 않는다
    (ON CONFLICT DO NOTHING — 테이블의 모든 UNIQUE 제약을 대상으로 함).

    실패(= 두 제약과 무관한 다른 오류)는 로깅하고 개별 트랜잭션으로 격리한다(한 건의 오류가
    나머지 건의 적재를 막지 않도록). 커밋 전후 COUNT(*)로 실제 반영 행 수를 검증한다.

    source는 호출부가 명시적으로 지정한다(기본값 없음 — 여러 소스가 이 함수를 공유하므로
    묵시적 기본값을 두면 잘못된 source로 적재될 위험이 있다).

    반환: dict(attempted, inserted, duplicate, failed, failed_detail)
    """
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM daily_news WHERE ticker = %s", (ticker,))
        count_before = cur.fetchone()[0]

    attempted = len(articles)
    inserted = 0
    duplicate = 0
    failed_detail = []

    for art in articles:
        if art["published_at"] is None:
            failed_detail.append({"title": art["title"], "reason": "published_at 파싱 실패"})
            continue
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO daily_news (ticker, date, title, summary, press, published_at, article_url, source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                    RETURNING id
                    """,
                    (
                        ticker,
                        art["published_at"].date(),
                        art["title"],
                        art["title"],  # summary: 종목뉴스 목록엔 별도 요약이 없어 제목으로 채움
                        art["press"],
                        art["published_at"],
                        art["article_url"],
                        source,
                    ),
                )
                row = cur.fetchone()
            conn.commit()
            if row is not None:
                inserted += 1
            else:
                duplicate += 1
        except Exception as e:
            conn.rollback()
            failed_detail.append({"title": art["title"], "url": art["article_url"], "reason": str(e)})
            print(f"      ❌ [{ticker}] 적재 실패: {art['title'][:40]!r} — {e}")

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM daily_news WHERE ticker = %s", (ticker,))
        count_after = cur.fetchone()[0]

    actual_delta = count_after - count_before
    if actual_delta != inserted:
        print(
            f"      ⚠️ [{ticker}] 검증 불일치: 카운트 기준 실제 증가분={actual_delta}, "
            f"upsert가 보고한 신규 삽입={inserted} — 코드 점검 필요"
        )

    return {
        "attempted": attempted,
        "inserted": inserted,
        "duplicate": duplicate,
        "failed": len(failed_detail),
        "failed_detail": failed_detail,
        "count_before": count_before,
        "count_after": count_after,
    }
