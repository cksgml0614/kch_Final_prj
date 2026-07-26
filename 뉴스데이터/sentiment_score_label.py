from db_manager import get_db_connection

def update_news_labels(ticker="005930"):
    """
    KOSPI 지수와 상관없이, 해당 종목의 당일 등락률(change_rate)만으로
    뉴스에 sentiment_score(정답 라벨)를 부여합니다.
    """
    conn = get_db_connection()
    if not conn: return

    try:
        with conn.cursor() as cur:
            # 1. 'sentiment_score'가 아직 없는 뉴스들만 가져옵니다.
            # 종목 가격 테이블(daily_stock_prices)과 JOIN하여 등락률을 바로 확보합니다.
            query = """
                    SELECT n.id, s.change_rate
                    FROM daily_news n
                    JOIN daily_stock_prices s ON n.ticker = s.ticker AND n.date = s.date
                    WHERE n.ticker = %s 
                      AND n.sentiment_score IS NULL;
                    """
            cur.execute(query, (ticker,))
            rows = cur.fetchall()

            if not rows:
                print(f"✨ [{ticker}] 새로 점수를 매길 뉴스가 없습니다. (모두 완료됨)")
                return

            print(f"📊 총 {len(rows)}건의 뉴스에 대해 개별 종목 등락률 기준 라벨링 시작...")

            updated_count = 0
            for news_id, s_rate in rows:
                # 등락률(s_rate)에 따른 5단계 점수 할당 (임계값은 기존과 동일)

                if s_rate >= 0.03:
                    score = 2
                elif s_rate >= 0.01:
                    score = 1
                elif s_rate <= -0.03:
                    score = -2
                elif s_rate <= -0.01:
                    score = -1
                else:
                    score = 0

                # 3. DB 업데이트
                update_query = "UPDATE daily_news SET sentiment_score = %s WHERE id = %s"
                cur.execute(update_query, (score, news_id))
                updated_count += 1

            conn.commit()
            print(f"✅ {updated_count}건의 sentiment_score 업데이트 완료!")

    except Exception as e:
        print(f"❌ 작업 중 오류 발생: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    # 삼성전자(005930) 라벨링 실행
    update_news_labels("005930")