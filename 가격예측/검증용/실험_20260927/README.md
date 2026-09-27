# 2026-09-27 실험 아카이브

2026-09-27 세션의 실험 스크립트를 결과 재검증용으로 보존한다. 같은 날 오전, 커밋 없이 삭제된 손실함수 실험
스크립트 4개를 재검증할 때 코드를 볼 수 없어 곤란했던 일 때문에, 결과가 문서에 수치로 남아 있어도 스크립트를 지우지
않고 여기 모았다.

- **코드만 git에 올라간다.** 입력·산출 데이터(npy, npz, parquet, csv, json, log, txt)는 이 폴더에 로컬 사본만 두고
  `.gitignore`로 제외했다.
- 모든 실험은 **fold3 val만** 쓴다. test는 평가하지 않았다.
- 대부분의 스크립트는 `OUT = 스크립트 폴더`에서 입력을 읽고 결과를 쓴다. 저장소 루트를 `sys.path`에 넣고
  `os.chdir`하도록 경로가 하드코딩돼 있다(`C:\kch_Final_prj`).
- 공용 평가 함수는 `가격예측/검증용/regime_eval.py`에 있다. 새 실험은 스크립트를 복붙하지 말고 이 모듈을 쓸 것.

| 스크립트 | 무엇을 | 결과가 기록된 곳 |
|---|---|---|
| `val_diag.py` | 게이트 집계 버그 확인(B1), val 분포(B2), 소수 종목 영향(B3), 19영업일 백테스트 RMSE 재계산 | CLAUDE.md "게이트 집계 버그" 절 |
| `arch_compare.py` | LSTM seed 42/43/44 vs Transformer seed 42 | CLAUDE.md. ⚠️ 이 스크립트의 게이트 판정은 **버그 수정 전 비대칭 방식**이다. 대칭 판정은 `loss_recheck.py` 결과를 볼 것 |
| `loss_recheck.py` | 급변구간 손실함수 3종 대칭 재검증 + LSTM 3 seed 대칭 게이트 | 결과_TaskT_급변구간_손실함수_시도.md [9] |
| `price_level_v2.py` | 종가(레벨) 예측 A/B vs naive(반올림 버그를 고친 v2) | CLAUDE.md "종가(레벨) 예측 실험" |
| `bigmove_clf.py` | 급변일 GBM 분류기 PR-AUC + 레이블 셔플 | CLAUDE.md "급변일 전용 분류기" |
| `excess5_clf.py` | 초과수익률 z 5클래스 GBM + 셔플 + confusion matrix | CLAUDE.md "초과수익률 z 5클래스" |
| `mi_check.py` | close_return/open_gap_ratio MI 착시 점검 + 단일 피처 GBM R² | 발표자료/diagnose_structure.py(C)의 후속, CLAUDE.md |
| `price_estimate_and_blend.py` | 가격 추정 비교, 그룹별 재집계, garch_sigma 블렌딩, GBM 스위치 전환율. 세션에선 인라인으로만 돌렸던 것을 `regime_eval`로 재구성했고 수치 재현을 확인함 | 결과 문서 [10], 발표자료/price_estimate_*.csv |
| `lstm_reg.py` | LSTM 규제 완화 4조합 | 결과 문서 [11] |
| `lstm_mbr.py` | 16번째 피처 `market_bigmove_ratio`(전날 값) 추가 LSTM | 결과 문서 [11] 끝, CLAUDE.md |
| `lstm_trainpred.py` → `postproc_eval.py` | 운영 LSTM의 train 예측 저장 → 상위 P% 예측 배율 후처리(P 10/20/30 × M 1.1/1.2/1.5/2.0/3.0) | 결과 문서 [12] |
| `req_scan.py` | 활성 코드 import 스캔 + 의존성 closure로 미사용 패키지 찾기(requirements 정리 도구) | 커밋 818062a |

**로컬에만 있는 데이터 파일(세션 산출물 사본)**
- `val_diag_rows.parquet`: val 행 + Transformer 예측 + baseline
- `lossrecheck_lstm_s42.npy` / `lstm_prod_*.npy`: 운영 LSTM val·train 예측
- `price_level_preds_B_*.npz`: 종가모델 B
- `bigmove_clf_*_pred.npy`: GBM 확률
- `lstm_reg_*`: 규제 스윕 기존 설정 예측과 행 키
- `actual_vol_0914_before.csv`: 9/14 actual_volatility 정정 전 값
- `pip_freeze_before_cleanup.txt`: requirements 정리 전 freeze. 롤백용

없으면 각 스크립트를 순서대로 다시 돌려 재생성한다(데이터 준비 약 15분).
