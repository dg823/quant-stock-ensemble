# Quant Stock Prediction Ensemble

주가 데이터를 기반으로 수익률/방향성/급락 여부를 예측하는 여러 버전의 앙상블 모델과,
실시간 예측 및 모니터링용 Streamlit 대시보드로 구성된 퀀트 ML 프로젝트입니다.

## 구성

- `two_step_ensemble.py` — 2단계 계층 필터링 앙상블(변동성 필터 → 방향성 분류). 날짜 기반 `TimeSeriesSplit`과 fold 내부 스케일링으로 데이터 누수를 방지.
- `dual_model_ensemble.py` — N일 뒤 수익률 회귀 모델 + 급락 감지 이진 분류 모델 앙상블.
- `jusik.py` — MLP 기반 회귀/분류 앙상블(원본 CSV 피처만 사용하는 제약 버전).
- `han.py` — XGBoost 분류 모델 학습 및 `RandomizedSearchCV` 하이퍼파라미터 탐색, feature importance 시각화.
- `hyper.py` — 회귀/분류 모델 공용 하이퍼파라미터 탐색 유틸리티.
- `test_3class.py` / `check_dist.py` — 매도/중립/매수 3클래스 라벨링 테스트 및 라벨 분포 점검.
- `realtime_predict.py` — `FinanceDataReader`로 실시간 시세를 받아 `two_step_ensemble` 모델로 신호를 생성.
- `web.py` — Streamlit + Plotly 기반 실시간 퀀트 모니터링 대시보드.

`*_analysis.png`, `*_predictions.csv`는 실행 결과 예시입니다.

## 실행

```bash
pip install -r requirements.txt
python two_step_ensemble.py      # 모델 학습/평가
python realtime_predict.py       # 실시간 신호 생성
streamlit run web.py             # 대시보드 실행
```

원본 CSV 데이터셋(`Quant_PRO_BigData_350.csv`, `Quant_Scaled_ML_Ready.csv`)은 용량 문제로 저장소에 포함하지 않았습니다.
