"""
[Project] 주가 변동성 기반 2단계 필터링 앙상블 예측 시스템
-------------------------------------------------------
본 코드는 머신러닝 기반 퀀트 트레이딩 모델로, 단순히 주가의 방향을 예측하는 것을 넘어 
'예측 신뢰도가 높은 시점'을 선별하여 신호를 생성하는 계층적 구조를 채택하고 있습니다.

주요 아키텍처 및 특징:
1. Two-Step Filtering Strategy (2단계 계층 구조)
   - Step 1 (Volatility Filter): 5일 내 유의미한 변동(abs > 1%)의 발생 여부를 먼저 예측합니다.
     시장 노이즈(횡보장)를 1차적으로 필터링하여 불필요한 거래 비용을 줄입니다.
   - Step 2 (Directional Model): 변동성이 예측된 종목에 대해서만 상승/하락 방향을 분류합니다.
     학습 시 실제 변동성이 컸던 시점의 데이터만 사용하여 방향성 학습의 해상도를 높였습니다.

2. Strict Data Leakage Prevention (엄격한 데이터 누수 방지)
   - Date-based TimeSeriesSplit: 동일 날짜의 다른 종목 정보가 훈련/테스트 셋에 섞여 
     미래 정보를 참조하는 '수평적 데이터 누수'를 원천적으로 차단합니다.
   - In-fold Scaling: 전체 데이터가 아닌, 각 검증 Fold의 훈련 데이터로만 스케일러를 학습(fit)시켜 
     검증 데이터의 통계 정보가 유출되는 것을 방지합니다.

3. Robust Data Engineering (강건한 데이터 전처리)
   - 무한대(Inf) 값 처리 및 결측치 제거를 통해 금융 데이터의 기술적 오류를 해결했습니다.
   - 수익률 클리핑(-50% ~ 50%)을 통해 이상치(Outlier)에 의한 모델 왜곡을 방어합니다.

4. Quantitative Signal Logic (퀀트 시그널 논리)
   - 단순 분류(0/1)가 아닌 확률(Probability) 기반 임계값을 적용(v>0.65, d>0.65)하여 
     하락장에서는 보수적인 관망 시그널을, 확신이 높은 시점에는 강력 매수 시그널을 생성합니다.

결과물:
 - two_step_predictions.csv: 종목별 변동성/상승 확률 및 최종 투자 등급
 - two_step_analysis.png: 각 단계별 피처 중요도 분석 시각화
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, precision_score
from sklearn.preprocessing import RobustScaler
import warnings

warnings.filterwarnings('ignore')

# ── 설정 ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Quant_PRO_BigData_350.csv")

FEATURES = [
    'retail_net_buy', 'foreign_net_buy', 'inst_net_buy', 'volume',
    'atr_14d', 'vol_change_1d', 'volatility_20d', 'disparity_20d',
    'rsi_14d', 'macd_osc', 'bb_lower_break', 'vkospi', 'relative_strength'
]

PRED_DAYS = 5
MOVE_THRESH = 0.01  # 1% 이상 움직임을 '유의미한 변동'으로 정의

# ── 데이터 로드 및 2단계 라벨링 ────────────────────────────────────────────────
def load_and_preprocess():
    print(f"[1/5] 데이터 로드 및 2단계 라벨링 생성")
    df = pd.read_csv(CSV_PATH, parse_dates=['date'])
    df = df.sort_values(['name', 'date']).reset_index(drop=True)
    
    # 5일 후 수익률
    df['target_return'] = df.groupby('name')['close'].transform(lambda x: x.shift(-PRED_DAYS) / x - 1)
    
    # 무한대 및 결측치 처리
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=FEATURES + ['target_return']).reset_index(drop=True)
    
    # 수익률 상하한 제한
    df['target_return'] = df['target_return'].clip(-0.5, 0.5)

    # Step 1 라벨: 변동성 발생 여부 (절대값 > 1%)
    df['label_step1'] = (df['target_return'].abs() >= MOVE_THRESH).astype(int)
    
    # Step 2 라벨: 방향성 (상승 1, 하락 0)
    df['label_step2'] = (df['target_return'] > 0).astype(int)
    
    print(f"  - 변동성 발생 비율 (Step 1): {df['label_step1'].mean():.2%}")
    print(f"  - 전체 상승 비율: {df['label_step2'].mean():.2%}")
    
    return df

# ── 날짜 기반 분할 ────────────────────────────────────────────────────────────
def get_date_splits(df, n_splits=5):
    unique_dates = np.sort(df['date'].unique())
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for train_date_idx, test_date_idx in tscv.split(unique_dates):
        tr_dates = unique_dates[train_date_idx]
        te_dates = unique_dates[test_date_idx]
        
        # 검증용 분리
        split_pt = int(len(tr_dates) * 0.8)
        yield (df.index[df['date'].isin(tr_dates[:split_pt])], 
               df.index[df['date'].isin(tr_dates[split_pt:])], 
               df.index[df['date'].isin(te_dates)])

# ── 모델 설정 ──────────────────────────────────────────────────────────────────
def get_model():
    return XGBClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        gamma=0.1, reg_alpha=0.1, reg_lambda=3.0,
        random_state=42, n_jobs=-1, eval_metric='logloss',
        early_stopping_rounds=50
    )

# ── 한글 폰트 설정 ─────────────────────────────────────────────────────────────
def set_korean_font():
    candidates = ['Malgun Gothic', 'Apple SD Gothic Neo',
                  'NanumGothic', 'NanumBarunGothic', 'DejaVu Sans']
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams['font.family'] = font
            break
    plt.rcParams['axes.unicode_minus'] = False

# ── 결합 시그널 로직 ───────────────────────────────────────────────────────────
def get_signal(v, d):
    """
    v: 변동성 발생 확률 (Step 1)
    d: 상승 방향 확률 (Step 2)
    """
    if v > 0.65 and d > 0.65: return '🟢 강력 매수'
    if v > 0.55 and d > 0.55: return '🟡 매수'
    if v > 0.65 and d < 0.35: return '🔴 강력 매도'
    return '⚪ 관망(중립)'

# ── 최종 모델 학습 ────────────────────────────────────────────────────────────
def train_final_models(df: pd.DataFrame, spw: float):
    X = df[FEATURES].values
    y1 = df['label_step1'].values
    y2 = df['label_step2'].values
    
    # 전체 데이터를 시간순으로 정렬하여 마지막 20%를 검증용으로 사용
    unique_dates = np.sort(df['date'].unique())
    split_pt = int(len(unique_dates) * 0.8)
    tr_dates = unique_dates[:split_pt]
    val_dates = unique_dates[split_pt:]
    
    tr_idx = df.index[df['date'].isin(tr_dates)]
    val_idx = df.index[df['date'].isin(val_dates)]
    
    scaler = RobustScaler()
    X_tr = scaler.fit_transform(X[tr_idx])
    X_val = scaler.transform(X[val_idx])
    
    m1 = get_model()
    m1.fit(X_tr, y1[tr_idx], eval_set=[(X_val, y1[val_idx])], verbose=False)
    
    move_mask_tr = (y1[tr_idx] == 1)
    move_mask_val = (y1[val_idx] == 1)
    m2 = get_model()
    m2.fit(X_tr[move_mask_tr], y2[tr_idx][move_mask_tr], 
           eval_set=[(X_val[move_mask_val], y2[val_idx][move_mask_val])], verbose=False)
           
    return m1, m2, scaler

# ── 메인 파이프라인 ────────────────────────────────────────────────────────────
def run_pipeline():
    set_korean_font()
    df = load_and_preprocess()
    X = df[FEATURES].values
    y1 = df['label_step1'].values
    y2 = df['label_step2'].values
    
    splits = list(get_date_splits(df))
    
    print(f"\n[2/5] 2단계 모델 교차 검증 시작")
    # ... (생략된 교차 검증 로직은 유지) ...
    for fold, (tr_idx, val_idx, te_idx) in enumerate(splits, 1):
        scaler = RobustScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_val = scaler.transform(X[val_idx])
        X_te = scaler.transform(X[te_idx])
        
        model_step1 = get_model()
        model_step1.fit(X_tr, y1[tr_idx], eval_set=[(X_val, y1[val_idx])], verbose=False)
        
        move_tr_mask = (y1[tr_idx] == 1)
        move_val_mask = (y1[val_idx] == 1)
        model_step2 = get_model()
        model_step2.fit(X_tr[move_tr_mask], y2[tr_idx][move_tr_mask], 
                        eval_set=[(X_val[move_val_mask], y2[val_idx][move_val_mask])], 
                        verbose=False)
        
        prob1 = model_step1.predict_proba(X_te)[:, 1]
        prob2 = model_step2.predict_proba(X_te)[:, 1]
        final_buy = (prob1 > 0.6) & (prob2 > 0.6)
        actual_up = (df.loc[te_idx, 'target_return'] >= MOVE_THRESH)
        prec = precision_score(actual_up, final_buy) if final_buy.any() else 0
        print(f"  Fold {fold} | 매수신호수: {final_buy.sum():>4} | 매수적적중률(Prec): {prec:.2%}")

    # --- 최종 모델 학습 ---
    print(f"\n[3/5] 최종 모델 학습 및 추론")
    y1_all = df['label_step1'].values
    spw = (y1_all == 0).sum() / (y1_all == 1).sum()
    m1, m2, scaler = train_final_models(df, spw)

    # --- 최근 데이터 예측 ---
    latest = df.sort_values('date').groupby('name').last().reset_index()
    X_lat = scaler.transform(latest[FEATURES].values)
    
    p1 = m1.predict_proba(X_lat)[:, 1]
    p2 = m2.predict_proba(X_lat)[:, 1]
    
    latest['vol_prob'] = p1
    latest['dir_prob'] = p2
    latest['signal'] = latest.apply(lambda r: get_signal(r['vol_prob'], r['dir_prob']), axis=1)
    
    print(f"\n[4/5] 최신 데이터 시그널 분포")
    print(latest['signal'].value_counts())
    
    # --- 시각화 및 결과 저장 ---
    print(f"\n[5/5] 결과 저장")
    res = latest[['name', 'date', 'close', 'vol_prob', 'dir_prob', 'signal']].sort_values('vol_prob', ascending=False)
    res.to_csv(os.path.join(BASE_DIR, 'two_step_predictions.csv'), index=False, encoding='utf-8-sig')
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    pd.Series(m1.feature_importances_, index=FEATURES).sort_values().plot(kind='barh', ax=axes[0], title='Step 1: 변동성 피처 중요도')
    pd.Series(m2.feature_importances_, index=FEATURES).sort_values().plot(kind='barh', ax=axes[1], title='Step 2: 방향성 피처 중요도')
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, 'two_step_analysis.png'))
    print("\n✅ 완료! two_step_predictions.csv 및 two_step_analysis.png 확인")

if __name__ == "__main__":
    run_pipeline()
