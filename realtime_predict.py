import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import os
from two_step_ensemble import load_and_preprocess, train_final_models, get_signal, FEATURES
from sklearn.preprocessing import RobustScaler
import warnings

warnings.filterwarnings('ignore')

# ── 설정 ──────────────────────────────────────────────────────────────────────
# 주요 종목 리스트 (삼성전자, SK하이닉스, 현대차, POSCO홀딩스, LG화학 등)
STOCK_CODES = {
    '삼성전자': '005930',
    'SK하이닉스': '000660',
    '현대차': '005380',
    'POSCO홀딩스': '005490',
    'LG화학': '051910',
    'NAVER': '035420',
    '카카오': '035720',
    '기아': '000270',
    '셀트리온': '068270',
    '삼성SDI': '006400'
}

def calculate_technical_indicators(df):
    """FinanceDataReader에서 가져온 OHLCV 데이터를 기반으로 모델 피처 계산"""
    # 1. ATR (14일)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr_14d'] = tr.rolling(window=14).mean()
    
    # 2. RSI (14일)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14d'] = 100 - (100 / (1 + rs))
    
    # 3. 변동성 (20일)
    df['volatility_20d'] = df['Close'].pct_change().rolling(window=20).std()
    
    # 4. 이격도 (Disparity 20일)
    df['disparity_20d'] = df['Close'] / df['Close'].rolling(window=20).mean()
    
    # 5. 거래량 변화율
    df['vol_change_1d'] = df['Volume'].pct_change()
    
    # 6. MACD Oscillator
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    df['macd_osc'] = macd - signal

    # 7. BB Lower Break (볼린저 밴드 하단 이탈)
    ma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['bb_lower_break'] = (df['Close'] < (ma20 - 2 * std20)).astype(int)

    return df

def run_realtime_predict():
    print("🚀 [최신 데이터 예측] 모델 학습 및 최신 주가 데이터 수집 중...")
    
    # 1. 기존 데이터로 최종 모델 학습
    df_base = load_and_preprocess()
    y_base = df_base['label_step1'].values
    spw = (y_base == 0).sum() / (y_base == 1).sum()
    m1, m2, scaler = train_final_models(df_base, spw)
    
    print("\n" + "="*80)
    print(f"{'종목명':<12} | {'현재가':>10} | {'변동확률':>8} | {'상승확률':>8} | {'최종 시그널'}")
    print("-" * 80)
    
    results = []
    for name, code in STOCK_CODES.items():
        try:
            # 2. 최신 주가 로드 (최근 100일)
            df_live = fdr.DataReader(code)
            if df_live.empty: continue
            
            df_live = calculate_technical_indicators(df_live)
            latest = df_live.iloc[-1]
            
            # 3. 피처 정렬 (FEATURES 리스트와 동일하게)
            # 수급(net_buy) 및 시장(vkospi, relative_strength)은 실시간 API 부재로 
            # 기존 학습 데이터의 최근 평균값 혹은 0으로 임시 보완
            input_row = {}
            for f in FEATURES:
                if f in latest:
                    input_row[f] = latest[f]
                else:
                    # 수급 데이터 등은 보수적으로 평균값 처리
                    input_row[f] = df_base[f].mean()
            
            input_df = pd.DataFrame([input_row])
            X_scaled = scaler.transform(input_df)
            
            # 4. 모델 예측
            v_prob = m1.predict_proba(X_scaled)[0, 1]
            d_prob = m2.predict_proba(X_scaled)[0, 1]
            signal = get_signal(v_prob, d_prob)
            
            print(f"{name:<12} | {df_live['Close'].iloc[-1]:>10,.0f}원 | {v_prob:>8.2%} | {d_prob:>8.2%} | {signal}")
            
        except Exception as e:
            print(f"[{name}] 데이터 로드 실패: {e}")

    print("="*80)
    print("✅ 최신 시세 기반 예측이 완료되었습니다.")

if __name__ == "__main__":
    run_realtime_predict()
