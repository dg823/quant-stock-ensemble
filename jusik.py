"""
jusik.py (Final Edition - Strict Features & MLP)
----------------------
회귀 모델 (3일 뒤 수익률 예측) + 이진 분류 모델 (하락 감지) 앙상블
목표: Acc, Prec, Rec, F1 > 0.5
제약: 원래 제공된 CSV 컬럼만 사용 (추가 피처 생성 금지)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             precision_score, recall_score, f1_score, accuracy_score, roc_auc_score)

# ── 경로 ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Quant_Scaled_ML_Ready.csv")

# ── 피처 (CSV 헤더에 있는 것만 사용) ──────────────────────────────────────────
FEATURES = [
    'retail_net_buy', 'foreign_net_buy', 'inst_net_buy',
    'close', 'volume', 'atr_14d', 'volatility_20d', 
    'disparity_20d', 'rsi_14d', 'vol_change_1d', 
    'macd_osc', 'bb_lower_break', 'vkospi', 'relative_strength'
]

# ── 임계값 ────────────────────────────────────────────────────────────────────
STRONG_BUY_THRESH  =  0.03
BUY_THRESH         =  0.01
SELL_THRESH        = -0.01
DROP_LABEL_THRESH  = -0.01  # 하락 라벨 기준 (-1% 이하 → 1)
SELL_PROB_THRESH   =  0.50  # 분류 모델 매도 확률 기준

# ── 한글 폰트 ─────────────────────────────────────────────────────────────────
def set_korean_font():
    candidates = ['Malgun Gothic', 'Apple SD Gothic Neo', 'NanumGothic', 'NanumBarunGothic', 'DejaVu Sans']
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams['font.family'] = font
            break
    plt.rcParams['axes.unicode_minus'] = False

# ── 전처리 ────────────────────────────────────────────────────────────────────
def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    print("[1/6] 데이터 로드 & 전처리")
    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df.sort_values(['name', 'date']).reset_index(drop=True)

    # 타겟 생성
    df['target_return'] = df.groupby('name')['close'].transform(lambda x: x.shift(-3) / x - 1)
    df['target_drop'] = (df['target_return'] <= DROP_LABEL_THRESH).astype(int)

    # 결측치 제거
    df = df.dropna(subset=FEATURES + ['target_return'])
    return df

# ── 분류 CV ───────────────────────────────────────────────────────────────────
def cv_classifier(df: pd.DataFrame) -> dict:
    print(f"\n[2/6] 이진 분류 모델 TimeSeriesSplit CV")
    df_cv = df.sort_values('date').reset_index(drop=True)
    X, y = df_cv[FEATURES].values, df_cv['target_drop'].values
    tscv = TimeSeriesSplit(n_splits=5)
    metrics = []

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X), 1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_te, y_te = X[te_idx], y[te_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        m = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42, early_stopping=True)
        m.fit(X_tr_s, y_tr)

        prob = m.predict_proba(X_te_s)[:, 1]
        pred = (prob >= SELL_PROB_THRESH).astype(int)
        
        acc = accuracy_score(y_te, pred)
        prec = precision_score(y_te, pred, zero_division=0)
        rec = recall_score(y_te, pred, zero_division=0)
        f1 = f1_score(y_te, pred, zero_division=0)
        auc = roc_auc_score(y_te, prob)

        metrics.append({'fold': fold, 'Acc': acc, 'Prec': prec, 'Rec': rec, 'F1': f1, 'AUC': auc})
        print(f"      Fold {fold} | Acc {acc:.4f} | Prec {prec:.4f} | Rec {rec:.4f} | F1 {f1:.4f} | AUC {auc:.4f}")

    mdf = pd.DataFrame(metrics)
    print(f"  평균 → Acc {mdf['Acc'].mean():.4f} | Prec {mdf['Prec'].mean():.4f} | Rec {mdf['Rec'].mean():.4f} | F1 {mdf['F1'].mean():.4f}")
    return mdf.to_dict('list')

# ── 최종 모델 학습 ────────────────────────────────────────────────────────────
def train_final_models(df: pd.DataFrame):
    print(f"\n[3/6] 최종 모델 학습 (MLP)")
    df_s = df.sort_values('date').reset_index(drop=True)
    X = df_s[FEATURES].values
    y_reg = df_s['target_return'].values
    y_cls = df_s['target_drop'].values

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    reg = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42, early_stopping=True)
    reg.fit(X_s, y_reg)

    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42, early_stopping=True)
    clf.fit(X_s, y_cls)

    return reg, clf, scaler

# ── 결합 시그널 ───────────────────────────────────────────────────────────────
def combined_signal(pred_ret: float, sell_prob: float) -> str:
    if sell_prob >= SELL_PROB_THRESH:
        return '🔴 강력 매도'
    if pred_ret >= STRONG_BUY_THRESH:
        return '🟢 강력 매수'
    if pred_ret >= BUY_THRESH:
        return '🟡 매수'
    if pred_ret >= SELL_THRESH:
        return '⚪ 중립'
    return '🔵 매도'

# ── 최근 데이터 예측 ──────────────────────────────────────────────────────────
def predict_latest(df: pd.DataFrame, reg, clf, scaler) -> pd.DataFrame:
    print(f"\n[4/6] 최근 데이터 기준 종목별 예측")
    latest = df.sort_values('date').groupby('name').last().reset_index()
    X_lat  = scaler.transform(latest[FEATURES].values)

    latest['pred_return']   = reg.predict(X_lat)
    latest['sell_prob']     = clf.predict_proba(X_lat)[:, 1]
    latest['signal'] = latest.apply(
        lambda r: combined_signal(r['pred_return'], r['sell_prob']), axis=1
    )

    result = latest[['name', 'date', 'close',
                      'pred_return', 'sell_prob', 'signal']].sort_values(
        'pred_return', ascending=False
    ).reset_index(drop=True)

    print(f"\n  ── 투자 등급별 종목 수 ──")
    print(result['signal'].value_counts().to_string())
    return result

# ── 시각화 ────────────────────────────────────────────────────────────────────
def visualize(result: pd.DataFrame, cv_clf: dict):
    print(f"\n[5/6] 시각화 저장")
    set_korean_font()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('한국 주식 위험 경보 — MLP 이중 모델', fontsize=14, fontweight='bold')

    # 1. 투자 등급 분포
    ax = axes[0]
    sig_counts = result['signal'].value_counts()
    color_map = {
        '🔴 강력 매도': '#c0392b', '🔵 매도': '#2980b9',
        '⚪ 중립': '#95a5a6', '🟡 매수': '#e67e22', '🟢 강력 매수': '#27ae60'
    }
    colors = [color_map.get(s, '#95a5a6') for s in sig_counts.index]
    sig_counts.plot(kind='bar', ax=ax, color=colors, edgecolor='white')
    ax.set_title('투자 등급별 종목 분포')
    ax.set_ylabel('종목 수')

    # 2. CV 성능 추이
    ax = axes[1]
    folds = cv_clf['fold']
    ax.plot(folds, cv_clf['Acc'], 'o-', label='Accuracy')
    ax.plot(folds, cv_clf['F1'], 's-', label='F1-Score')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, label='목표 (0.5)')
    ax.set_title('CV Fold별 성능')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Score')
    ax.set_xticks(folds)
    ax.legend()

    plt.tight_layout()
    save_path = os.path.join(BASE_DIR, 'dual_model_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      저장: {save_path}")

# ── 메인 ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df = load_and_preprocess(CSV_PATH)

    cv_clf_results    = cv_classifier(df)
    reg, clf, scaler  = train_final_models(df)
    result            = predict_latest(df, reg, clf, scaler)

    visualize(result, cv_clf_results)

    out_csv = os.path.join(BASE_DIR, 'dual_model_predictions.csv')
    result.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"\n  결과 CSV: {out_csv}")

    print("\n✅ 완료!")
    print("   · dual_model_analysis.png")
    print("   · dual_model_predictions.csv")
