"""
dual_model_ensemble.py
----------------------
회귀 모델 (N일 뒤 수익률 예측) +
이진 분류 모델 (N일 내 치명적 하락 감지) 앙상블

[퀀트/ML 최적화 버전]
 - 엄격한 날짜 기반 TimeSeriesSplit (수평적 데이터 누수 방지)
 - Fold 내부 스케일링 (Data Leakage 완벽 차단)
 - XGBoost 최신 API 및 정규화 파라미터 적용
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             classification_report, roc_auc_score, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score,
                             ConfusionMatrixDisplay)
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# ── 경로 ──────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 중요: 수익률 계산을 위해 스케일링 전 원본 데이터(Raw Price)를 사용합니다.
CSV_PATH = os.path.join(BASE_DIR, "Quant_PRO_BigData_350.csv")

# ── 피처 및 글로벌 상수 ────────────────────────────────────────────────────────
FEATURES = [
    'retail_net_buy', 'foreign_net_buy', 'inst_net_buy', 'volume',
    'atr_14d', 'vol_change_1d', 'volatility_20d', 'disparity_20d',
    'rsi_14d', 'macd_osc', 'bb_lower_break', 'vkospi', 'relative_strength'
]

PRED_DAYS          =  5      # 기존 3일 -> 5일(1주간) 노이즈 감소 및 주간 모멘텀 포착
STRONG_BUY_THRESH  =  0.03   # 회귀: 강력 매수 (+3%)
BUY_THRESH         =  0.01   # 회귀: 매수 (+1%)
SELL_THRESH        = -0.01   # 회귀: 매도 (-1%)

DROP_LABEL_THRESH  = -0.01   # 분류: 하락 라벨 기준 (-1% 이하)
SELL_PROB_THRESH   =  0.70   # 분류: 앙상블 강력 매도 발동 확률 (주석과 수치 일치)

# ── 한글 폰트 ─────────────────────────────────────────────────────────────────
def set_korean_font():
    candidates = ['Malgun Gothic', 'Apple SD Gothic Neo',
                  'NanumGothic', 'NanumBarunGothic', 'DejaVu Sans']
    available = {f.name for f in fm.fontManager.ttflist}
    for font in candidates:
        if font in available:
            plt.rcParams['font.family'] = font
            break
    plt.rcParams['axes.unicode_minus'] = False

# ── 전처리 (스케일링 제외) ──────────────────────────────────────────────────────
def load_and_preprocess(csv_path: str) -> pd.DataFrame:
    print(f"[1/6] 데이터 로드 & 타겟 생성 (예측기간: {PRED_DAYS}일)")

    df = pd.read_csv(csv_path, parse_dates=['date'])
    df = df.sort_values(['name', 'date']).reset_index(drop=True)
    
    print(f"  원본 shape : {df.shape}")

    # ===== 타겟 생성 =====
    # 원본 종가(Raw Close)를 사용하여 정확한 수익률 계산
    df['target_return'] = (
        df.groupby('name')['close']
        .transform(lambda x: x.shift(-PRED_DAYS) / x - 1)
    )
    
    # DROP_LABEL_THRESH 상수 적용
    df['target_drop'] = (
        df['target_return'] <= DROP_LABEL_THRESH
    ).astype(int)

    # ===== 결측 및 이상치 처리 =====
    # 피처 데이터에서 무한대 값 제거
    df[FEATURES] = df[FEATURES].replace([np.inf, -np.inf], np.nan)
    
    # 타겟과 피처에 결측치가 있는 행 제거
    df = df.dropna(subset=FEATURES + ['target_return']).reset_index(drop=True)

    # 수익률 상하한 제한 (비정상적 변동 제거)
    df['target_return'] = df['target_return'].clip(-0.5, 0.5)

    drop_ratio = df['target_drop'].mean()
    print(f"  전처리 후 shape : {df.shape}")
    print(f"  하락 라벨 비율  : {drop_ratio:.2%}")

    return df

# ── 교차 검증 헬퍼 (날짜 기반 분할) ───────────────────────────────────────────
def get_date_based_splits(df: pd.DataFrame, n_splits: int = 5, val_ratio: float = 0.2):
    """
    동일 날짜의 데이터 누수(Cross-sectional Leakage)를 방지하기 위해 
    row 단위가 아닌 unique 'date' 단위로 TimeSeriesSplit을 수행합니다.
    """
    unique_dates = np.sort(df['date'].unique())
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    splits = []
    for train_date_idx, test_date_idx in tscv.split(unique_dates):
        train_dates_all = unique_dates[train_date_idx]
        test_dates = unique_dates[test_date_idx]
        
        # Train dates를 다시 Train / Validation으로 분할
        split_point = int(len(train_dates_all) * (1 - val_ratio))
        train_dates = train_dates_all[:split_point]
        val_dates = train_dates_all[split_point:]
        
        # 날짜를 인덱스로 매핑
        tr_idx = df.index[df['date'].isin(train_dates)].to_numpy()
        val_idx = df.index[df['date'].isin(val_dates)].to_numpy()
        te_idx = df.index[df['date'].isin(test_dates)].to_numpy()
        
        splits.append((tr_idx, val_idx, te_idx))
        
    return splits

# ── XGBoost 공통 파라미터 ─────────────────────────────────────────────────────
# 금융 데이터의 노이즈를 고려한 보수적 정규화 파라미터 세팅
xgb_base_params = {
    'n_estimators': 1000,
    'learning_rate': 0.03,
    'max_depth': 4,
    'min_child_weight': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,           # 리프 노드 추가 분할을 위한 최소 손실 감소율
    'reg_alpha': 0.1,       # L1 정규화 (Sparsity 유도)
    'reg_lambda': 3.0,      # L2 정규화 (가중치 폭발 방지)
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}

# ── 회귀 모델 교차 검증 ───────────────────────────────────────────────────────
def cv_regression(df: pd.DataFrame, n_splits: int = 5) -> dict:
    print(f"\n[2/6] 회귀 모델 교차 검증 (Date-based CV, n={n_splits})")
    
    splits = get_date_based_splits(df, n_splits=n_splits)
    X = df[FEATURES].values
    y = df['target_return'].values
    
    metrics = []
    for fold, (tr_idx, val_idx, te_idx) in enumerate(splits, 1):
        # 1. Fold 내부 스케일링 (Data Leakage 차단)
        scaler = RobustScaler()
        X_tr  = scaler.fit_transform(X[tr_idx])
        X_val = scaler.transform(X[val_idx])
        X_te  = scaler.transform(X[te_idx])
        
        y_tr, y_val, y_te = y[tr_idx], y[val_idx], y[te_idx]

        # 2. 모델 학습 (최신 Early Stopping API 적용)
        m = XGBRegressor(
            **xgb_base_params,
            objective='reg:squarederror',
            early_stopping_rounds=50,
            eval_metric='rmse'
        )
        
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        pred = m.predict(X_te)
        
        # 3. 평가
        rmse = np.sqrt(mean_squared_error(y_te, pred))
        mae  = mean_absolute_error(y_te, pred)
        r2   = r2_score(y_te, pred)
        metrics.append({'fold': fold, 'RMSE': rmse, 'MAE': mae, 'R2': r2})
        print(f"      Fold {fold} | RMSE {rmse:.4f} | MAE {mae:.4f} | R² {r2:.4f}")

    mdf = pd.DataFrame(metrics)
    print(f"  평균 → RMSE {mdf['RMSE'].mean():.4f} | MAE {mdf['MAE'].mean():.4f} | R² {mdf['R2'].mean():.4f}")
    return mdf.to_dict('list')

# ── 분류 모델 교차 검증 ───────────────────────────────────────────────────────
def cv_classifier(df: pd.DataFrame, n_splits: int = 5) -> tuple:
    print(f"\n[3/6] 분류 모델 교차 검증 (Date-based CV, n={n_splits})")
    
    y = df['target_drop'].values
    spw = (y == 0).sum() / (y == 1).sum()
    print(f"      scale_pos_weight (안전/하락 비율) = {spw:.2f}")

    splits = get_date_based_splits(df, n_splits=n_splits)
    X = df[FEATURES].values
    
    metrics = []
    y_te_all = []
    pred_all = []
    
    for fold, (tr_idx, val_idx, te_idx) in enumerate(splits, 1):
        # 1. Fold 내부 스케일링
        scaler = RobustScaler()
        X_tr  = scaler.fit_transform(X[tr_idx])
        X_val = scaler.transform(X[val_idx])
        X_te  = scaler.transform(X[te_idx])
        
        y_tr, y_val, y_te = y[tr_idx], y[val_idx], y[te_idx]

        # 2. 모델 학습
        m = XGBClassifier(
            **xgb_base_params,
            objective='binary:logistic',
            scale_pos_weight=spw,
            max_delta_step=1,       # 불균형 데이터 로지스틱 예측 안정화
            early_stopping_rounds=50,
            eval_metric='auc'
        )
        
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        prob = m.predict_proba(X_te)[:, 1]
        pred = (prob >= SELL_PROB_THRESH).astype(int)
        
        # 3. 평가
        auc = roc_auc_score(y_te, prob)
        acc = accuracy_score(y_te, pred)
        mac_f1 = f1_score(y_te, pred, average='macro')
        
        metrics.append({'fold': fold, 'AUC': auc, 'Accuracy': acc, 'Macro_F1': mac_f1})
        print(f"      Fold {fold} | AUC {auc:.4f} | Acc {acc:.4f} | Macro F1 {mac_f1:.4f}")
        
        # 마지막 Fold의 Confusion Matrix 기록을 위해 저장
        if fold == n_splits:
            cm = confusion_matrix(y_te, pred)

    mdf = pd.DataFrame(metrics)
    print(f"  평균 → AUC {mdf['AUC'].mean():.4f} | Acc {mdf['Accuracy'].mean():.4f} | Macro F1 {mdf['Macro_F1'].mean():.4f}")
    return mdf.to_dict('list'), spw, cm

# ── 최종 모델 학습 ────────────────────────────────────────────────────────────
def train_final_models(df: pd.DataFrame, spw: float):
    print(f"\n[4/6] 전체 데이터 최종 모델 학습 (Validation Split: 20%)")
    
    # 시간순 최신 20%를 Validation으로 사용
    unique_dates = np.sort(df['date'].unique())
    split_point = int(len(unique_dates) * 0.8)
    
    train_dates = unique_dates[:split_point]
    val_dates = unique_dates[split_point:]
    
    tr_idx = df.index[df['date'].isin(train_dates)]
    val_idx = df.index[df['date'].isin(val_dates)]
    
    X = df[FEATURES].values
    y_reg = df['target_return'].values
    y_cls = df['target_drop'].values
    
    # 최종 스케일러 생성 및 학습
    final_scaler = RobustScaler()
    X_tr = final_scaler.fit_transform(X[tr_idx])
    X_val = final_scaler.transform(X[val_idx])
    
    # 회귀 모델 최종 학습
    reg = XGBRegressor(
        **xgb_base_params, objective='reg:squarederror', 
        early_stopping_rounds=50, eval_metric='rmse'
    )
    reg.fit(X_tr, y_reg[tr_idx], eval_set=[(X_val, y_reg[val_idx])], verbose=False)
    print(f"      회귀 최종 모델  best_iteration: {reg.best_iteration}")

    # 분류 모델 최종 학습
    clf = XGBClassifier(
        **xgb_base_params, objective='binary:logistic', scale_pos_weight=spw, 
        max_delta_step=1, early_stopping_rounds=50, eval_metric='auc'
    )
    clf.fit(X_tr, y_cls[tr_idx], eval_set=[(X_val, y_cls[val_idx])], verbose=False)
    print(f"      분류 최종 모델  best_iteration: {clf.best_iteration}")

    return reg, clf, final_scaler

# ── 결합 시그널 ───────────────────────────────────────────────────────────────
def combined_signal(pred_ret: float, sell_prob: float) -> str:
    """분류 모델의 강력 매도 신호를 우선 적용 후, 회귀 모델 수익률에 따라 등급 분류"""
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
    print(f"\n[5/6] 최신 시점 전 종목 예측 추론")
    
    # 종목별 가장 마지막 날짜의 데이터만 추출
    latest = df.sort_values('date').groupby('name').last().reset_index()
    X_lat = scaler.transform(latest[FEATURES].values)

    latest['pred_return'] = reg.predict(X_lat)
    latest['sell_prob']   = clf.predict_proba(X_lat)[:, 1]
    latest['signal']      = latest.apply(
        lambda r: combined_signal(r['pred_return'], r['sell_prob']), axis=1
    )

    result = latest[['name', 'date', 'close', 'pred_return', 'sell_prob', 'signal']].sort_values(
        'pred_return', ascending=False
    ).reset_index(drop=True)

    print(f"\n  ── 투자 등급별 종목 수 ──")
    print(result['signal'].value_counts().to_string())
    
    print(f"\n  ── 🔴 강력 매도 Top 5 (위험 회피 추천) ──")
    strong_sell = result[result['signal'] == '🔴 강력 매도'].sort_values('sell_prob', ascending=False).head(5)
    print(strong_sell[['name', 'pred_return', 'sell_prob']].to_string(index=False))
    
    print(f"\n  ── 🟢 강력 매수 Top 5 (기대 수익률 상위) ──")
    strong_buy = result[result['signal'] == '🟢 강력 매수'].head(5)
    print(strong_buy[['name', 'pred_return', 'sell_prob']].to_string(index=False))

    return result

# ── 시각화 ────────────────────────────────────────────────────────────────────
def visualize(result: pd.DataFrame, reg, clf, cv_reg: dict, cv_clf: dict, last_cm: np.ndarray):
    print(f"\n[6/6] 성과 지표 및 대시보드 시각화 저장")
    set_korean_font()
    
    # 3x3 그리드로 확장하여 상세 지표 추가
    fig = plt.figure(figsize=(24, 16))
    fig.suptitle('한국 주식 퀀트 이중 모델 대시보드 (수익률 & 리스크 방어)', fontsize=18, fontweight='bold')

    # 1. 회귀 피처 중요도
    ax1 = plt.subplot(3, 3, 1)
    imp1 = pd.Series(reg.feature_importances_, index=FEATURES).sort_values()
    imp1.plot(kind='barh', ax=ax1, color='#3498db')
    ax1.set_title('회귀 모델 피처 중요도 (기대 수익률)')

    # 2. 분류 피처 중요도
    ax2 = plt.subplot(3, 3, 2)
    imp2 = pd.Series(clf.feature_importances_, index=FEATURES).sort_values()
    imp2.plot(kind='barh', ax=ax2, color='#e74c3c')
    ax2.set_title('분류 모델 피처 중요도 (하락 위험 감지)')

    # 3. 투자 등급 분포
    ax3 = plt.subplot(3, 3, 3)
    sig_counts = result['signal'].value_counts()
    color_map = {'🔴 강력 매도': '#c0392b', '🔵 매도': '#2980b9', '⚪ 중립': '#95a5a6', '🟡 매수': '#e67e22', '🟢 강력 매수': '#27ae60'}
    colors = [color_map.get(s, '#95a5a6') for s in sig_counts.index]
    sig_counts.plot(kind='bar', ax=ax3, color=colors, edgecolor='white')
    ax3.set_title('전체 마켓 투자 등급별 분포')
    ax3.tick_params(axis='x', rotation=0)

    # 4. 수익률 분포
    ax4 = plt.subplot(3, 3, 4)
    ax4.hist(result['pred_return'] * 100, bins=50, color='#3498db', alpha=0.8)
    ax4.axvline(STRONG_BUY_THRESH * 100, color='green', linestyle='--', label=f'강력매수 {STRONG_BUY_THRESH*100}%')
    ax4.axvline(SELL_THRESH * 100, color='red', linestyle='--', label=f'매도 {SELL_THRESH*100}%')
    ax4.set_title('최신 데이터 예측 수익률 분포 (%)')
    ax4.legend(fontsize=9)

    # 5. 하락 확률 분포
    ax5 = plt.subplot(3, 3, 5)
    ax5.hist(result['sell_prob'], bins=50, color='#e74c3c', alpha=0.8)
    ax5.axvline(SELL_PROB_THRESH, color='black', linestyle='--', label=f'강력매도 임계값 {SELL_PROB_THRESH}')
    ax5.set_title('최신 데이터 하락 확률 분포')
    ax5.legend(fontsize=9)

    # 6. CV Fold 별 스코어 흐름
    ax6 = plt.subplot(3, 3, 6)
    folds = cv_reg['fold']
    ax6.plot(folds, cv_reg['RMSE'], 'o-', color='#3498db', label='회귀 RMSE')
    ax6_2 = ax6.twinx()
    ax6_2.plot(folds, cv_clf['AUC'], 's--', color='#e74c3c', label='분류 AUC')
    ax6.set_title('Date-based CV 성과 흐름')
    ax6.set_xticks(folds)
    lines, labels = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_2.get_legend_handles_labels()
    ax6.legend(lines + lines2, labels + labels2, loc='upper left')

    # 7. Confusion Matrix (마지막 Fold 기준)
    ax7 = plt.subplot(3, 3, 7)
    disp = ConfusionMatrixDisplay(confusion_matrix=last_cm, display_labels=['Safe (0)', 'Drop (1)'])
    disp.plot(ax=ax7, cmap='Blues', colorbar=False)
    ax7.set_title('Confusion Matrix (최종 Validation)')

    plt.tight_layout()
    save_path = os.path.join(BASE_DIR, 'dual_model_analysis.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"      저장: {save_path}")

# ── 메인 ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df = load_and_preprocess(CSV_PATH)

    cv_reg            = cv_regression(df)
    cv_clf, spw, cm   = cv_classifier(df)

    reg, clf, scaler  = train_final_models(df, spw)
    result            = predict_latest(df, reg, clf, scaler)

    visualize(result, reg, clf, cv_reg, cv_clf, cm)

    out_csv = os.path.join(BASE_DIR, 'dual_model_predictions.csv')
    result.to_csv(out_csv, index=False, encoding='utf-8-sig')
    print(f"\n  결과 CSV: {out_csv}")
    print("\n✅ 파이프라인 분석 및 예측 완료!")
