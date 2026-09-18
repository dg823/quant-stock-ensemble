import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
import warnings
warnings.filterwarnings('ignore')

def set_korean_font():
    candidates = ['Malgun Gothic', 'Apple SD Gothic Neo', 'NanumGothic', 'NanumBarunGothic']
    for font in candidates:
        if font in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams['font.family'] = font
            break
    plt.rcParams['axes.unicode_minus'] = False
    plt.style.use('dark_background')

# =====================================================================
# 1. 데이터 로드
# =====================================================================
print("1. 데이터 로드...")
df = pd.read_csv('Quant_Scaled_ML_Ready.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['name', 'date']).reset_index(drop=True)

df['close_change_1d']  = df.groupby('name')['close'].pct_change(1)
df['close_change_20d'] = df.groupby('name')['close'].pct_change(20)

# =====================================================================
# 2. 종속변수 생성
# =====================================================================
print("2. 종속변수 생성 (중간 60% 제거)...")
df['return_5d'] = df.groupby('name')['close'].shift(-5) / df['close'] - 1
df = df.replace([np.inf, -np.inf], np.nan)

features = [
    'retail_net_buy', 'foreign_net_buy', 'inst_net_buy', 'close', 'volume',
    'atr_14d', 'volatility_20d', 'disparity_20d', 'rsi_14d', 'vol_change_1d',
    'macd_osc', 'bb_lower_break', 'vkospi', 'relative_strength',
    'close_change_1d', 'close_change_20d'
]

df = df.dropna(subset=features + ['return_5d']).copy()

q_low  = df['return_5d'].quantile(0.20)
q_high = df['return_5d'].quantile(0.80)

df['label'] = np.nan
df.loc[df['return_5d'] <= q_low,  'label'] = 0
df.loc[df['return_5d'] >= q_high, 'label'] = 1
df = df.dropna(subset=['label']).copy()
df['label'] = df['label'].astype(int)

# =====================================================================
# 3. Train/Test 분할
# =====================================================================
print("3. Train/Test 분할...")
df = df.sort_values('date').reset_index(drop=True)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df  = df.iloc[split_idx:]

X_train, y_train = train_df[features], train_df['label']
X_test,  y_test  = test_df[features],  test_df['label']
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# =====================================================================
# 4. XGBoost 튜닝
# =====================================================================
print("\n4. XGBoost 튜닝 시작... (약 10~15분)")

model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    tree_method='hist',
    random_state=42,
    n_jobs=-1
)

param_dist = {
    'n_estimators':     [100, 200, 300, 500],
    'learning_rate':    [0.01, 0.05, 0.1, 0.2],
    'max_depth':        [3, 4, 6, 8],
    'subsample':        [0.6, 0.7, 0.8, 0.9],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
    'min_child_weight': [1, 3, 5, 7],
    'gamma':            [0, 0.1, 0.3, 0.5]
}

tscv = TimeSeriesSplit(n_splits=3)
random_search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=30,
    scoring='f1',
    cv=tscv,
    verbose=2,
    random_state=42,
    n_jobs=1
)
random_search.fit(X_train, y_train, sample_weight=sample_weights)
best_params = random_search.best_params_
print(f"\n최적 파라미터: {best_params}")

best_model = random_search.best_estimator_
proba = best_model.predict_proba(X_test)[:, 1]

# =====================================================================
# 5. 성능 리포트
# =====================================================================
y_pred = (proba >= 0.5).astype(int)
print("\n" + "="*60)
print("                   [ 성능 리포트 (threshold=0.5) ]")
print("="*60)
print(classification_report(y_test, y_pred, target_names=['매도(0)', '매수(1)']))

# =====================================================================
# 6. 시각화 1: Threshold별 성능 비교
# =====================================================================
print("\n시각화 1: Threshold별 성능 비교...")
set_korean_font()

thresholds = np.arange(0.30, 0.75, 0.05)
sell_precision_list, buy_precision_list = [], []
sell_recall_list,    buy_recall_list    = [], []

for t in thresholds:
    pred = (proba >= t).astype(int)
    if len(np.unique(pred)) < 2:
        sell_precision_list.append(np.nan)
        buy_precision_list.append(np.nan)
        sell_recall_list.append(np.nan)
        buy_recall_list.append(np.nan)
        continue
    r = classification_report(y_test, pred, output_dict=True, zero_division=0)
    sell_precision_list.append(r['0']['precision'])
    buy_precision_list.append(r['1']['precision'])
    sell_recall_list.append(r['0']['recall'])
    buy_recall_list.append(r['1']['recall'])

fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='#0B0E14')
fig.patch.set_facecolor('#0B0E14')

ax1 = axes[0]
ax1.set_facecolor('#131722')
ax1.plot(thresholds, sell_precision_list, 'o-', color='#F23645', label='매도 Precision', linewidth=2)
ax1.plot(thresholds, buy_precision_list,  'o-', color='#2962FF', label='매수 Precision', linewidth=2)
ax1.axhline(y=0.5, color='white', linestyle='--', alpha=0.5, label='0.5 기준선')
ax1.set_title('Threshold별 Precision', color='white', fontsize=14, fontweight='bold')
ax1.set_xlabel('Threshold', color='#D1D4DC')
ax1.set_ylabel('Precision', color='#D1D4DC')
ax1.tick_params(colors='#D1D4DC')
ax1.legend(facecolor='#131722', labelcolor='white')
ax1.grid(color='#2A2E39', linestyle='--', alpha=0.5)

ax2 = axes[1]
ax2.set_facecolor('#131722')
ax2.plot(thresholds, sell_recall_list, 'o-', color='#F23645', label='매도 Recall', linewidth=2)
ax2.plot(thresholds, buy_recall_list,  'o-', color='#2962FF', label='매수 Recall', linewidth=2)
ax2.axhline(y=0.5, color='white', linestyle='--', alpha=0.5, label='0.5 기준선')
ax2.set_title('Threshold별 Recall', color='white', fontsize=14, fontweight='bold')
ax2.set_xlabel('Threshold', color='#D1D4DC')
ax2.set_ylabel('Recall', color='#D1D4DC')
ax2.tick_params(colors='#D1D4DC')
ax2.legend(facecolor='#131722', labelcolor='white')
ax2.grid(color='#2A2E39', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('threshold_analysis.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()
print("✅ threshold_analysis.png 저장 완료!")

# =====================================================================
# 7. 시각화 2: max_depth별 성능 비교
# =====================================================================
print("\n시각화 2: max_depth별 성능 비교... (약 5분)")

max_depths = [2, 3, 4, 5, 6, 7, 8, 10]
depth_results = []

for depth in max_depths:
    m = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
        n_estimators=best_params['n_estimators'],
        learning_rate=best_params['learning_rate'],
        subsample=best_params['subsample'],
        colsample_bytree=best_params['colsample_bytree'],
        min_child_weight=best_params['min_child_weight'],
        gamma=best_params['gamma'],
        max_depth=depth
    )
    m.fit(X_train, y_train, sample_weight=sample_weights)
    p = m.predict_proba(X_test)[:, 1]
    pred = (p >= 0.5).astype(int)
    r = classification_report(y_test, pred, output_dict=True, zero_division=0)
    depth_results.append({
        'max_depth': depth,
        'sell_precision': r['0']['precision'],
        'buy_precision':  r['1']['precision'],
        'sell_recall':    r['0']['recall'],
        'buy_recall':     r['1']['recall'],
        'accuracy':       r['accuracy']
    })
    print(f"  max_depth={depth} | 매도P: {r['0']['precision']:.3f} | 매수P: {r['1']['precision']:.3f} | 매도R: {r['0']['recall']:.3f} | 매수R: {r['1']['recall']:.3f}")

depth_df = pd.DataFrame(depth_results)

fig, axes = plt.subplots(1, 2, figsize=(16, 6), facecolor='#0B0E14')
fig.patch.set_facecolor('#0B0E14')

ax1 = axes[0]
ax1.set_facecolor('#131722')
ax1.plot(depth_df['max_depth'], depth_df['sell_precision'], 'o-', color='#F23645', label='매도 Precision', linewidth=2)
ax1.plot(depth_df['max_depth'], depth_df['buy_precision'],  'o-', color='#2962FF', label='매수 Precision', linewidth=2)
ax1.plot(depth_df['max_depth'], depth_df['accuracy'],       'o-', color='#FFD700', label='Accuracy',       linewidth=2)
ax1.set_title('max_depth별 Precision', color='white', fontsize=14, fontweight='bold')
ax1.set_xlabel('max_depth', color='#D1D4DC')
ax1.set_ylabel('Score', color='#D1D4DC')
ax1.tick_params(colors='#D1D4DC')
ax1.legend(facecolor='#131722', labelcolor='white')
ax1.grid(color='#2A2E39', linestyle='--', alpha=0.5)

ax2 = axes[1]
ax2.set_facecolor('#131722')
ax2.plot(depth_df['max_depth'], depth_df['sell_recall'], 'o-', color='#F23645', label='매도 Recall', linewidth=2)
ax2.plot(depth_df['max_depth'], depth_df['buy_recall'],  'o-', color='#2962FF', label='매수 Recall', linewidth=2)
ax2.set_title('max_depth별 Recall', color='white', fontsize=14, fontweight='bold')
ax2.set_xlabel('max_depth', color='#D1D4DC')
ax2.set_ylabel('Score', color='#D1D4DC')
ax2.tick_params(colors='#D1D4DC')
ax2.legend(facecolor='#131722', labelcolor='white')
ax2.grid(color='#2A2E39', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('max_depth_analysis.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()
print("✅ max_depth_analysis.png 저장 완료!")

# =====================================================================
# 8. 시각화 3: 하이퍼파라미터별 성능 비교
# =====================================================================
print("\n시각화 3: 주요 하이퍼파라미터별 성능 비교... (약 10분)")

param_experiments = {
    'n_estimators':    [100, 200, 300, 500],
    'learning_rate':   [0.01, 0.05, 0.1, 0.2],
    'subsample':       [0.6, 0.7, 0.8, 0.9],
    'min_child_weight':[1, 3, 5, 7],
}

fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='#0B0E14')
fig.patch.set_facecolor('#0B0E14')
axes = axes.flatten()

for idx, (param_name, param_values) in enumerate(param_experiments.items()):
    results = []
    for val in param_values:
        params = {
            'n_estimators':     best_params['n_estimators'],
            'learning_rate':    best_params['learning_rate'],
            'max_depth':        best_params['max_depth'],
            'subsample':        best_params['subsample'],
            'colsample_bytree': best_params['colsample_bytree'],
            'min_child_weight': best_params['min_child_weight'],
            'gamma':            best_params['gamma'],
        }
        params[param_name] = val

        m = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            tree_method='hist',
            random_state=42,
            n_jobs=-1,
            **params
        )
        m.fit(X_train, y_train, sample_weight=sample_weights)
        p = m.predict_proba(X_test)[:, 1]
        pred = (p >= 0.5).astype(int)
        r = classification_report(y_test, pred, output_dict=True, zero_division=0)
        results.append({
            'value':         val,
            'sell_recall':   r['0']['recall'],
            'buy_precision': r['1']['precision'],
            'accuracy':      r['accuracy']
        })
        print(f"  {param_name}={val} | 매도R: {r['0']['recall']:.3f} | 매수P: {r['1']['precision']:.3f}")

    res_df = pd.DataFrame(results)
    ax = axes[idx]
    ax.set_facecolor('#131722')
    ax.plot(range(len(param_values)), res_df['sell_recall'],   'o-', color='#F23645', label='매도 Recall',    linewidth=2)
    ax.plot(range(len(param_values)), res_df['buy_precision'], 'o-', color='#2962FF', label='매수 Precision', linewidth=2)
    ax.plot(range(len(param_values)), res_df['accuracy'],      'o-', color='#FFD700', label='Accuracy',       linewidth=2)
    ax.set_title(f'{param_name}별 성능', color='white', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(param_values)))
    ax.set_xticklabels([str(v) for v in param_values], color='#D1D4DC')
    ax.set_xlabel(param_name, color='#D1D4DC')
    ax.set_ylabel('Score', color='#D1D4DC')
    ax.tick_params(colors='#D1D4DC')
    ax.legend(facecolor='#131722', labelcolor='white', fontsize=9)
    ax.grid(color='#2A2E39', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('hyperparam_analysis.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()
print("✅ hyperparam_analysis.png 저장 완료!")

# =====================================================================
# 9. Feature Importance
# =====================================================================
importance = pd.Series(best_model.feature_importances_, index=features).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 8), facecolor='#0B0E14')
ax.set_facecolor('#131722')
colors = ['#2962FF' if 'buy' in f or 'inst' in f
          else '#F23645' if 'strength' in f or 'break' in f
          else '#FFD700' for f in importance.index]
importance.plot(kind='barh', ax=ax, color=colors, edgecolor='none')
ax.set_title('Feature 중요도', color='white', fontsize=14, fontweight='bold', pad=15)
ax.tick_params(colors='#D1D4DC', labelsize=10)
ax.grid(axis='x', color='#2A2E39', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()
print("✅ feature_importance.png 저장 완료!")