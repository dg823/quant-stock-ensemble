from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from xgboost import XGBRegressor, XGBClassifier

def tune_hyperparameters(df, spw):
    df_cv = df.sort_values('date').reset_index(drop=True)
    X = df_cv[FEATURES].values
    y_reg = df_cv['target_return'].values
    y_cls = df_cv['target_drop'].values

    tscv = TimeSeriesSplit(n_splits=5)

    param_grid = {
        'max_depth': [3, 4, 5, 6, 7],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.6, 0.7, 0.8, 0.9],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
        'n_estimators': [300, 500, 1000]
    }

    reg = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
    reg_search = RandomizedSearchCV(
        estimator=reg,
        param_distributions=param_grid,
        n_iter=10,
        scoring='neg_root_mean_squared_error',
        cv=tscv,
        random_state=42,
        n_jobs=-1
    )
    reg_search.fit(X, y_reg)

    clf = XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=spw,
        random_state=42,
        n_jobs=-1
    )
    clf_search = RandomizedSearchCV(
        estimator=clf,
        param_distributions=param_grid,
        n_iter=10,
        scoring='roc_auc',
        cv=tscv,
        random_state=42,
        n_jobs=-1
    )
    clf_search.fit(X, y_cls)

    return reg_search.best_params_, clf_search.best_params_