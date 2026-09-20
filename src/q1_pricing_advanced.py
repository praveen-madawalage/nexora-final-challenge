import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import joblib
import math
import os
import warnings
import json

warnings.filterwarnings('ignore')

TEST_HORIZON = 30
FEATURES = ['lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14', 'lag_30',
            'roll_mean_7d', 'roll_mean_30d', 'roll_std_30d',
            'dayofweek', 'month', 'day_sin', 'day_cos']
CATEGORICAL = ['market_encoded']


def add_features_vectorized(df):
    """Adds time series features vectorially to the entire dataframe."""
    df = df.copy()
    
    # Time features
    df['dayofweek'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day_sin'] = np.sin(2 * np.pi * df['date'].dt.day / 31.0)
    df['day_cos'] = np.cos(2 * np.pi * df['date'].dt.day / 31.0)
    
    # Sort carefully before grouping
    df = df.sort_values(['market', 'date'])
    
    # Lag features
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f'lag_{lag}'] = df.groupby('market')['price'].shift(lag)
        
    # Rolling features
    df['roll_mean_7d'] = df.groupby('market')['price'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    df['roll_mean_30d'] = df.groupby('market')['price'].transform(lambda x: x.rolling(window=30, min_periods=1).mean())
    df['roll_std_30d'] = df.groupby('market')['price'].transform(lambda x: x.rolling(window=30, min_periods=1).std().fillna(0.0))
    
    # Fill initial NaNs in lags with the first available price (or bfill)
    for lag in [1, 2, 3, 7, 14, 30]:
        df[f'lag_{lag}'] = df.groupby('market')[f'lag_{lag}'].bfill()
        
    return df


def load_and_split_data(filepath: str):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    
    # Add features before splitting!
    df = add_features_vectorized(df)
    
    train_list = []
    test_list = []
    
    for market, grp in df.groupby('market'):
        grp = grp.sort_values('date')
        if len(grp) <= TEST_HORIZON:
            raise ValueError(f"Not enough data for market {market}")
        
        train_list.append(grp.iloc[:-TEST_HORIZON])
        test_list.append(grp.iloc[-TEST_HORIZON:])
        
    df_train = pd.concat(train_list).reset_index(drop=True)
    df_test = pd.concat(test_list).reset_index(drop=True)
    return df, df_train, df_test


def evaluate_baselines(df_train, df_test):
    results = []
    forecast_dfs = []
    
    arima_orders = [(1,1,1), (0,1,1), (5,1,0)]
    
    for market in df_train['market'].unique():
        market_train = df_train[df_train['market'] == market]
        market_test = df_test[df_test['market'] == market]
        currency = market_train['currency'].iloc[0]
        
        train_prices = market_train['price'].values
        test_prices = market_test['price'].values
        test_dates = market_test['date'].values
        
        # 1. Naive
        naive_pred = np.full(TEST_HORIZON, train_prices[-1])
        
        # 2. ES
        try:
            es = ExponentialSmoothing(train_prices, trend='add', seasonal=None, initialization_method='estimated').fit()
            es_pred = es.forecast(TEST_HORIZON)
        except:
            es_pred = naive_pred
            
        # 3. ARIMA Grid Search
        best_arima_pred = naive_pred
        best_aic = float('inf')
        for order in arima_orders:
            try:
                model = ARIMA(train_prices, order=order).fit()
                if model.aic < best_aic:
                    best_aic = model.aic
                    best_arima_pred = model.forecast(TEST_HORIZON)
            except:
                continue
                
        for name, preds in [('Naive Persistence', naive_pred), ('Exponential Smoothing', es_pred), ('Best ARIMA', best_arima_pred)]:
            rmse = np.sqrt(mean_squared_error(test_prices, preds))
            mape = mean_absolute_percentage_error(test_prices, preds)
            results.append({'market': market, 'model_name': name, 'rmse': rmse, 'mape': mape, 'test_records': TEST_HORIZON})
            
            forecast_dfs.append(pd.DataFrame({
                'market': market, 'currency': currency, 'date': test_dates, 
                'actual_price': test_prices, 'predicted_price': preds, 
                'error': test_prices - preds, 'model_name': name, 
                'horizon_step': np.arange(1, TEST_HORIZON + 1)
            }))
            
    return pd.DataFrame(results), pd.concat(forecast_dfs, ignore_index=True)


def train_lightgbm_direct(df_train):
    market_mapping = {m: i for i, m in enumerate(df_train['market'].unique())}
    df_train['market_encoded'] = df_train['market'].map(market_mapping)
    
    models = {}
    
    # Train 30 separate models for direct forecasting
    for h in range(1, TEST_HORIZON + 1):
        # We want to predict y_{t+h} using x_t
        df_train_h = df_train.copy()
        df_train_h['target'] = df_train_h.groupby('market')['price'].shift(-h)
        
        # Drop rows where we don't have a future target
        df_train_h = df_train_h.dropna(subset=['target'] + FEATURES)
        
        X_train = df_train_h[FEATURES + CATEGORICAL]
        y_train = df_train_h['target']
        
        lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        lgb_model.fit(X_train, y_train, categorical_feature=['market_encoded'])
        
        models[h] = lgb_model
        
    return models, market_mapping


def forecast_lightgbm_direct(models, market_mapping, df_train, df_test):
    results = []
    forecast_dfs = []
    
    for market in df_train['market'].unique():
        market_train = df_train[df_train['market'] == market]
        market_test = df_test[df_test['market'] == market]
        market_enc = market_mapping[market]
        currency = market_train['currency'].iloc[0]
        
        # Get the VERY LAST row of training data for this market (time T)
        last_train_row = market_train.iloc[-1:].copy()
        last_train_row['market_encoded'] = market_enc
        
        X_infer = last_train_row[FEATURES + CATEGORICAL]
        
        preds = []
        # Predict all 30 steps using the 30 models
        for h in range(1, TEST_HORIZON + 1):
            pred = models[h].predict(X_infer)[0]
            preds.append(pred)
            
        preds = np.array(preds)
        test_prices = market_test['price'].values
        
        rmse = np.sqrt(mean_squared_error(test_prices, preds))
        mape = mean_absolute_percentage_error(test_prices, preds)
        
        # Ensemble (Direct LGBM + Naive)
        naive_pred = np.full(TEST_HORIZON, market_train['price'].values[-1])
        ensemble_preds = (preds + naive_pred) / 2.0
        ens_rmse = np.sqrt(mean_squared_error(test_prices, ensemble_preds))
        ens_mape = mean_absolute_percentage_error(test_prices, ensemble_preds)

        for name, p, r, m in [('Direct LightGBM', preds, rmse, mape), ('Ensemble (LGBM+Naive)', ensemble_preds, ens_rmse, ens_mape)]:
            results.append({'market': market, 'model_name': name, 'rmse': r, 'mape': m, 'test_records': TEST_HORIZON})
            
            forecast_dfs.append(pd.DataFrame({
                'market': market, 'currency': currency, 'date': market_test['date'].values, 
                'actual_price': test_prices, 'predicted_price': p, 
                'error': test_prices - p, 'model_name': name, 
                'horizon_step': np.arange(1, TEST_HORIZON + 1)
            }))
        
    return pd.DataFrame(results), pd.concat(forecast_dfs, ignore_index=True)


def run_pipeline(data_path='data/processed/prices_clean.csv'):
    df, df_train, df_test = load_and_split_data(data_path)
    
    baseline_res, baseline_forecasts = evaluate_baselines(df_train, df_test)
    
    models, market_mapping = train_lightgbm_direct(df_train)
    lgb_res, lgb_forecasts = forecast_lightgbm_direct(models, market_mapping, df_train, df_test)
    
    final_res = pd.concat([baseline_res, lgb_res], ignore_index=True)
    final_forecasts = pd.concat([baseline_forecasts, lgb_forecasts], ignore_index=True)
    
    os.makedirs('data/outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    final_forecasts.to_csv('data/outputs/q1_price_forecasts_advanced.csv', index=False)
    joblib.dump({'models': models, 'features': FEATURES, 'categorical': CATEGORICAL, 'market_mapping': market_mapping}, 'models/carbon_price_advanced_lgbm.pkl')
    
    ensemble_res = final_res[final_res['model_name'] == 'Ensemble (LGBM+Naive)']
    
    output_contract = {
        "module": "Q1.1_Carbon_Price",
        "entity": "ALL_MARKETS",
        "as_of_date": "2026-04-30",
        "horizon": "30_days",
        "target": "price",
        "metrics": {
            "rmse": float(ensemble_res['rmse'].mean()),
            "mape": float(ensemble_res['mape'].mean())
        },
        "features_used": FEATURES,
        "model_name": "Direct LGBM + Naive Ensemble"
    }
    return output_contract

if __name__ == '__main__':
    contract = run_pipeline()
    print(json.dumps(contract, indent=4))
