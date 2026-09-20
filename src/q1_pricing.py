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

warnings.filterwarnings('ignore')

TEST_HORIZON = 30
FEATURES = ['lag_1', 'lag_2', 'lag_3', 'lag_7', 'lag_14', 'lag_30',
            'roll_mean_7d', 'roll_mean_30d', 'roll_std_30d',
            'dayofweek', 'month', 'day_sin', 'day_cos']
CATEGORICAL = ['market_encoded']


def load_and_split_data(filepath: str):
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(by=['market', 'date']).reset_index(drop=True)
    
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
            
        # 3. ARIMA
        try:
            arima = ARIMA(train_prices, order=(1,1,1)).fit()
            arima_pred = arima.forecast(TEST_HORIZON)
        except:
            arima_pred = naive_pred
            
        for name, preds in [('Naive Persistence', naive_pred), ('Exponential Smoothing', es_pred), ('ARIMA', arima_pred)]:
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


def train_lightgbm(df_train):
    df_train_lgb = df_train.dropna(subset=FEATURES).copy()
    market_mapping = {m: i for i, m in enumerate(df_train['market'].unique())}
    df_train_lgb['market_encoded'] = df_train_lgb['market'].map(market_mapping)
    
    X_train = df_train_lgb[FEATURES + CATEGORICAL]
    y_train = df_train_lgb['price']
    
    lgb_model = lgb.LGBMRegressor(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
    lgb_model.fit(X_train, y_train, categorical_feature=['market_encoded'])
    
    return lgb_model, market_mapping


def generate_features_for_step(history_df, current_date):
    history_df = history_df.sort_values('date')
    dayofweek, month = current_date.dayofweek, current_date.month
    day_sin = math.sin(2 * math.pi * current_date.day / 31.0)
    day_cos = math.cos(2 * math.pi * current_date.day / 31.0)
    
    lag_1 = history_df['price'].iloc[-1]
    lag_2 = history_df['price'].iloc[-2] if len(history_df) >= 2 else lag_1
    lag_3 = history_df['price'].iloc[-3] if len(history_df) >= 3 else lag_2
    lag_7 = history_df['price'].iloc[-7] if len(history_df) >= 7 else history_df['price'].iloc[0]
    lag_14 = history_df['price'].iloc[-14] if len(history_df) >= 14 else history_df['price'].iloc[0]
    lag_30 = history_df['price'].iloc[-30] if len(history_df) >= 30 else history_df['price'].iloc[0]
    
    roll_mean_7d = history_df['price'].iloc[-7:].mean() if len(history_df) >= 7 else history_df['price'].mean()
    roll_mean_30d = history_df['price'].iloc[-30:].mean() if len(history_df) >= 30 else history_df['price'].mean()
    roll_std_30d = history_df['price'].iloc[-30:].std() if len(history_df) >= 30 else history_df['price'].std()
    if pd.isna(roll_std_30d): roll_std_30d = 0.0
    
    return [lag_1, lag_2, lag_3, lag_7, lag_14, lag_30, roll_mean_7d, roll_mean_30d, roll_std_30d, dayofweek, month, day_sin, day_cos]


def forecast_lightgbm(lgb_model, market_mapping, df_train, df_test):
    results = []
    forecast_dfs = []
    
    for market in df_train['market'].unique():
        market_train = df_train[df_train['market'] == market].copy()
        market_test = df_test[df_test['market'] == market].copy()
        market_enc = market_mapping[market]
        currency = market_train['currency'].iloc[0]
        
        history = market_train[['date', 'price']].copy()
        preds = []
        
        for _, row in market_test.iterrows():
            current_date = row['date']
            feats = generate_features_for_step(history, current_date)
            
            X_infer = pd.DataFrame([feats + [market_enc]], columns=FEATURES + CATEGORICAL)
            X_infer['market_encoded'] = X_infer['market_encoded'].astype('int')
            pred = lgb_model.predict(X_infer)[0]
            preds.append(pred)
            
            history = pd.concat([history, pd.DataFrame({'date': [current_date], 'price': [pred]})], ignore_index=True)
            
        preds = np.array(preds)
        test_prices = market_test['price'].values
        
        rmse = np.sqrt(mean_squared_error(test_prices, preds))
        mape = mean_absolute_percentage_error(test_prices, preds)
        results.append({'market': market, 'model_name': 'Autoregressive LightGBM', 'rmse': rmse, 'mape': mape, 'test_records': TEST_HORIZON})
        
        forecast_dfs.append(pd.DataFrame({
            'market': market, 'currency': currency, 'date': market_test['date'].values, 
            'actual_price': test_prices, 'predicted_price': preds, 
            'error': test_prices - preds, 'model_name': 'Autoregressive LightGBM', 
            'horizon_step': np.arange(1, TEST_HORIZON + 1)
        }))
        
    return pd.DataFrame(results), pd.concat(forecast_dfs, ignore_index=True)


def run_pipeline(data_path='data/processed/prices_clean.csv'):
    df, df_train, df_test = load_and_split_data(data_path)
    
    baseline_res, baseline_forecasts = evaluate_baselines(df_train, df_test)
    
    lgb_model, market_mapping = train_lightgbm(df_train)
    lgb_res, lgb_forecasts = forecast_lightgbm(lgb_model, market_mapping, df_train, df_test)
    
    final_res = pd.concat([baseline_res, lgb_res], ignore_index=True)
    final_forecasts = pd.concat([baseline_forecasts, lgb_forecasts], ignore_index=True)
    
    os.makedirs('data/outputs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    final_forecasts.to_csv('data/outputs/q1_price_forecasts.csv', index=False)
    joblib.dump({'model': lgb_model, 'features': FEATURES, 'categorical': CATEGORICAL, 'market_mapping': market_mapping}, 'models/carbon_price_lgbm.pkl')
    
    output_contract = {
        "module": "Q1.1_Carbon_Price",
        "entity": "ALL_MARKETS",
        "as_of_date": "2026-04-30",
        "horizon": "30_days",
        "target": "price",
        "metrics": {
            "rmse": float(lgb_res['rmse'].mean()),
            "mape": float(lgb_res['mape'].mean())
        },
        "features_used": FEATURES,
        "model_name": "Autoregressive LightGBM"
    }
    return output_contract

if __name__ == '__main__':
    contract = run_pipeline()
    import json
    print(json.dumps(contract, indent=4))
