import base64
import datetime
import logging
import os

import joblib
import numpy as np
import redis
import uvicorn
from fastapi import FastAPI
import yfinance as yf
import pickle
import io
import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel
from starlette import status
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger("unicorn")

app = FastAPI()


class FinanceModel:

    def __init__(self):
        self.model = None
        self.standart = None

    def import_model(self, file_path):
        with open(file_path, 'rb') as f:
            self.model = joblib.load(f)

    def prepare_df(self, df):
        _df = df
        _df['future_price'] = _df['Close'].shift(-1) / _df['Close']

        return _df

    def create_features(self, _df):
        # создание новых признаков
        # среднее занчения за 3 5 7 14 30 90 дней
        _df['rolling_3D_Open_mean'] = _df['Open'].rolling(window=3).mean()
        _df['rolling_5D_Open_mean'] = _df['Open'].rolling(window=5).mean()
        _df['rolling_week_Open_mean'] = _df['Open'].rolling(window=7).mean()
        _df['rolling_2weeks_Open_mean'] = _df['Open'].rolling(window=14).mean()
        _df['rolling_mounth_Open_mean'] = _df['Open'].rolling(window=30).mean()
        _df['rolling_quartal_Open_mean'] = _df['Open'].rolling(window=90).mean()

        _df['rolling_3D_Close_mean'] = _df['Close'].rolling(window=3).mean()
        _df['rolling_5D_Close_mean'] = _df['Close'].rolling(window=5).mean()
        _df['rolling_week_Close_mean'] = _df['Close'].rolling(window=7).mean()
        _df['rolling_2weeks_Close_mean'] = _df['Close'].rolling(window=14).mean()
        _df['rolling_mounth_Close_mean'] = _df['Close'].rolling(window=30).mean()
        _df['rolling_quartal_Close_mean'] = _df['Close'].rolling(window=90).mean()

        _df['rolling_3D_High_mean'] = _df['High'].rolling(window=3).mean()
        _df['rolling_5D_High_mean'] = _df['High'].rolling(window=5).mean()
        _df['rolling_week_High_mean'] = _df['High'].rolling(window=7).mean()
        _df['rolling_2weeks_High_mean'] = _df['High'].rolling(window=14).mean()
        _df['rolling_mounth_High_mean'] = _df['High'].rolling(window=30).mean()
        _df['rolling_quartal_High_mean'] = _df['High'].rolling(window=90).mean()

        _df['rolling_3D_Low_mean'] = _df['Low'].rolling(window=3).mean()
        _df['rolling_5D_Low_mean'] = _df['Low'].rolling(window=5).mean()
        _df['rolling_week_Low_mean'] = _df['Low'].rolling(window=7).mean()
        _df['rolling_2weeks_Low_mean'] = _df['Low'].rolling(window=14).mean()
        _df['rolling_mounth_Low_mean'] = _df['Low'].rolling(window=30).mean()
        _df['rolling_quartal_Low_mean'] = _df['Low'].rolling(window=90).mean()
        #     дисперсия по окнам за 3 5 7 14 30 90 дней
        _df['disp_3D_Open'] = _df['Open'].rolling(window=3).var()
        _df['disp_5D_Open'] = _df['Open'].rolling(window=5).var()
        _df['disp_week_Open'] = _df['Open'].rolling(window=7).var()
        _df['disp_2weeks_Open'] = _df['Open'].rolling(window=14).var()
        _df['disp_mounth_Open'] = _df['Open'].rolling(window=30).var()
        _df['disp_quartal_Open'] = _df['Open'].rolling(window=90).var()

        _df['disp_3D_Close'] = _df['Close'].rolling(window=3).var()
        _df['disp_5D_Close'] = _df['Close'].rolling(window=5).var()
        _df['disp_week_Close'] = _df['Close'].rolling(window=7).var()
        _df['disp_2weeks_Close'] = _df['Close'].rolling(window=14).var()
        _df['disp_mounth_Close'] = _df['Close'].rolling(window=30).var()
        _df['disp_quartal_Close'] = _df['Close'].rolling(window=90).var()

        _df['disp_3D_High'] = _df['High'].rolling(window=3).var()
        _df['disp_5D_High'] = _df['High'].rolling(window=5).var()
        _df['disp_week_High'] = _df['High'].rolling(window=7).var()
        _df['disp_2weeks_High'] = _df['High'].rolling(window=14).var()
        _df['disp_mounth_High'] = _df['High'].rolling(window=30).var()
        _df['disp_quartal_High'] = _df['High'].rolling(window=90).var()

        _df['disp_3D_Low'] = _df['Low'].rolling(window=3).var()
        _df['disp_5D_Low'] = _df['Low'].rolling(window=5).var()
        _df['disp_week_Low'] = _df['Low'].rolling(window=7).var()
        _df['disp_2weeks_Low'] = _df['Low'].rolling(window=14).var()
        _df['disp_mounth_Low'] = _df['Low'].rolling(window=30).var()
        _df['disp_quartal_Low'] = _df['Low'].rolling(window=90).var()
        # относительное изменение за день
        _df['relative_change_Open'] = _df['Open'] / _df['Open'].shift(1)
        _df['relative_change_Close'] = _df['Close'] / _df['Close'].shift(1)
        _df['relative_change_High'] = _df['High'] / _df['High'].shift(1)
        _df['relative_change_Low'] = _df['Low'] / _df['Low'].shift(1)
        #   вычисление логарифмической доходности
        _df['Log_Returns'] = np.log(_df['Close'] / _df['Open'])
        #   Стандартное отклонение
        _df['Volatility_3D'] = _df['Log_Returns'].rolling(window=3).std()
        _df['Volatility_5D'] = _df['Log_Returns'].rolling(window=5).std()
        _df['Volatility_week'] = _df['Log_Returns'].rolling(window=7).std()
        _df['Volatility_2week'] = _df['Log_Returns'].rolling(window=14).std()
        _df['Volatility_mounth'] = _df['Log_Returns'].rolling(window=30).std()
        _df['Volatility_quartal'] = _df['Log_Returns'].rolling(window=90).std()
        #   изменение цены акции
        _df['Price Change'] = _df['Close'].diff()
        # период RSI
        n = 14
        # положительное и отрицательное изменение цены
        _df['Gain'] = _df['Price Change'].apply(lambda x: x if x > 0 else 0)
        _df['Loss'] = _df['Price Change'].apply(lambda x: -x if x < 0 else 0)
        # среднее значение изменения цены для заданного периода
        _df['Avg Gain'] = _df['Gain'].rolling(window=n).mean()
        _df['Avg Loss'] = _df['Loss'].rolling(window=n).mean()
        # отношение среднего прироста к среднему падению
        relative_strength = _df['Avg Gain'] / _df['Avg Loss']
        # RSI
        _df['RSI'] = 100 - (100 / (1 + relative_strength))
        #    ежедневное изменение цены в %
        _df['% daily price change'] = _df['Close'].pct_change() * 100

        return _df

    def create_featured_df(self, df):
        _df = self.prepare_df(df)
        X_f = self.create_features(_df)
        X_f = X_f.loc[:, X_f.columns != 'future_price']
        X_f = X_f.dropna()
        return X_f

    def predict(self, X_df):
        predicted_percent = self.model.predict(X_df)
        future_price = predicted_percent[0] * X_df['Close']
        return future_price


class FinService:

    def get_history(self, ticker_name, period='max'):
        data_history = yf.download(ticker_name, period=period)

        return data_history


class StatusCheck(BaseModel):
    status: str = "OK"


class PredictResponse(BaseModel):
    future_price: float


class Price(BaseModel):
    yesterday_price: float


redis_storage = redis.Redis(host=os.getenv('REDIS_HOST'), port=int(os.getenv('REDIS_PORT')), decode_responses=True)


@app.get('/predict/{ticker}')
async def predict(ticker):
    fin_ser = FinService()
    today_date = datetime.datetime.now().date()
    _key = f'{ticker} - {today_date}'
    if redis_storage.exists(_key):
        future_price = redis_storage.get(_key)
        return PredictResponse(future_price=round(float(future_price), 2))
    history = fin_ser.get_history(ticker, '90d')
    fin_model = FinanceModel()
    fin_model.import_model('model_1.pkl')
    X_df = fin_model.create_featured_df(history)
    future_price_series = fin_model.predict(X_df)
    future_price = future_price_series[0]
    if not redis_storage.exists(_key):
        redis_storage.set(_key, future_price)

    return PredictResponse(future_price=round(future_price, 2))


@app.get('/history/{ticker}')
async def show_history(ticker):
    fin_ser = FinService()
    history = fin_ser.get_history(ticker, '1d')
    return Price(yesterday_price=round(history['Close'], 2))


@app.get(
    "/status",
    tags=["statuscheck"],
    summary="Perform a Status Check",
    response_description="Return HTTP Status Code 200 (OK)",
    status_code=status.HTTP_200_OK,
    response_model=StatusCheck,
)
def get_health() -> StatusCheck:
    return StatusCheck(status="OK")


templates = Jinja2Templates(directory="templates")


@app.get('/graph/{ticker}')
async def graph(request: Request, ticker):
    fin_ser = FinService()
    history = fin_ser.get_history(ticker, 'max')

    # Create the graph
    plt.figure(figsize=(12, 5))
    plt.plot(history.Close)
    plt.title(f'{ticker} Stock Price History')
    plt.xlabel('Date')
    plt.ylabel('Price')

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)

    base64_encoded_image = base64.b64encode(buffer.read()).decode("utf-8")

    return templates.TemplateResponse("graph.html", {"request": request, "graph": base64_encoded_image})


if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8099)
