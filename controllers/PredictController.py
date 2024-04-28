from flask import request, jsonify
import numpy as np
import pandas as pd
from tensorflow import keras
from keras import models
from keras.models import model_from_json
from sklearn.preprocessing import MinMaxScaler
from prophet import Prophet
import requests
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
server_dir = os.path.dirname(os.path.dirname(script_dir))
# 2465663

class PredictController:
    def predictTempProphet():
      try:
          # Fetch data from ThingSpeak URL
          url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'
          response = requests.get(url)
          data = response.json()

          # Extract data
          tempTime = [entry['created_at'] for entry in data['feeds']]
          tempData = [float(entry['field1']) for entry in data['feeds']]

          # Convert to pandas DataFrame
          dataset = pd.DataFrame({'ds': pd.to_datetime(tempTime), 'y': tempData})

          # Remove timezone from 'ds' column
          dataset['ds'] = dataset['ds'].dt.tz_localize(None)

          model_prophet = Prophet()
          model_prophet.fit(dataset)

          # Make a future dataframe for 1 hour later (5 minutes each)
          future = model_prophet.make_future_dataframe(periods=12, freq='5T')
          forecast = model_prophet.predict(future)

          # Get last 12 rows
          forecast = forecast.head(12)

          # Get only ds and yhat
          forecast = forecast[['ds', 'yhat']]
          forecast.reset_index(drop=True, inplace=True)

          # Convert to numpy array
          arrayForecast = forecast['yhat'].values

          # Round up to 2 decimal places
          arrayForecast = np.around(arrayForecast, decimals=2)

          # Get the date and time of the forecast
          forecast_dates = forecast['ds'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()

          # Prepare JSON response with forecast and forecast dates
          response_data = {'forecast_dates': forecast_dates, 'forecast_values': arrayForecast.tolist()}

      except Exception as e:
          response_data = {'error': str(e)}

      return jsonify(response_data)