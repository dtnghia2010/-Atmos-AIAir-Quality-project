from flask import request, jsonify
import numpy as np
import pandas as pd
from tensorflow import keras
from keras import models
from keras.models import model_from_json
from keras.models import Sequential  # Import Sequential class
from keras.layers import LSTM, Dense  # Import LSTM and Dense layers
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
'''
    def predictTempProphetLSTM():
        if request.method == 'POST':
            try:
                data = request.json
                objectFormat = data['dataTemp']

                # push data to array
                tempTime = []
                for i in objectFormat['time']:
                    tempTime.append(i)

                tempData = []
                for i in objectFormat['value']:
                    tempData.append(i)

                arrayData = np.array(tempData)
                arrayTime = np.array(tempTime)
                datetimeTemp = pd.to_datetime(arrayTime)

                dataset = pd.DataFrame({'ds': datetimeTemp, 'y': arrayData})
                dataset = dataset.set_index('ds')
                dataset = dataset.resample('5T').ffill()
                dataset = dataset.dropna()
                dataset = dataset.iloc[1:]
                dataset.reset_index(inplace=True)

                scaler = MinMaxScaler()
                scaled_temp = scaler.fit_transform(dataset[['y']])
                sequence_length = 12
                if len(scaled_temp) < sequence_length:
                    padded_temp = np.pad(scaled_temp, ((sequence_length - len(scaled_temp), 0), (0, 0)), mode='constant')
                else:
                    padded_temp = scaled_temp[-sequence_length:]
                input_data = padded_temp.reshape((1, 1, sequence_length))

                # Load model LSTM
                temp_lstm_json = os.path.join(server_dir, 'server/datasets/models/lstm/test-lstm.json')
                temp_lstm_weight = os.path.join(server_dir, 'server/datasets/models/lstm/test_lstm_weight.h5')
                with open(temp_lstm_json, 'r') as json_file:
                    loaded_model_json_lstm = json_file.read()

                loaded_model_lstm = Sequential()  # Define Sequential model
                loaded_model_lstm.add(LSTM(units=256, input_shape=(1, sequence_length), return_sequences=True))  # Add LSTM layer
                loaded_model_lstm.add(LSTM(units=128))  # Add second LSTM layer
                loaded_model_lstm.add(Dense(units=12, activation='linear'))  # Add Dense layer
                loaded_model_lstm.compile(optimizer='adam', loss='mean_squared_error')  # Compile model
                loaded_model_lstm.load_weights(temp_lstm_weight)  # Load weights

                # Load model BPNN (json and h5)
                temp_bpnn_json = os.path.join(server_dir, 'server/datasets/models/prophet_lstm/temp-bpnn-model.json')
                temp_bpnn_weight = os.path.join(server_dir, 'server/datasets/models/prophet_lstm/temp-bpnn-model.h5')
                with open(temp_bpnn_json, 'r') as json_file:
                    loaded_model_json_bpnn = json_file.read()

                loaded_model_bpnn = model_from_json(loaded_model_json_bpnn)
                loaded_model_bpnn.load_weights(temp_bpnn_weight)

                if os.path.exists(temp_lstm_weight) and os.path.exists(temp_bpnn_weight):
                    #-----------lstm-----------
                    print("--------model loaded lstm---------")
                    predictions = loaded_model_lstm.predict(input_data)
                    predictions_inv = scaler.inverse_transform(predictions)[0]
                    arrayForecast = np.array(predictions_inv)
                    arrayForecast = np.around(arrayForecast, decimals=4)
                    lstmForecast = arrayForecast

                    #-----------prophet-----------
                    print("--------model loaded prophet---------")
                    dataset['ds'] = dataset['ds'].dt.tz_localize(None)

                    model_prophet = Prophet()
                    model_prophet.fit(dataset)

                    future = model_prophet.make_future_dataframe(periods=12, freq='5T')
                    prophetForecast = model_prophet.predict(future)
                    prophetForecast = prophetForecast.tail(12)

                    # get only ds and yhat
                    prophetForecast = prophetForecast[['ds', 'yhat']]
                    prophetForecast = prophetForecast.set_index('ds')
                    prophetForecast.reset_index(inplace=True)

                    #-----------bpnn-----------
                    dataset_bpnn = dataset.copy().tail(12)
                    dataset_bpnn['ds'] = pd.to_datetime(dataset_bpnn['ds'])
                    dataset_bpnn['hour'] = dataset_bpnn['ds'].dt.hour
                    dataset_bpnn['minute'] = dataset_bpnn['ds'].dt.minute
                    dataset_bpnn['day_of_week'] = dataset_bpnn['ds'].dt.dayofweek

                    # drop ds and y column
                    dataset_bpnn = dataset_bpnn.drop(['ds', 'y'], axis=1)

                    #add lstm forecast to dataset
                    dataset_bpnn['lstm'] = lstmForecast

                    # add prophet forecast to dataset
                    dataset_bpnn['prophet'] = prophetForecast['yhat'].values

                    print("--------model loaded bpnn---------")
                    predictions = loaded_model_bpnn.predict(dataset_bpnn)

                    # convert 2D to 1D array
                    predictions = predictions.flatten()

                    # round up to 2 decimal
                    arrayForecast = np.around(predictions, decimals=4)

                    # convert to list
                    listForecast = arrayForecast.tolist()

                    # convert to json
                    objectFormat['forecast'] = listForecast

                    return jsonify(objectFormat)

                else:
                    print(f"File not found: {temp_lstm_weight} or {temp_bpnn_weight}")
                    return jsonify({'error': f"File not found: {temp_lstm_weight} or {temp_bpnn_weight}"})
            except Exception as e:
                print(e)
                return jsonify({'error': str(e)})
'''