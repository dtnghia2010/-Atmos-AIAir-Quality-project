from flask import request, jsonify
import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression 
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import LSTM, Dense
import requests

import warnings
warnings.filterwarnings('ignore')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

script_dir = os.path.dirname(os.path.abspath(__file__))
server_dir = os.path.dirname(os.path.dirname(script_dir))

p_gb = {'n_estimators': 500, 'max_depth': 10, 'min_samples_split': 2,'learning_rate': 0.09, 'loss': 'squared_error', 'random_state': RANDOM_SEED}
p_xgb = {'n_estimators': 700, 'max_depth': 12, 'learning_rate': 0.09, 'random_state': RANDOM_SEED}
p_rf = {'n_estimators': 1000, 'max_depth': 10, 'random_state': RANDOM_SEED}
p_knn = {'n_neighbors': 3}



script_dir = os.path.dirname(os.path.abspath(__file__))
server_dir = os.path.dirname(os.path.dirname(script_dir))
# 2465663

class PredictController:
    
    def predictTempLSTM():
        try:
            # Get data from URL
            url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'
            response = requests.get(url)
            data = response.json()

            # Extract relevant data from the response
            feeds = data['feeds']
            tempTime = [feed['created_at'] for feed in feeds]
            tempData = [float(feed['field1']) for feed in feeds]

            # Create DataFrame from extracted data
            datetimeTemp = pd.to_datetime(tempTime)
            dataset = pd.DataFrame({'ds': datetimeTemp, 'y': tempData})
            dataset = dataset.set_index('ds')
            dataset = dataset.resample('5T').ffill()
            dataset = dataset.dropna()
            dataset = dataset.iloc[1:]
            dataset.reset_index(inplace=True)

            # Scale the data to be between 0 and 1
            scaler = MinMaxScaler()
            scaled_temp = scaler.fit_transform(dataset[['y']])

            # Ensure the sequence length matches the model's input (12 time steps)
            sequence_length = 12

            # Pad or truncate the sequence to match the model's input sequence length
            if len(scaled_temp) < sequence_length:
                padded_temp = np.pad(scaled_temp, ((sequence_length - len(scaled_temp), 0), (0, 0)), mode='constant')
            else:
                padded_temp = scaled_temp[-sequence_length:]

            # Reshape the data to be suitable for LSTM (samples, time steps, features)
            input_data = padded_temp.reshape((1, 1, sequence_length))

             # Load model architecture from JSON file
            temp_lstm_json = os.path.join(server_dir, 'server\datasets\models\lstm\_test-lstm.json')
            temp_lstm_weight = os.path.join(server_dir, 'server\datasets\models\lstm\_test_lstm_weight.h5')
            with open(temp_lstm_json, 'r') as json_file:
                loaded_model_json = json_file.read()

            # Load model json
            loaded_model = model_from_json(loaded_model_json)

            # Load model weights
            loaded_model.load_weights(temp_lstm_weight)

            if os.path.exists(temp_lstm_weight) and os.path.exists(temp_lstm_json):
                print("--------model loaded---------")
                predictions = loaded_model.predict(input_data)

                # Inverse transform the predictions to get original scale
                predictions_inv = scaler.inverse_transform(predictions)[0]

                # Get data from predictions
                arrayForecast = np.array(predictions_inv)

                # Round up to 2 decimal
                arrayForecast = np.around(arrayForecast, decimals=4)

                # Get the date and time of the forecast
                forecast_dates = pd.date_range(datetimeTemp[-1], periods=12, freq='5T')

                # Prepare JSON response with forecast and forecast dates
                response_data = {'forecast_dates': forecast_dates.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                                'forecast_values': arrayForecast.tolist()}

            else:
                response_data = {'error': 'Model not found'}

        except Exception as e:
            print(e)
            response_data = {'error': str(e)}

        return jsonify(response_data)
    
    def predictHumiLSTM():
        try:
            # Get data from URL
            url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'
            response = requests.get(url)
            data = response.json()

            # Extract relevant data from the response
            feeds = data['feeds']
            humiTime = [feed['created_at'] for feed in feeds]
            humiData = [float(feed['field2']) for feed in feeds]

            arrayData = np.array(humiData)
            arrayTime = np.array(humiTime)
            datetimeHumi = pd.to_datetime(arrayTime)

            dataset = pd.DataFrame({'ds': datetimeHumi, 'y': arrayData})
            dataset = dataset.set_index('ds')
            dataset = dataset.resample('5T').ffill()
            dataset = dataset.dropna()
            dataset = dataset.iloc[1:]
            dataset.reset_index(inplace=True)

            scaler = MinMaxScaler()
            scaled_humi = scaler.fit_transform(dataset[['y']])

            sequence_length = 100
            if len(scaled_humi) < sequence_length:
                padded_humi = np.pad(scaled_humi, ((sequence_length - len(scaled_humi), 0), (0, 0)), mode='constant')
            else:
                padded_humi = scaled_humi[-sequence_length:]
            input_data = padded_humi.reshape((1, 1, sequence_length))
            
            humi_lstm_json = os.path.join(server_dir, 'server/datasets/models/lstm/humi-lstm.json')
            humi_lstm_weight = os.path.join(server_dir, 'server/datasets/models/lstm/humi_lstm_weight.h5')
            with open(humi_lstm_json, 'r') as json_file:
                loaded_model_json = json_file.read()
            
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(humi_lstm_weight)

            if os.path.exists(humi_lstm_weight):
                predictions = loaded_model.predict(input_data)
                predictions_inv = scaler.inverse_transform(predictions)[0]
                arrayForecast = np.array(predictions_inv)
                arrayForecast = np.around(arrayForecast, decimals=4)
                forecast_dates = pd.date_range(datetimeHumi[-1], periods=12, freq='5T')

                # Prepare JSON response with forecast and forecast dates
                response_data = {'forecast_dates': forecast_dates.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                                'forecast_values': arrayForecast.tolist()}

            else:
                response_data = {'error': 'Model not found'}

        except Exception as e:
            print(e)
            response_data = {'error': str(e)}

        return jsonify(response_data)

    
    def predictCO2LSTM():
        try:
            # Get data from URL
            url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'
            response = requests.get(url)
            data = response.json()

            # Extract relevant data from the response
            feeds = data['feeds']
            co2Time = [feed['created_at'] for feed in feeds]
            co2Data = [float(feed['field3']) for feed in feeds]  # Assuming CO2 data is in field3

            # Create DataFrame from extracted data
            datetimeCO2 = pd.to_datetime(co2Time)
            dataset = pd.DataFrame({'ds': datetimeCO2, 'y': co2Data})
            dataset = dataset.set_index('ds')
            dataset = dataset.resample('5T').ffill()
            dataset = dataset.dropna()
            dataset = dataset.iloc[1:]
            dataset.reset_index(inplace=True)

            # Scale the data to be between 0 and 1
            scaler = MinMaxScaler()
            scaled_co2 = scaler.fit_transform(dataset[['y']])

            # Ensure the sequence length matches the model's input (12 time steps)
            sequence_length = 100

            # Pad or truncate the sequence to match the model's input sequence length
            if len(scaled_co2) < sequence_length:
                padded_co2 = np.pad(scaled_co2, ((sequence_length - len(scaled_co2), 0), (0, 0)), mode='constant')
            else:
                padded_co2 = scaled_co2[-sequence_length:]

            # Reshape the data to be suitable for LSTM (samples, time steps, features)
            input_data = padded_co2.reshape((1, 1, sequence_length))

            # Load model architecture from JSON file
            co2_lstm_json = os.path.join(server_dir, 'server\datasets\models\lstm\co2-lstm.json')
            co2_lstm_weight = os.path.join(server_dir, 'server\datasets\models\lstm\co2_lstm_weight.h5')
            
            if os.path.exists(co2_lstm_weight) and os.path.exists(co2_lstm_json):
                with open(co2_lstm_json, 'r') as json_file:
                    loaded_model_json = json_file.read()

                # Load model json
                loaded_model = model_from_json(loaded_model_json)

                # Load model weights
                loaded_model.load_weights(co2_lstm_weight)
                print("--------Model loaded successfully---------")

                # Make predictions
                predictions = loaded_model.predict(input_data)

                # Inverse transform the predictions to get original scale
                predictions_inv = scaler.inverse_transform(predictions)[0]

                # Round up to 2 decimal
                predictions_rounded = np.around(predictions_inv, decimals=4)

                # Get the date and time of the forecast
                forecast_dates = pd.date_range(datetimeCO2[-1], periods=12, freq='5T')

                # Prepare JSON response with forecast and forecast dates
                response_data = {
                    'forecast_dates': forecast_dates.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                    'forecast_values': predictions_rounded.tolist()
                }

            else:
                print(f"File not found: {co2_lstm_weight}")
                response_data = {'error': 'Model not found'}

        except Exception as e:
            print(e)
            response_data = {'error': str(e)}

        return jsonify(response_data)

    def predictCOLSTM():
            try:
                # Get data from URL
                url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'
                response = requests.get(url)
                data = response.json()

                # Extract relevant data from the response
                feeds = data['feeds']
                tempTime = [feed['created_at'] for feed in feeds]
                tempData = [float(feed['field4']) for feed in feeds]

                # Create DataFrame from extracted data
                datetimeTemp = pd.to_datetime(tempTime)
                dataset = pd.DataFrame({'ds': datetimeTemp, 'y': tempData})
                dataset = dataset.set_index('ds')
                dataset = dataset.resample('5T').ffill()
                dataset = dataset.dropna()
                dataset = dataset.iloc[1:]
                dataset.reset_index(inplace=True)

                # Scale the data to be between 0 and 1
                scaler = MinMaxScaler()
                scaled_temp = scaler.fit_transform(dataset[['y']])

                # Ensure the sequence length matches the model's input (12 time steps)
                sequence_length = 100

                # Pad or truncate the sequence to match the model's input sequence length
                if len(scaled_temp) < sequence_length:
                    padded_temp = np.pad(scaled_temp, ((sequence_length - len(scaled_temp), 0), (0, 0)), mode='constant')
                else:
                    padded_temp = scaled_temp[-sequence_length:]

                # Reshape the data to be suitable for LSTM (samples, time steps, features)
                input_data = padded_temp.reshape((1, 1, sequence_length))

                # Load model architecture from JSON file
                temp_lstm_json = os.path.join(server_dir, 'server\datasets\models\lstm\co-lstm.json')
                temp_lstm_weight = os.path.join(server_dir, 'server\datasets\models\lstm\co_lstm_weight.h5')
                with open(temp_lstm_json, 'r') as json_file:
                    loaded_model_json = json_file.read()

                # Load model json
                loaded_model = model_from_json(loaded_model_json)

                # Load model weights
                loaded_model.load_weights(temp_lstm_weight)

                if os.path.exists(temp_lstm_weight) and os.path.exists(temp_lstm_json):
                    print("--------model loaded---------")
                    predictions = loaded_model.predict(input_data)

                    # Inverse transform the predictions to get original scale
                    predictions_inv = scaler.inverse_transform(predictions)[0]

                    # Get data from predictions
                    arrayForecast = np.array(predictions_inv)

                    # Round up to 2 decimal
                    arrayForecast = np.around(arrayForecast, decimals=4)

                    # Get the date and time of the forecast
                    forecast_dates = pd.date_range(datetimeTemp[-1], periods=12, freq='5T')

                    # Prepare JSON response with forecast and forecast dates
                    response_data = {'forecast_dates': forecast_dates.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                                    'forecast_values': arrayForecast.tolist()}

                else:
                    print(f"File not found: {temp_lstm_weight}")
                    response_data = {'error': 'Model not found'}

            except Exception as e:
                print(e)
                response_data = {'error': str(e)}

            return jsonify(response_data)

    def predictUVLSTM():
        try:
            # Get data from URL
            url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'
            response = requests.get(url)
            data = response.json()

            # Extract relevant data from the response
            feeds = data['feeds']
            uvTime = [feed['created_at'] for feed in feeds]
            uvData = [float(feed['field5']) for feed in feeds]  # Assuming UV data is in field5

            arrayData = np.array(uvData)
            arrayTime = np.array(uvTime)
            datetimeUV = pd.to_datetime(arrayTime)

            dataset = pd.DataFrame({'ds': datetimeUV, 'y': arrayData})
            dataset = dataset.set_index('ds')
            dataset = dataset.resample('5T').ffill()
            dataset.reset_index(inplace=True)

            scaler = MinMaxScaler()
            scaled_uv = scaler.fit_transform(dataset[['y']])

            sequence_length = 100
            if len(scaled_uv) < sequence_length:
                padded_uv = np.pad(scaled_uv, ((sequence_length - len(scaled_uv), 0), (0, 0)), mode='constant')
            else:
                padded_uv = scaled_uv[-sequence_length:]
            input_data = padded_uv.reshape((1, 1, sequence_length))
            
            uv_lstm_json = os.path.join(server_dir, r'server\datasets\models\lstm\uv-lstm.json')
            uv_lstm_weight = os.path.join(server_dir, r'server\datasets\models\lstm\uv_lstm_weight.h5')
            with open(uv_lstm_json, 'r') as json_file:
                loaded_model_json = json_file.read()
            
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(uv_lstm_weight)

            if os.path.exists(uv_lstm_weight):
                predictions = loaded_model.predict(input_data)
                predictions_inv = scaler.inverse_transform(predictions)[0]
                arrayForecast = np.array(predictions_inv)
                arrayForecast = np.around(arrayForecast, decimals=4)

                # Get the date and time of the forecast
                forecast_dates = pd.date_range(datetimeUV[-1], periods=12, freq='5T')

                # Prepare JSON response with forecast and forecast dates
                response_data = {
                    'forecast_dates': forecast_dates.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                    'forecast_values': arrayForecast.tolist()
                }

            else:
                print(f"File not found: {uv_lstm_weight}")
                response_data = {'error': 'Model not found'}

        except Exception as e:
            print(e)
            response_data = {'error': str(e)}

        return jsonify(response_data)

    def predictPMLSTM():
        try:
            # Get data from URL
            url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'
            response = requests.get(url)
            data = response.json()

            # Extract relevant data from the response
            feeds = data['feeds']
            pm25Time = [feed['created_at'] for feed in feeds]
            pm25Data = [float(feed['field6']) for feed in feeds]  # Assuming PM2.5 data is in field6

            arrayData = np.array(pm25Data)
            arrayTime = np.array(pm25Time)
            datetimePM25 = pd.to_datetime(arrayTime)

            dataset = pd.DataFrame({'ds': datetimePM25, 'y': arrayData})
            dataset = dataset.set_index('ds')
            dataset = dataset.resample('5T').ffill()
            dataset.reset_index(inplace=True)

            scaler = MinMaxScaler()
            scaled_pm25 = scaler.fit_transform(dataset[['y']])

            sequence_length = 100
            if len(scaled_pm25) < sequence_length:
                padded_pm25 = np.pad(scaled_pm25, ((sequence_length - len(scaled_pm25), 0), (0, 0)), mode='constant')
            else:
                padded_pm25 = scaled_pm25[-sequence_length:]
            input_data = padded_pm25.reshape((1, 1, sequence_length))
            
            pm25_lstm_json = os.path.join(server_dir, 'server/datasets/models/lstm/pm25-lstm.json')
            pm25_lstm_weight = os.path.join(server_dir, 'server/datasets/models/lstm/pm25_lstm_weight.h5')
            with open(pm25_lstm_json, 'r') as json_file:
                loaded_model_json = json_file.read()
            
            loaded_model = model_from_json(loaded_model_json)
            loaded_model.load_weights(pm25_lstm_weight)

            if os.path.exists(pm25_lstm_weight):
                predictions = loaded_model.predict(input_data)
                predictions_inv = scaler.inverse_transform(predictions)[0]
                arrayForecast = np.array(predictions_inv)
                arrayForecast = np.absolute(arrayForecast)
                arrayForecast = np.around(arrayForecast, decimals=4)

                # Get the date and time of the forecast
                forecast_dates = pd.date_range(datetimePM25[-1], periods=12, freq='5T')

                # Prepare JSON response with forecast and forecast dates
                response_data = {
                    'forecast_dates': forecast_dates.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                    'forecast_values': arrayForecast.tolist()
                }

            else:
                print(f"File not found: {pm25_lstm_weight}")
                response_data = {'error': 'Model not found'}

        except Exception as e:
            print(e)
            response_data = {'error': str(e)}

        return jsonify(response_data)
    
    def predictTempProphetLSTM():
        try:
            # Fetch data from ThingSpeak
            url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'
            response = requests.get(url)
            data = response.json()

            # Extracting time and temperature data from the response
            tempTime = []
            tempData = []
            for entry in data['feeds']:
                tempTime.append(entry['created_at'])
                tempData.append(float(entry['field1']))

            # Prepare the dataset
            dataset = pd.DataFrame({'ds': tempTime, 'y': tempData})
            dataset['ds'] = pd.to_datetime(dataset['ds'])
            dataset = dataset.set_index('ds')
            dataset = dataset.resample('5T').ffill()
            dataset = dataset.dropna()
            dataset.reset_index(inplace=True)

            # Continue with existing code for model loading and forecasting
            scaler = MinMaxScaler()
            scaled_temp = scaler.fit_transform(dataset[['y']])
            sequence_length = 12
            if len(scaled_temp) < sequence_length:
                padded_temp = np.pad(scaled_temp, ((sequence_length - len(scaled_temp), 0), (0, 0)), mode='constant')
            else:
                padded_temp = scaled_temp[-sequence_length:]
            input_data = padded_temp.reshape((1, 1, sequence_length))

            # Load LSTM model
            temp_lstm_json = os.path.join(server_dir, 'server\datasets\models\lstm\_test-lstm.json')
            temp_lstm_weight = os.path.join(server_dir, 'server\datasets\models\lstm\_test_lstm_weight.h5')
            with open(temp_lstm_json, 'r') as json_file:
                loaded_model_json_lstm = json_file.read()

            loaded_model_lstm = Sequential()  
            loaded_model_lstm.add(LSTM(units=256, input_shape=(1, sequence_length), return_sequences=True))  
            loaded_model_lstm.add(LSTM(units=128))  
            loaded_model_lstm.add(Dense(units=12, activation='linear'))  
            loaded_model_lstm.compile(optimizer='adam', loss='mean_squared_error')  
            loaded_model_lstm.load_weights(temp_lstm_weight)  

            # Load BPNN model
            temp_bpnn_json = os.path.join(server_dir, 'server\datasets\models\prophet-lstm\_temp-bpnn-model.json')
            temp_bpnn_weight = os.path.join(server_dir, 'server\datasets\models\prophet-lstm\_temp-bpnn-model.h5')
            with open(temp_bpnn_json, 'r') as json_file:
                loaded_model_json_bpnn = json_file.read()

            loaded_model_bpnn = model_from_json(loaded_model_json_bpnn)
            loaded_model_bpnn.load_weights(temp_bpnn_weight)

            if os.path.exists(temp_lstm_weight) and os.path.exists(temp_bpnn_weight):
                # LSTM
                predictions = loaded_model_lstm.predict(input_data)
                predictions_inv = scaler.inverse_transform(predictions)[0]
                arrayForecast = np.array(predictions_inv)
                arrayForecast = np.around(arrayForecast, decimals=4)
                lstmForecast = arrayForecast

                # Prophet
                dataset['ds'] = dataset['ds'].dt.tz_localize(None)
                model_prophet = Prophet()
                model_prophet.fit(dataset)
                future = model_prophet.make_future_dataframe(periods=12, freq='5T')
                prophetForecast = model_prophet.predict(future)
                prophetForecast = prophetForecast.tail(12)
                prophetForecast = prophetForecast[['ds', 'yhat']]
                prophetForecast = prophetForecast.set_index('ds')
                prophetForecast.reset_index(inplace=True)

                # BPNN
                dataset_bpnn = dataset.copy().tail(12)
                dataset_bpnn['ds'] = pd.to_datetime(dataset_bpnn['ds'])
                dataset_bpnn['hour'] = dataset_bpnn['ds'].dt.hour
                dataset_bpnn['minute'] = dataset_bpnn['ds'].dt.minute
                dataset_bpnn['day_of_week'] = dataset_bpnn['ds'].dt.dayofweek
                dataset_bpnn = dataset_bpnn.drop(['ds', 'y'], axis=1)
                dataset_bpnn['lstm'] = lstmForecast
                dataset_bpnn['prophet'] = prophetForecast['yhat'].values

                predictions = loaded_model_bpnn.predict(dataset_bpnn)
                predictions = predictions.flatten()
                bpnnForecast = np.around(predictions, decimals=2)
                forecast_dates = prophetForecast['ds'].astype(str).tolist()
                forecast_values = bpnnForecast.tolist()

                return jsonify({'forecast_dates': forecast_dates, 'forecast_values': forecast_values})

            else:
                return jsonify({'error': f"File not found: {temp_lstm_weight} or {temp_bpnn_weight}"})

        except Exception as e:
            return jsonify({'error': str(e)})

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

                forecast = forecast.tail(12)

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

    def predictHumiProphet():
            try:
                # Fetch data from ThingSpeak URL
                url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'
                response = requests.get(url)
                data = response.json()

                # Extract data
                tempTime = [entry['created_at'] for entry in data['feeds']]
                tempData = [float(entry['field2']) for entry in data['feeds']]

                # Convert to pandas DataFrame
                dataset = pd.DataFrame({'ds': pd.to_datetime(tempTime), 'y': tempData})

                # Remove timezone from 'ds' column
                dataset['ds'] = dataset['ds'].dt.tz_localize(None)

                model_prophet = Prophet()
                model_prophet.fit(dataset)

                # Make a future dataframe for 1 hour later (5 minutes each)
                future = model_prophet.make_future_dataframe(periods=12, freq='5T')
                forecast = model_prophet.predict(future)

                forecast = forecast.tail(12)

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

    def predictCOProphet():
        try:
            # Fetch data from ThingSpeak URL
            url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'
            response = requests.get(url)
            data = response.json()

            # Extract data
            tempTime = [entry['created_at'] for entry in data['feeds']]
            tempData = [float(entry['field3']) for entry in data['feeds']]

            # Convert to pandas DataFrame
            dataset = pd.DataFrame({'ds': pd.to_datetime(tempTime), 'y': tempData})

            # Remove timezone from 'ds' column
            dataset['ds'] = dataset['ds'].dt.tz_localize(None)

            model_prophet = Prophet()
            model_prophet.fit(dataset)

            # Make a future dataframe for 1 hour later (5 minutes each)
            future = model_prophet.make_future_dataframe(periods=12, freq='5T')
            forecast = model_prophet.predict(future)

            forecast = forecast.tail(12)

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
    
    def predictCO2Prophet():
            try:
                # Fetch data from ThingSpeak URL
                url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'
                response = requests.get(url)
                data = response.json()

                # Extract data
                tempTime = [entry['created_at'] for entry in data['feeds']]
                tempData = [float(entry['field4']) for entry in data['feeds']]

                # Convert to pandas DataFrame
                dataset = pd.DataFrame({'ds': pd.to_datetime(tempTime), 'y': tempData})

                # Remove timezone from 'ds' column
                dataset['ds'] = dataset['ds'].dt.tz_localize(None)

                model_prophet = Prophet()
                model_prophet.fit(dataset)

                # Make a future dataframe for 1 hour later (5 minutes each)
                future = model_prophet.make_future_dataframe(periods=12, freq='5T')
                forecast = model_prophet.predict(future)

                forecast = forecast.tail(12)

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
    
    def predictUVProphet():
        try:
            # Fetch data from ThingSpeak URL
            url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'
            response = requests.get(url)
            data = response.json()

            # Extract data
            tempTime = [entry['created_at'] for entry in data['feeds']]
            tempData = [float(entry['field5']) for entry in data['feeds']]

            # Convert to pandas DataFrame
            dataset = pd.DataFrame({'ds': pd.to_datetime(tempTime), 'y': tempData})

            # Remove timezone from 'ds' column
            dataset['ds'] = dataset['ds'].dt.tz_localize(None)

            model_prophet = Prophet()
            model_prophet.fit(dataset)

            # Make a future dataframe for 1 hour later (5 minutes each)
            future = model_prophet.make_future_dataframe(periods=12, freq='5T')
            forecast = model_prophet.predict(future)

            forecast = forecast.tail(12)

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
    
    def predictPMProphet():
        try:
            # Fetch data from ThingSpeak URL
            url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'
            response = requests.get(url)
            data = response.json()

            # Extract data
            tempTime = [entry['created_at'] for entry in data['feeds']]
            tempData = [float(entry['field6']) for entry in data['feeds']]

            # Convert to pandas DataFrame
            dataset = pd.DataFrame({'ds': pd.to_datetime(tempTime), 'y': tempData})

            # Remove timezone from 'ds' column
            dataset['ds'] = dataset['ds'].dt.tz_localize(None)

            model_prophet = Prophet()
            model_prophet.fit(dataset)

            # Make a future dataframe for 1 hour later (5 minutes each)
            future = model_prophet.make_future_dataframe(periods=12, freq='5T')
            forecast = model_prophet.predict(future)

            forecast = forecast.tail(12)

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


      #-------------------LR-------------------
    def predictLRTemp():
        try:
            # GET method to fetch data
            url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'
            response = requests.get(url)
            data = response.json()

            # Extract relevant data from the response
            feeds = data['feeds']
            tempData = [float(feed['field1']) for feed in feeds]
            tempTime = [feed['created_at'] for feed in feeds]

            # convert to numpy array and pandas dataframe
            arrayData = np.array(tempData)
            arrayTime = np.array(tempTime)
            datetimeTemp = pd.to_datetime(arrayTime)

            dataset = pd.DataFrame({'ds': datetimeTemp, 'y': arrayData})
            dataset = dataset.set_index('ds')
            dataset = dataset.resample('5T').ffill()
            dataset = dataset.dropna()
            dataset = dataset.iloc[1:]
            dataset['time'] = np.arange(len(dataset))

            X = dataset[['time']]
            y = dataset['y']

            model_lr = LinearRegression()
            model_lr.fit(X, y)

            # get the last timestamp in the dataset
            last_timestamp = dataset.index[-1]

            # Generate timestamps for the next hour with 5-minute intervals
            next_hour_timestamps = pd.date_range(last_timestamp, periods=12, freq='5T')

            # Reshape timestamps to be used as features for prediction
            next_hour_features = pd.DataFrame({'date': next_hour_timestamps})
            next_hour_features.set_index('date', inplace=True)
            next_hour_features['time'] = np.arange(len(next_hour_features))

            # Use the trained model to predict temperature for the next hour
            predicted_counts = model_lr.predict(next_hour_features[['time']])

            # Round the predictions to 8 decimal places
            arrayForecast = np.around(predicted_counts, decimals=8)

            # Prepare JSON response with forecast and forecast dates
            forecast_dates = next_hour_timestamps.strftime('%Y-%m-%d %H:%M:%S').tolist()
            response_data = {'forecast_dates': forecast_dates, 'forecast_values': arrayForecast.tolist()}

        except Exception as e:
            print(e)
            response_data = {'error': str(e)}

        return jsonify(response_data)
    
    def predictLRHumi():
        try:
            # GET method to fetch data
            url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'
            response = requests.get(url)
            data = response.json()

            # Extract relevant data from the response
            feeds = data['feeds']
            tempData = [float(feed['field2']) for feed in feeds]
            tempTime = [feed['created_at'] for feed in feeds]

            # convert to numpy array and pandas dataframe
            arrayData = np.array(tempData)
            arrayTime = np.array(tempTime)
            datetimeTemp = pd.to_datetime(arrayTime)

            dataset = pd.DataFrame({'ds': datetimeTemp, 'y': arrayData})
            dataset = dataset.set_index('ds')
            dataset = dataset.resample('5T').ffill()
            dataset = dataset.dropna()
            dataset = dataset.iloc[1:]
            dataset['time'] = np.arange(len(dataset))

            X = dataset[['time']]
            y = dataset['y']

            model_lr = LinearRegression()
            model_lr.fit(X, y)

            # get the last timestamp in the dataset
            last_timestamp = dataset.index[-1]

            # Generate timestamps for the next hour with 5-minute intervals
            next_hour_timestamps = pd.date_range(last_timestamp, periods=12, freq='5T')

            # Reshape timestamps to be used as features for prediction
            next_hour_features = pd.DataFrame({'date': next_hour_timestamps})
            next_hour_features.set_index('date', inplace=True)
            next_hour_features['time'] = np.arange(len(next_hour_features))

            # Use the trained model to predict temperature for the next hour
            predicted_counts = model_lr.predict(next_hour_features[['time']])

            # Round the predictions to 8 decimal places
            arrayForecast = np.around(predicted_counts, decimals=8)

            # Prepare JSON response with forecast and forecast dates
            forecast_dates = next_hour_timestamps.strftime('%Y-%m-%d %H:%M:%S').tolist()
            response_data = {'forecast_dates': forecast_dates, 'forecast_values': arrayForecast.tolist()}

        except Exception as e:
            print(e)
            response_data = {'error': str(e)}

        return jsonify(response_data)
    
    def predictLRCO2():
        try:
            # GET method to fetch data
            url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'
            response = requests.get(url)
            data = response.json()

            # Extract relevant data from the response
            feeds = data['feeds']
            tempData = [float(feed['field3']) for feed in feeds]
            tempTime = [feed['created_at'] for feed in feeds]

            # convert to numpy array and pandas dataframe
            arrayData = np.array(tempData)
            arrayTime = np.array(tempTime)
            datetimeTemp = pd.to_datetime(arrayTime)

            dataset = pd.DataFrame({'ds': datetimeTemp, 'y': arrayData})
            dataset = dataset.set_index('ds')
            dataset = dataset.resample('5T').ffill()
            dataset = dataset.dropna()
            dataset = dataset.iloc[1:]
            dataset['time'] = np.arange(len(dataset))

            X = dataset[['time']]
            y = dataset['y']

            model_lr = LinearRegression()
            model_lr.fit(X, y)

            # get the last timestamp in the dataset
            last_timestamp = dataset.index[-1]

            # Generate timestamps for the next hour with 5-minute intervals
            next_hour_timestamps = pd.date_range(last_timestamp, periods=12, freq='5T')

            # Reshape timestamps to be used as features for prediction
            next_hour_features = pd.DataFrame({'date': next_hour_timestamps})
            next_hour_features.set_index('date', inplace=True)
            next_hour_features['time'] = np.arange(len(next_hour_features))

            # Use the trained model to predict temperature for the next hour
            predicted_counts = model_lr.predict(next_hour_features[['time']])

            # Round the predictions to 8 decimal places
            arrayForecast = np.around(predicted_counts, decimals=8)

            # Prepare JSON response with forecast and forecast dates
            forecast_dates = next_hour_timestamps.strftime('%Y-%m-%d %H:%M:%S').tolist()
            response_data = {'forecast_dates': forecast_dates, 'forecast_values': arrayForecast.tolist()}

        except Exception as e:
            print(e)
            response_data = {'error': str(e)}

        return jsonify(response_data)
    
    def predictLRCO():
        try:
            # GET method to fetch data
            url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'
            response = requests.get(url)
            data = response.json()

            # Extract relevant data from the response
            feeds = data['feeds']
            tempData = [float(feed['field4']) for feed in feeds]
            tempTime = [feed['created_at'] for feed in feeds]

            # convert to numpy array and pandas dataframe
            arrayData = np.array(tempData)
            arrayTime = np.array(tempTime)
            datetimeTemp = pd.to_datetime(arrayTime)

            dataset = pd.DataFrame({'ds': datetimeTemp, 'y': arrayData})
            dataset = dataset.set_index('ds')
            dataset = dataset.resample('5T').ffill()
            dataset = dataset.dropna()
            dataset = dataset.iloc[1:]
            dataset['time'] = np.arange(len(dataset))

            X = dataset[['time']]
            y = dataset['y']

            model_lr = LinearRegression()
            model_lr.fit(X, y)

            # get the last timestamp in the dataset
            last_timestamp = dataset.index[-1]

            # Generate timestamps for the next hour with 5-minute intervals
            next_hour_timestamps = pd.date_range(last_timestamp, periods=12, freq='5T')

            # Reshape timestamps to be used as features for prediction
            next_hour_features = pd.DataFrame({'date': next_hour_timestamps})
            next_hour_features.set_index('date', inplace=True)
            next_hour_features['time'] = np.arange(len(next_hour_features))

            # Use the trained model to predict temperature for the next hour
            predicted_counts = model_lr.predict(next_hour_features[['time']])

            # Round the predictions to 8 decimal places
            arrayForecast = np.around(predicted_counts, decimals=8)

            # Prepare JSON response with forecast and forecast dates
            forecast_dates = next_hour_timestamps.strftime('%Y-%m-%d %H:%M:%S').tolist()
            response_data = {'forecast_dates': forecast_dates, 'forecast_values': arrayForecast.tolist()}

        except Exception as e:
            print(e)
            response_data = {'error': str(e)}

        return jsonify(response_data)
    
    def predictLRUV():
        try:
            # GET method to fetch data
            url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'
            response = requests.get(url)
            data = response.json()

            # Extract relevant data from the response
            feeds = data['feeds']
            tempData = [float(feed['field5']) for feed in feeds]
            tempTime = [feed['created_at'] for feed in feeds]

            # convert to numpy array and pandas dataframe
            arrayData = np.array(tempData)
            arrayTime = np.array(tempTime)
            datetimeTemp = pd.to_datetime(arrayTime)

            dataset = pd.DataFrame({'ds': datetimeTemp, 'y': arrayData})
            dataset = dataset.set_index('ds')
            dataset = dataset.resample('5T').ffill()
            dataset = dataset.dropna()
            dataset = dataset.iloc[1:]
            dataset['time'] = np.arange(len(dataset))

            X = dataset[['time']]
            y = dataset['y']

            model_lr = LinearRegression()
            model_lr.fit(X, y)

            # get the last timestamp in the dataset
            last_timestamp = dataset.index[-1]

            # Generate timestamps for the next hour with 5-minute intervals
            next_hour_timestamps = pd.date_range(last_timestamp, periods=12, freq='5T')

            # Reshape timestamps to be used as features for prediction
            next_hour_features = pd.DataFrame({'date': next_hour_timestamps})
            next_hour_features.set_index('date', inplace=True)
            next_hour_features['time'] = np.arange(len(next_hour_features))

            # Use the trained model to predict temperature for the next hour
            predicted_counts = model_lr.predict(next_hour_features[['time']])

            # Round the predictions to 8 decimal places
            arrayForecast = np.around(predicted_counts, decimals=8)

            # Prepare JSON response with forecast and forecast dates
            forecast_dates = next_hour_timestamps.strftime('%Y-%m-%d %H:%M:%S').tolist()
            response_data = {'forecast_dates': forecast_dates, 'forecast_values': arrayForecast.tolist()}

        except Exception as e:
            print(e)
            response_data = {'error': str(e)}

        return jsonify(response_data)
    
    def predictLRPM():
        try:
            # GET method to fetch data
            url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'
            response = requests.get(url)
            data = response.json()

            # Extract relevant data from the response
            feeds = data['feeds']
            tempData = [float(feed['field6']) for feed in feeds]
            tempTime = [feed['created_at'] for feed in feeds]

            # convert to numpy array and pandas dataframe
            arrayData = np.array(tempData)
            arrayTime = np.array(tempTime)
            datetimeTemp = pd.to_datetime(arrayTime)

            dataset = pd.DataFrame({'ds': datetimeTemp, 'y': arrayData})
            dataset = dataset.set_index('ds')
            dataset = dataset.resample('5T').ffill()
            dataset = dataset.dropna()
            dataset = dataset.iloc[1:]
            dataset['time'] = np.arange(len(dataset))

            X = dataset[['time']]
            y = dataset['y']

            model_lr = LinearRegression()
            model_lr.fit(X, y)

            # get the last timestamp in the dataset
            last_timestamp = dataset.index[-1]

            # Generate timestamps for the next hour with 5-minute intervals
            next_hour_timestamps = pd.date_range(last_timestamp, periods=12, freq='5T')

            # Reshape timestamps to be used as features for prediction
            next_hour_features = pd.DataFrame({'date': next_hour_timestamps})
            next_hour_features.set_index('date', inplace=True)
            next_hour_features['time'] = np.arange(len(next_hour_features))

            # Use the trained model to predict temperature for the next hour
            predicted_counts = model_lr.predict(next_hour_features[['time']])

            # Round the predictions to 8 decimal places
            arrayForecast = np.around(predicted_counts, decimals=8)

            # Prepare JSON response with forecast and forecast dates
            forecast_dates = next_hour_timestamps.strftime('%Y-%m-%d %H:%M:%S').tolist()
            response_data = {'forecast_dates': forecast_dates, 'forecast_values': arrayForecast.tolist()}

        except Exception as e:
            print(e)
            response_data = {'error': str(e)}

        return jsonify(response_data)
    

    def predictGBTemp():
        # Define the URL for GET request
        url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'

        # Make a GET request to the API
        response = requests.get(url)

        if response.status_code == 200:
            # Extract data from the response
            data = response.json()
            tempData = [float(entry['field1']) for entry in data['feeds']]
            tempTime = [entry['created_at'] for entry in data['feeds']]

            # Convert time to datetime
            datetimeTemp = pd.to_datetime(tempTime)

            dataset = pd.DataFrame({'ds': datetimeTemp, 'y': tempData})
            dataset = dataset.set_index('ds')
            dataset = dataset.resample('5T').ffill()
            dataset = dataset.dropna()
            dataset = dataset.iloc[1:]
            dataset['time'] = np.arange(len(dataset))

            X = dataset[['time']]
            y = dataset['y']

            model_gb = GradientBoostingRegressor(**p_gb)
            model_gb.fit(X, y)

            last_timestamp = dataset.index[-1]
            next_hour_timestamps = pd.date_range(last_timestamp, periods=12, freq='5T')
            next_hour_features = pd.DataFrame({'date': next_hour_timestamps})
            next_hour_features.set_index('date', inplace=True)
            next_hour_features['time'] = np.arange(len(next_hour_features))

            predicted_counts = model_gb.predict(next_hour_features)

            # Round predictions
            arrayForecast = np.around(predicted_counts, decimals=8)

            # Convert arrays to lists
            forecast_dates = next_hour_timestamps.astype(str).tolist()
            forecast_values = arrayForecast.tolist()

            # Create JSON response
            response_data = {
                "forecast_dates": forecast_dates,
                "forecast_values": forecast_values
            }

            return jsonify(response_data)

        else:
            # If the GET request fails, return an error message
            return jsonify({"error": "Failed to retrieve data from the API"}), 500
        
    def predictGBHumi():
        # Define the URL for GET request
        url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'

        # Make a GET request to the API
        response = requests.get(url)

        if response.status_code == 200:
            # Extract data from the response
            data = response.json()
            tempData = [float(entry['field2']) for entry in data['feeds']]
            tempTime = [entry['created_at'] for entry in data['feeds']]

            # Convert time to datetime
            datetimeTemp = pd.to_datetime(tempTime)

            dataset = pd.DataFrame({'ds': datetimeTemp, 'y': tempData})
            dataset = dataset.set_index('ds')
            dataset = dataset.resample('5T').ffill()
            dataset = dataset.dropna()
            dataset = dataset.iloc[1:]
            dataset['time'] = np.arange(len(dataset))

            X = dataset[['time']]
            y = dataset['y']

            model_gb = GradientBoostingRegressor(**p_gb)
            model_gb.fit(X, y)

            last_timestamp = dataset.index[-1]
            next_hour_timestamps = pd.date_range(last_timestamp, periods=12, freq='5T')
            next_hour_features = pd.DataFrame({'date': next_hour_timestamps})
            next_hour_features.set_index('date', inplace=True)
            next_hour_features['time'] = np.arange(len(next_hour_features))

            predicted_counts = model_gb.predict(next_hour_features)

            # Round predictions
            arrayForecast = np.around(predicted_counts, decimals=8)

            # Convert arrays to lists
            forecast_dates = next_hour_timestamps.astype(str).tolist()
            forecast_values = arrayForecast.tolist()

            # Create JSON response
            response_data = {
                "forecast_dates": forecast_dates,
                "forecast_values": forecast_values
            }

            return jsonify(response_data)

        else:
            # If the GET request fails, return an error message
            return jsonify({"error": "Failed to retrieve data from the API"}), 500
        
    def predictGBCO2():
        # Define the URL for GET request
        url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'

        # Make a GET request to the API
        response = requests.get(url)

        if response.status_code == 200:
            # Extract data from the response
            data = response.json()
            tempData = [float(entry['field3']) for entry in data['feeds']]
            tempTime = [entry['created_at'] for entry in data['feeds']]

            # Convert time to datetime
            datetimeTemp = pd.to_datetime(tempTime)

            dataset = pd.DataFrame({'ds': datetimeTemp, 'y': tempData})
            dataset = dataset.set_index('ds')
            dataset = dataset.resample('5T').ffill()
            dataset = dataset.dropna()
            dataset = dataset.iloc[1:]
            dataset['time'] = np.arange(len(dataset))

            X = dataset[['time']]
            y = dataset['y']

            model_gb = GradientBoostingRegressor(**p_gb)
            model_gb.fit(X, y)

            last_timestamp = dataset.index[-1]
            next_hour_timestamps = pd.date_range(last_timestamp, periods=12, freq='5T')
            next_hour_features = pd.DataFrame({'date': next_hour_timestamps})
            next_hour_features.set_index('date', inplace=True)
            next_hour_features['time'] = np.arange(len(next_hour_features))

            predicted_counts = model_gb.predict(next_hour_features)

            # Round predictions
            arrayForecast = np.around(predicted_counts, decimals=8)

            # Convert arrays to lists
            forecast_dates = next_hour_timestamps.astype(str).tolist()
            forecast_values = arrayForecast.tolist()

            # Create JSON response
            response_data = {
                "forecast_dates": forecast_dates,
                "forecast_values": forecast_values
            }

            return jsonify(response_data)

        else:
            # If the GET request fails, return an error message
            return jsonify({"error": "Failed to retrieve data from the API"}), 500
        
    def predictGBCO():
        # Define the URL for GET request
        url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'

        # Make a GET request to the API
        response = requests.get(url)

        if response.status_code == 200:
            # Extract data from the response
            data = response.json()
            tempData = [float(entry['field4']) for entry in data['feeds']]
            tempTime = [entry['created_at'] for entry in data['feeds']]

            # Convert time to datetime
            datetimeTemp = pd.to_datetime(tempTime)

            dataset = pd.DataFrame({'ds': datetimeTemp, 'y': tempData})
            dataset = dataset.set_index('ds')
            dataset = dataset.resample('5T').ffill()
            dataset = dataset.dropna()
            dataset = dataset.iloc[1:]
            dataset['time'] = np.arange(len(dataset))

            X = dataset[['time']]
            y = dataset['y']

            model_gb = GradientBoostingRegressor(**p_gb)
            model_gb.fit(X, y)

            last_timestamp = dataset.index[-1]
            next_hour_timestamps = pd.date_range(last_timestamp, periods=12, freq='5T')
            next_hour_features = pd.DataFrame({'date': next_hour_timestamps})
            next_hour_features.set_index('date', inplace=True)
            next_hour_features['time'] = np.arange(len(next_hour_features))

            predicted_counts = model_gb.predict(next_hour_features)

            # Round predictions
            arrayForecast = np.around(predicted_counts, decimals=8)

            # Convert arrays to lists
            forecast_dates = next_hour_timestamps.astype(str).tolist()
            forecast_values = arrayForecast.tolist()

            # Create JSON response
            response_data = {
                "forecast_dates": forecast_dates,
                "forecast_values": forecast_values
            }

            return jsonify(response_data)

        else:
            # If the GET request fails, return an error message
            return jsonify({"error": "Failed to retrieve data from the API"}), 500
    
    def predictGBUV():
        # Define the URL for GET request
        url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'

        # Make a GET request to the API
        response = requests.get(url)

        if response.status_code == 200:
            # Extract data from the response
            data = response.json()
            tempData = [float(entry['field5']) for entry in data['feeds']]
            tempTime = [entry['created_at'] for entry in data['feeds']]

            # Convert time to datetime
            datetimeTemp = pd.to_datetime(tempTime)

            dataset = pd.DataFrame({'ds': datetimeTemp, 'y': tempData})
            dataset = dataset.set_index('ds')
            dataset = dataset.resample('5T').ffill()
            dataset = dataset.dropna()
            dataset = dataset.iloc[1:]
            dataset['time'] = np.arange(len(dataset))

            X = dataset[['time']]
            y = dataset['y']

            model_gb = GradientBoostingRegressor(**p_gb)
            model_gb.fit(X, y)

            last_timestamp = dataset.index[-1]
            next_hour_timestamps = pd.date_range(last_timestamp, periods=12, freq='5T')
            next_hour_features = pd.DataFrame({'date': next_hour_timestamps})
            next_hour_features.set_index('date', inplace=True)
            next_hour_features['time'] = np.arange(len(next_hour_features))

            predicted_counts = model_gb.predict(next_hour_features)

            # Round predictions
            arrayForecast = np.around(predicted_counts, decimals=8)

            # Convert arrays to lists
            forecast_dates = next_hour_timestamps.astype(str).tolist()
            forecast_values = arrayForecast.tolist()

            # Create JSON response
            response_data = {
                "forecast_dates": forecast_dates,
                "forecast_values": forecast_values
            }

            return jsonify(response_data)

        else:
            # If the GET request fails, return an error message
            return jsonify({"error": "Failed to retrieve data from the API"}), 500
    
    def predictGBPM():
        # Define the URL for GET request
        url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'

        # Make a GET request to the API
        response = requests.get(url)

        if response.status_code == 200:
            # Extract data from the response
            data = response.json()
            tempData = [float(entry['field6']) for entry in data['feeds']]
            tempTime = [entry['created_at'] for entry in data['feeds']]

            # Convert time to datetime
            datetimeTemp = pd.to_datetime(tempTime)

            dataset = pd.DataFrame({'ds': datetimeTemp, 'y': tempData})
            dataset = dataset.set_index('ds')
            dataset = dataset.resample('5T').ffill()
            dataset = dataset.dropna()
            dataset = dataset.iloc[1:]
            dataset['time'] = np.arange(len(dataset))

            X = dataset[['time']]
            y = dataset['y']

            model_gb = GradientBoostingRegressor(**p_gb)
            model_gb.fit(X, y)

            last_timestamp = dataset.index[-1]
            next_hour_timestamps = pd.date_range(last_timestamp, periods=12, freq='5T')
            next_hour_features = pd.DataFrame({'date': next_hour_timestamps})
            next_hour_features.set_index('date', inplace=True)
            next_hour_features['time'] = np.arange(len(next_hour_features))

            predicted_counts = model_gb.predict(next_hour_features)

            # Round predictions
            arrayForecast = np.around(predicted_counts, decimals=8)

            # Convert arrays to lists
            forecast_dates = next_hour_timestamps.astype(str).tolist()
            forecast_values = arrayForecast.tolist()

            # Create JSON response
            response_data = {
                "forecast_dates": forecast_dates,
                "forecast_values": forecast_values
            }

            return jsonify(response_data)

        else:
            # If the GET request fails, return an error message
            return jsonify({"error": "Failed to retrieve data from the API"}), 500
        
    # ---XGB---

    def predictXGBTemp():
        # Define the URL for GET request
        url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'

        # Make a GET request to the API
        response = requests.get(url)

        if response.status_code == 200:
            try:
                # Extract data from the response
                data = response.json()
                tempData = [float(entry['field1']) for entry in data['feeds']]
                tempTime = [entry['created_at'] for entry in data['feeds']]

                # Convert time to datetime
                datetimeTemp = pd.to_datetime(tempTime)

                dataset = pd.DataFrame({'ds': datetimeTemp, 'y': tempData})
                dataset = dataset.set_index('ds')
                dataset = dataset.resample('5T').ffill()
                dataset = dataset.dropna()
                dataset = dataset.iloc[1:]
                dataset['time'] = np.arange(len(dataset))

                X = dataset[['time']]
                y = dataset['y']

                model_gb = XGBRegressor(**p_gb)
                model_gb.fit(X, y)

                last_timestamp = dataset.index[-1]
                next_hour_timestamps = pd.date_range(last_timestamp, periods=12, freq='5T')
                next_hour_features = pd.DataFrame({'date': next_hour_timestamps})
                next_hour_features.set_index('date', inplace=True)
                next_hour_features['time'] = np.arange(len(next_hour_features))

                predicted_counts = model_gb.predict(next_hour_features)

                # Round predictions
                arrayForecast = np.around(predicted_counts, decimals=8)

                # Convert arrays to lists
                forecast_dates = next_hour_timestamps.astype(str).tolist()
                forecast_values = arrayForecast.tolist()

                # Create JSON response
                response_data = {
                    "forecast_dates": forecast_dates,
                    "forecast_values": forecast_values
                }

                return jsonify(response_data)

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        else:
            # If the GET request fails, return an error message
            return jsonify({"error": "Failed to retrieve data from the API"}), 500
        

    def predictXGBHumi():
        # Define the URL for GET request
        url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'

        # Make a GET request to the API
        response = requests.get(url)

        if response.status_code == 200:
            try:
                # Extract data from the response
                data = response.json()
                tempData = [float(entry['field2']) for entry in data['feeds']]
                tempTime = [entry['created_at'] for entry in data['feeds']]

                # Convert time to datetime
                datetimeTemp = pd.to_datetime(tempTime)

                dataset = pd.DataFrame({'ds': datetimeTemp, 'y': tempData})
                dataset = dataset.set_index('ds')
                dataset = dataset.resample('5T').ffill()
                dataset = dataset.dropna()
                dataset = dataset.iloc[1:]
                dataset['time'] = np.arange(len(dataset))

                X = dataset[['time']]
                y = dataset['y']

                model_gb = XGBRegressor(**p_gb)
                model_gb.fit(X, y)

                last_timestamp = dataset.index[-1]
                next_hour_timestamps = pd.date_range(last_timestamp, periods=12, freq='5T')
                next_hour_features = pd.DataFrame({'date': next_hour_timestamps})
                next_hour_features.set_index('date', inplace=True)
                next_hour_features['time'] = np.arange(len(next_hour_features))

                predicted_counts = model_gb.predict(next_hour_features)

                # Round predictions
                arrayForecast = np.around(predicted_counts, decimals=8)

                # Convert arrays to lists
                forecast_dates = next_hour_timestamps.astype(str).tolist()
                forecast_values = arrayForecast.tolist()

                # Create JSON response
                response_data = {
                    "forecast_dates": forecast_dates,
                    "forecast_values": forecast_values
                }

                return jsonify(response_data)

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        else:
            # If the GET request fails, return an error message
            return jsonify({"error": "Failed to retrieve data from the API"}), 500
        
    def predictXGBCO2():
        # Define the URL for GET request
        url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'

        # Make a GET request to the API
        response = requests.get(url)

        if response.status_code == 200:
            try:
                # Extract data from the response
                data = response.json()
                tempData = [float(entry['field3']) for entry in data['feeds']]
                tempTime = [entry['created_at'] for entry in data['feeds']]

                # Convert time to datetime
                datetimeTemp = pd.to_datetime(tempTime)

                dataset = pd.DataFrame({'ds': datetimeTemp, 'y': tempData})
                dataset = dataset.set_index('ds')
                dataset = dataset.resample('5T').ffill()
                dataset = dataset.dropna()
                dataset = dataset.iloc[1:]
                dataset['time'] = np.arange(len(dataset))

                X = dataset[['time']]
                y = dataset['y']

                model_gb = XGBRegressor(**p_gb)
                model_gb.fit(X, y)

                last_timestamp = dataset.index[-1]
                next_hour_timestamps = pd.date_range(last_timestamp, periods=12, freq='5T')
                next_hour_features = pd.DataFrame({'date': next_hour_timestamps})
                next_hour_features.set_index('date', inplace=True)
                next_hour_features['time'] = np.arange(len(next_hour_features))

                predicted_counts = model_gb.predict(next_hour_features)

                # Round predictions
                arrayForecast = np.around(predicted_counts, decimals=8)

                # Convert arrays to lists
                forecast_dates = next_hour_timestamps.astype(str).tolist()
                forecast_values = arrayForecast.tolist()

                # Create JSON response
                response_data = {
                    "forecast_dates": forecast_dates,
                    "forecast_values": forecast_values
                }

                return jsonify(response_data)

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        else:
            # If the GET request fails, return an error message
            return jsonify({"error": "Failed to retrieve data from the API"}), 500
    
    def predictXGBCO():
        # Define the URL for GET request
        url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'

        # Make a GET request to the API
        response = requests.get(url)

        if response.status_code == 200:
            try:
                # Extract data from the response
                data = response.json()
                tempData = [float(entry['field4']) for entry in data['feeds']]
                tempTime = [entry['created_at'] for entry in data['feeds']]

                # Convert time to datetime
                datetimeTemp = pd.to_datetime(tempTime)

                dataset = pd.DataFrame({'ds': datetimeTemp, 'y': tempData})
                dataset = dataset.set_index('ds')
                dataset = dataset.resample('5T').ffill()
                dataset = dataset.dropna()
                dataset = dataset.iloc[1:]
                dataset['time'] = np.arange(len(dataset))

                X = dataset[['time']]
                y = dataset['y']

                model_gb = XGBRegressor(**p_gb)
                model_gb.fit(X, y)

                last_timestamp = dataset.index[-1]
                next_hour_timestamps = pd.date_range(last_timestamp, periods=12, freq='5T')
                next_hour_features = pd.DataFrame({'date': next_hour_timestamps})
                next_hour_features.set_index('date', inplace=True)
                next_hour_features['time'] = np.arange(len(next_hour_features))

                predicted_counts = model_gb.predict(next_hour_features)

                # Round predictions
                arrayForecast = np.around(predicted_counts, decimals=8)

                # Convert arrays to lists
                forecast_dates = next_hour_timestamps.astype(str).tolist()
                forecast_values = arrayForecast.tolist()

                # Create JSON response
                response_data = {
                    "forecast_dates": forecast_dates,
                    "forecast_values": forecast_values
                }

                return jsonify(response_data)

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        else:
            # If the GET request fails, return an error message
            return jsonify({"error": "Failed to retrieve data from the API"}), 500
        
    
    def predictXGBUV():
        # Define the URL for GET request
        url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'

        # Make a GET request to the API
        response = requests.get(url)

        if response.status_code == 200:
            try:
                # Extract data from the response
                data = response.json()
                tempData = [float(entry['field5']) for entry in data['feeds']]
                tempTime = [entry['created_at'] for entry in data['feeds']]

                # Convert time to datetime
                datetimeTemp = pd.to_datetime(tempTime)

                dataset = pd.DataFrame({'ds': datetimeTemp, 'y': tempData})
                dataset = dataset.set_index('ds')
                dataset = dataset.resample('5T').ffill()
                dataset = dataset.dropna()
                dataset = dataset.iloc[1:]
                dataset['time'] = np.arange(len(dataset))

                X = dataset[['time']]
                y = dataset['y']

                model_gb = XGBRegressor(**p_gb)
                model_gb.fit(X, y)

                last_timestamp = dataset.index[-1]
                next_hour_timestamps = pd.date_range(last_timestamp, periods=12, freq='5T')
                next_hour_features = pd.DataFrame({'date': next_hour_timestamps})
                next_hour_features.set_index('date', inplace=True)
                next_hour_features['time'] = np.arange(len(next_hour_features))

                predicted_counts = model_gb.predict(next_hour_features)

                # Round predictions
                arrayForecast = np.around(predicted_counts, decimals=8)

                # Convert arrays to lists
                forecast_dates = next_hour_timestamps.astype(str).tolist()
                forecast_values = arrayForecast.tolist()

                # Create JSON response
                response_data = {
                    "forecast_dates": forecast_dates,
                    "forecast_values": forecast_values
                }

                return jsonify(response_data)

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        else:
            # If the GET request fails, return an error message
            return jsonify({"error": "Failed to retrieve data from the API"}), 500
        
    
    def predictXGBPM():
        # Define the URL for GET request
        url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'

        # Make a GET request to the API
        response = requests.get(url)

        if response.status_code == 200:
            try:
                # Extract data from the response
                data = response.json()
                tempData = [float(entry['field6']) for entry in data['feeds']]
                tempTime = [entry['created_at'] for entry in data['feeds']]

                # Convert time to datetime
                datetimeTemp = pd.to_datetime(tempTime)

                dataset = pd.DataFrame({'ds': datetimeTemp, 'y': tempData})
                dataset = dataset.set_index('ds')
                dataset = dataset.resample('5T').ffill()
                dataset = dataset.dropna()
                dataset = dataset.iloc[1:]
                dataset['time'] = np.arange(len(dataset))

                X = dataset[['time']]
                y = dataset['y']

                model_gb = XGBRegressor(**p_gb)
                model_gb.fit(X, y)

                last_timestamp = dataset.index[-1]
                next_hour_timestamps = pd.date_range(last_timestamp, periods=12, freq='5T')
                next_hour_features = pd.DataFrame({'date': next_hour_timestamps})
                next_hour_features.set_index('date', inplace=True)
                next_hour_features['time'] = np.arange(len(next_hour_features))

                predicted_counts = model_gb.predict(next_hour_features)

                # Round predictions
                arrayForecast = np.around(predicted_counts, decimals=8)

                # Convert arrays to lists
                forecast_dates = next_hour_timestamps.astype(str).tolist()
                forecast_values = arrayForecast.tolist()

                # Create JSON response
                response_data = {
                    "forecast_dates": forecast_dates,
                    "forecast_values": forecast_values
                }

                return jsonify(response_data)

            except Exception as e:
                return jsonify({"error": str(e)}), 500

        else:
            # If the GET request fails, return an error message
            return jsonify({"error": "Failed to retrieve data from the API"}), 500
    
#--------RF----------------
    def predictRFTemp():
        try:
            # Make GET request to retrieve data from the specified URL
            url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'
            response = requests.get(url)
            data = response.json()

            # Extract relevant data for forecasting
            timestamps = []
            values = []
            for entry in data['feeds']:
                timestamps.append(entry['created_at'])
                values.append(float(entry['field1']))

            # Create pandas DataFrame
            dataset = pd.DataFrame({'ds': timestamps, 'y': values})
            dataset['ds'] = pd.to_datetime(dataset['ds'])

            # Resample and preprocess the data (similar to previous code)
            dataset = dataset.set_index('ds')
            dataset = dataset.resample('5T').ffill()
            dataset = dataset.dropna()
            dataset = dataset.iloc[1:]
            dataset['time'] = np.arange(len(dataset))

            X = dataset[['time']]
            y = dataset['y']

            # Train the model (similar to previous code)
            model_rf = RandomForestRegressor(**p_rf)
            model_rf.fit(X, y)

            # Generate future timestamps for forecasting
            last_timestamp = dataset.index[-1]
            next_hour_timestamps = pd.date_range(last_timestamp, periods=12, freq='5T')
            next_hour_features = pd.DataFrame({'date': next_hour_timestamps})
            next_hour_features.set_index('date', inplace=True)
            next_hour_features['time'] = np.arange(len(next_hour_features))

            # Make predictions
            predicted_counts = model_rf.predict(next_hour_features)

            # Format predictions as required
            forecast_dates = [str(date) for date in next_hour_timestamps]
            forecast_values = predicted_counts.tolist()

            # Construct JSON response
            response_json = {
                "forecast_dates": forecast_dates,
                "forecast_values": forecast_values
            }

            return jsonify(response_json)
        except Exception as e:
            print(e)
    
    def predictRFHumi():
        try:
            # Make GET request to retrieve data from the specified URL
            url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'
            response = requests.get(url)
            data = response.json()

            # Extract relevant data for forecasting
            timestamps = []
            values = []
            for entry in data['feeds']:
                timestamps.append(entry['created_at'])
                values.append(float(entry['field2']))

            # Create pandas DataFrame
            dataset = pd.DataFrame({'ds': timestamps, 'y': values})
            dataset['ds'] = pd.to_datetime(dataset['ds'])

            # Resample and preprocess the data (similar to previous code)
            dataset = dataset.set_index('ds')
            dataset = dataset.resample('5T').ffill()
            dataset = dataset.dropna()
            dataset = dataset.iloc[1:]
            dataset['time'] = np.arange(len(dataset))

            X = dataset[['time']]
            y = dataset['y']

            # Train the model (similar to previous code)
            model_rf = RandomForestRegressor(**p_rf)
            model_rf.fit(X, y)

            # Generate future timestamps for forecasting
            last_timestamp = dataset.index[-1]
            next_hour_timestamps = pd.date_range(last_timestamp, periods=12, freq='5T')
            next_hour_features = pd.DataFrame({'date': next_hour_timestamps})
            next_hour_features.set_index('date', inplace=True)
            next_hour_features['time'] = np.arange(len(next_hour_features))

            # Make predictions
            predicted_counts = model_rf.predict(next_hour_features)

            # Format predictions as required
            forecast_dates = [str(date) for date in next_hour_timestamps]
            forecast_values = predicted_counts.tolist()

            # Construct JSON response
            response_json = {
                "forecast_dates": forecast_dates,
                "forecast_values": forecast_values
            }

            return jsonify(response_json)
        except Exception as e:
            print(e)
    
    def predictRFCO2():
        try:
            # Make GET request to retrieve data from the specified URL
            url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'
            response = requests.get(url)
            data = response.json()

            # Extract relevant data for forecasting
            timestamps = []
            values = []
            for entry in data['feeds']:
                timestamps.append(entry['created_at'])
                values.append(float(entry['field3']))

            # Create pandas DataFrame
            dataset = pd.DataFrame({'ds': timestamps, 'y': values})
            dataset['ds'] = pd.to_datetime(dataset['ds'])

            # Resample and preprocess the data (similar to previous code)
            dataset = dataset.set_index('ds')
            dataset = dataset.resample('5T').ffill()
            dataset = dataset.dropna()
            dataset = dataset.iloc[1:]
            dataset['time'] = np.arange(len(dataset))

            X = dataset[['time']]
            y = dataset['y']

            # Train the model (similar to previous code)
            model_rf = RandomForestRegressor(**p_rf)
            model_rf.fit(X, y)

            # Generate future timestamps for forecasting
            last_timestamp = dataset.index[-1]
            next_hour_timestamps = pd.date_range(last_timestamp, periods=12, freq='5T')
            next_hour_features = pd.DataFrame({'date': next_hour_timestamps})
            next_hour_features.set_index('date', inplace=True)
            next_hour_features['time'] = np.arange(len(next_hour_features))

            # Make predictions
            predicted_counts = model_rf.predict(next_hour_features)

            # Format predictions as required
            forecast_dates = [str(date) for date in next_hour_timestamps]
            forecast_values = predicted_counts.tolist()

            # Construct JSON response
            response_json = {
                "forecast_dates": forecast_dates,
                "forecast_values": forecast_values
            }

            return jsonify(response_json)
        except Exception as e:
            print(e)
    
    def predictRFCO():
        try:
            # Make GET request to retrieve data from the specified URL
            url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'
            response = requests.get(url)
            data = response.json()

            # Extract relevant data for forecasting
            timestamps = []
            values = []
            for entry in data['feeds']:
                timestamps.append(entry['created_at'])
                values.append(float(entry['field3']))

            # Create pandas DataFrame
            dataset = pd.DataFrame({'ds': timestamps, 'y': values})
            dataset['ds'] = pd.to_datetime(dataset['ds'])

            # Resample and preprocess the data (similar to previous code)
            dataset = dataset.set_index('ds')
            dataset = dataset.resample('5T').ffill()
            dataset = dataset.dropna()
            dataset = dataset.iloc[1:]
            dataset['time'] = np.arange(len(dataset))

            X = dataset[['time']]
            y = dataset['y']

            # Train the model (similar to previous code)
            model_rf = RandomForestRegressor(**p_rf)
            model_rf.fit(X, y)

            # Generate future timestamps for forecasting
            last_timestamp = dataset.index[-1]
            next_hour_timestamps = pd.date_range(last_timestamp, periods=12, freq='5T')
            next_hour_features = pd.DataFrame({'date': next_hour_timestamps})
            next_hour_features.set_index('date', inplace=True)
            next_hour_features['time'] = np.arange(len(next_hour_features))

            # Make predictions
            predicted_counts = model_rf.predict(next_hour_features)

            # Format predictions as required
            forecast_dates = [str(date) for date in next_hour_timestamps]
            forecast_values = predicted_counts.tolist()

            # Construct JSON response
            response_json = {
                "forecast_dates": forecast_dates,
                "forecast_values": forecast_values
            }

            return jsonify(response_json)
        except Exception as e:
            print(e)
    
    def predictRFUV():
        try:
            # Make GET request to retrieve data from the specified URL
            url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'
            response = requests.get(url)
            data = response.json()

            # Extract relevant data for forecasting
            timestamps = []
            values = []
            for entry in data['feeds']:
                timestamps.append(entry['created_at'])
                values.append(float(entry['field5']))

            # Create pandas DataFrame
            dataset = pd.DataFrame({'ds': timestamps, 'y': values})
            dataset['ds'] = pd.to_datetime(dataset['ds'])

            # Resample and preprocess the data (similar to previous code)
            dataset = dataset.set_index('ds')
            dataset = dataset.resample('5T').ffill()
            dataset = dataset.dropna()
            dataset = dataset.iloc[1:]
            dataset['time'] = np.arange(len(dataset))

            X = dataset[['time']]
            y = dataset['y']

            # Train the model (similar to previous code)
            model_rf = RandomForestRegressor(**p_rf)
            model_rf.fit(X, y)

            # Generate future timestamps for forecasting
            last_timestamp = dataset.index[-1]
            next_hour_timestamps = pd.date_range(last_timestamp, periods=12, freq='5T')
            next_hour_features = pd.DataFrame({'date': next_hour_timestamps})
            next_hour_features.set_index('date', inplace=True)
            next_hour_features['time'] = np.arange(len(next_hour_features))

            # Make predictions
            predicted_counts = model_rf.predict(next_hour_features)

            # Format predictions as required
            forecast_dates = [str(date) for date in next_hour_timestamps]
            forecast_values = predicted_counts.tolist()

            # Construct JSON response
            response_json = {
                "forecast_dates": forecast_dates,
                "forecast_values": forecast_values
            }

            return jsonify(response_json)
        except Exception as e:
            print(e)

    def predictRFPM():
        try:
            # Make GET request to retrieve data from the specified URL
            url = 'https://api.thingspeak.com/channels/2465663/feeds.json?results=100'
            response = requests.get(url)
            data = response.json()

            # Extract relevant data for forecasting
            timestamps = []
            values = []
            for entry in data['feeds']:
                timestamps.append(entry['created_at'])
                values.append(float(entry['field6']))

            # Create pandas DataFrame
            dataset = pd.DataFrame({'ds': timestamps, 'y': values})
            dataset['ds'] = pd.to_datetime(dataset['ds'])

            # Resample and preprocess the data (similar to previous code)
            dataset = dataset.set_index('ds')
            dataset = dataset.resample('5T').ffill()
            dataset = dataset.dropna()
            dataset = dataset.iloc[1:]
            dataset['time'] = np.arange(len(dataset))

            X = dataset[['time']]
            y = dataset['y']

            # Train the model (similar to previous code)
            model_rf = RandomForestRegressor(**p_rf)
            model_rf.fit(X, y)

            # Generate future timestamps for forecasting
            last_timestamp = dataset.index[-1]
            next_hour_timestamps = pd.date_range(last_timestamp, periods=12, freq='5T')
            next_hour_features = pd.DataFrame({'date': next_hour_timestamps})
            next_hour_features.set_index('date', inplace=True)
            next_hour_features['time'] = np.arange(len(next_hour_features))

            # Make predictions
            predicted_counts = model_rf.predict(next_hour_features)

            # Format predictions as required
            forecast_dates = [str(date) for date in next_hour_timestamps]
            forecast_values = predicted_counts.tolist()

            # Construct JSON response
            response_json = {
                "forecast_dates": forecast_dates,
                "forecast_values": forecast_values
            }

            return jsonify(response_json)
        except Exception as e:
            print(e)
            
            
            
            