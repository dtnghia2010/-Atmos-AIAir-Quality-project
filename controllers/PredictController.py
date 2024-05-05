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
    '''
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
'''
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
            
            
            
            