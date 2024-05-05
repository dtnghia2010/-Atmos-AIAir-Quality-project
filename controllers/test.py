'''
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
                print(f"File not found: {temp_lstm_weight}")
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
            humiData = [float(feed['field2']) for feed in feeds]  # Assuming humidity data is in field1

            # Create DataFrame from extracted data
            datetimeHumi = pd.to_datetime(humiTime)
            dataset = pd.DataFrame({'ds': datetimeHumi, 'y': humiData})
            dataset = dataset.set_index('ds')
            dataset = dataset.resample('5T').ffill()
            dataset = dataset.dropna()
            dataset = dataset.iloc[1:]
            dataset.reset_index(inplace=True)

            # Scale the data to be between 0 and 1
            scaler = MinMaxScaler()
            scaled_humi = scaler.fit_transform(dataset[['y']])

            # Ensure the sequence length matches the model's input (12 time steps)
            sequence_length = 12

            # Pad or truncate the sequence to match the model's input sequence length
            if len(scaled_humi) < sequence_length:
                padded_humi = np.pad(scaled_humi, ((sequence_length - len(scaled_humi), 0), (0, 0)), mode='constant')
            else:
                padded_humi = scaled_humi[-sequence_length:]

            # Reshape the data to be suitable for LSTM (samples, time steps, features)
            input_data = padded_humi.reshape((1, 1, sequence_length))

            # Load model architecture from JSON file
            humi_lstm_json = os.path.join(server_dir, 'server\datasets\models\lstm\humi-lstm.json')
            humi_lstm_weight = os.path.join(server_dir, 'server\datasets\models\lstm\humi_lstm_weight.h5')
            
            if os.path.exists(humi_lstm_weight) and os.path.exists(humi_lstm_json):
                with open(humi_lstm_json, 'r') as json_file:
                    loaded_model_json = json_file.read()

                # Load model json
                loaded_model = model_from_json(loaded_model_json)

                # Load model weights
                loaded_model.load_weights(humi_lstm_weight)
                print("--------Model loaded successfully---------")

                # Make predictions
                predictions = loaded_model.predict(input_data)

                # Inverse transform the predictions to get original scale
                predictions_inv = scaler.inverse_transform(predictions)[0]

                # Round up to 2 decimal
                predictions_rounded = np.around(predictions_inv, decimals=4)

                # Get the date and time of the forecast
                forecast_dates = pd.date_range(datetimeHumi[-1], periods=12, freq='5T')

                # Prepare JSON response with forecast and forecast dates
                response_data = {
                    'forecast_dates': forecast_dates.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                    'forecast_values': predictions_rounded.tolist()
                }

            else:
                print(f"File not found: {humi_lstm_weight}")
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
            sequence_length = 12

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
                sequence_length = 12

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

            # Create DataFrame from extracted data
            datetimeUV = pd.to_datetime(uvTime)
            dataset = pd.DataFrame({'ds': datetimeUV, 'y': uvData})
            dataset = dataset.set_index('ds')
            dataset = dataset.resample('5T').ffill()
            dataset = dataset.dropna()
            dataset = dataset.iloc[1:]
            dataset.reset_index(inplace=True)

            # Scale the data to be between 0 and 1
            scaler = MinMaxScaler()
            scaled_uv = scaler.fit_transform(dataset[['y']])

            # Ensure the sequence length matches the model's input (12 time steps)
            sequence_length = 12

            # Pad or truncate the sequence to match the model's input sequence length
            if len(scaled_uv) < sequence_length:
                padded_uv = np.pad(scaled_uv, ((sequence_length - len(scaled_uv), 0), (0, 0)), mode='constant')
            else:
                padded_uv = scaled_uv[-sequence_length:]

            # Reshape the data to be suitable for LSTM (samples, time steps, features)
            input_data = padded_uv.reshape((1, 1, sequence_length))

            # Load model architecture from JSON file
            uv_lstm_json = os.path.join(server_dir, r'server\datasets\models\lstm\uv-lstm.json')
            uv_lstm_weight = os.path.join(server_dir, r'server\datasets\models\lstm\uv_lstm_weight.h5')
            
            if os.path.exists(uv_lstm_weight) and os.path.exists(uv_lstm_json):
                with open(uv_lstm_json, 'r') as json_file:
                    loaded_model_json = json_file.read()

                # Load model json
                loaded_model = model_from_json(loaded_model_json)

                # Load model weights
                loaded_model.load_weights(uv_lstm_weight)
                print("--------Model loaded successfully---------")

                # Make predictions
                predictions = loaded_model.predict(input_data)

                # Inverse transform the predictions to get original scale
                predictions_inv = scaler.inverse_transform(predictions)[0]

                # Round up to 2 decimal
                predictions_rounded = np.around(predictions_inv, decimals=4)

                # Get the date and time of the forecast
                forecast_dates = pd.date_range(datetimeUV[-1], periods=12, freq='5T')

                # Prepare JSON response with forecast and forecast dates
                response_data = {
                    'forecast_dates': forecast_dates.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                    'forecast_values': predictions_rounded.tolist()
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

            # Create DataFrame from extracted data
            datetimePM25 = pd.to_datetime(pm25Time)
            dataset = pd.DataFrame({'ds': datetimePM25, 'y': pm25Data})
            dataset = dataset.set_index('ds')
            dataset = dataset.resample('5T').ffill()
            dataset = dataset.dropna()
            dataset = dataset.iloc[1:]
            dataset.reset_index(inplace=True)

            # Scale the data to be between 0 and 1
            scaler = MinMaxScaler()
            scaled_pm25 = scaler.fit_transform(dataset[['y']])

            # Ensure the sequence length matches the model's input (12 time steps)
            sequence_length = 12

            # Pad or truncate the sequence to match the model's input sequence length
            if len(scaled_pm25) < sequence_length:
                padded_pm25 = np.pad(scaled_pm25, ((sequence_length - len(scaled_pm25), 0), (0, 0)), mode='constant')
            else:
                padded_pm25 = scaled_pm25[-sequence_length:]

            # Reshape the data to be suitable for LSTM (samples, time steps, features)
            input_data = padded_pm25.reshape((1, 1, sequence_length))

            # Load model architecture from JSON file
            pm25_lstm_json = os.path.join(server_dir, 'server\datasets\models\lstm\pm25-lstm.json')
            pm25_lstm_weight = os.path.join(server_dir, 'server\datasets\models\lstm\pm25_lstm_weight.h5')
            
            if os.path.exists(pm25_lstm_weight) and os.path.exists(pm25_lstm_json):
                with open(pm25_lstm_json, 'r') as json_file:
                    loaded_model_json = json_file.read()

                # Load model json
                loaded_model = model_from_json(loaded_model_json)

                # Load model weights
                loaded_model.load_weights(pm25_lstm_weight)
                print("--------Model loaded successfully---------")

                # Make predictions
                predictions = loaded_model.predict(input_data)

                # Inverse transform the predictions to get original scale
                predictions_inv = scaler.inverse_transform(predictions)[0]

                # Round up to 2 decimal
                predictions_rounded = np.around(predictions_inv, decimals=4)

                # Get the date and time of the forecast
                forecast_dates = pd.date_range(datetimePM25[-1], periods=12, freq='5T')

                # Prepare JSON response with forecast and forecast dates
                response_data = {
                    'forecast_dates': forecast_dates.strftime('%Y-%m-%d %H:%M:%S').tolist(),
                    'forecast_values': predictions_rounded.tolist()
                }

            else:
                print(f"File not found: {pm25_lstm_weight}")
                response_data = {'error': 'Model not found'}

        except Exception as e:
            print(e)
            response_data = {'error': str(e)}

        return jsonify(response_data)
'''

