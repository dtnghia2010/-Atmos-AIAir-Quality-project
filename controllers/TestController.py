import json
from bson import json_util
from flask import jsonify
import requests
import csv
from datetime import datetime, timedelta
import pandas as pd
import os

class TestController:
    def getSampleData():
            channel_id = "2465663"
            api_key = "MP0MEWPWMADVCPMG"
            results = "20"

            url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json?api_key={api_key}&results={results}"
            
            response = requests.get(url)
            data = response.json()['feeds']

            # Prepare sample data
            sample_data = []
            for entry in data:
                created_time = datetime.strptime(entry['created_at'], '%Y-%m-%dT%H:%M:%SZ').timestamp()
                sample_entry = {
                    'createdTime': created_time,
                    'Gas': entry['field3'],
                    'CO': entry['field4'],
                    'PM2.5': entry['field6'],
                    'UV': entry['field5'],
                    'Temperature': entry['field1'],
                    'Humidity': entry['field2']
                }
                sample_data.append(sample_entry)

            return jsonify({
                'message': 'Successfully fetched sample data.',
                'data': sample_data
            })
  
    def fetchOfflineData():
            def fetchData(field):
                url = f"https://api.thingspeak.com/channels/{channel_id}/fields/{field}.json?api_key={api_key}&results={results}"
                response = requests.get(url)
                try:
                    data = response.json()
                    return data
                except json.JSONDecodeError:
                    print(f"Error: Failed to decode JSON response for field {field}.")
                    return None

            def fetchAndSaveData(field, field_name):
                data = fetchData(field)
                if data and 'feeds' in data:
                    json_file_path = os.path.join("datasets", f"{field_name}.json")
                    formatted_data = [{"ts": int(datetime.fromisoformat(feed['created_at'].replace('Z', '+00:00')).timestamp()), "value": feed[f'field{field}']} for feed in data['feeds']]
                    with open(json_file_path, "w") as json_file:
                        json.dump(formatted_data, json_file, indent=2)
                else:
                    print(f"Error: Missing 'feeds' key in the response for field {field}.")

            def fetch_data_and_export_csv():
                channel_id = "2465663"
                api_key = "MP0MEWPWMADVCPMG"
                results = "8000"

                url = f"https://api.thingspeak.com/channels/{channel_id}/feeds.json?api_key={api_key}&results={results}"
                
                response = requests.get(url)
                data = response.json()['feeds']

                # Prepare CSV data
                csv_data = []
                for entry in data:
                    csv_row = {
                        'Time': entry['created_at'],
                        'Gas': entry['field3'],
                        'CO': entry['field4'],
                        'PM25': entry['field6'],
                        'UV': entry['field5'],
                        'TEMP': entry['field1'],
                        'HUMI': entry['field2']
                    }
                    csv_data.append(csv_row)

                # Export CSV file
                csv_file_path = 'datasets/output/data.csv'
                with open(csv_file_path, mode='w', newline='') as csv_file:
                    field_names = ['Time', 'Gas', 'CO', 'PM25', 'UV', 'TEMP', 'HUMI']
                    writer = csv.DictWriter(csv_file, fieldnames=field_names)
                    writer.writeheader()
                    writer.writerows(csv_data)

                

            channel_id = "2465663"
            api_key = "MP0MEWPWMADVCPMG"
            results = "8000"

            fields = {"1": "temp", "2": "humi", "3": "gas", "4": "co", "6": "pm2.5", "5": "uv_index"}  # Field IDs on ThingSpeak mapped to specific names
            for field, field_name in fields.items():
                fetchAndSaveData(field, field_name)
            
            fetch_data_and_export_csv()

            return jsonify({
                'message': 'Successfully fetched offline data and exported to JSON files.'
            })
  
    def calculate_aqi_from_csv():
        def calculate_aqi(pm25, co):
            # AQI breakpoints and corresponding pollutant concentration ranges
            pm25_breakpoints = [0, 30.4, 60.4, 90.4, 120.4, 250.4, 350.4]
            pm25_concentration_ranges = [(0, 30.4), (30.5, 60.4), (60.5, 90.4), (90.5, 120.4), (120.5, 250.4), (250.5, 350.4), (350.5, 500.4)]

            co_breakpoints = [0, 1.0, 2.0, 10.0, 17.0, 34.0, 50.4]
            co_concentration_ranges = [(0, 1.0), (1.1, 2.0), (2.1, 10.0), (10.1, 17.0), (17.1, 34.0), (34.1, 50.4), (50.5, 60)]

            # Calculate AQI for PM2.5
            def calculate_pm25_aqi(pm25):
                for i in range(len(pm25_breakpoints) - 1):
                    if pm25_breakpoints[i] <= pm25 <= pm25_breakpoints[i + 1]:
                        bp_low, bp_high = pm25_concentration_ranges[i]
                        aqi_low, aqi_high = pm25_breakpoints[i], pm25_breakpoints[i + 1]
                        aqi = ((aqi_high - aqi_low) / (bp_high - bp_low)) * (pm25 - bp_low) + aqi_low
                        return round(aqi)

            # Calculate AQI for CO
            def calculate_co_aqi(co):
                molecular_weight_co = 28.01  # g/mol
                co_mg_per_m3 = (co * molecular_weight_co) / 24.45

                # AQI calculation logic for CO in mg/m³
                for i in range(len(co_breakpoints) - 1):
                    if co_breakpoints[i] <= co_mg_per_m3 <= co_breakpoints[i + 1]:
                        bp_low, bp_high = co_concentration_ranges[i]
                        aqi_low, aqi_high = co_breakpoints[i], co_breakpoints[i + 1]
                        aqi = ((aqi_high - aqi_low) / (bp_high - bp_low)) * (co_mg_per_m3 - bp_low) + aqi_low
                        return round(aqi)

            pm25_aqi = calculate_pm25_aqi(pm25)
            co_aqi = calculate_co_aqi(co)

            return pm25_aqi, co_aqi
    
        def map_aqi_range_and_category(overall_aqi, pollutant):
            if pollutant == "CO":
                if overall_aqi <= 50.4:
                    return "Good", "0-50"
                elif overall_aqi <= 100.4:
                    return "Satisfactory", "51-100"
                elif overall_aqi <= 200.4:
                    return "Moderately Polluted", "101-200"
                elif overall_aqi <= 300.4:
                    return "Poor", "201-300"
                elif overall_aqi <= 400.4:
                    return "Very Poor", "301-400"
                else:
                    return "Severe", "401+"
            else:
                # Assume PM2.5 AQI range
                if overall_aqi <= 50.4:
                    return "Good", "0-50"
                elif overall_aqi <= 100.4:
                    return "Satisfactory", "51-100"
                elif overall_aqi <= 200.4:
                    return "Moderately Polluted", "101-200"
                elif overall_aqi <= 300.4:
                    return "Poor", "201-300"
                elif overall_aqi <= 400.4:
                    return "Very Poor", "301-400"
                else:
                    return "Severe", "401+"

        def map_aqi_score(aqi):
            if aqi < 0 or aqi > 500:
                return None
            
            # Map AQI score using the equation y = (-x/50) + 10
            aqi_score = (-aqi / 50) + 10
            
            # Round down to the nearest integer
            rounded_aqi_score = int(aqi_score)
            return rounded_aqi_score
        
        file_path = 'datasets/output/data.csv'
        try:
            with open(file_path, 'r') as csv_file:
                reader = csv.DictReader(csv_file)
                data = [row for row in reader if all(row.values())]  # Skip rows with missing values
        except FileNotFoundError:
            return "File not found."

        # Convert timestamps to datetime objects
        timestamps = [datetime.strptime(row['Time'], '%Y-%m-%dT%H:%M:%SZ') for row in data]

        # Calculate the start time for the 24-hour and 8-hour periods
        current_time = max(timestamps)
        pm25_start_time = current_time - timedelta(hours=24)
        co_start_time = current_time - timedelta(hours=8)

        # Filter data within the specified time frames
        pm25_data = [float(row['PM25']) for row, ts in zip(data, timestamps) if ts >= pm25_start_time]
        co_data = [float(row['CO']) for row, ts in zip(data, timestamps) if ts >= co_start_time]

        # Calculate 24-hour average for PM2.5 and 8-hour average for CO
        if pm25_data:
            pm25_avg = sum(pm25_data) / len(pm25_data)
        else:
            pm25_avg = None

        if co_data:
            co_avg = sum(co_data) / len(co_data)
        else:
            co_avg = None

        # Calculate AQI
        pm25_aqi, co_aqi = calculate_aqi(pm25_avg, co_avg)
        
        # Determine the overall AQI based on CO if it's not None, otherwise use PM2.5
        if co_aqi is not None:
            overall_aqi = co_aqi
        elif pm25_aqi is not None:
            overall_aqi = pm25_aqi
        else:
            # Handle the case where both AQI values are None
            return {'error': 'Failed to calculate AQI for both pollutants.'}
        
        # Map the overall AQI to air quality category and AQI range
        category, aqi_range = map_aqi_range_and_category(overall_aqi, "CO" if co_aqi is not None else "PM2.5")
        aqi_score = map_aqi_score(overall_aqi)
        return {
            'PM2.5 AQI': pm25_aqi, 
            'CO AQI': co_aqi, 
            'Overall AQI': overall_aqi,
            'Air Quality Category': category,
            'AQI Range': aqi_range,
            'AQI Score:': aqi_score
        }