import json
from flask import jsonify
import requests
import csv
from datetime import datetime, timedelta
import pandas as pd
import os
import time

class TestController:
    def getSampleData():
        token = "eyJhbGciOiJIUzUxMiJ9.eyJzdWIiOiJ0ZW5hbnRAdGhpbmdzYm9hcmQub3JnIiwidXNlcklkIjoiODkwODJiMDAtNDAyYi0xMWVmLWEwZWEtZDkzN2RlNDlmOTFjIiwic2NvcGVzIjpbIlRFTkFOVF9BRE1JTiJdLCJzZXNzaW9uSWQiOiI5MGMwMjE3ZC1iYTQ0LTQwMjUtOWE3NS0wOWUwZjQ1ODA1YTIiLCJleHAiOjE3MjgyMDgyMTMsImlzcyI6InRoaW5nc2JvYXJkLmlvIiwiaWF0IjoxNzI4MTk5MjEzLCJlbmFibGVkIjp0cnVlLCJpc1B1YmxpYyI6ZmFsc2UsInRlbmFudElkIjoiODY4ZGI1MjAtNDAyYi0xMWVmLWEwZWEtZDkzN2RlNDlmOTFjIiwiY3VzdG9tZXJJZCI6IjEzODE0MDAwLTFkZDItMTFiMi04MDgwLTgwODA4MDgwODA4MCJ9.DiUlmNlmVugiffRAAhFfxtqukuxwyw-2N6AZnx80o4hZyaZwGOGL-waASgReC_yYgwVM_7XHgbFlkJvlcYtBZg"
        fields = ["temperature", "humidity", "dust", "mq135", "mq3", "uv"]
        deviceID = "bf64aa90-4033-11ef-ac72-df8184d14926"
        limit = 20
        sort = "DESC"
        host = "192.168.31.243:8080"

        # Calculate the current time as endTs
        end = int(time.time() * 1000)  # current time in milliseconds
        start = 0  # starting from the oldest possible time

        url = f"http://{host}/api/plugins/telemetry/DEVICE/{deviceID}/values/timeseries?keys={','.join(fields)}&startTs={start}&endTs={end}&limit={limit}&sortOrder={sort}"
        headers = {
            'Accept': 'application/json',
            'Authorization': f"Bearer {token}"
        }
        response = requests.get(url, headers=headers)

        if response.status_code != 200:
            return jsonify({"error": "Failed to fetch data"}), response.status_code

        data = response.json()

        # Check if data is a dictionary with lists as values
        if not isinstance(data, dict):
            return jsonify({"error": "Unexpected data format"}), 500

        try:
            # Create a structure that matches your expected output format
            formatted_data = []

            for i in range(limit):  # Limit to requested records
                record = {}
                
                # Extract values from the fetched data (using 'nan' as placeholder if data is missing)
                record["CO"] = str(data.get("mq135", [{}])[i].get("value", "nan"))
                record["Gas"] = str(data.get("mq3", [{}])[i].get("value", "nan"))
                record["Humidity"] = str(data.get("humidity", [{}])[i].get("value", "nan"))
                record["PM2.5"] = str(data.get("dust", [{}])[i].get("value", "nan"))
                record["Temperature"] = str(data.get("temperature", [{}])[i].get("value", "nan"))
                record["UV"] = str(data.get("uv", [{}])[i].get("value", "nan"))

                # Format the timestamp
                timestamp = data.get("temperature", [{}])[i].get("ts", None)
                if timestamp:
                    record["createdTime"] = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')
                else:
                    record["createdTime"] = "Unknown"

                formatted_data.append(record)

        except Exception as e:
            return jsonify({"error": str(e)}), 500

        return jsonify(formatted_data)
  
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
            pm25_breakpoints = [0, 12.0, 35.4, 55.4, 150.4, 250.4, 500.4]
            pm25_concentration_ranges = [(0, 12.0), (12.1, 35.4), (35.5, 55.4), (55.5, 150.4), (150.5, 250.4), (250.5, 500.4)]

            co_breakpoints = [0, 4.4, 9.4, 12.4, 15.4, 30.4, 50.4]
            co_concentration_ranges = [(0, 4.4), (4.5, 9.4), (9.5, 12.4), (12.5, 15.4), (15.5, 30.4), (30.5, 50.4)]

            # Calculate AQI for PM2.5
            def calculate_pm25_aqi(pm25):
                # AQI calculation logic for PM2.5 in μg/m³
                if 0 <= pm25 <= 12.0:
                    aqi_low, aqi_high = 0, 50
                elif 12.1 <= pm25 <= 35.4:
                    aqi_low, aqi_high = 51, 100
                elif 35.5 <= pm25 <= 55.4:
                    aqi_low, aqi_high = 101, 150
                elif 55.5 <= pm25 <= 150.4:
                    aqi_low, aqi_high = 151, 200
                elif 150.5 <= pm25 <= 250.4:
                    aqi_low, aqi_high = 201, 300
                else:
                    aqi_low, aqi_high = 301, 500
                
                # Find the corresponding PM2.5 concentration range
                for i in range(len(pm25_breakpoints) - 1):
                    if pm25_breakpoints[i] <= pm25 <= pm25_breakpoints[i + 1]:
                        bp_low, bp_high = pm25_concentration_ranges[i]
                        aqi = ((aqi_high - aqi_low) / (bp_high - bp_low)) * (pm25 - bp_low) + aqi_low
                        return round(aqi)


            # Calculate AQI for CO
            def calculate_co_aqi(co):
                # AQI calculation logic for CO in ppm
                if 0 <= co <= 4.4:
                    aqi_low, aqi_high = 0, 50
                elif 4.5 <= co <= 9.4:
                    aqi_low, aqi_high = 51, 100
                elif 9.5 <= co <= 12.4:
                    aqi_low, aqi_high = 101, 150
                elif 12.5 <= co <= 15.4:
                    aqi_low, aqi_high = 151, 200
                elif 15.5 <= co <= 30.4:
                    aqi_low, aqi_high = 201, 300
                else:
                    aqi_low, aqi_high = 301, 500
                
                # Find the corresponding CO concentration range
                for i in range(len(co_breakpoints) - 1):
                    if co_breakpoints[i] <= co <= co_breakpoints[i + 1]:
                        bp_low, bp_high = co_concentration_ranges[i]
                        aqi = ((aqi_high - aqi_low) / (bp_high - bp_low)) * (co - bp_low) + aqi_low
                        return round(aqi)


            pm25_aqi = calculate_pm25_aqi(pm25)
            co_aqi = calculate_co_aqi(co)

            return pm25_aqi, co_aqi

        def map_aqi_range_and_category(overall_aqi, pollutant):
            if pollutant == "CO":
                if overall_aqi <= 50:
                    return "Good", "0-50"
                elif overall_aqi <= 100:
                    return "Moderate", "51-100"
                elif overall_aqi <= 150:
                    return "Unhealthy for Sensitive Groups", "101-150"
                elif overall_aqi <= 200:
                    return "Unhealthy", "151-200"
                elif overall_aqi <= 300:
                    return "Very Unhealthy", "201-300"
                else:
                    return "Hazardous", "301-500"
            else:
                # Assume PM2.5 AQI range
                if overall_aqi <= 50:
                    return "Good", "0-50"
                elif overall_aqi <= 100:
                    return "Moderate", "51-100"
                elif overall_aqi <= 150:
                    return "Unhealthy for Sensitive Groups", "101-150"
                elif overall_aqi <= 200:
                    return "Unhealthy", "151-200"
                elif overall_aqi <= 300:
                    return "Very Unhealthy", "201-300"
                else:
                    return "Hazardous", "301-500"

        def map_aqi_score(aqi):
            if aqi is None or aqi < 0 or aqi > 500:
                return None

            # Map AQI score using the equation y = (-x/50) + 10
            aqi_score = (-aqi / 50) + 10

            # Round to one decimal place
            rounded_aqi_score = round(aqi_score, 1)
            return rounded_aqi_score
        
        channel_id = '2465663'
        read_api_key = 'MP0MEWPWMADVCPMG'
        results = requests.get(f'https://api.thingspeak.com/channels/{channel_id}/feeds.csv', params={'api_key': read_api_key})
        if results.status_code != 200:
            return "Failed to fetch data from ThingSpeak."
        
        csv_data = results.text.splitlines()
        reader = csv.DictReader(csv_data)
        data = list(reader)
        
        timestamps = [datetime.strptime(row['created_at'], '%Y-%m-%d %H:%M:%S %Z') for row in data]
        current_time = max(timestamps)
        pm25_start_time = current_time - timedelta(hours=24)
        co_start_time = current_time - timedelta(hours=8)

        pm25_data = [float(row['field6']) for row, ts in zip(data, timestamps) if ts >= pm25_start_time]
        co_data = [float(row['field4']) for row, ts in zip(data, timestamps) if ts >= co_start_time]

        pm25_avg = sum(pm25_data) / len(pm25_data) if pm25_data else None
        co_avg = sum(co_data) / len(co_data) if co_data else None
        print(pm25_avg, co_avg)

        pm25_aqi, co_aqi = calculate_aqi(pm25_avg, co_avg)
        
        overall_aqi = max(co_aqi, pm25_aqi) if co_aqi is not None and pm25_aqi is not None else co_aqi if co_aqi is not None else pm25_aqi if pm25_aqi is not None else None
        
        if overall_aqi is None:
            return {'error': 'Failed to calculate AQI for both pollutants.'}
        
        category, aqi_range = map_aqi_range_and_category(overall_aqi, "CO" if co_aqi is not None else "PM2.5")
        aqi_score = map_aqi_score(overall_aqi)

        return {
            'pM25_aqi': pm25_aqi,
            'co_aqi': co_aqi,
            'overall_aqi': overall_aqi,
            'air_quality_category': category,
            'aqi_range': aqi_range,
            'aqi_score': aqi_score
        }
