import requests


def get_nwp_data(latitude, longitude):
    url = "https://api.open-meteo.com/v1/forecast"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": [
            "precipitation",
            "temperature_2m",
            "relative_humidity_2m",
            "surface_pressure",
            "wind_speed_10m"
        ],
        "forecast_days": 1,
        "timezone": "auto"
    }

    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()

    data = response.json()

    return data


if __name__ == "__main__":
    data = get_nwp_data(16.3067, 80.4365)

    print("NWP data fetched successfully.")
    print("Available forecast fields:")
    print(data["hourly"].keys())