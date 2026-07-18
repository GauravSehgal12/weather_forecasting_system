



import requests
from datetime import datetime
from fastapi import HTTPException
from app.config import OPEN_METEO_URL, TIMEZONE, FORECAST_DAYS
from app.utils.weather_utils import WEATHER_MAP

def get_forecast(latitude: float, longitude: float) -> list[dict]:
    """
    Fetches 5-day weather forecast data from Open-Meteo API.

    Args:
        latitude  (float): Geographic latitude of the city
        longitude (float): Geographic longitude of the city

    Returns:
        list[dict]:
            List containing daily forecast information such as:
            - date
            - min/max temperature
            - weather description
            - icon code
            - wind speed
    """

    params = {
        "latitude": latitude,
        "longitude": longitude,

        
        "daily": ",".join([
            "temperature_2m_max",
            "temperature_2m_min",
            "weathercode",
            "windspeed_10m_max",
        ]),

     
        "timezone": TIMEZONE,

     
        "forecast_days": FORECAST_DAYS,
    }


    try:
        response = requests.get(
            OPEN_METEO_URL,
            params=params,
            timeout=10
        )

        response.raise_for_status()

    except requests.RequestException as e:

        # Convert request errors into FastAPI HTTPException
        raise HTTPException(
            status_code=503,
            detail=f"Forecast service unavailable: {str(e)}"
        )

   
    data = response.json()

    daily_data = data.get("daily", {})

    
    dates = daily_data.get("time", [])
    temp_min = daily_data.get("temperature_2m_min", [])
    temp_max = daily_data.get("temperature_2m_max", [])
    weather_codes = daily_data.get("weathercode", [])
    wind_speeds = daily_data.get("windspeed_10m_max", [])

   
    forecasts = []

   
    for i in range(len(dates)):

        
        weather_code = weather_codes[i]

        description, icon = WEATHER_MAP.get(
            weather_code,
            ("Unknown", "01d")
        )

        
        formatted_date = datetime.strptime(
            dates[i],
            "%Y-%m-%d"
        ).strftime("%a, %b %d")

        forecasts.append({

           
            "date": formatted_date,

            
            "time": "12:00",

         
            "temp_min": round(temp_min[i]),

         
            "temp_max": round(temp_max[i]),

            
            "description": description,

            
            "icon": icon,

            
            "humidity": None,

            "wind_speed": round(wind_speeds[i]),
        })

   
    return forecasts