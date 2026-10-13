import requests
import math
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
import os

load_dotenv(override=True)

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

search_tool = TavilySearchResults(
    api_key=TAVILY_API_KEY,
    max_results=3,
    topic="general",
    search_depth="advanced")

@tool
def calculator(expression:str)->str:
    """
    Useful for simple match calculations.
    Input should be valid math expression.
    Example: 2+ 4, math.sqrt(16), 10 * 5
    """
    try:
        allowed = {
            "math":math,
            "abs":abs,            
            "min":min,
            "max":max,
            "sum":sum
        }
        result = eval(expression, {"__builtins__":{}}, allowed)
        return str(result)
    except Exception as e:
        return f"Calculation error: {str(e)}"

@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbod (e.g. 'AAPL','TSLA')
    Using Alpha Vantage with API Key in the URL
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    r = requests.get(url)
    return r.json()

@tool
def get_weather_data(city: str) -> str:
    """Fetch current weather data for a city from weatherstak.com"""    
    if not WEATHERSTACK_API_KEY:
        raise ValueError("WEATHERSTACK_API_KEY is not set in the environment variables.")

    url = f"http://api.weatherstack.com/current?access_key={WEATHERSTACK_API_KEY}&query={city}"
    response = requests.get(url)
    data = response.json()

    if "error" in data:
        return f"Error fetching weather data: {data['error']['info']}"

    if "current" not in data:
        return "Error: No current weather data found."
    current_weather = data.get("current", {})
    temperature = current_weather.get("temperature")
    humidity = current_weather.get("humidity")
    weather_descriptions = current_weather.get("weather_descriptions", [])
    description = weather_descriptions[0] if weather_descriptions else "No description available"

    return {
        "city": city,
        "temperature": temperature,
        "humidity": humidity,
        "description": description
    }

tools_list = [search_tool,get_stock_price, get_weather_data,calculator]