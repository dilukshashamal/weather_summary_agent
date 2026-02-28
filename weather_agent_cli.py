import boto3
import json
import subprocess
import time
from datetime import datetime

def call_claude_sonnet(prompt):
    """
    Send prompt to Claude 4.5 Sonnet via Amazon Bedrock.
    
    Args:
        prompt: Text prompt for Claude
    
    Returns:
        tuple: (success: bool, response: str)
    """
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-west-2'
    )
    
    try:
        response = bedrock.converse(
            modelId='us.anthropic.claude-sonnet-4-5-20250929-v1:0',
            messages=[{
                "role": "user",
                "content": [{"text": prompt}]
            }],
            inferenceConfig={
                "maxTokens": 2000,
                "temperature": 0.7
            }
        )
        return True, response['output']['message']['content'][0]['text']
    except Exception as e:
        return False, f"Error calling Claude: {str(e)}"

def execute_curl_command(url):
    """
    Execute curl command to fetch API data.
    
    Args:
        url: API endpoint URL
    
    Returns:
        tuple: (success: bool, response: str)
    """
    try:
        result = subprocess.run(
            ['curl', '-s', url],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return True, result.stdout
        else:
            return False, f"Curl command failed: {result.stderr}"
    except subprocess.TimeoutExpired:
        return False, "Request timed out after 30 seconds"
    except Exception as e:
        return False, f"Error executing curl: {str(e)}"

def generate_weather_api_calls(location):
    """
    Use Claude to generate NWS API calls for a location.
    
    Args:
        location: Location name, ZIP code, or description
    
    Returns:
        tuple: (success: bool, api_calls: list)
    """
    prompt = f"""
You are an expert at working with the National Weather Service (NWS) API.

Your task: Generate the NWS API URL to get weather forecast data for "{location}".

Instructions:
1. First, determine the approximate latitude and longitude coordinates for this location
2. Generate the NWS Points API URL: https://api.weather.gov/points/{{lat}},{{lon}}

For the coordinates, use your knowledge to estimate:
- Major cities: Use well-known coordinates
- ZIP codes: Estimate based on the area
- States: Use approximate center coordinates
- In case a location description is provided instead of a location name, please use the most likely city and state name as the location for the coordinates

Example for Seattle:
https://api.weather.gov/points/47.6062,-122.3321

Example for largest city in USA:
Based on your knowledge, you will establish location is New York City
https://api.weather.gov/points/40.7128,-74.0060

Now generate the API call (Points API) for the established location. 
Return ONLY the complete Points API URL, nothing else.
Format: https://api.weather.gov/points/LAT,LON
"""
    
    print(f"AI is analyzing '{location}' and generating weather API calls...")
    success, response = call_claude_sonnet(prompt)
    
    if success:
        api_url = response.strip()
        if api_url.startswith('https://api.weather.gov/points/'):
            return True, [api_url]
        else:
            return False, f"AI generated invalid URL: {api_url}"
    else:
        return False, response

def get_forecast_url_from_points_response(points_json):
    """
    Extract forecast URL from Points API response.
    
    Args:
        points_json: JSON response string
    
    Returns:
        tuple: (success: bool, forecast_url: str)
    """
    try:
        data = json.loads(points_json)
        forecast_url = data['properties']['forecast']
        return True, forecast_url
    except (json.JSONDecodeError, KeyError) as e:
        return False, f"Error parsing Points API response: {str(e)}"

def process_weather_response(raw_json, location):
    """
    Use Claude to convert NWS API JSON into readable summary.
    
    Args:
        raw_json: Raw JSON response
        location: Location name for context
    
    Returns:
        tuple: (success: bool, summary: str)
    """
    prompt = f"""
You are a weather information specialist. I have raw National Weather Service forecast data for "{location}" that needs to be converted into a clear, helpful summary for a general audience.

Raw NWS API Response:
{raw_json}

Please create a weather summary that includes:
1. A brief introduction with the location
2. Current conditions and today's forecast
3. The next 2-3 days outlook with key details (temperature, precipitation, wind)
4. Any notable weather patterns or alerts
5. Format the response to be easy to read and understand

Make it informative and practical for someone planning their activities. Focus on being helpful and clear.
"""
    
    print("AI is processing weather data and creating summary...")
    success, response = call_claude_sonnet(prompt)
    
    return success, response

def run_weather_agent():
    """
    Main orchestration function for the AI weather agent.
    """
    print("Welcome to the Weather AI Agent!")
    print("This agent uses Claude 4.5 Sonnet to help you get weather forecasts.")
    print("=" * 60)
    
    while True:
        location = input("\nEnter a location name or description (or 'quit' to exit): ").strip()
        
        if location.lower() in ['quit', 'exit', 'q']:
            print("Thanks for using the Weather Agent!")
            break
            
        if not location:
            print("[ERROR] Please enter a location name or description.")
            continue
            
        print(f"\nStarting weather analysis for '{location}'...")
        print("-" * 40)
        
        print("Step 1: AI Planning Phase")
        success, api_calls = generate_weather_api_calls(location)
        
        if not success:
            print(f"[ERROR] Failed to generate API calls: {api_calls}")
            continue
            
        points_url = api_calls[0]
        print(f"[SUCCESS] Generated Points API URL: {points_url}")
        
        print("\nStep 2: Points API Execution")
        print("Fetching location data from National Weather Service...")
        success, points_response = execute_curl_command(points_url)
        
        if not success:
            print(f"[ERROR] Failed to fetch points data: {points_response}")
            continue
            
        print("[SUCCESS] Received points data")
        
        print("\nStep 3: Extracting Forecast URL")
        success, forecast_url = get_forecast_url_from_points_response(points_response)
        
        if not success:
            print(f"[ERROR] Failed to extract forecast URL: {forecast_url}")
            continue
            
        print(f"[SUCCESS] Forecast URL: {forecast_url[:60]}...")
        
        print("\nStep 4: Forecast API Execution")
        print("Fetching weather forecast data...")
        success, forecast_response = execute_curl_command(forecast_url)
        
        if not success:
            print(f"[ERROR] Failed to fetch forecast data: {forecast_response}")
            continue
            
        print(f"[SUCCESS] Received {len(forecast_response)} characters of forecast data")
        
        print("\nStep 5: AI Analysis Phase")
        success, summary = process_weather_response(forecast_response, location)
        
        if not success:
            print(f"[ERROR] Failed to process data: {summary}")
            continue
            
        print("\nStep 6: Weather Forecast")
        print("=" * 60)
        print(summary)
        print("=" * 60)
        
        print(f"\n[SUCCESS] Weather analysis complete for '{location}'!")
if __name__ == "__main__":
    run_weather_agent()