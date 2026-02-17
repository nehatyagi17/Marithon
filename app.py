import streamlit as st
import pandas as pd
import requests
import math
from datetime import datetime, timezone, timedelta
import traceback
import io
import plotly.graph_objects as go
import plotly.express as px

# ==============================================================================
# SECTION 0: APP SETUP AND HARDCODED DATA
# ==============================================================================
st.set_page_config(layout="wide", page_title="Marithon - Voyage Simulation")

# Hardcoded API key and route data for convenience
# For a production application, it's a security best practice to manage API keys
# as environment variables.
OPENWEATHER_API_KEY = "0f292dd4f556d0ca5c8c2bbf5ce7c950"

ROUTE_DATA_1 = """latitude,longitude
30.213982,32.557983
30.318359,32.382202
30.945814,32.306671
31.298117,32.387159
31.7,32.1
32.316071,30.408377
32.863395,28.905525
33.115811,28.212434
33.219565,27.927542
33.328,27.6298
33.748752,26.306431
34.011915,25.478721
34.187436,24.926664
34.8,23.0
35.126694,21.407365
35.845726,17.902084
36.086854,16.726588
36.4,15.2
36.907095,13.263819
37.209117,12.110644
37.212689,12.097004
37.215493,12.086301
37.283186,11.827836
37.454891,11.172235
37.5,11.0
37.489085,10.372293
37.420822,9.655883
37.2861,8.9197
37.2104,8.4552
37.0601,7.9906
36.7578,7.3197
36.4024,6.7118
36.1738,6.2995
35.7972,5.727
35.3409,5.0886
34.8587,4.5674
34.3312,4.0759
33.7486,3.697
33.0805,3.4682
32.3486,3.4617
31.5458,3.6766
30.6868,4.1106
29.7828,4.7214"""

ROUTE_DATA_2 = """latitude,longitude
30.213982,32.557983
29.7,32.6
27.9,33.75
27,34.5
23.6,20.75
38.9,16.3
41.2,15
12.7,43.3
12.40439,43.746586
12,13
12.577758,53.059021
12.2395,54.7085
11.4317,58.3951
11.083455,59.894005
10.866984,60.825733
10.5802,62.0601
10.031585,64.303249
9.934828,64.698862
9.862937,64.992809
9.6889,65.7044
"""

# Weather data cache to reduce API calls
weather_cache = {}

# ==============================================================================
# SECTION 1: SHIP PROFILE & CORE CALCULATIONS
# ==============================================================================
SHIP_PROFILES = {
"Handysize": {
"dwt": 25000, "type": "tanker", "ship_type_factor": 0.90, "windage_factor": 1.00,
"hull_factor": 1.00, "power_factor": 0.80, "min_speed": 6.0, "max_speed": 14.0, "default_STW": 13.0
},
"Panamax": {
"dwt": 70000, "type": "bulk_carrier", "ship_type_factor": 1.00, "windage_factor": 1.10,
"hull_factor": 1.00, "power_factor": 1.00, "min_speed": 8.0, "max_speed": 20.0, "default_STW": 14.0
},
"Suezmax": {
"dwt": 150000, "type": "tanker", "ship_type_factor": 1.10, "windage_factor": 0.90,
"hull_factor": 1.10, "power_factor": 1.20, "min_speed": 9.0, "max_speed": 22.0, "default_STW": 13.5
},
"VLCC": {
"dwt": 250000, "type": "tanker", "ship_type_factor": 1.30, "windage_factor": 0.80,
"hull_factor": 1.20, "power_factor": 1.40, "min_speed": 10.0, "max_speed": 17.0, "default_STW": 13.0
},
"ULCC": {
"dwt": 400000, "type": "tanker", "ship_type_factor": 1.50, "windage_factor": 0.70,
"hull_factor": 1.30, "power_factor": 1.50, "min_speed": 10.0, "max_speed": 16.0, "default_STW": 13.0
},
"ULCV": {
"dwt": 200000, "type": "container", "ship_type_factor": 1.40, "windage_factor": 1.50,
"hull_factor": 1.20, "power_factor": 1.60, "min_speed": 12.0, "max_speed": 25.0, "default_STW": 16.0
}
}

class ShipProfile:
    """A data class to hold the ship's physical characteristics."""
    def __init__(self, ship_type: str, dwt: int, loading_condition: str):
        self.type = ship_type
        self.dwt = dwt
        self.loading = loading_condition

def calculate_ship_coefficients(ship_profile: ShipProfile, ship_attrs: dict) -> dict:
    """Translates ship data into performance coefficients using SHIP_PROFILES."""
    base_ship_coeff = 0.020 if ship_profile.type == 'tanker' else 0.015
    base_swell_coeff = 0.15 if ship_profile.type == 'tanker' else 0.12
    if ship_profile.loading == 'ballast':
        base_ship_coeff *= 1.1
    size_factor = 1 - ((ship_profile.dwt - 150000) / 500000) * 0.1
    return {
        "ship_coeff": base_ship_coeff * size_factor * ship_attrs["windage_factor"],
        "swell_coeff": base_swell_coeff * size_factor * ship_attrs["hull_factor"]
    }

def calculate_sog(v_stw: float, weather: dict, ship_coeffs: dict, heading: float) -> float:
    """Calculates the final Speed Over Ground (SOG)."""
    def safe(val):
        return val if isinstance(val, (int, float)) else 0.0

    wind_speed = safe(weather.get("wind_speed_10m"))
    wind_direction = safe(weather.get("wind_direction_10m"))
    wind_wave_height = safe(weather.get("wind_wave_height"))
    swell_wave_height = safe(weather.get("swell_wave_height"))
    swell_wave_direction = safe(weather.get("swell_wave_direction"))
    ocean_current_velocity = safe(weather.get("ocean_current_velocity"))
    ocean_current_direction = safe(weather.get("ocean_current_direction"))
    visibility = safe(weather.get("visibility", 10000))

    wind_knots = wind_speed * 1.944
    rel_wind_angle = abs((heading - wind_direction + 180) % 360 - 180)
    dv_wind_wave = 0
    if rel_wind_angle < 90:
        dv_wind_wave = (
            ship_coeffs["ship_coeff"]
            * math.pow(wind_knots, 1.2)
            * math.pow(wind_wave_height, 0.8)
            * math.cos(math.radians(rel_wind_angle))
        )

    rel_swell_angle = abs((heading - swell_wave_direction + 180) % 360 - 180)
    dv_swell = 0
    if rel_swell_angle < 90:
        dv_swell = (
            ship_coeffs["swell_coeff"]
            * math.pow(swell_wave_height, 1.5)
            * math.cos(math.radians(rel_swell_angle))
        )

    dv_visibility = 0
    if visibility <= 1000:
        dv_visibility = v_stw * 0.25
    elif visibility <= 5000:
        dv_visibility = v_stw * 0.10

    speed_after_loss = v_stw - dv_wind_wave - dv_swell - dv_visibility

    current_knots = ocean_current_velocity * 1.944
    rel_current_angle = heading - ocean_current_direction
    v_current_assist = current_knots * math.cos(math.radians(rel_current_angle))

    return max(1.0, speed_after_loss + v_current_assist)

def calculate_fuel_and_cost(log: list, ship_attrs: dict) -> dict:
    """Calculates total fuel and cost based on voyage log."""
    FUEL_K_COEFFICIENT = 0.0374
    FUEL_BETA_EXPONENT = 2.738
    FUEL_PRICE_PER_TON_USD = 650.0
    total_fuel_tons = 0
    def safe(val):
        return val if isinstance(val, (int, float)) else 0.0

    for entry in log:
        sog_knots = entry["sog_knots"]
        weather = entry["weather"]
        time_hours = entry["time_hours"]
        base_daily_burn = FUEL_K_COEFFICIENT * math.pow(sog_knots, FUEL_BETA_EXPONENT) * ship_attrs["power_factor"]
        wind_speed = safe(weather.get("wind_speed_10m", 0))
        wind_wave_height = safe(weather.get("wind_wave_height", 0))
        weather_factor = 1 + (wind_speed / 50) + (wind_wave_height / 10)
        actual_daily_burn = base_daily_burn * weather_factor
        fuel_in_interval = (actual_daily_burn / 24) * time_hours
        total_fuel_tons += fuel_in_interval
    return {"total_fuel_tons": total_fuel_tons, "total_cost_usd": total_fuel_tons * FUEL_PRICE_PER_TON_USD}

# ==============================================================================
# SECTION 2: GEOSPATIAL FUNCTIONS
# ==============================================================================

def haversine_distance(p1, p2):
    """Calculates distance between two points in nautical miles."""
    R = 3440.065
    lat1, lon1, lat2, lon2 = map(math.radians, [p1[0], p1[1], p2[0], p2[1]])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def calculate_bearing(p1, p2):
    """Calculates bearing from point1 to point2."""
    lat1, lon1, lat2, lon2 = map(math.radians, [p1[0], p1[1], p2[0], p2[1]])
    dLon = lon2 - lon1
    x = math.sin(dLon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360

def calculate_destination(point, bearing, distance_nm):
    """Calculates a destination point."""
    R = 3440.065
    lat1, lon1 = math.radians(point[0]), math.radians(point[1])
    bearing_rad = math.radians(bearing)
    lat2 = math.asin(math.sin(lat1) * math.cos(distance_nm / R) +
    math.cos(lat1) * math.sin(distance_nm / R) * math.cos(bearing_rad))
    lon2 = lon1 + math.atan2(math.sin(bearing_rad) * math.sin(distance_nm / R) * math.cos(lat1),
    math.cos(distance_nm / R) - math.sin(lat1) * math.sin(lat2))
    return (math.degrees(lat2), math.degrees(lon2))

# ==============================================================================
# SECTION 3: DYNAMIC WEATHER FETCHERS & ALERTS
# ==============================================================================

def fetch_dynamic_weather(lat, lon, api_key, target_time=None):
    """Fetches dynamic marine + wind + current data and detects storm/cyclone alerts."""
    def safe(val):
        return val if isinstance(val, (int, float)) else 0.0

    if target_time is None:
        target_time = datetime.now(timezone.utc)

    cache_key = (round(lat, 1), round(lon, 1), target_time.replace(minute=0, second=0, microsecond=0).replace(hour=target_time.hour // 3 * 3).isoformat())
    if cache_key in weather_cache:
        return weather_cache[cache_key]

    weather_data = {}
    alerts = []

    # Fetch marine and current data
    url_marine = (
        f"https://marine-api.open-meteo.com/v1/marine?"
        f"latitude={lat}&longitude={lon}&hourly=wind_wave_height,swell_wave_height,swell_wave_direction,ocean_current_velocity,ocean_current_direction&timezone=auto"
    )
    try:
        res = requests.get(url_marine, timeout=10)
        res.raise_for_status()
        data = res.json()
        times = [datetime.fromisoformat(t).replace(tzinfo=timezone.utc) for t in data.get("hourly", {}).get("time", [])]
        if not times:
            raise ValueError("No marine forecast data")
        closest_index = min(range(len(times)), key=lambda i: abs(times[i] - target_time))
        hourly_data = data["hourly"]
        weather_data.update({
        "wind_wave_height": safe(hourly_data.get("wind_wave_height", [None])[closest_index]),
        "swell_wave_height": safe(hourly_data.get("swell_wave_height", [None])[closest_index]),
        "swell_wave_direction": safe(hourly_data.get("swell_wave_direction", [None])[closest_index]),
        "ocean_current_velocity": safe(hourly_data.get("ocean_current_velocity", [None])[closest_index]),
        "ocean_current_direction": safe(hourly_data.get("ocean_current_direction", [None])[closest_index]),
        })
    except Exception as e:
        st.warning(f"Warning: Error fetching marine data, using defaults: {e}")
        weather_data.update({
        "wind_wave_height": 1.0, "swell_wave_height": 1.5, "swell_wave_direction": 90.0,
        "ocean_current_velocity": 0.5, "ocean_current_direction": 90.0
        })

    # Fetch wind and visibility
    url_wind = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={api_key}&units=metric"
    try:
        res = requests.get(url_wind, timeout=10)
        res.raise_for_status()
        data = res.json()
        forecasts = data.get("list", [])
        if not forecasts:
            raise ValueError("No wind forecast data")
        forecast_times = [datetime.fromtimestamp(f['dt'], tz=timezone.utc) for f in forecasts]
        closest_idx = min(range(len(forecast_times)), key=lambda i: abs(forecast_times[i] - target_time))
        forecast = forecasts[closest_idx]
        weather_data.update({
        "wind_speed_10m": safe(forecast.get('wind', {}).get('speed')),
        "wind_direction_10m": safe(forecast.get('wind', {}).get('deg')),
        "visibility": safe(forecast.get('visibility', 10000))
        })
    except Exception as e:
        st.warning(f"Warning: Error fetching wind data, using defaults: {e}")
        weather_data.update({
        "wind_speed_10m": 6.0, "wind_direction_10m": 180.0, "visibility": 10000
        })

    # Check for alerts with lowered thresholds
    if weather_data.get("wind_speed_10m", 0) > 5:
        alerts.append("⚠ Gale Warning - High Wind")
    if weather_data.get("wind_wave_height", 0) > 1 or weather_data.get("swell_wave_height", 0) > 1.5:
        alerts.append("🌊 High Seas Alert - Large Waves")
    weather_data["alerts"] = alerts

    weather_cache[cache_key] = weather_data
    return weather_data

def find_alternative_waypoint(current_pos, next_waypoint, api_key, target_time):
    """Finds a safer waypoint to avoid weather alerts."""
    def safe(val):
        return val if isinstance(val, (int, float)) else 0.0

    candidates = []
    offsets = [-1.0, 1.0]
    for dlat in offsets:
        for dlon in offsets:
            new_lat = current_pos[0] + dlat
            new_lon = current_pos[1] + dlon
            if -90 <= new_lat <= 90 and -180 <= new_lon <= 180:
                dist_to_next = haversine_distance((new_lat, new_lon), next_waypoint)
                if dist_to_next < haversine_distance(current_pos, next_waypoint) * 1.5:
                    candidates.append((new_lat, new_lon))

    if not candidates:
        return current_pos

    best_candidate = current_pos
    best_score = float('inf')
    for candidate in candidates:
        weather = fetch_dynamic_weather(candidate[0], candidate[1], api_key, target_time)
        if not weather["alerts"]:
            score = safe(weather.get("wind_speed_10m", 0)) + safe(weather.get("wind_wave_height", 0)) + safe(weather.get("swell_wave_height", 0))
            if score < best_score:
                best_score = score
                best_candidate = candidate
    return best_candidate

# ==============================================================================
# SECTION 4: SIMULATION ENGINE
# ==============================================================================

def run_simulation(waypoints: list, ship_profile: object, ship_attrs: dict, commanded_speed_knots: float, api_key: str, start_time: datetime, optimize_route: bool = False):
    """Runs the voyage simulation, with optional route optimization."""
    ship_coeffs = calculate_ship_coefficients(ship_profile, ship_attrs)
    CHECK_DISTANCE_NM = 50.0
    MAX_WAYPOINTS = 43 if len(waypoints) >= 43 else len(waypoints)

    log = []
    total_time_hours = 0.0
    total_dist_nm = 0.0
    current_pos = tuple(waypoints[0])
    time_nominal_h = 0.0
    current_waypoint_index = 0
    dynamic_waypoints = [current_pos]

    status_placeholder = st.empty()
    progress_bar = st.progress(0)

    for current_waypoint_index in range(len(waypoints) - 1):
        segment_end = tuple(waypoints[current_waypoint_index + 1])
        segment_dist_nm = haversine_distance(current_pos, segment_end)
        segment_heading = calculate_bearing(current_pos, segment_end)
        dist_covered_on_segment_nm = 0.0

        time_nominal_h += segment_dist_nm / commanded_speed_knots if commanded_speed_knots > 0 else 0

        while dist_covered_on_segment_nm < segment_dist_nm:
            chunk_dist_nm = min(CHECK_DISTANCE_NM, segment_dist_nm - dist_covered_on_segment_nm)
            if chunk_dist_nm < 1.0:
                break
            
            target_time = start_time + timedelta(hours=total_time_hours)
            weather = fetch_dynamic_weather(current_pos[0], current_pos[1], api_key, target_time)

            if optimize_route and weather.get("alerts"):
                status_placeholder.warning(f"Weather alert detected. Attempting to reroute...")
                new_pos = find_alternative_waypoint(current_pos, segment_end, api_key, target_time)
                if new_pos != current_pos:
                    current_pos = new_pos
                    dynamic_waypoints.append(current_pos)
                    segment_dist_nm = haversine_distance(current_pos, segment_end)
                    segment_heading = calculate_bearing(current_pos, segment_end)
                    dist_covered_on_segment_nm = 0.0
                    weather = fetch_dynamic_weather(current_pos[0], current_pos[1], api_key, target_time)
                else:
                    status_placeholder.info("No safer route found, proceeding with caution.")

            sog_knots = calculate_sog(commanded_speed_knots, weather, ship_coeffs, segment_heading)
            sog_knots = min(max(sog_knots, ship_attrs["min_speed"]), ship_attrs["max_speed"])
            time_for_chunk_hours = chunk_dist_nm / sog_knots if sog_knots > 0 else float('inf')

            log.append({
                "time_hours": time_for_chunk_hours,
                "sog_knots": sog_knots,
                "pos": current_pos,
                "weather": weather
            })

            total_time_hours += time_for_chunk_hours
            total_dist_nm += chunk_dist_nm
            dist_covered_on_segment_nm += chunk_dist_nm
            current_pos = calculate_destination(current_pos, segment_heading, chunk_dist_nm)
            
            # Update progress bar
            progress = (current_waypoint_index + (dist_covered_on_segment_nm / segment_dist_nm)) / (len(waypoints) - 1)
            progress_bar.progress(min(progress, 1.0))

        if haversine_distance(current_pos, segment_end) > 0.1:
            current_pos = segment_end
            dynamic_waypoints.append(current_pos)
            
    progress_bar.empty()
    status_placeholder.empty()

    eta_actual = start_time + timedelta(hours=total_time_hours)
    eta_nominal = start_time + timedelta(hours=time_nominal_h)

    suggested_stw = None
    if total_time_hours > time_nominal_h and time_nominal_h > 0:
        required_avg_speed = total_dist_nm / time_nominal_h
        suggested_stw = min(round(required_avg_speed * 1.1, 2), ship_attrs["max_speed"])

    cost_results = calculate_fuel_and_cost(log, ship_attrs)
    total_distance_net_speed = 0.0
    distance_sog_sum = 0.0
    for entry in log:
        sog = entry["sog_knots"]
        distance_nm_segment = entry["time_hours"] * sog
        total_distance_net_speed += distance_nm_segment
        distance_sog_sum += sog * distance_nm_segment
    net_speed = distance_sog_sum / total_distance_net_speed if total_distance_net_speed > 0 else 0.0

    return {
        "log": log,
        "total_dist_nm": total_dist_nm,
        "total_time_actual_h": total_time_hours,
        "total_time_nominal_h": time_nominal_h,
        "eta_actual": eta_actual.isoformat(),
        "eta_nominal": eta_nominal.isoformat(),
        "total_delay_h": total_time_hours - time_nominal_h,
        "suggested_STW": suggested_stw,
        "dynamic_waypoints": dynamic_waypoints,
        "cost_results": cost_results,
        "net_speed": net_speed
    }

# ==============================================================================
# SECTION 5: STREAMLIT APP UI
# ==============================================================================
st.title("🚢 Marithon - Maritime Voyage Simulation Tool")
st.markdown("A tool for simulating and optimizing ship voyages based on dynamic weather conditions.")
st.divider()

with st.sidebar:
    st.header("Voyage Parameters")
    st.warning("The OpenWeatherMap API key is hardcoded. For a production environment, this should be an environment variable.")
    
    route_source = st.selectbox(
        "Select Route Source:",
        ("Pre-defined Routes", "Upload Custom CSV")
    )
    
    waypoints = None
    if route_source == "Pre-defined Routes":
        route_choice = st.selectbox(
            "Choose a pre-defined route:",
            ("Route 1", "Route 2")
        )
        if route_choice == "Route 1":
            waypoints = pd.read_csv(io.StringIO(ROUTE_DATA_1))
        else:
            waypoints = pd.read_csv(io.StringIO(ROUTE_DATA_2))
    else:
        uploaded_file = st.file_uploader("Upload your own CSV route file", type=["csv"])
        if uploaded_file is not None:
            waypoints = pd.read_csv(uploaded_file)
    
    ship_name = st.selectbox(
        "Select Ship Type:",
        list(SHIP_PROFILES.keys())
    )
    
    loading_condition = st.selectbox(
        "Loading Condition:",
        ("loaded", "ballast")
    )
    
    ship_attrs = SHIP_PROFILES[ship_name]
    commanded_speed = st.number_input(
        "Commanded Speed (knots):",
        min_value=ship_attrs["min_speed"],
        max_value=ship_attrs["max_speed"],
        value=ship_attrs["default_STW"],
        step=0.1
    )
    
    start_time = st.date_input("Simulation Start Date", value=datetime.now(timezone.utc))
    start_time_time = st.time_input("Simulation Start Time", value=datetime.now(timezone.utc))
    start_datetime = datetime.combine(start_time, start_time_time, tzinfo=timezone.utc)

    st.markdown("---")
    sim_button = st.button("Run Simulation")
    opt_button = st.button("Run Route Optimization")

# ==============================================================================
# SECTION 6: RESULTS DISPLAY LOGIC
# ==============================================================================
tab1, tab2 = st.tabs(["Voyage Simulation", "Route Optimization"])

with tab1:
    st.header("Weather Effects on a Route")
    if sim_button:
        if waypoints is None:
            st.error("Please select or upload a valid route file.")
        else:
            try:
                my_ship = ShipProfile(ship_attrs["type"], ship_attrs["dwt"], loading_condition)
                waypoints_list = waypoints[['latitude', 'longitude']].values.tolist()
                
                with st.spinner("Running voyage simulation..."):
                    sim_results = run_simulation(waypoints_list, my_ship, ship_attrs, commanded_speed, OPENWEATHER_API_KEY, start_datetime, optimize_route=False)
                
                st.success("Simulation Complete!")
                st.subheader("Simulation Results")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Distance", f"{sim_results['total_dist_nm']:.2f} NM")
                col2.metric("Nominal Time", f"{sim_results['total_time_nominal_h']/24:.2f} days")
                col3.metric("Actual Time", f"{sim_results['total_time_actual_h']/24:.2f} days")

                col4, col5 = st.columns(2)
                col4.metric("Fuel Consumed", f"{sim_results['cost_results']['total_fuel_tons']:.2f} tons")
                col5.metric("Voyage Cost", f"${sim_results['cost_results']['total_cost_usd']:,.2f}")

                st.metric("Net Speed", f"{sim_results['net_speed']:.2f} knots")
                st.metric("ETA Nominal", sim_results['eta_nominal'])
                st.metric("ETA Actual", sim_results['eta_actual'])

                st.subheader("Route Map")
                
                # Convert waypoints to a DataFrame for Plotly
                route_df = pd.DataFrame(sim_results['dynamic_waypoints'], columns=['latitude', 'longitude'])
                
                fig = go.Figure(go.Scattermapbox(
                    lat=route_df['latitude'],
                    lon=route_df['longitude'],
                    mode='lines+markers',
                    marker={'size': 8, 'color': 'blue'},
                    line={'width': 4, 'color': 'blue'},
                    name='Voyage Route'
                ))
                
                fig.update_layout(
                    mapbox_style="open-street-map",
                    mapbox_zoom=3,
                    mapbox_center={"lat": route_df['latitude'].mean(), "lon": route_df['longitude'].mean()}
                )

                st.plotly_chart(fig, use_container_width=True)

                with st.expander("Show Voyage Log"):
                    st.json(sim_results['log'])
                
                alerts = [{"time_hours": entry['time_hours'], "alerts": entry['weather']['alerts'], "pos": entry['pos']} 
                          for entry in sim_results['log'] if entry['weather'].get('alerts')]
                
                if alerts:
                    st.subheader("Weather Alerts Encountered")
                    for alert in alerts:
                        st.warning(f"Alert at ({alert['pos'][0]:.2f}, {alert['pos'][1]:.2f}) after {alert['time_hours']:.2f} hours: {', '.join(alert['alerts'])}")
                else:
                    st.info("No weather alerts encountered on this route.")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.code(traceback.format_exc())

with tab2:
    st.header("Route Optimization Analysis")
    if opt_button:
        if waypoints is None:
            st.error("Please select or upload a valid route file.")
        else:
            try:
                my_ship = ShipProfile(ship_attrs["type"], ship_attrs["dwt"], loading_condition)
                waypoints_list = waypoints[['latitude', 'longitude']].values.tolist()
                
                # Run both simulations with a spinner
                with st.spinner("Running route optimization and comparison..."):
                    original_sim = run_simulation(waypoints_list[:], my_ship, ship_attrs, commanded_speed, OPENWEATHER_API_KEY, start_datetime, optimize_route=False)
                    optimized_sim = run_simulation(waypoints_list[:], my_ship, ship_attrs, commanded_speed, OPENWEATHER_API_KEY, start_datetime, optimize_route=True)

                st.success("Optimization Analysis Complete!")

                # Cost Efficiency Analysis
                original_cost = original_sim['cost_results']['total_cost_usd']
                optimized_cost = optimized_sim['cost_results']['total_cost_usd']
                original_time = original_sim['total_time_actual_h'] / 24
                optimized_time = optimized_sim['total_time_actual_h'] / 24

                cost_difference = optimized_cost - original_cost
                cost_percentage_change = (cost_difference / original_cost) * 100 if original_cost > 0 else 0
                time_difference = optimized_time - original_time
                time_percentage_change = (time_difference / original_time) * 100 if original_time > 0 else 0
                
                st.subheader("Comparison of Routes")
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Route")
                    st.metric("Total Distance", f"{original_sim['total_dist_nm']:.2f} NM")
                    st.metric("Total Time", f"{original_time:.2f} days")
                    st.metric("Total Cost", f"${original_cost:,.2f}")
                
                with col2:
                    st.subheader("Optimized Route")
                    st.metric("Total Distance", f"{optimized_sim['total_dist_nm']:.2f} NM")
                    st.metric("Total Time", f"{optimized_time:.2f} days")
                    st.metric("Total Cost", f"${optimized_cost:,.2f}")

                st.subheader("Cost & Time Efficiency Analysis")
                col1, col2 = st.columns(2)
                col1.metric("Cost Change", f"${cost_difference:,.2f}", f"{cost_percentage_change:.2f}%")
                col2.metric("Time Change", f"{time_difference:.2f} days", f"{time_percentage_change:.2f}%")
                
                st.subheader("Route Maps")
                
                # Convert waypoints to DataFrames for Plotly
                original_df = pd.DataFrame(original_sim['dynamic_waypoints'], columns=['latitude', 'longitude'])
                optimized_df = pd.DataFrame(optimized_sim['dynamic_waypoints'], columns=['latitude', 'longitude'])
                
                fig = go.Figure()
                
                # Add Original Route trace
                fig.add_trace(go.Scattermapbox(
                    lat=original_df['latitude'],
                    lon=original_df['longitude'],
                    mode='lines',
                    line=dict(width=4, color='rgba(255, 0, 0, 0.5)'),
                    name='Original Route'
                ))
                
                # Add Optimized Route trace
                fig.add_trace(go.Scattermapbox(
                    lat=optimized_df['latitude'],
                    lon=optimized_df['longitude'],
                    mode='lines',
                    line=dict(width=4, color='rgba(0, 255, 0, 0.5)'),
                    name='Optimized Route'
                ))

                # Add start and end points for both routes
                fig.add_trace(go.Scattermapbox(
                    lat=[original_df.iloc[0]['latitude'], optimized_df.iloc[0]['latitude']],
                    lon=[original_df.iloc[0]['longitude'], optimized_df.iloc[0]['longitude']],
                    mode='markers',
                    marker=dict(size=10, color='red'),
                    name='Start Points'
                ))
                fig.add_trace(go.Scattermapbox(
                    lat=[original_df.iloc[-1]['latitude'], optimized_df.iloc[-1]['latitude']],
                    lon=[original_df.iloc[-1]['longitude'], optimized_df.iloc[-1]['longitude']],
                    mode='markers',
                    marker=dict(size=10, color='green'),
                    name='End Points'
                ))
                
                fig.update_layout(
                    mapbox_style="open-street-map",
                    mapbox_zoom=3,
                    mapbox_center={"lat": original_df['latitude'].mean(), "lon": original_df['longitude'].mean()}
                )

                st.plotly_chart(fig, use_container_width=True)

                # Display alerts
                alerts_orig = [{"pos": entry['pos'], "time_hours": entry['time_hours'], "alerts": entry['weather']['alerts']}
                              for entry in original_sim["log"] if entry["weather"].get("alerts")]
                alerts_opt = [{"pos": entry['pos'], "time_hours": entry['time_hours'], "alerts": entry['weather']['alerts']}
                              for entry in optimized_sim["log"] if entry["weather"].get("alerts")]
                
                st.subheader("Weather Alerts")
                st.write("**Original Route Alerts:**")
                if alerts_orig:
                    for alert in alerts_orig:
                        st.warning(f"Alert at ({alert['pos'][0]:.2f}, {alert['pos'][1]:.2f}) after {alert['time_hours']:.2f} hours: {', '.join(alert['alerts'])}")
                else:
                    st.info("No weather alerts encountered on this route.")
                
                st.write("**Optimized Route Alerts:**")
                if alerts_opt:
                    for alert in alerts_opt:
                        st.warning(f"Alert at ({alert['pos'][0]:.2f}, {alert['pos'][1]:.2f}) after {alert['time_hours']:.2f} hours: {', '.join(alert['alerts'])}")
                else:
                    st.info("No weather alerts encountered on this optimized route.")
            
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.code(traceback.format_exc())
