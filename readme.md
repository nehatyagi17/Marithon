# TechX Voyage Simulator 🚢

[Live Demo](https://techx-voyage-simulator.streamlit.app/)

**Tagline:** Simulate and optimize maritime voyages with real-time weather effects, fuel costs, and route optimization.

---

## Overview
TechX Voyage Simulator is a Streamlit-based tool designed to simulate maritime voyages based on dynamic weather conditions. Users can explore how oceanic conditions such as wind, waves, and currents affect ship speed, fuel consumption, and overall voyage cost. The simulator also offers dynamic route optimization to minimize both time and cost.

---

## Features
- **Pre-defined & Custom Routes:** Use built-in routes or upload your own CSV of waypoints.  
- **Ship Support:** 6 ship types supported: Handysize, Panamax, Suezmax, VLCC, ULCC, ULCV. Each ship can be simulated under *loaded* or *ballast* conditions.  
- **Dynamic Weather Impact:** Calculates voyage effects based on wind, waves, swell, ocean currents, and visibility.  
- **Route Optimization:** Suggests alternative waypoints to avoid bad weather, reducing overall time and cost.  
- **Fuel & Cost Estimation:** Computes fuel consumption and voyage cost for each simulation.  
- **Interactive Map Visualization:** Shows the voyage route, including optimized paths, using Plotly.  
- **API Integration:** Uses multiple weather and marine data APIs to power the simulation:
  - **OpenWeather API:** Provides real-time weather data and forecasts.  
  - **Storm Glass API:** Provides marine-specific data such as waves, tides, wind, and ocean currents.
  - **Open-Meteo API:** Provides detailed weather forecasts including wind, visibility, and ocean conditions.

---

## Installation (Local)
1. Clone the repository:
    ```bash
    git clone <repo_url>
    cd techx-voyage-simulator
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the app:
    ```bash
    streamlit run app.py
    ```

---

## Dependencies
- Python 3.10+ (or compatible)
- [Streamlit](https://streamlit.io/)
- [Pandas](https://pandas.pydata.org/)
- [Requests](https://docs.python-requests.org/en/latest/)
- [Plotly](https://plotly.com/python/)

---

## Usage
1. Open the app (locally or via the [live demo](https://techx-voyage-simulator.streamlit.app/)).  
2. Select a pre-defined route or upload a CSV of waypoints.  
3. Choose a ship type and loading condition (loaded/ballast).  
4. Set the commanded speed and start date/time.  
5. Run the **Simulation** or **Route Optimization**.  
6. Explore the results: voyage duration, fuel cost, dynamic route map, and any weather alerts.
