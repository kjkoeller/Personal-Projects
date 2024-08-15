import pandas as pd
import numpy as np
import random

# Constants
NUM_DAYS = 365  # Simulate over a year
MINUTES_PER_DAY = 24 * 60
TOTAL_INTERVALS = NUM_DAYS * MINUTES_PER_DAY

# Road segments characteristics
ROAD_SEGMENTS = {
    'Highway': {'base_volume': (100, 500), 'base_speed': (60, 120), 'base_occupancy': (10, 50)},
    'Urban': {'base_volume': (200, 1000), 'base_speed': (20, 50), 'base_occupancy': (30, 90)},
    'Residential': {'base_volume': (50, 300), 'base_speed': (10, 30), 'base_occupancy': (20, 70)},
}

# Vehicle types characteristics
VEHICLE_TYPES = {
    'Car': {'speed_factor': 1.0, 'volume_factor': 1.0},
    'Truck': {'speed_factor': 0.8, 'volume_factor': 0.2},
    'Bus': {'speed_factor': 0.6, 'volume_factor': 0.1},
}

# Simulate dynamic traffic waves and congestion propagation
def traffic_wave_effect(current_volume):
    wave_prob = min(current_volume / 1000, 0.3)  # Higher probability of waves with higher volumes
    if random.random() < wave_prob:
        wave_strength = np.random.uniform(0.8, 1.2)  # Waves can either increase or decrease speed
        return wave_strength
    return 1.0

# Time-of-day effects
def time_of_day_effect(hour):
    if 7 <= hour <= 9 or 16 <= hour <= 18:
        return np.random.uniform(1.5, 2.0)  # Rush hour
    elif 12 <= hour <= 14:
        return np.random.uniform(0.7, 0.9)  # Midday lull
    elif 0 <= hour <= 5:
        return np.random.uniform(0.3, 0.5)  # Nighttime
    else:
        return np.random.uniform(1.0, 1.2)  # Normal hours

# Day-of-week effects
def day_of_week_effect(day_of_week):
    if day_of_week >= 5:  # Saturday and Sunday
        return np.random.uniform(0.6, 0.9)  # Less traffic on weekends
    elif day_of_week == 0:  # Monday
        return np.random.uniform(1.2, 1.5)  # High traffic on Monday
    else:
        return np.random.uniform(1.0, 1.2)  # Other weekdays

# Seasonal effects
def seasonal_effect(month):
    if month in [12, 1, 2]:  # Winter
        return np.random.uniform(0.8, 1.0)  # Lower traffic in winter
    elif month in [6, 7, 8]:  # Summer
        return np.random.uniform(1.2, 1.5)  # Higher traffic in summer
    else:
        return 1.0  # No significant seasonal effect

# Special event effects (randomly applied)
def special_event_effect(day):
    if random.random() < 0.05:  # 5% chance of a special event
        event_intensity = np.random.uniform(1.3, 2.0)
        event_duration = np.random.randint(60, 180)  # Lasts 1 to 3 hours
        return event_intensity, event_duration
    else:
        return 1.0, 0

# Advanced weather effects
def weather_effect():
    weather_types = ['clear', 'rain', 'fog', 'snow', 'storm', 'localized snow']
    weather_pattern = np.random.choice(weather_types, p=[0.5, 0.2, 0.1, 0.1, 0.05, 0.05])
    if weather_pattern == 'rain':
        return 0.8, np.random.randint(30, 180)  # Rain lasts 30 minutes to 3 hours
    elif weather_pattern == 'fog':
        return 0.85, np.random.randint(30, 180)  # Fog lasts 30 minutes to 3 hours
    elif weather_pattern == 'snow':
        return 0.6, np.random.randint(60, 240)  # Snow lasts 1 to 4 hours
    elif weather_pattern == 'storm':
        return 0.5, np.random.randint(60, 240)  # Storm lasts 1 to 4 hours
    elif weather_pattern == 'localized snow':
        return 0.7, np.random.randint(60, 120)  # Snow in specific segments
    else:
        return 1.0, 0  # Clear weather

# Traffic incident effects
def traffic_incident_effect():
    if random.random() < 0.05:  # 5% chance of an incident
        severity = np.random.choice(['minor', 'major', 'severe'], p=[0.5, 0.3, 0.2])
        if severity == 'minor':
            return 0.85, np.random.randint(10, 30)  # Minor incident lasts 10-30 minutes
        elif severity == 'major':
            return 0.7, np.random.randint(30, 60)  # Major incident lasts 30-60 minutes
        else:
            return 0.5, np.random.randint(60, 120)  # Severe incident lasts 1-2 hours
    else:
        return 1.0, 0  # No incident

# Road layout effect (e.g., Traffic Light vs. Roundabout)
def road_layout_effect(layout_type, current_volume, current_speed):
    if layout_type == 'Traffic Light':
        if current_volume > 200:  # Simulate delay for high traffic volumes
            speed_reduction = np.random.uniform(0.7, 0.9)
        else:
            speed_reduction = np.random.uniform(0.9, 1.0)
        return speed_reduction * current_speed, np.random.uniform(0.8, 1.0) * current_volume
    elif layout_type == 'Roundabout':
        if current_volume > 200:  # Simulate efficiency gain for high traffic volumes
            speed_increase = np.random.uniform(1.0, 1.2)
        else:
            speed_increase = np.random.uniform(0.9, 1.1)
        return speed_increase * current_speed, np.random.uniform(0.9, 1.1) * current_volume
    else:
        return current_speed, current_volume  # Default to no effect

# Generate sophisticated traffic data
np.random.seed(42)
timestamps = pd.date_range(start='2022-01-01', periods=TOTAL_INTERVALS, freq='T')

data_records = []

# Initialize conditions
weather_multiplier, weather_duration = weather_effect()
incident_multiplier, incident_duration = traffic_incident_effect()
event_multiplier, event_duration = special_event_effect(0)

# Simulated road layouts
road_layouts = ['Traffic Light', 'Roundabout', 'Stop Sign']

for i in range(TOTAL_INTERVALS):
    timestamp = timestamps[i]
    hour = timestamp.hour
    day_of_week = timestamp.dayofweek
    month = timestamp.month

    # Apply daily effects
    tod_multiplier = time_of_day_effect(hour)
    dow_multiplier = day_of_week_effect(day_of_week)
    season_multiplier = seasonal_effect(month)

    # Update special event effects
    if event_duration > 0:
        event_duration -= 1
    else:
        event_multiplier, event_duration = special_event_effect(i // MINUTES_PER_DAY)

    # Update weather effects
    if weather_duration > 0:
        weather_duration -= 1
    else:
        weather_multiplier, weather_duration = weather_effect()

# Update traffic incident effects
if incident_duration > 0:
    incident_duration -= 1
else:
    incident_multiplier, incident_duration = traffic_incident_effect()

for segment_name, segment_params in ROAD_SEGMENTS.items():
    for vehicle_type, vehicle_params in VEHICLE_TYPES.items():
        # Randomly choose a road layout for this segment
        road_layout = random.choice(road_layouts)

        base_volume = np.random.randint(*segment_params['base_volume'])
        base_speed = np.random.uniform(*segment_params['base_speed'])
        base_occupancy = np.random.uniform(*segment_params['base_occupancy'])

        # Apply multipliers for various conditions
        adjusted_volume = base_volume * tod_multiplier * dow_multiplier * season_multiplier * event_multiplier * incident_multiplier * np.random.uniform(0.95, 1.05) * vehicle_params['volume_factor']
        adjusted_speed = base_speed * tod_multiplier * dow_multiplier * season_multiplier * weather_multiplier * incident_multiplier * np.random.uniform(0.95, 1.05) * vehicle_params['speed_factor']

        # Apply traffic wave effects
        wave_multiplier = traffic_wave_effect(adjusted_volume)
        adjusted_speed *= wave_multiplier

        # Apply road layout effects
        adjusted_speed, adjusted_volume = road_layout_effect(road_layout, adjusted_volume, adjusted_speed)

        adjusted_occupancy = base_occupancy * tod_multiplier * dow_multiplier * np.random.uniform(0.95, 1.05)

        data_records.append({
            'timestamp': timestamp,
            'road_segment': segment_name,
            'vehicle_type': vehicle_type,
            'traffic_volume': np.round(adjusted_volume, 2),
            'average_speed': np.round(adjusted_speed, 2),
            'occupancy': np.round(adjusted_occupancy, 2),
            'road_layout': road_layout  # Include the road layout in the data
        })

# Create DataFrame and save to CSV

data = pd.DataFrame(data_records)
data.to_csv('historical_traffic_data.csv', index=False)
print("Highly advanced synthetic traffic data generated and saved to 'data/historical_traffic_data.csv'")
