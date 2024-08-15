import pandas as pd
import numpy as np

# Parameters
num_days = 30
num_intervals_per_day = 24 * 60  # 1-minute intervals
total_intervals = num_days * num_intervals_per_day

# Generate timestamps
timestamps = pd.date_range(start='2022-01-01', periods=total_intervals, freq='T')

# Time-of-day effects (morning and evening rush hours, midday lull)
def time_of_day_effect(hour):
    if 7 <= hour <= 9 or 16 <= hour <= 18:
        return np.random.uniform(1.4, 1.8)  # Rush hour multiplier
    elif 12 <= hour <= 13:
        return np.random.uniform(0.7, 0.9)  # Midday lull
    elif 0 <= hour <= 5:
        return np.random.uniform(0.3, 0.6)  # Nighttime lower traffic
    else:
        return np.random.uniform(1.0, 1.2)  # Normal traffic

# Day-of-week effects (weekday vs. weekend, specific days)
def day_of_week_effect(day_of_week):
    if day_of_week >= 5:  # Saturday and Sunday
        return np.random.uniform(0.6, 0.8)  # Less traffic on weekends
    elif day_of_week == 0:  # Monday (higher traffic)
        return np.random.uniform(1.1, 1.3)
    elif day_of_week == 4:  # Friday (rush to leave the city)
        return np.random.uniform(1.2, 1.4)
    else:
        return np.random.uniform(0.9, 1.1)  # Normal traffic on other weekdays

# Seasonal effects (e.g., winter vs. summer)
def seasonal_effect(month):
    if month in [12, 1, 2]:  # Winter
        return np.random.uniform(0.8, 1.0)  # Slightly lower traffic in winter
    elif month in [6, 7, 8]:  # Summer
        return np.random.uniform(1.0, 1.2)  # Higher traffic in summer
    else:
        return 1.0  # No seasonal effect in other months

# Weather conditions effect with complex patterns
def weather_effect():
    weather_types = ['clear', 'rain', 'fog', 'snow', 'storm']
    weather_pattern = np.random.choice(weather_types, p=[0.5, 0.2, 0.1, 0.1, 0.1])
    duration = np.random.choice([30, 60, 120], p=[0.5, 0.3, 0.2])  # Duration of the weather event in minutes
    if weather_pattern == 'rain':
        return 0.8, duration  # Moderate reduction in traffic speed
    elif weather_pattern == 'fog':
        return 0.85, duration  # Slight reduction in traffic speed
    elif weather_pattern == 'snow':
        return 0.6, duration  # Significant reduction in traffic speed
    elif weather_pattern == 'storm':
        return 0.5, duration  # Severe reduction in traffic speed
    else:
        return 1.0, duration  # No effect in clear weather

# Traffic incidents (e.g., accidents)
def traffic_incident():
    incident_prob = np.random.rand()
    if incident_prob < 0.05:  # 5% chance of an incident
        severity = np.random.choice(['minor', 'major', 'severe'], p=[0.5, 0.3, 0.2])
        if severity == 'minor':
            return 0.85, np.random.randint(10, 30)  # Minor incident lasts 10-30 minutes
        elif severity == 'major':
            return 0.7, np.random.randint(30, 60)  # Major incident lasts 30-60 minutes
        else:
            return 0.5, np.random.randint(60, 120)  # Severe incident lasts 1-2 hours
    else:
        return 1.0, 0  # No incident

# Generate fake traffic data with more complexity
np.random.seed(42)
traffic_volume = []
average_speed = []
occupancy = []
road_segment = []  # If simulating multiple road segments

weather_mult, weather_duration = weather_effect()
incident_mult, incident_duration = traffic_incident()

for i in range(total_intervals):
    timestamp = timestamps[i]
    hour = timestamp.hour
    day_of_week = timestamp.dayofweek
    month = timestamp.month

    tod_effect = time_of_day_effect(hour)
    dow_effect = day_of_week_effect(day_of_week)
    season_mult = seasonal_effect(month)

    if weather_duration > 0:
        weather_duration -= 1
    else:
        weather_mult, weather_duration = weather_effect()

    if incident_duration > 0:
        incident_duration -= 1
    else:
        incident_mult, incident_duration = traffic_incident()

    base_volume = np.random.randint(50, 300)
    base_speed = np.random.uniform(30, 80)
    base_occupancy = np.random.uniform(20, 80)

    # Adjust based on the effects
    adjusted_volume = base_volume * tod_effect * dow_effect * season_mult * incident_mult * np.random.uniform(0.95, 1.05)
    adjusted_speed = base_speed * tod_effect * dow_effect * season_mult * weather_mult * incident_mult * np.random.uniform(0.95, 1.05)
    adjusted_occupancy = base_occupancy * tod_effect * dow_effect * np.random.uniform(0.95, 1.05)

    traffic_volume.append(adjusted_volume)
    average_speed.append(adjusted_speed)
    occupancy.append(adjusted_occupancy)
    road_segment.append(np.random.choice(['Segment 1', 'Segment 2', 'Segment 3']))  # Simulate different road segments

# Create DataFrame
data = pd.DataFrame({
    'timestamp': timestamps,
    'traffic_volume': np.round(traffic_volume, 2),
    'average_speed': np.round(average_speed, 2),
    'occupancy': np.round(occupancy, 2),
    'road_segment': road_segment
})

# Save to CSV
data.to_csv('data/historical_traffic_data.csv', index=False)
print("Advanced synthetic traffic data generated and saved to 'data/historical_traffic_data.csv'")