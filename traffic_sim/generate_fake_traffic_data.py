import pandas as pd
import numpy as np

# Parameters
num_days = 30
num_intervals_per_day = 24 * 60  # 1-minute intervals
total_intervals = num_days * num_intervals_per_day

# Generate timestamps
timestamps = pd.date_range(start='2022-01-01', periods=total_intervals, freq='T')

# Time-of-day effects (morning and evening rush hours)
def time_of_day_effect(hour):
    if 7 <= hour <= 9 or 16 <= hour <= 18:
        return np.random.uniform(1.3, 1.7)  # Rush hour multiplier with some variability
    elif 0 <= hour <= 5:
        return np.random.uniform(0.4, 0.6)  # Nighttime lower traffic with some variability
    else:
        return np.random.uniform(0.9, 1.1)  # Normal traffic with slight variability

# Day-of-week effects (lower traffic on weekends)
def day_of_week_effect(day_of_week):
    if day_of_week >= 5:  # Saturday and Sunday
        return np.random.uniform(0.6, 0.8)  # Less traffic on weekends with variability
    else:
        return np.random.uniform(0.9, 1.1)  # Normal traffic on weekdays with slight variability

# Holiday effects
holidays = pd.to_datetime(['2022-01-01', '2022-01-15', '2022-02-12'])  # Example holidays
def holiday_effect(date):
    if pd.to_datetime(date.date()) in holidays:
        return np.random.uniform(0.3, 0.7)  # Significantly lower traffic on holidays
    else:
        return 1.0  # No effect if not a holiday

# Weather conditions effect with dynamic patterns
def weather_effect():
    weather_types = ['clear', 'rain', 'fog', 'snow']
    # Simulate weather conditions lasting for several hours
    weather_durations = np.random.choice([60, 120, 180, 240], p=[0.3, 0.4, 0.2, 0.1])
    weather_pattern = np.random.choice(weather_types, p=[0.6, 0.2, 0.1, 0.1])  # More likely to be clear
    if weather_pattern == 'rain':
        return 0.8, weather_durations  # Reduce speed in rain
    elif weather_pattern == 'fog':
        return 0.9, weather_durations  # Slightly reduce speed in fog
    elif weather_pattern == 'snow':
        return 0.6, weather_durations  # Significantly reduce speed in snow
    else:
        return 1.0, weather_durations  # No effect in clear weather

# Random traffic events (e.g., accidents)
def random_traffic_event():
    event_prob = np.random.rand()
    if event_prob < 0.05:  # 5% chance of a traffic event
        event_duration = np.random.choice([15, 30, 45], p=[0.4, 0.4, 0.2])  # Lasts 15, 30, or 45 minutes
        return 0.7, event_duration  # Reduce traffic volume and speed due to the event
    else:
        return 1.0, 0  # No event

# Generate fake traffic data with more complexity
np.random.seed(42)
traffic_volume = []
average_speed = []
occupancy = []

weather_mult, weather_duration = weather_effect()
event_mult, event_duration = random_traffic_event()

for i in range(total_intervals):
    timestamp = timestamps[i]
    hour = timestamp.hour
    day_of_week = timestamp.dayofweek

    tod_effect = time_of_day_effect(hour)
    dow_effect = day_of_week_effect(day_of_week)
    holiday_mult = holiday_effect(timestamp)

    if weather_duration > 0:
        weather_duration -= 1
    else:
        weather_mult, weather_duration = weather_effect()

    if event_duration > 0:
        event_duration -= 1
    else:
        event_mult, event_duration = random_traffic_event()

    base_volume = np.random.randint(50, 300)
    base_speed = np.random.uniform(30, 80)
    base_occupancy = np.random.uniform(20, 80)

    # Adjust based on the effects
    adjusted_volume = base_volume * tod_effect * dow_effect * holiday_mult * event_mult * np.random.uniform(0.95, 1.05)
    adjusted_speed = base_speed * tod_effect * dow_effect * weather_mult * holiday_mult * event_mult * np.random.uniform(0.95, 1.05)
    adjusted_occupancy = base_occupancy * tod_effect * dow_effect * np.random.uniform(0.95, 1.05)

    traffic_volume.append(adjusted_volume)
    average_speed.append(adjusted_speed)
    occupancy.append(adjusted_occupancy)

# Create DataFrame
data = pd.DataFrame({
    'timestamp': timestamps,
    'traffic_volume': np.round(traffic_volume, 2),
    'average_speed': np.round(average_speed, 2),
    'occupancy': np.round(occupancy, 2)
})

# Save to CSV
data.to_csv('data/historical_traffic_data.csv', index=False)
print("Complex fake traffic data generated and saved to 'data/historical_traffic_data.csv'")

