import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd

# Load the data
data = pd.read_csv('data/historical_traffic_data.csv')
data['timestamp'] = pd.to_datetime(data['timestamp'])

app = dash.Dash(__name__)

# Layout of the dashboard
app.layout = html.Div([
    html.H1('Traffic Data Dashboard'),

    html.Div([
        # Dropdown for selecting road segment
        html.Label('Select Road Segment:'),
        dcc.Dropdown(
            id='segment-dropdown',
            options=[{'label': seg, 'value': seg} for seg in data['road_segment'].unique()],
            value=data['road_segment'].unique()[0]
        ),
        # Dropdown for selecting vehicle type
        html.Label('Select Vehicle Type:'),
        dcc.Dropdown(
            id='vehicle-dropdown',
            options=[{'label': vt, 'value': vt} for vt in data['vehicle_type'].unique()],
            value=data['vehicle_type'].unique()[0]
        ),
        # Dropdown for selecting feature to plot
        html.Label('Select Feature:'),
        dcc.Dropdown(
            id='feature-dropdown',
            options=[
                {'label': 'Traffic Volume', 'value': 'traffic_volume'},
                {'label': 'Average Speed', 'value': 'average_speed'},
                {'label': 'Occupancy', 'value': 'occupancy'}
            ],
            value='traffic_volume'
        ),
        # Date range selector
        html.Label('Select Date Range:'),
        dcc.DatePickerRange(
            id='date-picker-range',
            start_date=data['timestamp'].min().date(),
            end_date=data['timestamp'].max().date(),
            display_format='YYYY-MM-DD',
        ),
    ]),

    dcc.Graph(id='traffic-graph')
])

@app.callback(
    Output('traffic-graph', 'figure'),
    [
        Input('segment-dropdown', 'value'),
        Input('vehicle-dropdown', 'value'),
        Input('feature-dropdown', 'value'),
        Input('date-picker-range', 'start_date'),
        Input('date-picker-range', 'end_date')
    ]
)
def update_graph(selected_segment, selected_vehicle, selected_feature, start_date, end_date):
    # Filter data based on selections
    filtered_data = data[
        (data['road_segment'] == selected_segment) &
        (data['vehicle_type'] == selected_vehicle) &
        (data['timestamp'] >= start_date) &
        (data['timestamp'] <= end_date)
    ]

    # Create the figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered_data['timestamp'],
        y=filtered_data[selected_feature],
        mode='lines+markers',
        name=selected_feature
    ))

    fig.update_layout(
        title=f'{selected_feature.replace("_", " ").title()} Over Time',
        xaxis_title='Time',
        yaxis_title=selected_feature.replace('_', ' ').title(),
        xaxis=dict(
            range=[filtered_data['timestamp'].min(), filtered_data['timestamp'].max()],
            rangeslider=dict(visible=True)
        )
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)