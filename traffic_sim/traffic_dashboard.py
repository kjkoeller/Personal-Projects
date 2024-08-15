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

    # Row with the traffic graph and the road network image
    html.Div([
        # Traffic Graph
        html.Div([
            dcc.Graph(id='traffic-graph'),
        ], style={'width': '70%', 'display': 'inline-block'}),

        # Example Road Network Image (replace with actual if needed)
        html.Div([
            html.Img(src=app.get_asset_url('road_network.png'), style={'width': '100%', 'height': 'auto'}),
        ], style={'width': '25%', 'display': 'inline-block', 'padding': '20px'}),
    ]),

    html.Div([
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
        html.Label('Select Date Range:'),
        dcc.DatePickerRange(
            id='date-picker-range',
            start_date=data['timestamp'].min().date(),
            end_date=data['timestamp'].max().date(),
            display_format='YYYY-MM-DD',
        ),
    ]),
])

@app.callback(
    Output('traffic-graph', 'figure'),
    [Input('feature-dropdown', 'value'),
     Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_graph(selected_feature, start_date, end_date):
    # Ensure the date range is correctly handled by pandas
    filtered_data = data[(data['timestamp'] >= pd.to_datetime(start_date)) &
                         (data['timestamp'] <= pd.to_datetime(end_date))]

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
            type="date",
            range=[filtered_data['timestamp'].min(), filtered_data['timestamp'].max()],
            rangeslider=dict(visible=True),  # Add a range slider for easier zooming
        ),
        transition_duration=500
    )

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)

