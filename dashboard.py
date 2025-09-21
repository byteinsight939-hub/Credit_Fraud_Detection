import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

import dash
from dash import dcc, html, Input, Output

# 1. Load data
df = pd.read_csv('creditcard.csv')  # adjust path

# 2. Preprocess / feature engineering
# E.g., convert “Time” to hours, maybe create bins for Amount
df['Hour'] = (df['Time'] // 3600) % 24  # or something like that

# Optional: sample down non‑fraud if too many, or oversample fraud for certain plots

# 3. Initialize Dash
app = dash.Dash(__name__)
app.title = "Credit Card Fraud Dashboard"

# 4. Layout
app.layout = html.Div([
    html.H1("Credit Card Fraud Analysis"),

    html.Div([
        html.Label("Select Feature for Histogram:"),
        dcc.Dropdown(
            id='hist-feature',
            options=[{'label': c, 'value': c} for c in df.columns if c not in ['Class','Time']],
            value='Amount'
        ),
    ], style={'width': '30%', 'display': 'inline-block'}),

    dcc.Graph(id='histogram'),

    html.Div([
        html.Label("Select Hour Range:"),
        dcc.RangeSlider(
            id='hour-slider',
            min=0,
            max=23,
            step=1,
            value=[0,23],
            marks={i: str(i) for i in range(0,24,2)}
        )
    ], style={'width': '60%', 'padding': '20px'}),

    dcc.Graph(id='tx_by_hour'),

    html.H2("PCA plot (2 components)"),
    dcc.Graph(id='pca-scatter'),
    
    html.H2("Correlation Heatmap"),
    dcc.Graph(id='corr-heatmap'),
])

# 5. Callbacks

@app.callback(
    Output('histogram', 'figure'),
    Input('hist-feature', 'value'),
    Input('hour-slider', 'value'),
)
def update_histogram(feature, hour_range):
    dff = df.loc[(df['Hour'] >= hour_range[0]) & (df['Hour'] <= hour_range[1])]
    fig = px.histogram(
        dff,
        x=feature,
        color='Class',
        nbins=50,
        title=f"Distribution of {feature} by Fraud vs Non‑Fraud"
    )
    return fig

@app.callback(
    Output('tx_by_hour', 'figure'),
    Input('hour-slider', 'value'),
)
def update_tx_by_hour(hour_range):
    dff = df.loc[(df['Hour'] >= hour_range[0]) & (df['Hour'] <= hour_range[1])]
    grouped = dff.groupby(['Hour','Class']).size().reset_index(name='count')
    fig = px.line(grouped, x='Hour', y='count', color='Class',
                  title="Transactions by Hour (Count) by Class")
    return fig

@app.callback(
    Output('pca-scatter', 'figure'),
    Input('hour-slider', 'value'),
)
def update_pca(hour_range):
    dff = df.loc[(df['Hour'] >= hour_range[0]) & (df['Hour'] <= hour_range[1])]
    features = [c for c in df.columns if c.startswith('V')]  # or whichever features
    pca = PCA(n_components=2)
    components = pca.fit_transform(dff[features].fillna(0))
    pc_df = pd.DataFrame(components, columns=['PC1','PC2'])
    pc_df['Class'] = dff['Class'].values
    fig = px.scatter(pc_df, x='PC1', y='PC2', color='Class',
                     title="PCA of Features colored by Fraud vs Non‑Fraud",
                     opacity=0.7)
    return fig

@app.callback(
    Output('corr-heatmap', 'figure'),
    Input('hour-slider', 'value'),  # optional: filter
)
def update_heatmap(hour_range):
    dff = df.loc[(df['Hour'] >= hour_range[0]) & (df['Hour'] <= hour_range[1])]
    features = [c for c in df.columns if c not in ['Time','Class']]
    corr = dff[features].corr()
    fig = px.imshow(corr, 
                    title="Correlation Heatmap of Features",
                    color_continuous_scale='RdBu_r',
                    zmin=-1, zmax=1)
    return fig

# 6. Run the app
if __name__ == '__main__':
    app.run(debug=True)
