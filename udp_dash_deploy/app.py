import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
import colorsys
from matplotlib import colors as mcolors

# === Load data ===
import os
import urllib.request

# Google Drive direct download links (replace with your actual IDs)
df_url = "https://drive.google.com/uc?export=download&id=1mx2ctKeLb0lJrddK3DWMqqAD1OGaEM7o"
gdf_url = "https://drive.google.com/uc?export=download&id=1HpUlHlOwXHOqJsuWgNOOalOamxGePKiY"

if not os.path.exists("df_merged.pkl"):
    urllib.request.urlretrieve(df_url, "df_merged.pkl")
if not os.path.exists("master_gdf.pkl"):
    urllib.request.urlretrieve(gdf_url, "master_gdf.pkl")

df = pd.read_pickle("df_merged.pkl")
master_gdf = gpd.read_pickle("master_gdf.pkl")


app = Dash(__name__)
app.title = "UDP Coefficient Explorer"

geo_rename_map = {
    "City_Buffer": "City of Atlanta or Intersecting Tracts",
    "Core_NonCity": "Core Counties Outside City"
}

color_map = {
    ("City_Buffer", "Vulnerable"): "#5081e4",
    ("City_Buffer", "Gentrifying"): "#e15c3c",
    ("Core_NonCity", "Vulnerable"): "#934fd6",
    ("Core_NonCity", "Gentrifying"): "#f0b400"
}

marker_symbols = {True: 'circle', False: 'circle-open'}

app.layout = html.Div([
    html.H1("UDP Typology Coefficient Explorer", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Select Target Variable:"),
        dcc.Dropdown(
            id='target_dropdown',
            options=[{'label': v.replace('_', ' ').title(), 'value': v} for v in df['target'].unique()],
            value='total_apps', clearable=False
        ),
        html.Label("Select Occupancy Type:", style={'marginTop': '1rem'}),
        dcc.Dropdown(
            id='occupancy_dropdown',
            options=[{'label': v.replace('_', ' '), 'value': v} for v in df['occupancy_label'].unique()],
            value='Principal_Residence', clearable=False
        ),
        html.Label("Select Year (for Map):", style={'marginTop': '1rem'}),
        dcc.Slider(
            id='year_slider',
            min=int(df['year'].min()),
            max=int(df['year'].max()),
            step=1,
            marks={int(y): str(int(y)) for y in df['year'].unique()},
            value=int(df['year'].min())
        )
    ], style={'width': '85%', 'margin': 'auto'}),

    dcc.Graph(id='coefficient_plot', style={'marginTop': '2rem'}),
    dcc.Graph(id='map_plot', style={'marginTop': '2rem'}),

    html.Div([
        html.P("Note:", style={'fontWeight': 'bold'}),
        html.P("• Census tracts are only shown if at least one mortgage application of the selected type occurred that year."),
        html.P("• “City of Atlanta or Intersecting Tracts” includes census tracts inside or intersecting city limits."),
        html.P("• “Core Counties Outside City” refers to Fulton, DeKalb, Gwinnett, Cobb, or Clayton tracts outside the city."),
        html.P("• UDP 'Vulnerable' group includes: Low-Income/Susceptible to Displacement, At Risk of Gentrification, Early/Ongoing Gentrification."),
        html.P("• UDP 'Gentrifying' group includes: Ongoing Displacement, Advanced Gentrification.")
    ], style={'width': '85%', 'margin': '2rem auto', 'fontSize': '0.9rem'})
])

@app.callback(
    [Output('coefficient_plot', 'figure'),
     Output('map_plot', 'figure')],
    [Input('target_dropdown', 'value'),
     Input('occupancy_dropdown', 'value'),
     Input('year_slider', 'value')]
)
def update_figures(target, occupancy, year):
    dff = df[(df['target'] == target) & (df['occupancy_label'] == occupancy)].copy()
    gdf_year = master_gdf[
        (master_gdf['year'] == year) &
        (master_gdf['occupancy_label'] == occupancy) &
        (
            ((master_gdf['in_city_or_intersect']) & master_gdf['udp_group'].isin(['Vulnerable', 'Gentrifying'])) |
            ((~master_gdf['in_city_or_intersect']) &
             master_gdf['county'].isin(['Fulton', 'DeKalb', 'Gwinnett', 'Cobb', 'Clayton']) &
             master_gdf['udp_group'].isin(['Vulnerable', 'Gentrifying']))
        )
    ].copy()

    fig1 = go.Figure()
    for geo in dff['geo_group'].unique():
        for group in ['Vulnerable', 'Gentrifying']:
            dfg = dff[dff['geo_group'] == geo]
            sig_col = f'sig_{group.lower()}'
            coef_col = f'coef_udp_{group}'
            caution_col = f'caution_{group.lower()}'
            mean_col = f'{target}_mean'
            legend_label = f"{geo_rename_map.get(geo, geo)} — {group}"

            fig1.add_trace(go.Scatter(
                x=dfg['year'], y=dfg[coef_col], mode='lines',
                name=legend_label, legendgroup=f"{geo}-{group}", showlegend=True,
                line=dict(color=color_map[(geo, group)], width=2), hoverinfo='skip'
            ))

            fig1.add_trace(go.Scatter(
                x=dfg['year'], y=dfg[coef_col], mode='markers', showlegend=False,
                legendgroup=f"{geo}-{group}",
                marker=dict(
                    symbol=[marker_symbols[sig] for sig in dfg[sig_col]],
                    size=10, color=color_map[(geo, group)],
                    line=dict(width=1, color='black')
                ),
                hovertext=[
                    f"Year: {yr}<br>Coefficient: {coef:.2f}<br>{'Significant' if sig else 'Not Significant'}<br>"
                    f"{caution or 'No cautions'}<br>Ref. Mean: {ref:.2f}<br>"
                    f"≈{abs(coef) / ref * 100:.0f}% of baseline" if pd.notna(ref) and ref != 0 else ""
                    for yr, coef, sig, caution, ref in zip(
                        dfg['year'], dfg[coef_col], dfg[sig_col], dfg[caution_col], dfg[mean_col]
                    )
                ]
            ))

    fig1.update_layout(
        title=f"Regression Coefficients over Time for {occupancy.replace('_', ' ')} — {target.replace('_', ' ')}",
        xaxis_title='Year', yaxis_title='Coefficient', template='plotly_white'
    )

    gdf_year = gdf_year[pd.notna(gdf_year[target])].copy()
    gdf_year['geo_group'] = gdf_year.apply(lambda row: 'City_Buffer' if row['in_city_or_intersect'] else 'Core_NonCity', axis=1)
    gdf_year['base_color'] = gdf_year.apply(lambda row: color_map.get((row['geo_group'], row['udp_group']), 'lightgray'), axis=1)

    vmin, vmax = gdf_year[target].min(), gdf_year[target].max()
    def adjust_color(base, value):
        if pd.isna(value) or pd.isna(vmin) or pd.isna(vmax): return '#dddddd'
        norm_val = (value - vmin) / (vmax - vmin) if vmax != vmin else 0.5
        r, g, b = mcolors.to_rgb(base)
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        l = 0.95 - 0.5 * norm_val
        l = max(0.3, min(1.0, l))
        r_new, g_new, b_new = colorsys.hls_to_rgb(h, l, s)
        return mcolors.to_hex((r_new, g_new, b_new))

    gdf_year['fill_color'] = gdf_year.apply(lambda row: adjust_color(row['base_color'], row[target]), axis=1)
    gdf_year['hover'] = gdf_year.apply(lambda row: f"County: {row['county']}<br>{target.replace('_', ' ').title()}: {row[target]:,.2f}", axis=1)

    fig2 = px.choropleth_mapbox(
        gdf_year,
        geojson=gdf_year.geometry.__geo_interface__,
        locations=gdf_year.index.astype(str),
        color=gdf_year['fill_color'],
        color_discrete_map={c: c for c in gdf_year['fill_color'].unique()},
        custom_data=["hover"]
    )

    fig2.update_traces(
        marker_line_color='black',
        marker_line_width=0.5,
        hovertemplate="%{customdata[0]}<extra></extra>"
    )

    fig2.update_layout(
        mapbox_style="carto-positron",
        mapbox_center={"lat": 33.75, "lon": -84.39},
        mapbox_zoom=9,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        showlegend=False
    )

    return fig1, fig2

if __name__ == '__main__':
    app.run(debug=True)
