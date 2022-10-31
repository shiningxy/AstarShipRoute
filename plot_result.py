import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def draw_exist():
    data = pd.read_csv("data/path.csv")
    token = "pk.eyJ1IjoieGlhbmd5dXdhbmciLCJhIjoiY2wwbWI1Y3psMTRyZDNla2N4dGtodnRybCJ9.mwx99M60eZAPv6aYuhe5yg"
    fig = px.line_mapbox(data, lat="lat", lon="lon", color="color")
    fig.add_trace(go.Scattermapbox(
        mode="markers+lines",
        lon=data.lon,
        lat=data.lat,
        marker={'size': 12}))
    fig.update_layout(
        mapbox={'accesstoken': token, 'center': {'lon': 120, 'lat': 35}, 'style': "satellite-streets", 'zoom': 5},
        margin={'l': 0, 'r': 0, 't': 0, 'b': 0})
    fig.show()

if __name__ == "__main__":
    draw_exist()
