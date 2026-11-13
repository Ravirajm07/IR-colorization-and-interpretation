from __future__ import annotations

import base64
import io
from pathlib import Path

import folium
import numpy as np
import rasterio
import streamlit as st
from PIL import Image
from rasterio.warp import transform_bounds
from streamlit_folium import st_folium


def render_map(job: dict) -> None:
    geotiff = job.get("output_paths", {}).get("geotiff")
    if not geotiff or not Path(geotiff).exists():
        st.info("A georeferenced prediction is required for the GIS viewer.")
        return
    with rasterio.open(geotiff) as source:
        data = source.read()
        bounds = transform_bounds(source.crs, "EPSG:4326", *source.bounds)
    west, south, east, north = bounds
    image = Image.fromarray(np.moveaxis(data, 0, -1))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    basemap = st.selectbox("Basemap", ["OpenStreetMap", "CartoDB positron", "CartoDB Voyager"])
    opacity = st.slider("Generated layer opacity", 0.1, 1.0, 0.85)
    map_object = folium.Map(location=[(south + north) / 2, (west + east) / 2], zoom_start=9, tiles=basemap)
    folium.raster_layers.ImageOverlay(
        image=f"data:image/png;base64,{encoded}", bounds=[[south, west], [north, east]], opacity=opacity, name="Generated RGB"
    ).add_to(map_object)
    folium.Rectangle([[south, west], [north, east]], color="#00b8d9", weight=2, fill=False, tooltip="Scene footprint").add_to(map_object)
    folium.LayerControl(collapsed=False).add_to(map_object)
    result = st_folium(map_object, use_container_width=True, height=620)
    if result.get("last_clicked"):
        point = result["last_clicked"]
        st.caption(f"Coordinate: {point['lat']:.6f}, {point['lng']:.6f}")
    with st.expander("Raster metadata"):
        st.json({"path": geotiff, "bounds_wgs84": bounds, "shape": list(data.shape)})
