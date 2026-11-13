import numpy as np
import rasterio
from rasterio.transform import from_origin

from app.services.geospatial_service import export_rgb_geotiff


def test_geotiff_export_preserves_spatial_reference(tmp_path):
    output = export_rgb_geotiff(
        np.full((3, 8, 9), 0.5, dtype=np.float32),
        tmp_path / "generated.tif",
        "EPSG:32643",
        from_origin(700000, 2100000, 30, 30),
        np.ones((8, 9), dtype=bool),
    )
    with rasterio.open(output) as source:
        assert source.count == 3
        assert source.crs.to_string() == "EPSG:32643"
        assert source.res == (30.0, 30.0)
        assert source.read().shape == (3, 8, 9)
