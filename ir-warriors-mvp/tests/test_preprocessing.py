import numpy as np

from app.services.preprocessing_service import (
    normalize_thermal,
    preprocess_landsat_scene,
    qa_valid_mask,
    scale_reflectance,
    scale_temperature,
)
from app.utils.validators import discover_scene


def test_landsat_file_discovery(landsat_scene_factory):
    scene = discover_scene(landsat_scene_factory())
    assert scene.thermal.name.endswith("_ST_B10.TIF")
    assert scene.red.name.endswith("_SR_B4.TIF")
    assert scene.green.name.endswith("_SR_B3.TIF")
    assert scene.blue.name.endswith("_SR_B2.TIF")
    assert scene.qa.name.endswith("_QA_PIXEL.TIF")


def test_collection_two_scale_factors():
    values = np.array([0, 10000], dtype=np.uint16)
    np.testing.assert_allclose(scale_reflectance(values), [-0.2, 0.075], atol=1e-6)
    np.testing.assert_allclose(scale_temperature(values), [149.0, 183.1802], atol=1e-4)
    np.testing.assert_allclose(normalize_thermal(np.array([250.0, 290.0, 330.0]), 250, 330), [0, 0.5, 1])


def test_qa_mask_rejects_required_bits():
    qa = np.array([[0, 1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5]], dtype=np.uint16)
    assert qa_valid_mask(qa).tolist() == [[True, False, False, False, False, False, False]]


def test_preprocessing_builds_aligned_pair(landsat_scene_factory):
    scene = preprocess_landsat_scene(landsat_scene_factory())
    assert scene.thermal.shape == (32, 32)
    assert scene.rgb.shape == (3, 32, 32)
    assert not scene.valid_mask[0, 0]
    assert scene.valid_mask[1, 1]
    assert scene.metadata.crs == "EPSG:32643"
