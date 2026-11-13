from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_prefix="IRW_", extra="ignore")

    project_name: str = "IR Warriors MVP"
    model_version: str = "pix2pixhd-inspired-v1"
    api_url: str = "http://localhost:8000"
    data_dir: Path = ROOT / "data"
    output_dir: Path = ROOT / "outputs"
    checkpoint_dir: Path = ROOT / "checkpoints"
    temp_dir: Path = ROOT / ".tmp"
    max_upload_mb: int = 4096
    thermal_min_kelvin: float = 250.0
    thermal_max_kelvin: float = 330.0
    reflectance_min: float = 0.0
    reflectance_max: float = 0.4
    gamma: float = 1.0 / 2.2
    patch_size: int = 256
    stride: int = 192
    min_valid_fraction: float = 0.75
    generator_filters: int = 64
    generator_res_blocks: int = 6

    def ensure_directories(self) -> None:
        for path in (
            self.data_dir / "raw",
            self.data_dir / "processed",
            self.data_dir / "patches" / "train",
            self.data_dir / "patches" / "val",
            self.data_dir / "patches" / "test",
            self.data_dir / "boundaries",
            self.data_dir / "samples",
            self.output_dir / "predictions",
            self.output_dir / "geotiff",
            self.output_dir / "comparisons",
            self.output_dir / "reports",
            self.output_dir / "metrics",
            self.checkpoint_dir,
            self.temp_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings
