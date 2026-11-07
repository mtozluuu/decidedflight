"""Shared configurable limits for point and regional weather decisions."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DroneLimit(BaseModel):
    model_config = ConfigDict(allow_inf_nan=False)
    cloud_cover_ok_max_pct: float = Field(default=50, ge=0, le=100)
    wind_ok_max_knots: float = Field(default=15, ge=0)
    wind_risky_max_knots: float = Field(default=25, ge=0)
    visibility_ok_min_km: float = Field(default=5, ge=0)
    visibility_risky_min_km: float = Field(default=1, ge=0)
    cloud_base_ok_min_ft: float = Field(default=500, ge=0)
    cloud_base_risky_min_ft: float = Field(default=200, ge=0)
    cloud_ceiling_ok_min_ft: float = Field(default=1000, ge=0)
    cloud_ceiling_risky_min_ft: float = Field(default=500, ge=0)
    humidity_ok_max_pct: float = Field(default=85, ge=0, le=100)
    humidity_risky_max_pct: float = Field(default=95, ge=0, le=100)
    temp_ok_min_c: float = Field(default=0, allow_inf_nan=False)
    temp_ok_max_c: float = Field(default=40, allow_inf_nan=False)
    temp_risky_min_override_c: float | None = Field(default=None, allow_inf_nan=False)
    temp_risky_max_override_c: float | None = Field(default=None, allow_inf_nan=False)

    @property
    def temp_risky_min_c(self) -> float:
        return (
            self.temp_risky_min_override_c
            if self.temp_risky_min_override_c is not None
            else self.temp_ok_min_c - 5
        )

    @property
    def temp_risky_max_c(self) -> float:
        return (
            self.temp_risky_max_override_c
            if self.temp_risky_max_override_c is not None
            else self.temp_ok_max_c + 5
        )

    @model_validator(mode="after")
    def ordered_thresholds(self):
        pairs = (
            (self.wind_ok_max_knots, self.wind_risky_max_knots),
            (self.visibility_risky_min_km, self.visibility_ok_min_km),
            (self.cloud_base_risky_min_ft, self.cloud_base_ok_min_ft),
            (self.cloud_ceiling_risky_min_ft, self.cloud_ceiling_ok_min_ft),
            (self.humidity_ok_max_pct, self.humidity_risky_max_pct),
            (self.temp_ok_min_c, self.temp_ok_max_c),
            (self.temp_risky_min_c, self.temp_ok_min_c),
            (self.temp_ok_max_c, self.temp_risky_max_c),
        )
        if any(low > high for low, high in pairs):
            raise ValueError("Uygun ve riskli limitlerin sıralaması geçersiz.")
        return self
