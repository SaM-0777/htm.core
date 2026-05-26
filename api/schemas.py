from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class CloudLayer(BaseModel):
    coverage: str
    height_ft: Optional[int] = None
    type: Optional[str] = None


class MetarDataInput(BaseModel):
    time_recorded: datetime
    temperature_c: float
    pressure_hpa: float
    dew_point_c: float
    visibility: float
    wind_direction_deg: float
    wind_speed_kt: float
    wind_gust_kt: float
    is_wind_variable: bool = False
    cloud_layers: List[CloudLayer] = []
