from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime


class CloudLayer(BaseModel):
    coverage: str
    height_ft: Optional[int] = None
    type: Optional[str] = None


class MetarDataInput(BaseModel):
    time_recorded: datetime
    temperature_c: Optional[float] = None
    dew_point_c: Optional[float] = None
    pressure_hpa: Optional[float] = None
    wind_direction_deg: Optional[float] = None
    wind_speed_kt: Optional[float] = None
    cloud_layers: List[CloudLayer] = []
