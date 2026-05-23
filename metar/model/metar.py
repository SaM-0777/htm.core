from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class MetarRecord:
    cityName: str
    icaoId: str
    dataProvider: str
    name: str
    temperature: float
    rawMetarCode: str
    recordedTime: str
    updatedAt: str