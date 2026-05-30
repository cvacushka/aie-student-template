from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


RoomType = Literal["lecture", "seminar", "lab", "coworking"]
BuildingZone = Literal["north", "south", "east", "west", "central"]
LessonType = Literal["lecture", "practice", "lab", "exam", "none"]
RiskLevel = Literal["low", "medium", "high"]


class AirGuardRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "timestamp": "2026-03-17T10:35:00",
                "room_id": "B-421",
                "room_type": "lecture",
                "building_zone": "central",
                "lesson_type": "lecture",
                "room_area_m2": 72.0,
                "capacity": 70,
                "occupancy_count": 61,
                "hvac_airflow_m3h": 360.0,
                "ventilation_level": 0.38,
                "outdoor_temp_c": 6.0,
                "indoor_temp_c": 25.8,
                "humidity_pct": 58.0,
                "co2_ppm_current": 1040.0,
                "minutes_since_lesson_start": 48,
                "minutes_until_break": 32,
                "noise_level_db": 59.0,
                "is_exam_period": False,
                "floor": 4,
            }
        }
    )

    timestamp: datetime
    room_id: str = Field(min_length=1, max_length=32)
    room_type: RoomType
    building_zone: BuildingZone
    lesson_type: LessonType
    room_area_m2: float = Field(gt=10, le=250)
    capacity: int = Field(ge=5, le=250)
    occupancy_count: int = Field(ge=0, le=300)
    hvac_airflow_m3h: float = Field(ge=0, le=3000)
    ventilation_level: float = Field(ge=0, le=1)
    outdoor_temp_c: float = Field(ge=-40, le=45)
    indoor_temp_c: float = Field(ge=10, le=40)
    humidity_pct: float = Field(ge=5, le=95)
    co2_ppm_current: float = Field(ge=350, le=3000)
    minutes_since_lesson_start: int = Field(ge=0, le=180)
    minutes_until_break: int = Field(ge=0, le=180)
    noise_level_db: float = Field(ge=20, le=95)
    is_exam_period: bool
    floor: int = Field(ge=1, le=20)


class AirGuardResponse(BaseModel):
    room_id: str
    risk_probability: float = Field(ge=0, le=1)
    risk_level: RiskLevel
    model_threshold: float = Field(ge=0, le=1)
    model_name: str
    prediction_horizon_minutes: int = 30
    recommendation: str
