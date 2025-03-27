# api/models.py
from pydantic import BaseModel
from typing import Optional

class CropFeatures(BaseModel):
    N: Optional[int] = None
    P: Optional[int] = None
    K: Optional[int] = None
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    ph: Optional[float] = None
    rainfall: Optional[float] = None
