from pydantic import BaseModel
from typing import List

class PredictionInput(BaseModel):
    values: List[float]