from typing import List

from pydantic import BaseModel


class FeatureInputDto(BaseModel):
    features: List[float]
    label: str