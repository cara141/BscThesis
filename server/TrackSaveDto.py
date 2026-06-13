from typing import List

from pydantic import BaseModel


class TrackSaveDto(BaseModel):
    user_id: int
    title: str
    main_genre: str
    sub_genre: str
    features: List[float]