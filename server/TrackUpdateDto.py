from typing import List

from pydantic import BaseModel


class TrackUpdateDto(BaseModel):
    id: int
    user_id: int
    title: str
    main_genre: str
    sub_genre: str
    features: List[float]