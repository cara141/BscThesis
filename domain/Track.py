import json
from dataclasses import dataclass, field
from typing import List

@dataclass
class Track:
    id: int = None
    user_id: int = None
    title: str = None
    main_genre: str = None
    sub_genre: str = None
    features: List[float] = field(default_factory=list)