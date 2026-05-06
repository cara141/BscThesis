from dataclasses import dataclass, field
from typing import List

@dataclass
class User:
    id: int = None
    username: str = ""
    email: str = ""
    password: str = ""