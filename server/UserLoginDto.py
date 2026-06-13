from pydantic import BaseModel

class UserLoginDto(BaseModel):
    username: str
    password: str