from pydantic import BaseModel

class UserRegisterDto(BaseModel):
    username: str
    email: str
    password: str