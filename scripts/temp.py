from pydantic import BaseModel, ConfigDict


class MyModel(BaseModel):
    model_config = ConfigDict(extra='allow', validate_assignment=True)
    a: int
    b: int
    c: int = 1


a = MyModel(a=1, b=2, d = '4')


print(len(a))



