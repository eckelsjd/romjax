from romjax import YamlLoader
from romjax.model import Model

from pydantic import field_validator

class UserModel(Model):
    opts: dict[str, int]
    detail: str

    @field_validator('detail')
    @classmethod
    def validate_detail(cls, field):
        if len(field) > 100:
            raise Exception

        return str(field)

    def evaluate():
        pass

    def solve():
        pass


my_config = YamlLoader.load('scripts/config.yml')
my_model = my_config['solver']
print(my_config)
