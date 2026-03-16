from pathlib import Path

import pytest
from pydantic import ValidationError

from romtools import YamlLoader
from romtools.typing import DictModel


class CustomSettings(DictModel, extra='forbid'):
    one: str
    two: str
    four: float = 1.1


def test_custom_settings():
    """Make sure subclasses of DictModel work"""
    settings = CustomSettings(one='1', two='2')

    # get, set, del, iter, len
    assert settings['four'] == 1.1
    settings['four'] = '3'
    assert settings['four'] == 3
    assert isinstance(settings['four'], float)
    del settings['one']
    
    with pytest.raises(KeyError):
        _ = settings['one']
    
    settings['one'] = '1'

    keys = []
    values = []
    for k, v in settings.items():
        keys.append(k)
        values.append(v)
    
    assert all([k in settings.keys() for k in keys])
    assert all([settings[k] == v for k,v in zip(keys, values)])
    
    assert len(keys) == len(settings) == 3
    assert 'one' in settings
    assert settings.get('five') is None

    with pytest.raises(ValidationError):
        settings['three'] = 'forbidden'


def test_plain_dict():
    """Make sure DictModel works without predefined schema."""
    d = {'a': 1, 'b': 2, 'c': 3}
    model = DictModel(**d)
    model.e = 'e'
    model['f'] = 42
    del model['a']
    model.update({'g': 'helo', 'h': 242.3})

    del d['a']
    d.update({'g': 'helo', 'h': 242.3, 'e': 'e', 'f': 42})

    assert len(d) == len(model)
    assert d == dict(model)


def test_from_yaml():
    """Make sure validation for DictModel works from file."""
    fixture_path = Path("tests/fixtures_typing.yml")
    data = YamlLoader.load(fixture_path)

    assert isinstance(data['settings'], CustomSettings)
    assert isinstance(data['configs'], DictModel)

    assert isinstance(data['settings']['four'], float)

    data['configs'].update({'1': 1, '2': 2})
    assert all(k in data['configs'].keys() for k in ['1', '2', 'one', 'two'])

    with pytest.raises(ValidationError):
        data['settings']['3'] = 'three'  # extras are forbidden
