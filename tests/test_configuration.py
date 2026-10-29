from pathlib import Path

import pytest
import yaml

import romjax
from romjax.configuration import ConfigAssignment, YamlTemplate


def template(text: str, source: Path | None = None) -> YamlTemplate:
    return YamlTemplate.from_source(romjax.YamlSource(yaml.compose(text), source))


def normalized_path(value: str | Path) -> str:
    return Path(value).as_posix()


@pytest.mark.parametrize('value', [None, True, 2, 0.001, 'null', 'false', '1e-3'])
def test_exact_assignment_preserves_types_and_neighbors(value):
    base = template('items: [1, 2, 3]\noptimizer: {kwargs: {}}')
    source = base.render([ConfigAssignment(path=['items', 1], value=value)])
    actual = romjax.load(source)['items']
    assert actual == [1, value, 3]
    assert type(actual[1]) is type(value)
    assert romjax.load(base.render([]))['items'] == [1, 2, 3]


def test_add_leaf_preserve_tag_and_source(tmp_path):
    base = template('!romx:Train\noptimizer: {kwargs: {}}\ndata: __parent__/data.h5', tmp_path / 'base.yml')
    source = base.render([ConfigAssignment(path=['optimizer', 'kwargs', 'learning_rate'], value=0.01)])
    assert source.node.tag == '!romx:Train'
    text = romjax.YamlLoader._node_to_yaml(source.node)
    assert normalized_path(tmp_path / 'data.h5') in text.replace('\\', '/')
    assert 'learning_rate: 0.01' in text


@pytest.mark.parametrize('path', [['missing', 'x'], ['items', 9], ['items', -1], ['items', True], []])
def test_invalid_paths_fail(path):
    with pytest.raises(ValueError):
        template('items: [1, 2]').render([ConfigAssignment(path=path, value=3)])


def test_conflicts_and_alias_isolation():
    base = template('a: &value {x: 1}\nb: *value')
    source = base.render([ConfigAssignment(path=['a', 'x'], value=9)])
    assert romjax.load(source) == {'a': {'x': 9}, 'b': {'x': 1}}
    with pytest.raises(ValueError, match='overlap'):
        base.render([ConfigAssignment(path=['a'], value={}), ConfigAssignment(path=['a', 'x'], value=2)])


def test_merge_and_fingerprint(tmp_path):
    path = tmp_path / 'base.yml'
    path.write_text('optimizer: {name: adam, kwargs: {b1: 0.9}}')
    base = YamlTemplate.from_source(path)
    merged = base.render([ConfigAssignment(path=['optimizer'], value={'name': 'sgd'}, mode='merge')])
    assert romjax.load(merged)['optimizer'] == {'name': 'sgd', 'kwargs': {'b1': 0.9}}
    path.write_text('optimizer: {name: sgd}')
    assert base.fingerprint() != YamlTemplate.from_source(path).fingerprint()


def test_override_chain_retains_each_declaring_source(tmp_path):
    original = tmp_path / 'original'
    overlay = tmp_path / 'overlay'
    original.mkdir()
    overlay.mkdir()
    (original / 'base.yml').write_text('data: __parent__/data.h5\noptimizer: {kwargs: {learning_rate: 0.1}}')
    path = overlay / 'base.yml'
    path.write_text('!overrides:__parent__/../original/base.yml\nextra: __parent__/extra.h5')
    value = romjax.load(YamlTemplate.from_source(path).render([]))
    assert normalized_path(value['data']) == normalized_path(original / 'data.h5')
    assert normalized_path(value['extra']) == normalized_path(overlay / 'extra.h5')


def test_inline_override_preserves_base_origin(tmp_path):
    folder = tmp_path / 'base'
    folder.mkdir()
    (folder / 'base.yml').write_text('data: __parent__/data.h5\nvalue: 1')
    path = tmp_path / 'study.yml'
    path.write_text('base: !overrides:__parent__/base/base.yml\n  value: 2')
    source = romjax.load(path)['base']
    actual = romjax.load(YamlTemplate.from_source(source).render([]))
    assert actual['value'] == 2
    assert normalized_path(actual['data']) == normalized_path(folder / 'data.h5')


def test_referenced_yaml_is_fingerprinted_and_snapshotted(tmp_path):
    graph = tmp_path / 'graph.yml'
    graph.write_text('width: 8\ndata: __parent__/data.h5')
    base = template(f'graph: {graph}')
    original = base.fingerprint()
    snapshot = base.snapshot(tmp_path / 'snapshot')
    snap_path = Path(romjax.load(snapshot.render([]))['graph'])
    graph.write_text('width: 16')
    assert base.fingerprint() != original
    assert romjax.load(snap_path)['width'] == 8
    assert normalized_path(romjax.load(snap_path)['data']) == normalized_path(tmp_path / 'data.h5')
