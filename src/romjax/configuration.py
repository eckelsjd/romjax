"""Typed assignments into YAML configurations before runtime construction."""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import ConfigDict, field_validator

import romjax
from romjax.typing import DictModel


class ConfigAssignment(DictModel):
    """An exact YAML path assignment.

    :param path: mapping keys and nonnegative sequence indices
    :param value: replacement value or YAML source
    :param mode: replace the target, or apply YAML merge semantics
    """

    model_config = ConfigDict(extra='forbid')
    path: tuple[str | int, ...]
    value: Any
    mode: Literal['replace', 'merge'] = 'replace'

    @field_validator('path', mode='before')
    @classmethod
    def _path(cls, value: Any) -> tuple[str | int, ...]:
        if not isinstance(value, (list, tuple)) or not value:
            raise ValueError('Assignment path must be a nonempty token sequence.')
        if any(type(x) not in (str, int) or isinstance(x, int) and x < 0 for x in value):
            raise ValueError('Path tokens must be strings or nonnegative integers (not booleans).')
        return tuple(value)


class YamlTemplate:
    """Unconstructed, source-resolved YAML supporting exact typed assignments."""

    def __init__(self, source: romjax.YamlSource) -> None:
        self.source = source

    @classmethod
    def from_source(cls, source: Path | romjax.YamlSource) -> YamlTemplate:
        """Resolve raw YAML without constructing user objects.

        :param source: YAML file or source node
        :return: independent template
        """
        if isinstance(source, romjax.YamlSource):
            node, path = source.node, source.source_path
        else:
            text, path = romjax.YamlLoader._read_stream(Path(source))
            node = yaml.compose(text, Loader=yaml.SafeLoader)
        if node is None:
            raise ValueError('Base YAML is empty.')
        return cls(romjax.YamlSource(cls._resolve(node, path), path))

    @classmethod
    def _resolve(cls, node: yaml.Node, path: Path | None) -> yaml.Node:
        # Resolve overrides at their declaring source, including nested fragments.
        loader = romjax.YamlLoader
        if loader._is_override_node(node):
            base_path = loader._resolve_override_path(node.tag[len(loader.OVERRIDES_TAG):], path)
            base = cls.from_source(base_path).source.node
            overlay = loader._copy_with_tag(node, loader._default_node_tag(node))
            return loader._merge_nodes(base, cls._resolve(overlay, path))
        if isinstance(node, yaml.MappingNode):
            return yaml.MappingNode(node.tag, [(cls._resolve(k, path), cls._resolve(v, path)) for k, v in node.value])
        if isinstance(node, yaml.SequenceNode):
            return yaml.SequenceNode(node.tag, [cls._resolve(v, path) for v in node.value])
        return loader._resolve_parent_refs(node, path)

    @staticmethod
    def validate_paths(assignments: Sequence[ConfigAssignment]) -> None:
        """Reject duplicate and ancestor/descendant assignments.

        :param assignments: assignments to validate together
        """
        paths = [item.path for item in assignments]
        for index, path in enumerate(paths):
            for other in paths[:index]:
                if path[:len(other)] == other or other[:len(path)] == path:
                    raise ValueError(f'Assignment paths overlap: {other!r} and {path!r}.')

    def render(self, assignments: Sequence[ConfigAssignment]) -> romjax.YamlSource:
        """Apply assignments to an independent node tree.

        :param assignments: typed target/value pairs
        :return: raw YAML ready for serialization or runtime loading
        """
        self.validate_paths(assignments)
        # Rebuild without memoization: YAML aliases must not cause collateral edits.
        root = self._resolve(self.source.node, self.source.source_path)
        for assignment in assignments:
            if isinstance(assignment.value, romjax.YamlSource):
                replacement = self.from_source(assignment.value).source.node
            else:
                replacement = yaml.compose(romjax.YamlLoader.dump(assignment.value, _preserve_yaml_sources=True,
                                                                _preserve_model_types=True))
                replacement = self._resolve(replacement, self.source.source_path)
            node = root
            for index, token in enumerate(assignment.path):
                last = index == len(assignment.path) - 1
                if isinstance(node, yaml.MappingNode) and isinstance(token, str):
                    found = next((i for i, (key, _) in enumerate(node.value)
                                  if key.tag == 'tag:yaml.org,2002:str' and key.value == token), None)
                    if found is None:
                        if not last:
                            raise ValueError(f'Missing ancestor in path {assignment.path!r}.')
                        node.value.append((yaml.ScalarNode('tag:yaml.org,2002:str', token), replacement))
                        break
                    key, child = node.value[found]
                    if last:
                        value = romjax.YamlLoader._merge_nodes(child, replacement) if assignment.mode == 'merge' \
                            else replacement
                        node.value[found] = key, value
                    node = child
                elif isinstance(node, yaml.SequenceNode) and type(token) is int and token < len(node.value):
                    child = node.value[token]
                    if last:
                        node.value[token] = romjax.YamlLoader._merge_nodes(child, replacement) \
                            if assignment.mode == 'merge' else replacement
                    node = child
                else:
                    raise ValueError(f'Invalid target in path {assignment.path!r}.')
        return romjax.YamlSource(root, self.source.source_path)

    def _references(self, node: yaml.Node, seen: frozenset[Path] = frozenset()):
        if isinstance(node, yaml.MappingNode):
            children = [value for _, value in node.value]
        elif isinstance(node, yaml.SequenceNode):
            children = node.value
        else:
            children = []
        for child in children:
            yield from self._references(child, seen)
        if isinstance(node, yaml.ScalarNode) and node.tag == 'tag:yaml.org,2002:str':
            path = Path(node.value)
            if path.suffix.lower() in ('.yaml', '.yml') and path.is_file():
                path = path.resolve()
                if path in seen:
                    raise ValueError(f'Cyclic YAML file reference: {path}')
                template = self.from_source(path)
                yield node, path, template
                yield from self._references(template.source.node, seen | {path})

    def snapshot(self, directory: Path) -> YamlTemplate:
        """Copy referenced YAML documents and redirect this independent template.

        :param directory: destination for immutable configuration dependencies
        :return: template referencing the snapshots; data/artifact paths stay external
        """
        source = self.render([])
        references = list(self._references(source.node))
        if references:
            directory.mkdir(parents=True, exist_ok=True)
        # Walk in reverse so nested references have already been rewritten when
        # their containing document is serialized.
        for node, path, template in reversed(references):
            name = hashlib.sha256(str(path).encode()).hexdigest() + '.yml'
            destination = directory.resolve() / name
            destination.write_text(romjax.YamlLoader._node_to_yaml(template.source.node), encoding='utf-8')
            node.value = str(destination)
        return YamlTemplate(source)

    def fingerprint(self) -> str:
        """Return SHA256 of resolved YAML and every referenced YAML document."""
        digest = hashlib.sha256(romjax.YamlLoader._node_to_yaml(self.source.node).encode())
        for _, path, template in self._references(self.source.node):
            digest.update(str(path).encode())
            digest.update(romjax.YamlLoader._node_to_yaml(template.source.node).encode())
        return digest.hexdigest()
