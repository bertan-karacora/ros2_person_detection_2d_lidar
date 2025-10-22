import json
import logging

import yaml

_LOGGER = logging.getLogger(__name__)
_INDENT_TAB = 4


def read_yaml(path):
    try:
        with open(path, "r") as file:
            data = yaml.safe_load(file)
    except yaml.YAMLError as exception:
        _LOGGER.exception(exception)

        raise exception

    return data


def write_yaml(dict, path):
    class DumperIndent(yaml.Dumper):
        def increase_indent(self, flow=False, indentless=False):
            return super().increase_indent(flow, False)

    try:
        with open(path, "w") as file:
            yaml.dump(dict, file, Dumper=DumperIndent, default_flow_style=False, indent=_INDENT_TAB)
    except yaml.YAMLError as exception:
        _LOGGER.exception(exception)

        raise exception


def read_json(path):
    try:
        with open(path, "r") as file:
            data = json.load(file)
    except json.JSONDecodeError as exception:
        _LOGGER.exception(exception)

        raise exception

    return data
