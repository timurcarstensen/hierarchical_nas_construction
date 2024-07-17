import json
import os
from pathlib import Path
from typing import Any, NamedTuple

support_types = ("str", "int", "bool", "float", "none")


def convert_param(original_lists):
    assert isinstance(
        original_lists, list,
    ), f"The type is not right : {original_lists}"
    ctype, value = original_lists[0], original_lists[1]
    assert (
        ctype in support_types
    ), f"Ctype={ctype}, support={support_types}"
    is_list = isinstance(value, list)
    if not is_list:
        value = [value]
    outs = []
    for x in value:
        if ctype == "int":
            x = int(x)
        elif ctype == "str":
            x = str(x)
        elif ctype == "bool":
            x = bool(int(x))
        elif ctype == "float":
            x = float(x)
        elif ctype == "none":
            if x.lower() != "none":
                raise ValueError(
                    f"For the none type, the value must be none instead of {x}",
                )
            x = None
        else:
            raise TypeError(f"Does not know this type : {ctype}")
        outs.append(x)
    if not is_list:
        outs = outs[0]
    return outs


def load_config(
    path: Path | str, extra: dict[str, Any] | None = None,
) -> NamedTuple:
    """Load configuration from a JSON file and return it as a named tuple."""
    path = str(path)
    assert os.path.exists(path), f"Can not find {path}"
    # Reading data back
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    content = {k: convert_param(v) for k, v in data.items()}
    assert extra is None or isinstance(
        extra, dict,
    ), f"invalid type of extra : {extra}"
    if isinstance(extra, dict):
        content = {**content, **extra}
    Arguments = NamedTuple(
        "Configure", [(k, Any) for k in content],
    )
    return Arguments(**content)
