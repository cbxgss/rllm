import yaml


class LiteralStr(str):
    """用于强制 YAML 使用 | 多行格式"""
    pass


def _literal_str_representer(dumper, data):
    return dumper.represent_scalar(
        "tag:yaml.org,2002:str",
        data,
        style="|"
    )


yaml.add_representer(LiteralStr, _literal_str_representer)


def _convert_multiline_str(obj):
    """
    递归遍历：
    - 只要是包含 \n 的字符串，就转成 LiteralStr
    """
    if isinstance(obj, str):
        if "\n" in obj:
            return LiteralStr(obj)
        return obj
    elif isinstance(obj, dict):
        return {k: _convert_multiline_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_multiline_str(i) for i in obj]
    else:
        return obj


def save_dict_to_yaml(
    data: dict,
    path: str,
    width: int = 512,
):
    """
    将 dict 保存为 YAML 文件：
    - 多行字符串使用 | 格式
    - 不自动排序 key
    - 支持中文
    """

    data = _convert_multiline_str(data)

    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
            width=width,
            default_flow_style=False,
        )
