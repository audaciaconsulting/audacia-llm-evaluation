import re

def format_dict_log(dictionary: dict):
    lines = '\n'.join(f"{k}: {v}" for k, v in dictionary.items())
    return f"\n\n{'*'*100}\n\n{lines}\n\n{'*'*100}\n\n"

def camel_to_snake(camel_str):
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', camel_str).lower()
