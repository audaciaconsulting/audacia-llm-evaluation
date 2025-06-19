import re

def format_dict_log(dictionary: dict, stars: int = 100) -> str:
    string = f"\n\n{'*' * stars}\n\n{'\n'.join(f"{k}: {v}" for k, v in dictionary.items())}\n\n{'*' * stars}\n\n"
    return string

def camel_to_snake(camel_str):
    return re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', camel_str).lower()
