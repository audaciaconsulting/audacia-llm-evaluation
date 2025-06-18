def format_dict_log(dictionary: dict, stars: int = 100) -> str:
    string = f"\n\n{'*' * stars}\n\n{'\n'.join(f"{k}: {v}" for k, v in dictionary.items())}\n\n{'*' * stars}\n\n"
    return string
