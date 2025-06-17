def format_dict_log(dictionary: dict):
    return f"\n\n{'*'*100}\n\n{'\n'.join(f"{k}: {v}" for k, v in dictionary.items())}\n\n{'*'*100}\n\n"
