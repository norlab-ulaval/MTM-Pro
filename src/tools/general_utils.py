# coding=utf-8


import re

def sanitize_dirname(text: str) -> str:
    """
    Sanitizes a directory name by replacing spaces, removing invalid characters, and normalizing underscores.

    :param text: The input string to be sanitized.
    :return: A sanitized string suitable for use as a directory name.
    """
    # Replace spaces with underscores
    text = text.replace(' ', '_')
    # Keep only alphanumeric, underscores, and hyphens
    text = re.sub(r'[^\w\-]', '', text)
    # Remove multiple consecutive underscores
    text = re.sub(r'_+', '_', text)
    # Remove leading/trailing underscores
    return text.strip('_')
