import re


def clean_and_tokenise(text: str) -> list:
    """
    Convert the text to tokens by:
    1. Lowercasing the text.
    2. Replacing everything except letters, digits, underscores, spaces, and apostrophes with a space.
    3. Optionally removing any apostrophes that are not between two alphanumeric characters.
    4. Collapsing multiple spaces and trimming the text.
    5. Splitting the text into tokens based on whitespace.
    """
    text = text.lower()
    # replace everything except letters/digits/underscore/space/apostrophe with space
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    # optional: remove any apostrophes that are not between two alphanumeric chars
    #text = re.sub(r"(?<![a-z0-9])'|'(?![a-z0-9])", " ", text)
    # collapse multiple spaces and trim
    text = re.sub(r"\s+", " ", text).strip()
    # split on whitespace
    return text.split()