"""Contains classes of document."""


class Doc:
    """Class-wrapper of document."""
    def __init__(self, text: str, tknzr):
        self.text_raw = text
        self.tokens = tknzr.tokenize(text)
