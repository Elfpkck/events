"""Tokenizers."""


class BaseTokenizer:
    """Tokenize texts."""
    CONFIG_NAME = "First"

    def tokenize(self, text: str) -> [[str]]:
        """Split text by tokens per sentence."""
        sent_tks = self._sent_tokenize(text)
        return self._tokens_tokenize(sent_tks)

    def _sent_tokenize(self, text: str) -> [str]:
        return [sent for sent in text.split('.') if sent]

    def _tokens_tokenize(self, sent_tks: [str])-> [[str]]:
        return [list(filter(bool, sent.split(' '))) for sent in sent_tks]


class CustomTokenizer(BaseTokenizer):
    def _tokens_tokenize(self, sent_tks: [str])-> [[str]]:
        result = []

        for sent in sent_tks:
            by_whitespaces = []

            for word in sent.split(' '):
                by_whitespaces += word.strip().split('-')

            result.append([x for x in by_whitespaces if x])

        return result
