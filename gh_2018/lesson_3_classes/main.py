from gh_2018.lesson_3_classes._doc import Doc
from gh_2018.lesson_3_classes._tokenizers import BaseTokenizer, CustomTokenizer


def main(text: str):
    """Main program functionality."""
    doc = Doc(text, BaseTokenizer())
    print(doc.text_raw)
    print(doc.tokens)

    doc_new = Doc(text, CustomTokenizer())
    print(doc_new.text_raw)
    print(doc_new.tokens)


if __name__ == '__main__':
    text = "Any text. Some-another text."
    main(text)
