""" from https://github.com/keithito/tacotron """
from text import cleaners
from text.symbols import symbols


# Mappings from symbol to numeric ID and vice versa:
def _symbol_to_id(symbols):
    return {s: i for i, s in enumerate(symbols)}


def _id_to_symbol(symbols):
    return {i: s for i, s in enumerate(symbols)}


def text_to_sequence(text, cleaner_names, symbols):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
      cleaner_names: names of the cleaner functions to run the text through
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = []
    clean_text = _clean_text(text, cleaner_names)
    symbol_to_id = _symbol_to_id(symbols)
    for symbol in clean_text:
        if symbol in symbol_to_id.keys():
            symbol_id = symbol_to_id[symbol]
            sequence += [symbol_id]
        else:
            continue
    return sequence


def cleaned_text_to_sequence(cleaned_text, symbols):
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    Args:
      text: string to convert to a sequence
    Returns:
      List of integers corresponding to the symbols in the text
    """
    sequence = []
    symbol_to_id = _symbol_to_id(symbols)

    for symbol in cleaned_text:
        if symbol in symbol_to_id.keys():
            symbol_id = symbol_to_id[symbol]
            sequence += [symbol_id]
        else:
            print(f"[!] Undefined symbol {symbol} in: {cleaned_text}")
            continue
    return sequence


def sequence_to_text(sequence, symbols):
    """Converts a sequence of IDs back to a string"""
    id_to_symbol = _id_to_symbol(symbols)
    result = ""
    for symbol_id in sequence:
        s = id_to_symbol[symbol_id]
        result += s
    return result


def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        cleaner = getattr(cleaners, name)
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text
