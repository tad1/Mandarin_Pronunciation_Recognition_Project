# based on: https://github.com/khughitt/anki-chinese/blob/master/ankichinese/pinyin.py

from typing import List
import re
from zhon import pinyin
from dragonmapper import hanzi
from enum import Enum

class Tone(Enum):
    """Enum for tones"""
    TONE1 = 1
    TONE2 = 2
    TONE3 = 3
    TONE4 = 4
    TONE5 = 5


tone1_re = re.compile("[āēīōūǖ]")
tone2_re = re.compile("[áéíóúǘ]")
tone3_re = re.compile("[ǎěǐǒǔǚ]")
tone4_re = re.compile("[àèìòùǜ]")

def get_tone(pinyin_word:str) -> Tone:
    if len(tone1_re.findall(pinyin_word)) > 0:
        return Tone.TONE1 
    if len(tone2_re.findall(pinyin_word)) > 0:
        return Tone.TONE2
    if len(tone3_re.findall(pinyin_word)) > 0:
        return Tone.TONE3
    if len(tone4_re.findall(pinyin_word)) > 0:
        return Tone.TONE4
    else:
        return Tone.TONE5

def get_tones(chinese_phrase:str) -> List[Tone]:
    # first, query pinyin for complete phrase
    pinyin_phrase = hanzi.to_pinyin(chinese_phrase)
    # split by syllable and add relevant html container elements
    # e.g. "niǔdài" -> ['niǔ', 'dài']
    pinyin_parts = re.findall(pinyin.syllable, pinyin_phrase)

    tones = [get_tone(x) for x in pinyin_parts]

    return tones


