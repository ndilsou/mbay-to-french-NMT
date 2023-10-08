from typing import Literal
import unicodedata

import regex
from .models import Entity, ObjectId, ValueObject


class Translation(ValueObject):
    translation: str
    key: str

    @classmethod
    def from_text(cls, text: str):
        return cls(translation=text, key=cls._create_index_key(text))

    @staticmethod
    def _create_index_key(text: str):
        lower_case = text.lower()
        clean_text = regex.sub(r"^['\(\{\[-]", "", lower_case)
        first_letter = clean_text[0]
        candidate = unicodedata.normalize("NFD", first_letter)
        candidate = regex.sub(r"\p{M}", "", candidate)
        if ord(candidate) >= 97 and ord(candidate) <= 122:
            index_key = candidate
        else:
            index_key = "MISC"
        return index_key


class ParentId(ValueObject):
    id: ObjectId
    type: Literal["entry", "expression"]


class Example(Entity):
    parent_id: ParentId
    mbay: str
    english: Translation
    french: Translation
    sound_filename: str | None = None


class Expression(Entity):
    entry_id: ObjectId
    mbay: str
    english: Translation
    french: Translation
    sound_filename: str | None = None
    example: Example | None = None


class Note(ValueObject):
    french: str
    english: str


class RelatedWord(ValueObject):
    text: str | None = None
    id: ObjectId | None = None


class Entry(Entity):
    headword: str
    part_of_speech: str | None = None
    sound_filename: str | None = None
    french: Translation
    english: Translation
    related_word: RelatedWord | None = None
    grammatical_note: Note | None = None

    examples: list[Example]
    expressions: list[Expression]
