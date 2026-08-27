"""PII detector.

Detecting personal data must hard-switch privacy to strict and route the
request to a local model, bypassing free tiers that train on data.
Regexes, not a neural net: faster, more precise, no false positives on
ordinary text.

Returns a set of PII types. An empty set means clean.
"""
from __future__ import annotations

import re

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(
    r"(?<!\d)(?:\+7|8|7)[\s\-]?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{2}[\s\-]?\d{2}(?!\d)"
)
CARD_RE = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
PASSPORT_RE = re.compile(
    r"(?i)(?:паспорт|passport|серия|серией)[^\d]{0,20}\d{2}[^\d]{0,20}\d{2}[^\d]{0,20}\d{6}"
)
IP_RE = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
INN_RE = re.compile(r"(?<!\d)\d{12}(?!\d)")  # individual INN (Russian tax ID)


def _luhn(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def detect(text: str) -> set[str]:
    found: set[str] = set()
    if EMAIL_RE.search(text):
        found.add("email")
    if PHONE_RE.search(text):
        found.add("phone")
    if IP_RE.search(text):
        found.add("ip")
    if PASSPORT_RE.search(text):
        found.add("passport")
    if INN_RE.search(text):
        found.add("inn")
    for m in CARD_RE.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and _luhn(digits):
            found.add("card")
            break
    return found


def has_pii(text: str) -> bool:
    return bool(detect(text))
