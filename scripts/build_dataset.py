#!/usr/bin/env python3
"""Generate a labeled dataset for the L1 classifier head.

Cold start: there are no own logs yet, so labels are assembled from
templates per task class. This is a weak teacher, but better than nothing:
the head trains on it, then retrains on real logs
(scripts/train_head.py --from-telemetry).

    python -m scripts.build_dataset --out var/labels.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# (class, complexity, templates)
TEMPLATES: dict[str, list[tuple[float, list[str]]]] = {
    "code": [
        (
            0.7,
            [
                "напиши функцию на python, которая {задача}",
                "напиши код на python для {задача}",
                "как реализовать {задача} на python",
                "найди баг в этом коде: {код}",
                "объясни этот код: {код}",
                "напиши скрипт, который {задача}",
                "оптимизируй эту функцию: {код}",
                "напиши sql-запрос, который {задача}",
                "напиши регулярное выражение для {задача}",
                "отсортируй массив на python",
            ],
        ),
    ],
    "reasoning": [
        (
            0.7,
            [
                "объясни почему {факт}",
                "докажи что {факт}",
                "проанализируй {факт}",
                "сравни {а} и {б}",
                "почему {факт}",
                "спроектируй решение для {задача}",
                "как работает {факт}",
                "в чём разница между {а} и {б}",
                "обоснуй {факт}",
                "рассуждай шаг за шагом: {задача}",
            ],
        ),
    ],
    "extraction": [
        (
            0.3,
            [
                "извлеки email из строки: {строка}",
                "переведи на английский: {строка}",
                "исправь ошибки в тексте: {строка}",
                "составь список из: {строка}",
                "классифицируй: {строка}",
                "извлеки дату из текста: {строка}",
                "сократи текст: {строка}",
                "выдели ключевые слова из: {строка}",
                "перефразируй: {строка}",
                "извлеки имена из: {строка}",
            ],
        ),
    ],
    "chat": [
        (
            0.2,
            [
                "привет",
                "как дела",
                "расскажи анекдот",
                "что нового",
                "спасибо",
                "пока",
                "как тебя зовут",
                "что ты умеешь",
                "расскажи о себе",
                "привет, как настроение",
            ],
        ),
    ],
}

FILL = {
    "задача": [
        "сортировки списка",
        "поиска дубликатов",
        "парсинга json",
        "обработки файлов",
        "работы с api",
    ],
    "код": [
        "def f(x): return x",
        "for i in range(10): print(i)",
        "class A: pass",
        "x = [1,2,3]",
    ],
    "факт": [
        "небо голубое",
        "вода кипит при 100 градусах",
        "земля круглая",
        "зимой холодно",
        "свет быстрее звука",
    ],
    "а": ["python", "java", "кошки", "чай", "зима"],
    "б": ["javascript", "go", "собаки", "кофе", "лето"],
    "строка": [
        "a@b.com",
        "привет мир",
        "12.05.2024",
        "Иван Петров",
        "красный синий зелёный",
    ],
}


def build(n_per_class: int = 40, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    out: list[dict] = []
    for task, groups in TEMPLATES.items():
        for _ in range(n_per_class):
            complexity, templates = rng.choice(groups)
            tpl = rng.choice(templates)
            text = tpl
            for key, vals in FILL.items():
                if "{" + key + "}" in text:
                    text = text.replace("{" + key + "}", rng.choice(vals))
            out.append({"text": text, "task_class": task, "complexity": complexity})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="var/labels.jsonl")
    ap.add_argument("--per-class", type=int, default=40)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = build(n_per_class=args.per_class)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"generated {len(rows)} examples → {args.out}")


if __name__ == "__main__":
    main()
