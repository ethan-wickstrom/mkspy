from __future__ import annotations

import ast
from typing import Any, Dict, List, Tuple

Example = Tuple[Any, Any]


def _parse_value(text: str) -> Any:
    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def _case(raw: str) -> Example:
    left, right = raw.split("->", 1)
    return _parse_value(left.strip()), _parse_value(right.strip())


def task(description: str, cases: List[str]) -> Dict[str, Any]:
    parsed: List[Example] = [_case(c) for c in cases]
    return {"description": description, "test_cases": parsed}


TASK_LIBRARY: List[Dict[str, Any]] = [
    task(
        "Classify sentiment in product reviews",
        [
            "The phone is fantastic and works great. -> positive",
            "Battery died after two days. -> negative",
        ],
    ),
    task(
        "Extract top keywords from a paragraph",
        [
            "Python is a popular programming language for data science. -> ['Python', 'programming', 'data science']",
            "Solar and wind energy are leading renewable sources. -> ['Solar', 'wind energy', 'renewable sources']",
        ],
    ),
    task(
        "Translate English sentences to Spanish",
        [
            "The library opens at nine o'clock. -> 'La biblioteca abre a las nueve.'",
            "Where is the nearest train station? -> '¿Dónde está la estación de tren más cercana?'",
        ],
    ),
    task(
        "Summarize a news article in one sentence",
        [
            "NASA successfully launched a new satellite to monitor climate change, aiming to provide more accurate data on global warming trends. -> 'NASA launched a satellite to monitor climate change.'",
            "The city council approved a new park downtown, promising more green space for residents. -> 'The city council approved a downtown park to add green space.'",
        ],
    ),
    task(
        "Identify named entities in a text",
        [
            "Barack Obama was born in Hawaii and served as President of the United States. -> ['Barack Obama', 'Hawaii', 'United States']",
            "Apple released the first iPhone in 2007 under Steve Jobs. -> ['Apple', 'iPhone', '2007', 'Steve Jobs']",
        ],
    ),
    task(
        "Convert CSV data to a list of dictionaries",
        [
            """name,age\nAlice,30\nBob,25 -> [{'name': 'Alice', 'age': '30'}, {'name': 'Bob', 'age': '25'}]""",
            """city,population\nParis,2148000\nRome,2873000 -> [{'city': 'Paris', 'population': '2148000'}, {'city': 'Rome', 'population': '2873000'}]""",
        ],
    ),
    task(
        "Generate SQL query from natural language request",
        [
            "Retrieve names of employees hired after 2020. -> 'SELECT name FROM employees WHERE hire_date > \'2020-12-31\';'",
            "Count how many orders were completed in March 2024. -> 'SELECT COUNT(*) FROM orders WHERE completed_at >= \'2024-03-01\' AND completed_at < \'2024-04-01\';'",
        ],
    ),
    task(
        "Answer multi-hop questions using provided documents",
        [
            "{'question': 'Which city is the capital of the country whose national animal is the kiwi?', 'documents': ['The kiwi is the national bird of New Zealand.', 'Wellington is the capital of New Zealand.', 'Canberra is the capital of Australia.']} -> 'Wellington (doc2)'",
        ],
    ),
    task(
        "Generate Python code from a specification",
        [
            "Write a function that returns the nth Fibonacci number using iteration. -> 'def fibonacci(n: int) -> int:\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a'",
        ],
    ),
    task(
        "Identify bugs in Python code",
        [
            "def add_item(item, items=[])\n    items.append(item)\n    return items -> [\"mutable default argument 'items'\"]",
        ],
    ),
]

__all__: List[str] = ["TASK_LIBRARY", "task"]

