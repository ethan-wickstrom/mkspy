from __future__ import annotations

from typing import Any, Dict, List

TASK_LIBRARY: List[Dict[str, Any]] = [
    {
        "description": "Classify sentiment in product reviews",
        "input_types": "str (review)",
        "output_requirements": "str (positive/negative/neutral)",
        "test_cases": [
            {"input": "The phone is fantastic and works great.", "expected": "positive"},
            {"input": "Battery died after two days.", "expected": "negative"},
        ],
        "complexity": "simple",
    },
    {
        "description": "Extract top keywords from a paragraph",
        "input_types": "str (paragraph)",
        "output_requirements": "list[str] (keywords)",
        "test_cases": [
            {
                "input": "Python is a popular programming language for data science.",
                "expected": ["Python", "programming", "data science"],
            },
            {
                "input": "Solar and wind energy are leading renewable sources.",
                "expected": ["Solar", "wind energy", "renewable sources"],
            },
        ],
        "complexity": "simple",
    },
    {
        "description": "Translate English sentences to Spanish",
        "input_types": "str (English sentence)",
        "output_requirements": "str (Spanish translation)",
        "test_cases": [
            {
                "input": "The library opens at nine o'clock.",
                "expected": "La biblioteca abre a las nueve.",
            },
            {
                "input": "Where is the nearest train station?",
                "expected": "¿Dónde está la estación de tren más cercana?",
            },
        ],
        "complexity": "simple",
    },
    {
        "description": "Summarize a news article in one sentence",
        "input_types": "str (article)",
        "output_requirements": "str (single-sentence summary)",
        "test_cases": [
            {
                "input": "NASA successfully launched a new satellite to monitor climate change, aiming to provide more accurate data on global warming trends.",
                "expected": "NASA launched a satellite to monitor climate change.",
            },
            {
                "input": "The city council approved a new park downtown, promising more green space for residents.",
                "expected": "The city council approved a downtown park to add green space.",
            },
        ],
        "complexity": "moderate",
    },
    {
        "description": "Identify named entities in a text",
        "input_types": "str (text)",
        "output_requirements": "list[str] (named entities)",
        "test_cases": [
            {
                "input": "Barack Obama was born in Hawaii and served as President of the United States.",
                "expected": ["Barack Obama", "Hawaii", "United States"],
            },
            {
                "input": "Apple released the first iPhone in 2007 under Steve Jobs.",
                "expected": ["Apple", "iPhone", "2007", "Steve Jobs"],
            },
        ],
        "complexity": "moderate",
    },
    {
        "description": "Convert CSV data to a list of dictionaries",
        "input_types": "str (CSV data)",
        "output_requirements": "list[dict[str, str]] (rows)",
        "test_cases": [
            {
                "input": "name,age\nAlice,30\nBob,25",
                "expected": [{"name": "Alice", "age": "30"}, {"name": "Bob", "age": "25"}],
            },
            {
                "input": "city,population\nParis,2148000\nRome,2873000",
                "expected": [{"city": "Paris", "population": "2148000"}, {"city": "Rome", "population": "2873000"}],
            },
        ],
        "complexity": "moderate",
    },
    {
        "description": "Generate SQL query from natural language request",
        "input_types": "str (request)",
        "output_requirements": "str (SQL query)",
        "test_cases": [
            {
                "input": "Retrieve names of employees hired after 2020.",
                "expected": "SELECT name FROM employees WHERE hire_date > '2020-12-31';",
            },
            {
                "input": "Count how many orders were completed in March 2024.",
                "expected": "SELECT COUNT(*) FROM orders WHERE completed_at >= '2024-03-01' AND completed_at < '2024-04-01';",
            },
        ],
        "complexity": "complex",
    },
    {
        "description": "Answer multi-hop questions using provided documents",
        "input_types": "dict{question:str, documents:list[str]}",
        "output_requirements": "str (answer citing source document)",
        "test_cases": [
            {
                "input": {
                    "question": "Which city is the capital of the country whose national animal is the kiwi?",
                    "documents": [
                        "The kiwi is the national bird of New Zealand.",
                        "Wellington is the capital of New Zealand.",
                        "Canberra is the capital of Australia.",
                    ],
                },
                "expected": "Wellington (doc2)",
            }
        ],
        "complexity": "advanced",
    },
    {
        "description": "Generate Python code from a specification",
        "input_types": "str (specification)",
        "output_requirements": "dspy.Code (function implementation)",
        "test_cases": [
            {
                "input": "Write a function that returns the nth Fibonacci number using iteration.",
                "expected": "def fibonacci(n: int) -> int:\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
            }
        ],
        "complexity": "complex",
    },
    {
        "description": "Identify bugs in Python code",
        "input_types": "dspy.Code",
        "output_requirements": "list[str] (identified issues)",
        "test_cases": [
            {
                "input": "def add_item(item, items=[]):\n    items.append(item)\n    return items",
                "expected": ["mutable default argument 'items'"],
            }
        ],
        "complexity": "advanced",
    },
]
