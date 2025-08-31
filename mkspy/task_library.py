from __future__ import annotations

from typing import Any, Dict, List

TASK_LIBRARY: List[Dict[str, Any]] = [
    {
        "description": "Create a sentiment analysis module for product reviews",
        "input_types": "str",
        "output_requirements": "str (positive/negative/neutral)",
        "test_cases": [
            {"input": "This product is amazing!", "expected": "positive"},
            {"input": "Terrible quality, very disappointed", "expected": "negative"},
        ],
        "complexity": "simple",
    },
    {
        "description": "Build a multi-hop QA system with retrieval",
        "input_types": "str (question)",
        "output_requirements": "str (answer with citations)",
        "test_cases": [],
        "complexity": "complex",
    },
    {
        "description": "Generate a code debugging assistant",
        "input_types": "dspy.Code",
        "output_requirements": "list[str] (identified issues)",
        "test_cases": [],
        "complexity": "advanced",
    },
]
