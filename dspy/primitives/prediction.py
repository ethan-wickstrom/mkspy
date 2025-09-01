from dataclasses import dataclass
from typing import Any


@dataclass
class Prediction:
    """Stub prediction object with minimal attributes used in tests."""
    code: str = ""
    signature: str = ""
    architecture: str = ""
    is_valid: bool = True
    errors: Any = None
    refined_code: str = ""
    implementation_code: str = ""
    signature_spec: str = ""
    module_architecture: str = ""
    validation_result: bool = True
