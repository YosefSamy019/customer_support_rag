from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
import json
from pydantic import ValidationError

# Not Complete


class ResponseScheme(ABC, BaseModel):
    def parse_llm_output(llm_text: str) -> 'ResponseScheme':
        try:
            data = json.loads(llm_text)
            return UserIntentResponse.model_validate(data)
        except json.JSONDecodeError:
            raise ValueError("LLM output is not valid JSON")
        except ValidationError as e:
            raise ValueError(f"Schema validation failed: {e}")


class UserIntentResponse(ResponseScheme):
    intent: str = Field(
        title="Intent",
        description="A clear rephrasing of the user's true intent"
    )
