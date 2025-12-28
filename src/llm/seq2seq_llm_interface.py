from abc import ABC, abstractmethod
from typing import List

from src.scheme.response_scheme import ResponseScheme
from src.templates.template_interface import *


class Seq2SeqLLMInterface(ABC):
    def __init__(self, llm_name: str):
        self._name = llm_name
        self._prefix_temps_list: List[Template] = []
        self._postfix_temps_list: List[Template] = []

    def add_prefix(self, prefix_template: Template):
        self._prefix_temps_list.append(prefix_template)

    def add_postfix(self, postfix_template: Template):
        self._postfix_temps_list.append(postfix_template)

    def construct_prompt(self, msg: Template) -> str:
        prompt = []

        for prefix_temp in self._prefix_temps_list:
            prompt.append(str(prefix_temp))

        prompt.append(str(msg))

        for postfix_temp in self._postfix_temps_list:
            prompt.append(str(postfix_temp))

        return "\n".join(prompt)

    @abstractmethod
    def call(self, msg: Template | str) -> str:
        pass
