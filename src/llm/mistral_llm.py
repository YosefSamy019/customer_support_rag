from src.llm.seq2seq_llm_interface import *
import src.helper.functions as functions
import src.config as config


class MistralLLM(Seq2SeqLLMInterface):
    def __init__(
            self,
            llm_name: str,
            max_length: int = 3500,
            num_return_sequences: int = 1,
            temperature: float = 0.3,
            top_k: int = 40,
            top_p: float = 0.9,
            do_sample: bool = True
    ):
        super().__init__(llm_name)
        # Store parameters as instance attributes
        self.max_length = max_length
        self.num_return_sequences = num_return_sequences
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.do_sample = do_sample

    def call(self, msg: Template | str) -> str:
        full_prompt: str = self.construct_prompt(msg)
        full_prompt_length: int = len(full_prompt)


        url = (config.NGROK_URL +
               r'/'
               r'generate_mistral')

        headers = {"Authorization": "Bearer 123"}
        payload = {
            "prompt": full_prompt,
            "max_length": self.max_length,
            "num_return_sequences": self.num_return_sequences,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "do_sample": self.do_sample
        }

        response = functions.endpoint_call(
            url=url,
            headers=headers,
            payload=payload
        )

        full_response =  response['response']
        llm_response = full_response[full_prompt_length:]
        return llm_response
