# llm_manager.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Dict, Any
from abc import ABC, abstractmethod
from huggingface_hub import login

login(token='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')

class BaseLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass

class HuggingFaceLlama32LLM(BaseLLM):
    def __init__(self, model_name: str = "Qwen/Qwen3-4B-Thinking-2507", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True
            )
            print(f"✅ Loaded model: {self.model_name}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise

    def generate(self, prompt: str, max_new_tokens: int = 500, temperature: float = 0.1) -> str:
        """Generate text using the loaded model"""
        if not self.tokenizer or not self.model:
            return "Model not loaded properly"

        try:
            # Format prompt for the model
            formatted_prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            inputs = self.tokenizer(formatted_prompt, return_tensors="pt", truncation=True, max_length=2048)

            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids.to(self.model.device),
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only the generated part
            generated_tokens = outputs[0][inputs.input_ids.shape[1]:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            return response.strip()

        except Exception as e:
            return f"Generation error: {str(e)}"