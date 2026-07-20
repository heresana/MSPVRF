"""
GuardPrompt - LLM Client Interfaces
-----------------------------------
Handles connections to local Ollama Base LLMs using the official ollama package.
"""

import ollama
import threading

class OllamaClient:
    def __init__(self, model_name="llama3.2:latest", timeout_sec=30, num_predict=512):
        self.model_name = model_name
        self.timeout_sec = timeout_sec
        self.num_predict = num_predict

    def _generate_with_timeout(self, prompt: str, options: dict) -> dict:
        """Wraps ollama.generate in a thread to enforce a strict hard timeout."""
        result = [None]
        exception = [None]

        def _target():
            try:
                result[0] = ollama.generate(
                    model=self.model_name,
                    prompt=prompt,
                    options=options
                )
            except Exception as e:
                exception[0] = e

        t = threading.Thread(target=_target, daemon=True)
        t.start()
        t.join(self.timeout_sec)

        if t.is_alive():
            raise TimeoutError(f"Model '{self.model_name}' exceeded {self.timeout_sec}s timeout")
        if exception[0]:
            raise exception[0]
            
        return result[0]

    def generate(self, prompt: str, temperature=0.0) -> str:
        """
        Sends the prompt to local Ollama and returns the generated string.
        Temperature defaults to 0.0 to match your baseline evaluation settings.
        """
        options = {
            'temperature': temperature,
            'num_predict': self.num_predict
        }
        
        try:
            response = self._generate_with_timeout(prompt, options)
            generated_text = response.get('response', '').strip()
            
            if not generated_text:
                return "[ERROR: NULL_OUTPUT_DETECTED]"
                
            return generated_text
            
        except TimeoutError as e:
            return f"[ERROR: TIMEOUT - {str(e)}]"
        except Exception as e:
            return f"[ERROR: RUNTIME EXCEPTION - {str(e)}]"