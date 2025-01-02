import os
import anthropic
import openai
from typing import Optional

class LLMProvider:
    """
    Provides a unified interface for interacting with different Language Model providers.
    Currently supports Anthropic's Claude, OpenAI's GPT models, and DeepSeek V3.
    """
    
    def __init__(self, model: str):
        """
        Initialize the LLM provider with the specified model.
        
        Args:
            model (str): Name of the model to use (e.g., "claude-3-5-sonnet-20241022", "gpt-4o", "deepseek-chat")
        """
        print(f"\nInitializing LLMProvider with model: {model}")
        print(f"OpenAI version: {openai.__version__}")
        self.client = None
        self.model = model  # Store the model name
        
        if "claude" in model.lower():
            self.client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))
            print("✓ Initialized Anthropic client")
        elif model in ["gpt-4o"] or "deepseek" in model.lower():
            api_key = os.getenv("OPENAI_API_KEY") if model in ["gpt-4o"] else os.getenv("DEEPSEEK_API_KEY")
            base_url = "https://api.deepseek.com" if "deepseek" in model.lower() else None
            print(f"Base URL: {base_url}")
            print(f"API key exists: {bool(api_key)}")
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_url
            )
            print("✓ Initialized " + ("OpenAI client" if model in ["gpt-4o"] else "DeepSeek client"))
        else:
            raise ValueError(f"Unsupported model: {model}")
           
        if not self.client:
            raise ValueError("Failed to initialize LLM client")

    def generate_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 4000
    ) -> str:
        """
        Generate a completion using the selected LLM.
        
        Args:
            prompt (str): Main prompt text
            system_prompt (Optional[str]): System-level instructions for the model
            max_tokens (int): Maximum number of tokens in the response
            
        Returns:
            str: Generated completion text
            
        Raises:
            Exception: If there's an error during completion generation
        """
        try:
            print(f"\nGenerating completion using {self.model}...")
            
            if "claude" in self.model.lower():
                print("✓ Using Anthropic API")
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )
                print("✓ Anthropic API response received")
                return response.content[0].text
            
            elif self.model in ["gpt-4o"] or "deepseek" in self.model.lower():
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                print("✓ Sending request to API...")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens
                )
                print("✓ Response received successfully")
                return response.choices[0].message.content

        except Exception as e:
            print(f"\n❌ Error during completion generation:")
            print(f"Error type: {type(e).__name__}")
            print(f"Error message: {str(e)}")
            raise