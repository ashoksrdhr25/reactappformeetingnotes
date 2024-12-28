# llm_providers.py
import os
import anthropic
import openai
from typing import Optional

class LLMProvider:
    """
    Provides a unified interface for interacting with different Language Model providers.
    Currently supports Anthropic's Claude and OpenAI's GPT models, handling the specific
    requirements and API calls for each provider.
    """
    
    def __init__(self, model: str):
        """
        Initialize the LLM provider with the specified model.
        
        Args:
            model (str): Name of the model to use (e.g., "claude-3-5-sonnet-20241022")
        """
        print(f"Initializing LLMProvider with model: {model}")
        self.client = None
        self.model = model  # Store the model name
        
        if "claude" in model.lower():
            self.client = anthropic.Client(api_key=os.getenv("ANTHROPIC_API_KEY"))
            print("Initialized Anthropic client")
        elif model in ["gpt-4o"]:
            self.client = openai.Client(api_key=os.getenv("OPENAI_API_KEY"))
            print("Initialized OpenAI client")
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
            print(f"Generating completion with model: {self.model}")
            
            if "claude" in self.model.lower():
                print("Using Anthropic API")
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )
                print("Anthropic API response received")
                return response.content[0].text
            
            elif self.model in ["gpt-4o"]:
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=max_tokens
                )
                return response.choices[0].message.content

        except Exception as e:
            raise Exception(f"Error generating completion: {str(e)}")