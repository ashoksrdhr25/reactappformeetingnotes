import os
from llm_providers import LLMProvider

def test_deepseek():
    try:
        # Print OpenAI version
        import openai
        print(f"OpenAI version: {openai.__version__}")
        
        # Check if API key exists
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            print("Warning: DEEPSEEK_API_KEY environment variable not found")
        
        # Initialize the LLMProvider
        print("Attempting to initialize LLMProvider...")
        provider = LLMProvider(model="deepseek-chat")
        
        # Generate a completion
        print("Attempting to generate completion...")
        completion = provider.generate_completion(
            prompt="Hello, how are you?",
            system_prompt="You are a helpful assistant."
        )
        
        # Print the response
        print("\nDeepSeek Response:")
        print(completion)
        
    except Exception as e:
        print(f"\nError type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        import traceback
        print("\nFull traceback:")
        print(traceback.format_exc())

if __name__ == "__main__":
    test_deepseek()