from llama_cpp import Llama

def test_local_llama():
    # Initialize the model
    llm = Llama(
        model_path="./models/llama-2-7b-chat.gguf",
        n_ctx=2048,
        n_threads=4
    )
    
    # Test prompt
    prompt = """
    Summarize this meeting transcript:
    Today we discussed the Q4 roadmap. Alice will handle the frontend work.
    Bob is responsible for backend changes. Due date is December 15th.
    """
    
    # Generate response
    output = llm.create_completion(
        prompt=prompt,
        max_tokens=200,
        temperature=0.7,
        top_p=0.95,
        stop=["###"]  # Optional stop token
    )
    
    print("Model Output:", output['choices'][0]['text'])

if __name__ == "__main__":
    test_local_llama()

    # test_llama.py
try:
    from llama_cpp import Llama
    print("Successfully imported llama_cpp!")
except ImportError as e:
    print(f"Import Error: {str(e)}")
    print("Current Python path:")
    import sys
    for path in sys.path:
        print(path)