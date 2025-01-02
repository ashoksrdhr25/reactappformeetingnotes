# meeting_summarizer.py
from typing import Optional
from llm_providers import LLMProvider

class MeetingSummarizer:
    """
    Handles the generation of meeting summaries using an LLM provider.
    This class constructs appropriate prompts and manages the interaction
    with the LLM to generate structured meeting summaries.
    """
    
    def __init__(self, llm_provider: LLMProvider):
        """
        Initialize the summarizer with an LLM provider.
        
        Args:
            llm_provider (LLMProvider): Instance of LLMProvider to use for generating summaries
        """
        self.llm = llm_provider

    def generate_summary(
        self,
        transcript: str,
        context: str = "",
        instructions: Optional[str] = None,
        max_tokens: int = 4000  # Add max_tokens parameter with a default value
    ) -> str:
        """
        Generate a structured summary of a meeting transcript.
        
        Args:
            transcript (str): The meeting transcript to summarize
            context (str): Additional context or reference material
            instructions (Optional[str]): Custom instructions for summary generation
            max_tokens (int): Maximum number of tokens in the response
            
        Returns:
            str: Generated meeting summary
            
        Raises:
            Exception: If there's an error during summary generation
        """
        # Construct system prompt
        system_prompt = """
        You are an expert meeting summarizer. Your task is to create clear, structured meeting notes.
        Focus on key points, action items, and decisions made. Use markdown formatting for structure.
        """
        
        # Construct main prompt
        prompt = f"""
        Please summarize the following meeting transcript into structured notes.
        
        {f'Additional Context:\n{context}\n' if context else ''}
        {f'Special Instructions:\n{instructions}\n' if instructions else ''}
        
        Meeting Transcript:
        {transcript}
        
        Please provide the summary in the following format:
        1. Key Takeaways & Highlights
        2. Discussion Points
        3. Action Items (with owners and due dates if mentioned)
        """
        
        try:
            print("Sending prompt to LLM...")
            print(f"System Prompt: {system_prompt}")
            print(f"Main Prompt: {prompt}")
            
            summary = self.llm.generate_completion(
                prompt=prompt,
                system_prompt=system_prompt,
                max_tokens=max_tokens  # Pass max_tokens to generate_completion
            )
            
            print("Received summary from LLM:")
            print(summary)
            
            return summary
        except Exception as e:
            raise Exception(f"Error generating summary: {str(e)}")