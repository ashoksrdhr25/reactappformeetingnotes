# main.py
# Import standard libraries
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, validator
from typing import List, Optional
import os
from dotenv import load_dotenv
import uuid
from pathlib import Path

# Import our custom modules - these classes are defined in their own files
from llm_providers import LLMProvider
from document_processor import DocumentProcessor
from meeting_summarizer import MeetingSummarizer

# Load and verify environment variables
load_dotenv()
print("\nEnvironment Check:")
print(f"ANTHROPIC_API_KEY exists: {'ANTHROPIC_API_KEY' in os.environ}")
print(f"ANTHROPIC_API_KEY length: {len(os.getenv('ANTHROPIC_API_KEY', ''))}")
print(f"OPENAI_API_KEY exists: {'OPENAI_API_KEY' in os.environ}")
print(f"OPENAI_API_KEY length: {len(os.getenv('OPENAI_API_KEY', ''))}")
print(f"DEEPSEEK_API_KEY exists: {'DEEPSEEK_API_KEY' in os.environ}")
print(f"DEEPSEEK_API_KEY length: {len(os.getenv('DEEPSEEK_API_KEY', ''))}")

# Initialize FastAPI app
app = FastAPI()

# Configure CORS for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create upload directory if it doesn't exist
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Define request model for summary generation
class SummaryRequest(BaseModel):
    transcript: str
    instructions: Optional[str] = None
    model: str = "deepseek-chat"  # Default to DeepSeek for faster response times
    reference_docs: Optional[List[str]] = None

    @validator('model')
    def validate_model(cls, v):
        allowed_models = ["claude-3-5-sonnet-20241022", "gpt-4o", "deepseek-chat"]
        if v not in allowed_models:
            raise ValueError(f"Model must be one of {allowed_models}")
        return v

@app.post("/api/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    """Handle file uploads for reference documents"""
    try:
        uploaded_files = []
        for file in files:
            # Generate unique filename
            file_extension = Path(file.filename).suffix
            unique_filename = f"{uuid.uuid4()}{file_extension}"
            file_path = UPLOAD_DIR / unique_filename
            
            # Save file
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            uploaded_files.append(unique_filename)
        
        return {"uploaded_files": uploaded_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/summarize")
def summarize_meeting(request: SummaryRequest):
    """Generate meeting summary using specified LLM"""
    try:
        print(f"Received summarize request with model: {request.model}")
        print(f"Transcript length: {len(request.transcript)}")
        print(f"Instructions: {request.instructions}")

        # Initialize components
        llm_provider = LLMProvider(request.model)
        doc_processor = DocumentProcessor(UPLOAD_DIR)
        summarizer = MeetingSummarizer(llm_provider)

        # Process reference documents if provided
        context = ""
        if request.reference_docs:
            print(f"Processing {len(request.reference_docs)} reference documents")
            context = doc_processor.process_documents(request.reference_docs)

        # Generate summary
        print("Generating summary...")
        # Add model-specific handling
        max_tokens = 4000  # Default
        if request.model == "deepseek-chat":
            max_tokens = 8192  # DeepSeek supports longer outputs
                
        summary = summarizer.generate_summary(
            transcript=request.transcript,
            context=context,
            instructions=request.instructions,
            max_tokens=max_tokens  # Pass max_tokens here
        )
        print("Summary generated successfully")

        return {"summary": summary}
    except Exception as e:
        import traceback
        print(f"Error in summarize endpoint: {str(e)}")
        print("Full traceback:")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# Run the server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)