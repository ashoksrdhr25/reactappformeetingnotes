# document_processor.py
from pathlib import Path
from typing import List
import pypdf
import docx2txt

class DocumentProcessor:
    """
    Handles the processing of different document types (PDF, DOCX, TXT) and combines their content.
    This class is responsible for extracting text from various document formats that can be used
    as reference material for meeting summaries.
    """
    
    def __init__(self, upload_dir: Path):
        """
        Initialize the DocumentProcessor with a directory path where uploaded files are stored.
        
        Args:
            upload_dir (Path): Directory path where uploaded documents are stored
        """
        self.upload_dir = upload_dir

    def process_documents(self, filenames: List[str]) -> str:
        """
        Process a list of documents and combine their text content.
        
        Args:
            filenames (List[str]): List of filenames to process
            
        Returns:
            str: Combined text from all processed documents
        """
        combined_text = []
        
        for filename in filenames:
            file_path = self.upload_dir / filename
            
            if not file_path.exists():
                continue
                
            # Process based on file type
            if file_path.suffix.lower() == '.pdf':
                text = self._process_pdf(file_path)
            elif file_path.suffix.lower() in ['.doc', '.docx']:
                text = self._process_docx(file_path)
            elif file_path.suffix.lower() == '.txt':
                text = self._process_txt(file_path)
            else:
                continue
                
            combined_text.append(text)
        
        return "\n\n".join(combined_text)

    def _process_pdf(self, file_path: Path) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            file_path (Path): Path to the PDF file
            
        Returns:
            str: Extracted text content
        """
        text = []
        with open(file_path, 'rb') as file:
            pdf = pypdf.PdfReader(file)
            for page in pdf.pages:
                text.append(page.extract_text())
        return "\n".join(text)

    def _process_docx(self, file_path: Path) -> str:
        """
        Extract text from a DOCX file.
        
        Args:
            file_path (Path): Path to the DOCX file
            
        Returns:
            str: Extracted text content
        """
        return docx2txt.process(file_path)

    def _process_txt(self, file_path: Path) -> str:
        """
        Read text from a TXT file.
        
        Args:
            file_path (Path): Path to the TXT file
            
        Returns:
            str: File content
        """
        return file_path.read_text()