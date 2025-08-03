import os
import PyPDF2
from typing import List, Dict
from .config import PDF_DIR, SUPPORTED_EXTENSIONS

class PDFProcessor:
    def __init__(self, pdf_dir: str = PDF_DIR):
        self.pdf_dir = pdf_dir
    
    def get_pdf_files(self) -> List[str]:
        """Get all PDF files from the configured directory"""
        if not os.path.exists(self.pdf_dir):
            raise FileNotFoundError(f"PDF directory not found: {self.pdf_dir}")
        
        pdf_files = []
        for file in os.listdir(self.pdf_dir):
            if any(file.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                pdf_files.append(os.path.join(self.pdf_dir, file))
        
        return pdf_files
    
    def validate_pdf_file(self, file_path: str) -> bool:
        """Validate if a file is a readable PDF"""
        try:
            with open(file_path, 'rb') as file:
                PyPDF2.PdfReader(file)
            return True
        except Exception:
            return False
    
    def get_pdf_info(self, file_path: str) -> Dict:
        """Extract basic information from a PDF file"""
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                return {
                    "filename": os.path.basename(file_path),
                    "pages": len(reader.pages),
                    "size_mb": os.path.getsize(file_path) / (1024 * 1024)
                }
        except Exception as e:
            return {"filename": os.path.basename(file_path), "error": str(e)}