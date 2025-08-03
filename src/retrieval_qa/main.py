from .vector_store import VectorStoreManager
from .query_interface import QueryInterface
from .pdf_processor import PDFProcessor
from .config import PDF_DIR, DEFAULT_STORE_NAME
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)

def upload_pdfs_to_vector_store(
    pdf_dir: str = PDF_DIR,
    store_name: str = DEFAULT_STORE_NAME,
    validate_only: bool = False
) -> Dict:
    """Upload PDFs to OpenAI Vector Store"""
    try:
        pdf_processor = PDFProcessor(pdf_dir)
        vector_store_manager = VectorStoreManager()
        
        pdf_files = pdf_processor.get_pdf_files()
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        valid_files = [f for f in pdf_files if pdf_processor.validate_pdf_file(f)]
        logger.info(f"Valid PDF files: {len(valid_files)}")
        
        if not valid_files:
            logger.error("No valid PDF files found")
            return {"error": "No valid PDF files found"}
        
        if validate_only:
            return {
                "valid_files": len(valid_files),
                "total_files": len(pdf_files),
                "status": "validation_complete"
            }
        
        vector_store_details = vector_store_manager.create_vector_store(store_name)
        if not vector_store_details:
            logger.error("Failed to create vector store")
            return {"error": "Failed to create vector store"}
        
        upload_stats = vector_store_manager.upload_multiple_pdfs(
            vector_store_details["id"], 
            pdf_dir
        )
        
        upload_stats["vector_store"] = vector_store_details
        
        logger.info(f"Upload completed: {upload_stats['successful_uploads']} successful, "
                   f"{upload_stats['failed_uploads']} failed")
        
        return upload_stats
            
    except Exception as e:
        logger.error(f"Error in upload process: {e}")
        return {"error": str(e)}

def create_query_interface(vector_store_id: str, api_key: Optional[str] = None) -> QueryInterface:
    """Create a query interface for the vector store"""
    return QueryInterface(vector_store_id, api_key)