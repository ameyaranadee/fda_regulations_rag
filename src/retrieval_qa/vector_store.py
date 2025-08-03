import os
import concurrent
from tqdm import tqdm
from openai import OpenAI
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from .config import OPENAI_API_KEY, DEFAULT_STORE_NAME, MAX_WORKERS

class VectorStoreManager:
    def __init__(self, api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or OPENAI_API_KEY)

    def create_vector_store(self, store_name: str = DEFAULT_STORE_NAME) -> Dict:
        """Create a new vector store"""
        try:
            vector_store = self.client.vector_stores.create(name=store_name)
            details = {
                "id": vector_store.id,
                "name": vector_store.name,
                "created_at": vector_store.created_at,
                "file_count": vector_store.file_counts.completed
            }
            print("Vector store created: ", details)
            return details
        except Exception as e:
            print(f"Error creating vector store: {e}")
            return {}

    def upload_single_pdf(self, file_path: str, vector_store_id: str) -> Dict:
        """Upload a single PDF file to the vector store"""
        try:
            file_name = os.path.basename(file_path)
            try:
                file_response = self.client.files.create(
                    file=open(file_path, 'rb'),
                    purpose="retrieval"
                )
                attach_response = self.client.vector_stores.files.create(
                    vector_store_id=vector_store_id,
                    file_id=file_response.id
                )

                return {"file": file_name, "status": "success", "file_id": file_response.id}
        except Exception as e:
            print(f"Error with {file_name}: {str(e)}")
            return {"file": file_name, "status": "failed", "error": str(e)}

    def upload_multiple_pdfs(self, vector_store_id: str, pdf_dir: str) -> Dict:
        """Upload multiple PDFs to the vector store"""
        pdf_files = [os.path.join(pdf_dir,f) for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')]

        stats = {
            "total_files": len(pdf_files), 
            "successful_uploads": 0, 
            "failed_uploads": 0, 
            "errors": [],
            "file_ids": []
        }

        print(f"{len(pdf_files)} PDF files to process. Uploading in parallel...")

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(self.upload_single_pdf, file_path, vector_store_id): file_path for file_path in pdf_files
            }

            for future in tqdm(concurrent.future.as_completed(futures), total=len(pdf_files)):
                result = future.result()
                if result["status"] == "success":
                    stats["successful_uploads"] += 1
                    stats["file_ids"].append(result["file_id"])
                else:
                    stats["failed_uploads"] += 1
                    stats["errors"].append(result["error"])
                    
        return stats

    def search_vector_store(self, vector_store_id: str, query: str, top_k: int = 5) -> Dict:
        """
        Standalone vector search - find relevant content without LLM integration
        """
        try:
            search_results = self.client.vector_stores.search(
                vector_store_id=vector_store_id,
                query=query
            )
            
            results = []
            for result in search_results.data:
                result_info = {
                    "filename": result.filename,
                    "score": result.score,
                    "content_length": len(result.content[0].text),
                    "content_preview": result.content[0].text[:200] + "..." if len(result.content[0].text) > 200 else result.content[0].text
                }
                results.append(result_info)
            
            return {
                "query": query,
                "total_results": len(results),
                "results": results
            }
        except Exception as e:
            return {"error": str(e)}

    def query_with_llm(self, vector_store_id: str, query: str, 
                       model: str = "gpt-4o-mini") -> Dict:
        """
        Integrated search and LLM response using the Responses API
        """
        try:
            response = self.client.responses.create(
                input=query,
                model=model,
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": [vector_store_id],
                }]
            )
            
            # Extract the LLM response
            llm_response = response.output[1].content[0].text
            
            # Extract annotations to see which files were used
            annotations = response.output[1].content[0].annotations
            retrieved_files = set([result.filename for result in annotations])
            
            return {
                "query": query,
                "response": llm_response,
                "files_used": list(retrieved_files),
                "total_files_used": len(retrieved_files)
            }
        except Exception as e:
            return {"error": str(e)}

    def batch_search(self, vector_store_id: str, queries: List[str], 
                    use_llm: bool = False, model: str = "gpt-4o-mini") -> List[Dict]:
        """
        Perform batch search operations
        """
        results = []
        for query in queries:
            if use_llm:
                result = self.query_with_llm(vector_store_id, query, model)
            else:
                result = self.search_vector_store(vector_store_id, query)
            results.append(result)
        return results