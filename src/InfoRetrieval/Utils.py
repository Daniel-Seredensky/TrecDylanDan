import subprocess
from openai import AsyncOpenAI
from typing import List


def run_bm25_search(self, queries: list[str], path: str) -> None:
        """
        Helper method to run the Java BM25 search with subprocess
        
        Args:
            queries: List of query strings to search for
        """
        try:
            # Prepare the command
            java_cmd = [
                "java", 
                "-cp", ".:lib/*",  # Adjust classpath as needed
                "src.InfoRetrieval.Search"
            ] + queries + [path]
            
            # Run the Java search
            result = subprocess.run(
                java_cmd,
                capture_output=False,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"Java search failed: {result.stderr}")
                
            print("Proctor: BM25 search completed successfully")
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("BM25 search timed out after 5 minutes")
        except Exception as e:
            raise RuntimeError(f"Failed to execute BM25 search: {e}")
        

async def get_embedding(client: AsyncOpenAI, text: str, model: str) -> List[float]:
        """Fetch an embedding vector from the OpenAI Embeddings API."""
        response = client.embeddings.create(
            model= model,
            input=text,
        )
        return response.data[0].embedding