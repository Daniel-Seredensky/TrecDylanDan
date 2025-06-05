import subprocess
import os
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

def reset_index():
    """
    Runs AzureJanitor.bash and passes in the absolute path to index_schema.json,
    which is assumed to live next to this .py file.
    """
    # Determine where this .py file lives
    base_dir = os.path.dirname(os.path.abspath(__file__))
    schema_path = os.path.join(base_dir, "index_schema.json")
    script_path = os.path.join(base_dir, "AzureJanitor.bash")

    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Could not find AzureJanitor.bash at {script_path}")
    if not os.path.isfile(schema_path):
        raise FileNotFoundError(f"Could not find index_schema.json at {schema_path}")

    # Invoke the bash script, passing schema_path as the first argument
    proc = subprocess.run(
        ["bash", script_path, schema_path],
        capture_output=True,
        text=True,
    )

    if proc.returncode != 0:
        err = proc.stderr.strip() or "Unknown error"
        raise RuntimeError(f"AzureJanitor.bash failed: {err}")

    print("AzureJanitor.bash ran successfully. Index has been reset.")
