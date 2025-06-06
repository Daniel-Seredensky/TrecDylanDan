"""
    Uility functions for the project
    Some are wrappers for curl scripts, some are wrappers for OpenAI calls
"""

import subprocess
import os
from openai import AsyncOpenAI
from typing import List
from pathlib import Path
import asyncio

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
        response = await client.embeddings.create(
            model= model,
            input=text,
        )
        return response.data[0].embedding

async def reset_index():
    """
    Asynchronously runs AzureJanitor.bash and passes in the absolute path to index_schema.json,
    which is assumed to live next to this .py file.
    """
    # Determine where this .py file lives
    base_dir = os.path.dirname(os.path.abspath(__file__))
    schema_path = os.path.join(base_dir, "index_schema.json")
    script_path = os.path.join(base_dir, "BashUtils/AzureJanitor.bash")
    
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Could not find AzureJanitor.bash at {script_path}")
    
    if not os.path.isfile(schema_path):
        raise FileNotFoundError(f"Could not find index_schema.json at {schema_path}")
    
    # Invoke the bash script asynchronously, passing schema_path as the first argument
    proc = await asyncio.create_subprocess_exec(
        "bash", script_path, schema_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        text=True
    )
    
    stdout, stderr = await proc.communicate()
    
    if proc.returncode != 0:
        err = stderr.strip() or "Unknown error"
        raise RuntimeError(f"AzureJanitor.bash failed: {err}")
    
    return stdout.strip() if stdout else None

async def _expand_query(query: str) -> List[str]:
    """
    Asynchronously return the synonym-expanded token list for `query`.
    """
    SCRIPT = Path(__file__).with_name("BashUtils/ExpandQuery.sh")  # assumes script is alongside .py
    
    try:
        proc = await asyncio.create_subprocess_exec(
            "bash", str(SCRIPT), query,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            text=True
        )
        
        stdout, stderr = await proc.communicate()
        
        if proc.returncode != 0:
            err = stderr.strip() or "Unknown error"
            raise RuntimeError(f"ExpandQuery.sh failed with return code {proc.returncode}: {err}")
        
        # Each line is one token
        return [tok for tok in stdout.splitlines() if tok]
        
    except FileNotFoundError:
        raise FileNotFoundError(f"Could not find ExpandQuery.sh at {SCRIPT}")
    except Exception as e:
        raise RuntimeError(f"Failed to expand query '{query}': {e}")
        
async def expand_queries(queries: List[str]) -> List[List[str]]:
    """Expand multiple queries concurrently"""
    tasks = [_expand_query(query) for query in queries]
    return await asyncio.gather(*tasks)
