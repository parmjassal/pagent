import os
import fnmatch
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

# Assuming these relative imports are structured correctly in your project
from ..storage.semantic_search_v2 import SemanticSearchEngine
from ..orch.state import AgentState

logger = logging.getLogger(__name__)

# Pre-defined noise patterns to ignore
DEFAULT_IGNORE_GLOBS = [
    "*.md", "*.pack", "**/*.md", "**/*.pack", "**/*.rst", "**/*.generated.*", 
    "**/*_generated.*", "**/*.gen.*", "**/*.pb.go", "**/*.pb.cc", "**/*.pb.h", 
    "**/*.pb.*.h", "**/*.pb.*.cc", "**/*.g.dart", "**/*.designer.cs", "**/*.min.js", 
    "**/*.bundle.js", "**/*.chunk.js", "**/*.map", "requirements.txt", "pipfile", 
    "pipfile.lock", "poetry.lock", "package.json", "package-lock.json", "yarn.lock", 
    "pnpm-lock.yaml", "tsconfig.json", "pom.xml", "build.gradle", "settings.gradle", 
    "gradlew", "gradlew.bat", "CMakeLists.txt", "Makefile", "Cargo.toml", "Cargo.lock", 
    "**/vendor/**", "**/third_party/**", "**/extern/**", "**/deps/**", "**/*.png", 
    "**/*.jpg", "**/*.jpeg", "**/*.gif", "**/*.svg", "**/*.ico", "**/*.pdf", 
    "**/*.zip", "**/*.tar", "**/*.gz", "**/*.mp4", "**/*.mp3", "**/.idea/**", 
    "**/.vscode/**", "**/*.iml", "**/__snapshots__/**", "**/fixtures/**", 
    "**/mocks/**", "**/migrations/**", "**/alembic/**", "**/db/migrate/**", 
    "**/coverage/**", "**/.nyc_output/**", "**/reports/**", "**/.cache/**", 
    "**/.pytest_cache/**", "**/.mypy_cache**", "**/*.log", "**/*.out", "**/*.err", 
    "**/*.pid", "**/*.seed", ".eslintrc", ".eslintrc.json", ".eslintrc.js", 
    ".prettierrc", ".flake8", ".pylintrc", ".gitignore", ".gitattributes", "*.lock", "*.env"
]

class SearchTools:
    def __init__(self, session_path: Path):
        self.session_path = session_path
        self.index_root = session_path / "semantic_indexes"
        self.index_root.mkdir(parents=True, exist_ok=True)

    def _should_ignore(self, path_str: str, patterns: List[str]) -> bool:
        """Helper to match a path against a list of glob patterns."""
        for pattern in patterns:
            # Handle both simple filenames and path-based globs
            if fnmatch.fnmatch(path_str, pattern) or fnmatch.fnmatch(os.path.basename(path_str), pattern):
                return True
        return False

    def build_all_indexes(
        self, 
        repo_path: str,
        index_to_keywords: Dict[str, List[str]], 
        negative_globs: Optional[List[str]] = None,
        state: Optional[AgentState] = None
    ) -> str:
        # 1. Merge provided globs with defaults and deduplicate
        combined_negatives = list(set(DEFAULT_IGNORE_GLOBS + (negative_globs or [])))
        
        engines = {}
        for name in index_to_keywords.keys():
            domain_slug = name.lower().replace(" ", "_")
            domain_path = self.index_root / domain_slug
            domain_path.mkdir(parents=True, exist_ok=True)
            engines[name] = SemanticSearchEngine(domain_path)
            
        indexed_count = 0
        repo_root = Path(repo_path).resolve()

        # 2. Single-Pass Repository Walk
        for root, dirs, files in os.walk(repo_path):
            # Convert current root to a relative path for pattern matching
            rel_root = os.path.relpath(root, repo_path)
            
            # --- PRUNING DIRECTORIES ---
            # Modify dirs in-place to skip ignored folders (e.g., .git, node_modules)
            # We check the directory name against patterns
            dirs[:] = [d for d in dirs if not self._should_ignore(os.path.join(rel_root, d), combined_negatives)]
            
            for file in files:
                rel_file_path = os.path.join(rel_root, file)
                
                # --- FILTERING FILES ---
                if self._should_ignore(rel_file_path, combined_negatives):
                    continue

                file_path = Path(root) / file
                try:
                    # Use a binary check or small read to avoid loading massive blobs into memory
                    # read_text(errors='ignore') is okay for most code, but fails gracefully here
                    content = file_path.read_text(errors='ignore')
                    lines = content.splitlines()
                    if not lines:
                        continue

                    chunk_size = 100
                    for i in range(0, len(lines), chunk_size):
                        chunk_content = "\n".join(lines[i : i + chunk_size])
                        
                        for domain_name, keywords in index_to_keywords.items():
                            # Case-insensitive density check
                            if any(k.lower() in chunk_content.lower() for k in keywords):
                                engines[domain_name].index_chunk(
                                    file_path=str(file_path), 
                                    content=chunk_content, 
                                    start_line=i + 1, 
                                    boost_keywords=keywords
                                )
                    
                    indexed_count += 1
                except Exception as e:
                    logger.error(f"Failed indexing for {file_path}: {e}")

        # 3. Flush to disk
        for engine in engines.values():
            if hasattr(engine, '_save_index'):
                engine._save_index()

        return f"Indexing Complete. Processed {indexed_count} files into {len(engines)} domains."

    def semantic_search(
        self, 
        query: str, 
        index_name: str, 
        top_k: int = 20, 
        state: Optional[AgentState] = None
    ) -> str:
        """
        Performs a semantic query against a specific domain index.
        """
        domain_path = self.index_root / index_name.lower().replace(" ", "_")
        
        if not domain_path.exists():
            return f"Error: Index '{index_name}' does not exist at {domain_path}."
        
        try:
            # Load the engine for the specific domain
            engine = SemanticSearchEngine.load(domain_path)
            results = engine.query(query, top_k=top_k)
            
            if not results:
                return f"No matching documents found in the '{index_name}' index."
                
            formatted = [f"## Results from {index_name} Index:"]
            for r in results:
                meta = r['metadata']
                snippet = meta.get('content', 'Snippet not available')
                
                formatted.append(
                    f"### {meta.get('path', 'Unknown Path')} "
                    f"(Lines: {meta.get('start_line', '?')}-{meta.get('end_line', '?')}) "
                    f"[Score: {r.get('score', 0.0):.2f}]\n{snippet}\n---"
                )
            
            return "\n".join(formatted)
        except Exception as e:
            logger.error(f"Search failed for index '{index_name}': {e}")
            return f"Search failed for index '{index_name}': {str(e)}"
        
    def list_documents(
        self, 
        index_name: str, 
        state: Optional[AgentState] = None
    ) -> str:
        """
        Lists all unique files currently stored within a specific semantic index.
        """
        # 1. Resolve domain path (matches logic in semantic_search)
        domain_slug = index_name.lower().replace(" ", "_")
        domain_path = self.index_root / domain_slug
        
        if not domain_path.exists():
            return f"Error: Index '{index_name}' does not exist at {domain_path}."
        
        try:
            # 2. Load the engine for the specific domain
            engine = SemanticSearchEngine.load(domain_path)
            files = engine.list_unique_files()
            
            if not files:
                return f"The index '{index_name}' is initialized but contains no documents."
                
            # 3. Format the output for the agent/user
            header = f"## Files indexed in '{index_name}':"
            file_list = "\n".join([f"- {f}" for f in files])
            
            return f"{header}\n{file_list}\n\nTotal: {len(files)} files."
            
        except Exception as e:
            logger.error(f"Failed to list documents for index '{index_name}': {e}")
            return f"Error accessing index '{index_name}': {str(e)}"