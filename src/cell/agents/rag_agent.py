"""RAG agent — document/policy/receipt retrieval.

Answers: "What do documents, logs, policies, tickets, manuals, repos say?"

Backends:
  - FGIP graph FTS5 (primary — 127MB, 32K claims indexed)
  - File search (fallback — grep over home directory)
"""
import os
import re
import sqlite3
from pathlib import Path

from cell.agents.base import AgentBase, AgentResult, Permission

FGIP_DB = os.path.expanduser("~/fgip-engine/fgip.db")


class RAGLookupAgent(AgentBase):
    name = "rag_lookup"
    description = "Search documents, policies, and receipts by keyword. Returns matching excerpts."
    permission = Permission.READ
    timeout_s = 10
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "scope": {
                "type": "string",
                "description": "Scope: claims, nodes, all (default: claims)",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default: 10)",
            },
        },
        "required": ["query"],
    }
    output_schema = {
        "type": "object",
        "properties": {
            "results": {"type": "array"},
            "count": {"type": "integer"},
            "source": {"type": "string"},
        },
    }

    def execute(self, args: dict) -> AgentResult:
        query = args["query"]
        scope = args.get("scope", "claims")
        limit = args.get("limit", 10)

        # Try FGIP FTS5 first
        if os.path.exists(FGIP_DB):
            try:
                results = self._fts_search(query, scope, limit)
                return AgentResult(output={
                    "results": results,
                    "count": len(results),
                    "source": "fgip.db FTS5",
                })
            except Exception as e:
                pass

        # Fallback: file grep
        results = self._file_search(query, limit)
        return AgentResult(output={
            "results": results,
            "count": len(results),
            "source": "file_search",
        })

    def _fts_search(self, query: str, scope: str, limit: int) -> list[dict]:
        """Search FGIP database via FTS5."""
        conn = sqlite3.connect(f"file:{FGIP_DB}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        results = []

        try:
            if scope in ("claims", "all"):
                cursor = conn.execute(
                    "SELECT rowid, * FROM claims_fts WHERE claims_fts MATCH ? LIMIT ?",
                    (query, limit))
                for row in cursor:
                    d = dict(row)
                    text = d.get("claim_text", "") or d.get("text", "")
                    results.append({
                        "type": "claim",
                        "text": text[:300],
                        "rowid": row["rowid"],
                    })

            if scope in ("nodes", "all") and len(results) < limit:
                remaining = limit - len(results)
                cursor = conn.execute(
                    "SELECT rowid, * FROM nodes_fts WHERE nodes_fts MATCH ? LIMIT ?",
                    (query, remaining))
                for row in cursor:
                    results.append({
                        "type": "node",
                        "text": dict(row).get("name", "")[:300],
                        "rowid": row["rowid"],
                    })
        except sqlite3.OperationalError:
            # FTS table might not exist or query syntax error
            # Fall through to return whatever we have
            pass
        finally:
            conn.close()

        return results

    def _file_search(self, query: str, limit: int) -> list[dict]:
        """Fallback: search markdown/json files by keyword."""
        import subprocess
        home = os.path.expanduser("~")
        results = []
        try:
            proc = subprocess.run(
                ["grep", "-rn", "--include=*.md", "--include=*.json",
                 "-l", query, home],
                capture_output=True, text=True, timeout=5,
            )
            files = [f for f in proc.stdout.strip().split("\n") if f][:limit]
            for f in files:
                results.append({"type": "file", "path": f})
        except Exception:
            pass
        return results


class RAGSearchAgent(AgentBase):
    name = "rag_search"
    description = "Search receipts directory for matching receipt files."
    permission = Permission.READ
    timeout_s = 5
    input_schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Receipt ID or keyword"},
        },
        "required": ["query"],
    }

    def execute(self, args: dict) -> AgentResult:
        query = args["query"]
        receipt_dir = Path(os.path.expanduser("~/receipts"))
        results = []
        if receipt_dir.exists():
            for f in sorted(receipt_dir.rglob("*.json"))[:100]:
                if re.search(query, f.name, re.IGNORECASE):
                    results.append({
                        "path": str(f),
                        "name": f.name,
                        "size": f.stat().st_size,
                    })
        return AgentResult(output={
            "results": results[:20],
            "count": len(results),
        })
