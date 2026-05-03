"""Graph agent — entity/evidence relationship queries.

Answers: "How are entities connected?"

Backend: FGIP evidence graph (~/fgip-engine/fgip.db)
  - nodes: 1905 entities
  - edges: 2378 relationships
  - claims: 32,900 evidence fragments
"""
import os
import sqlite3

from cell.agents.base import AgentBase, AgentResult, Permission

FGIP_DB = os.path.expanduser("~/fgip-engine/fgip.db")


class GraphLookupAgent(AgentBase):
    name = "graph_lookup"
    description = "Look up an entity in the evidence graph by name or ID. Returns node details."
    permission = Permission.READ
    timeout_s = 5
    input_schema = {
        "type": "object",
        "properties": {
            "entity": {"type": "string", "description": "Entity name or ID to look up"},
        },
        "required": ["entity"],
    }

    def execute(self, args: dict) -> AgentResult:
        entity = args["entity"]
        if not os.path.exists(FGIP_DB):
            return AgentResult(error="FGIP graph not available")

        conn = sqlite3.connect(f"file:{FGIP_DB}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            # Search by name (case-insensitive LIKE)
            cursor = conn.execute(
                "SELECT * FROM nodes WHERE name LIKE ? LIMIT 10",
                (f"%{entity}%",))
            raw = cursor.fetchall()
            nodes = [dict(row) for row in raw]

            if not nodes:
                return AgentResult(output={
                    "found": False,
                    "entity": entity,
                    "nodes": [],
                })

            return AgentResult(output={
                "found": True,
                "entity": entity,
                "nodes": nodes,
                "count": len(nodes),
            })
        finally:
            conn.close()


class GraphNeighborsAgent(AgentBase):
    name = "graph_neighbors"
    description = "Find entities connected to a given entity. Optionally filter by relation type."
    permission = Permission.READ
    timeout_s = 5
    input_schema = {
        "type": "object",
        "properties": {
            "entity_id": {"type": "string", "description": "Entity ID or name"},
            "relation_filter": {
                "type": "string",
                "description": "Filter by relation type (optional)",
            },
            "limit": {"type": "integer", "description": "Max results (default: 20)"},
        },
        "required": ["entity_id"],
    }

    def execute(self, args: dict) -> AgentResult:
        entity_id = args["entity_id"]
        relation_filter = args.get("relation_filter")
        limit = args.get("limit", 20)

        if not os.path.exists(FGIP_DB):
            return AgentResult(error="FGIP graph not available")

        conn = sqlite3.connect(f"file:{FGIP_DB}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        try:
            # Find edges where this entity is source or target
            if relation_filter:
                cursor = conn.execute("""
                    SELECT e.*, ns.name as source_name, nt.name as target_name
                    FROM edges e
                    LEFT JOIN nodes ns ON e.from_node_id = ns.node_id
                    LEFT JOIN nodes nt ON e.to_node_id = nt.node_id
                    WHERE (ns.name LIKE ? OR nt.name LIKE ?)
                      AND e.edge_type LIKE ?
                    LIMIT ?
                """, (f"%{entity_id}%", f"%{entity_id}%",
                      f"%{relation_filter}%", limit))
            else:
                cursor = conn.execute("""
                    SELECT e.*, ns.name as source_name, nt.name as target_name
                    FROM edges e
                    LEFT JOIN nodes ns ON e.from_node_id = ns.node_id
                    LEFT JOIN nodes nt ON e.to_node_id = nt.node_id
                    WHERE ns.name LIKE ? OR nt.name LIKE ?
                    LIMIT ?
                """, (f"%{entity_id}%", f"%{entity_id}%", limit))

            edges = [dict(row) for row in cursor]

            return AgentResult(output={
                "entity": entity_id,
                "neighbors": edges,
                "count": len(edges),
                "filtered_by": relation_filter,
            })
        except sqlite3.OperationalError as e:
            return AgentResult(error=f"Graph query error: {e}")
        finally:
            conn.close()


class GraphStatsAgent(AgentBase):
    name = "graph_stats"
    description = "Return graph statistics: node count, edge count, claim count."
    permission = Permission.READ
    timeout_s = 3
    input_schema = {"type": "object", "properties": {}}

    def execute(self, args: dict) -> AgentResult:
        if not os.path.exists(FGIP_DB):
            return AgentResult(error="FGIP graph not available")

        conn = sqlite3.connect(f"file:{FGIP_DB}?mode=ro", uri=True)
        try:
            counts = {}
            for table in ("nodes", "edges", "claims"):
                try:
                    counts[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                except sqlite3.OperationalError:
                    counts[table] = 0
            return AgentResult(output={
                "nodes": counts["nodes"],
                "edges": counts["edges"],
                "claims": counts["claims"],
                "db_path": FGIP_DB,
            })
        except sqlite3.OperationalError as e:
            return AgentResult(error=f"Graph stats error: {e}")
        finally:
            conn.close()
