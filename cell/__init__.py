"""Cell Runtime — multi-lane local AI orchestrator.

A cell is a self-contained runtime unit:
- Membrane (gateway): OpenAI-compatible API surface
- Organelles (lanes): specialized models (coder, security, reasoning)
- State (memory lane): rolling session context
- Metabolism (model pool): load/swap/serve models via llama-server
"""

__version__ = "0.1.0"
