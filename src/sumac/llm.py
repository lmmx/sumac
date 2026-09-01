"""Mistral.rs in-process agent runner.

This module initializes a mistral.rs model runner with tool callbacks that invoke
Sumac domain operations. The model is configured via environment or defaults.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from sumac import agent


# Model configuration (environment overrides defaults)
DEFAULT_MODEL_ID = "tinyllama"
DEFAULT_MODEL_PATH = "~/.cache/sumac-models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
DEFAULT_GPU_LAYERS = 0  # CPU-only by default


class AgentRunner:
    """Minimal wrapper around mistral.rs for Sumac.

    Handles model initialization, tool registration, and the agentic loop.
    """

    def __init__(self, data_dir: Path, key: bytes):
        """Initialize the agent runner.

        Args:
            data_dir: Sumac data directory
            key: Encryption key
        """
        self.data_dir = data_dir
        self.key = key

        # Import mistral here so it's only required if agent is actually used
        try:
            from mistralrs import Runner, Task
        except ImportError as e:
            raise ImportError(
                "mistral-rs not installed. Install with: pip install mistral-rs"
            ) from e

        self.Runner = Runner
        self.Task = Task

        # Get model path from environment or use default
        model_id = os.environ.get("SUMAC_MODEL_ID", DEFAULT_MODEL_ID)
        model_path = os.environ.get(
            "SUMAC_MODEL_PATH",
            os.path.expanduser(DEFAULT_MODEL_PATH),
        )
        gpu_layers = int(os.environ.get("SUMAC_GPU_LAYERS", DEFAULT_GPU_LAYERS))

        self.model_id = model_id
        self.model_path = model_path
        self.gpu_layers = gpu_layers

        # Check if model exists
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Download with: sumac model-download {model_id}"
            )

        # Initialize runner (deferred until first use to fail early if model missing)
        self._runner: Runner | None = None

    def _ensure_runner(self) -> Runner:
        """Lazily initialize the runner on first use."""
        if self._runner is None:
            self._runner = self.Runner(
                model_id=self.model_id,
                model_path=self.model_path,
                model_dtype="auto",  # Auto-detect based on model
                in_memory_files=[],  # No extra files
            )
            # Register tools
            self._register_tools()
        return self._runner

    def _register_tools(self) -> None:
        """Register tool callbacks with the runner."""
        runner = self._runner
        assert runner is not None

        # Tool definitions (schema understood by mistral.rs)
        tools_defs = [
            {
                "type": "function",
                "function": {
                    "name": "search_inventory",
                    "description": "Search for a product in inventory by name",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Product name or substring to search for",
                            }
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "consume",
                    "description": "Consume (use) a product from inventory",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_query": {
                                "type": "string",
                                "description": "Product name or substring to search for",
                            },
                            "amount": {
                                "type": "string",
                                "description": "Amount to consume (e.g. '1', '0.5')",
                            },
                        },
                        "required": ["product_query", "amount"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "move",
                    "description": "Move a product from one location to another",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "product_query": {
                                "type": "string",
                                "description": "Product name or substring to search for",
                            },
                            "to_location_query": {
                                "type": "string",
                                "description": "Target location name",
                            },
                            "amount": {
                                "type": "string",
                                "description": "Amount to move (optional; defaults to all)",
                            },
                        },
                        "required": ["product_query", "to_location_query"],
                    },
                },
            },
        ]

        # Register each tool with a callback
        for tool_def in tools_defs:
            runner.register_tool(
                tool_def,
                self._tool_callback,
            )

    def _tool_callback(self, tool_name: str, tool_input: dict) -> str:
        """Callback invoked by mistral.rs when the model calls a tool.

        Args:
            tool_name: Name of the tool being called
            tool_input: Arguments passed by the model

        Returns:
            String result to return to the model.
        """
        return agent.process_tool_call(tool_name, tool_input, data_dir=self.data_dir, key=self.key)

    def run(self, prompt: str) -> str:
        """Run the agent on a prompt.

        Args:
            prompt: User's natural language request

        Returns:
            Final response from the agent.
        """
        runner = self._ensure_runner()

        # Prepare messages for the model
        messages = [{"role": "user", "content": prompt}]

        # Run the agentic loop: model generates, tools execute, repeat until done
        max_iterations = 10  # Safety limit
        for iteration in range(max_iterations):
            # Get response from model
            response = runner.generate(messages)

            # Check if model is done (no tool calls)
            if not response.get("tool_calls"):
                # Model finished - return its message
                return response.get("content", "No response from model")

            # Process tool calls
            tool_results = []
            for tool_call in response.get("tool_calls", []):
                tool_name = tool_call.get("function", {}).get("name")
                tool_args = tool_call.get("function", {}).get("arguments", {})

                # Parse arguments if they're a JSON string
                if isinstance(tool_args, str):
                    try:
                        tool_args = json.loads(tool_args)
                    except json.JSONDecodeError:
                        tool_args = {}

                # Execute tool
                result = self._tool_callback(tool_name, tool_args)
                tool_results.append(
                    {
                        "tool_call_id": tool_call.get("id"),
                        "result": result,
                    }
                )

            # Add tool results back to messages
            messages.append({"role": "assistant", "content": response.get("content")})
            messages.append({"role": "user", "content": tool_results})

        return "Agent exceeded maximum iterations without completing the request"
