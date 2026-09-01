# Sumac AI Agent

This document describes the minimal in-process AI agent for Sumac that translates natural language into inventory operations.

## Quick Start

### 1. Install mistral.rs

```bash
pip install mistral-rs
```

### 2. Download a model

Download a small (~1-2 GB), quantized GGUF model. TinyLlama 1.1B is a good starting point:

```bash
mkdir -p ~/.cache/sumac-models
cd ~/.cache/sumac-models
wget https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf
```

### 3. Use the agent

```bash
sumac ask "where is the jam?"
sumac ask "consume 1 jar of jam"
sumac ask "move the ragu from freezer to fridge"
```

## Configuration

### Model Selection

By default, the agent uses:
- **Model ID**: `tinyllama`
- **Model path**: `~/.cache/sumac-models/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf`
- **GPU layers**: 0 (CPU-only)

Override via environment variables:

```bash
export SUMAC_MODEL_ID="mistral"
export SUMAC_MODEL_PATH="~/.cache/sumac-models/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
export SUMAC_GPU_LAYERS=20  # Use 20 layers on GPU (requires CUDA/Metal)

sumac ask "where is the milk?"
```

### Supported Models

Any GGUF model that supports:
- Function calling / tool use
- Instruct/chat format
- Reasonable instruction-following

**Recommended for CPU-only:**
- TinyLlama 1.1B (Q4_K_M quantization) — ~600 MB, fast
- Phi 2.7B (Q4_K_M) — ~1.6 GB, higher quality (marginal on CPU)

**If you have GPU:**
- Mistral 7B (Q4_K_M) — ~4 GB, high quality
- Neural Chat 3B (Q4_K_M) — ~1.8 GB, balanced

All are available on [HuggingFace](https://huggingface.co/models?sort=trending&search=GGUF).

## Architecture

### Design Principles

- **In-process**: No HTTP server, no separate daemon, no API keys.
- **Small**: ~400 lines of code (agent + LLM modules).
- **Minimal**: No LangChain, CrewAI, AutoGen, or MCP.
- **Safe**: Model can only call three explicit tools; all operations go through Sumac's validation.
- **Idiomatic**: Uses existing domain operations (consume, move, search).

### How It Works

1. **User**: `sumac ask "move the ragu to the fridge"`
2. **CLI** (`cli.py`): Loads vault, key, and initializes `AgentRunner`
3. **Agent Runner** (`llm.py`): Initializes mistral.rs with tool definitions
4. **Model**: Reads the prompt, decides which tools to call
5. **Tool callbacks** (`agent.py`): 
   - Load current inventory and config
   - Validate the operation via `decide.py` (same gate as CLI)
   - Apply writes to storage
   - Return result to model
6. **Model**: Generates a natural-language response
7. **CLI**: Display result to user

### Tool Functions

The model can call three tools:

#### `search_inventory(query: str) -> str`

Find a product by name (substring match).

Returns:
- Product name and all locations holding it (with quantities)
- If not found: suggestions or error
- If ambiguous: asks user to be more specific

**Example**: `"where is the jam?"` → searches for "jam"

#### `consume(product_query: str, amount: str) -> str`

Record consumption of a product from its location.

Constraints:
- Requires exactly one location (if multiple, asks which one)
- Amount is a string decimal (e.g., "1", "0.5")

**Example**: `"use 1 jar of jam"` → searches for "jam", consumes 1 from wherever it is

#### `move(product_query: str, to_location_query: str, amount: str | None) -> str`

Move a product between locations.

Constraints:
- Requires exactly one source location (if multiple, asks which one)
- Requires unambiguous target location (if multiple, asks which one)
- If `amount` is None, moves all of it

**Example**: `"move the ragu to the fridge shelf 2"` → searches for "ragu", target location "fridge shelf 2"

## Behavior

### What the Agent Does

- **Searches before acting**: Calls `search_inventory` first to find the product
- **Handles ambiguity gracefully**: Asks the user rather than guessing
- **Respects domain validation**: All operations go through `decide.py` (same validation as CLI)
- **Gives clear feedback**: Returns what was done, or why it couldn't

### What the Agent Cannot Do

- Directly access the database or filesystem
- Invent products, locations, or quantities
- Execute arbitrary Python or shell commands
- Modify the ledger without going through domain operations
- Create new locations or products (only search existing ones)

### Example Conversations

```
> sumac ask "where is the milk?"
✓ Milk: 2 l at Fridge, 0.5 l at Freezer

> sumac ask "consume 1 l of milk from the fridge"
✓ Consumed 1 l milk from fridge

> sumac ask "move the jam to the pantry"
✓ Moved 1.5 jar jam from fridge to pantry

> sumac ask "move the ragu"
Error: Ambiguous: 2 products match "ragu": bolognese ragu (ragu-bolognese), pesto ragu (ragu-pesto). Please be more specific.

> sumac ask "move the bolognese to the freezer"
✓ Moved 1 portion bolognese ragu from pantry to freezer
```

## Limitations & Future Work

### MVP Limitations

- **Tool calling**: Only three tools (search, consume, move)
- **Location ambiguity**: Model must be explicit; can't infer "fridge shelf 2" without exact name match
- **Quantity units**: Model sees and uses canonical units; no automatic unit conversion in prompts
- **No memory**: Each `ask` starts fresh; no conversation history
- **No REPL**: No interactive agent loop (only one-shot `ask` command)

### Known Issues

- **Large models on CPU**: A 7B model takes several seconds per turn
- **Quantization**: Lower quantization (Q4_K_M) trades quality for speed; consider Q6_K for higher quality
- **Model choice matters**: Tool calling works best with models fine-tuned for it (Mistral, Phi, etc.)

### Possible Extensions

- Add an interactive REPL mode (`sumac` → agent loop)
- Expose more tools (snapshot locations, add products, etc.)
- Integrate with real calendars/recipes for smart meal planning
- Multi-turn conversation with memory
- Voice input/output
- Embeddings for semantic search (e.g., "anything citrus-flavored?")

## Troubleshooting

### Model not found
```
FileNotFoundError: Model not found at ~/.cache/sumac-models/...
```
Download the model, or set `SUMAC_MODEL_PATH` to where you have it.

### Out of memory
Your model is too large for your hardware. Try:
- Smaller model (TinyLlama 1.1B instead of Mistral 7B)
- Higher quantization (Q2_K instead of Q4_K_M)
- Enable GPU layers with `SUMAC_GPU_LAYERS=20`

### Tool calls not working
Your model may not support function calling well. Try:
- Mistral, Phi, TinyLlama, Neural Chat (all tested)
- Avoid generic models (e.g., StableLM); they don't do tool calling
- Check model's own documentation for tool-calling examples

### Slow responses
- Smaller models are faster (but lower quality)
- Use quantized models (Q4_K_M or higher)
- Enable GPU if available (`SUMAC_GPU_LAYERS`)
- Pre-load model into RAM if possible

## Implementation Notes

### Code Organization

- `src/sumac/agent.py`: Tool implementations and orchestration (no I/O, no model)
- `src/sumac/llm.py`: mistral.rs integration (runner, tool registration, agentic loop)
- `src/sumac/cli.py`: `ask` command entry point
- `tests/test_agent.py`: Comprehensive tests for agent logic

### Why This Architecture?

1. **Separation of concerns**:
   - Domain logic (agent.py) is testable without the model
   - LLM logic (llm.py) is separate and swappable
   - CLI (cli.py) is thin and straightforward

2. **Testability**:
   - All tool functions take inventory/config as inputs, not I/O
   - No mocking needed; tests use real domain operations
   - Agent tests run in ~1s; don't require model

3. **Maintainability**:
   - Tool functions are pure(ish) — map inputs → outputs
   - Tool definitions live near implementations
   - Adding new tools requires touching only two places (agent.py, llm.py)

### Mistral.rs Details

mistral.rs is a Rust library for LLM inference with Python bindings. Key features:
- CPU and GPU support
- Quantization-aware (GGUF, GGML formats)
- Tool calling via function callbacks
- No separate server needed

The Python SDK (`mistral-rs` on PyPI) exposes:
- `Runner`: Model runner that handles tool callbacks
- Tool registration with function definitions (JSON Schema)
- `generate()` method that returns model output with tool calls

See [mistral.rs Python docs](https://docs.mistralrs.dev/examples/python/agentic-tools/) for the full API.

## Performance

### Latency (on 2023 MacBook Pro, Apple Silicon, CPU-only)

| Model | Quantization | Typical Time |
|-------|--------------|--------------|
| TinyLlama 1.1B | Q4_K_M | 1-2s |
| Phi 2.7B | Q4_K_M | 3-5s |
| Mistral 7B | Q4_K_M | 10-15s |

(First invocation loads model into RAM; subsequent invocations are faster.)

### Memory Usage

| Model | Quantization | RAM |
|-------|--------------|-----|
| TinyLlama 1.1B | Q4_K_M | ~600 MB |
| Phi 2.7B | Q4_K_M | ~1.8 GB |
| Mistral 7B | Q4_K_M | ~4-5 GB |

## Related

- **Domain model**: See `src/sumac/models.py`, `src/sumac/events.py`
- **Validation**: See `src/sumac/decide.py` (where operations are validated)
- **Storage**: See `src/sumac/store.py`, `src/sumac/ledger.py` (append-only encrypted log)
- **CLI**: See `src/sumac/cli.py` (other commands like `add`, `move`, `find`)
