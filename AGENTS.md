# AGENTS.md

## Cursor Cloud specific instructions

### Overview
FLAMO is a Python library for frequency-domain differentiable audio processing built on PyTorch. It is a standalone scientific/ML package with no web services, databases, or containers.

### Running the library
- Install in editable mode: `pip install -e .` from the repo root.
- Requires Python >= 3.10 and `libsndfile1` (system package, pre-installed on Ubuntu).
- No GPU required; CPU-only PyTorch works fine for all examples.

### Testing
- There is no formal test suite (no pytest, unittest, etc.) and no linter/formatter configured.
- Validation is done by running example scripts in `examples/`. Use `MPLBACKEND=Agg` to avoid display-related errors in headless environments.
- The `Trainer` class requires the `train_dir` directory to exist before instantiation; create it with `mkdir -p` first.

### Running examples
- Individual examples: `MPLBACKEND=Agg python examples/e0_siso.py`
- All examples: `MPLBACKEND=Agg python examples/run_all.py`
- Examples default to `--device cuda` and fall back to `cpu` if CUDA is unavailable.

### Known gotchas
- `flamo/auxiliary/minimize.py` emits a `SyntaxWarning: invalid escape sequence '\s'` on import — this is harmless and present in the upstream code.
- The `flamo` package does not expose `__version__`; do not try to access `flamo.__version__`.
