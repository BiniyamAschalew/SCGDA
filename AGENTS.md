# AGENTS.md

## Purpose
This repository prefers minimal, readable, research-oriented code that matches the existing implementation style.
When making changes, prioritize reuse of existing modules, functions, and utilities over introducing new abstractions.

## Core coding rules
- Match the repository's existing code style, naming, structure, and formatting.
- Prefer concise and modular code.
- Do not introduce industry-style defensive programming unless explicitly requested.
- Avoid unnecessary validation, excessive assertions, redundant error handling, and generic safety checks.
- Assume the intended research environment is already correctly set up unless the surrounding code clearly requires otherwise.
- Do not add checks like device-availability checks, excessive type guards, or configuration validation unless they are necessary for correctness in this repository.
- Keep implementations short, direct, and easy to read.
- Prefer small helper functions over large monolithic blocks, but do not over-fragment the code.
- Reuse existing utilities, models, training code, and helper functions whenever possible.
- Before creating a new function or class, search the repository for an existing one that already fits or can be slightly extended.

## Reuse-first policy
Before writing new code:
1. Inspect relevant directories and existing modules.
2. Identify existing models, layers, utils, training helpers, and evaluation functions.
3. Reuse them whenever possible.
4. If a new implementation is necessary, make it consistent with nearby files.

## Repository understanding requirement
When starting work on a task:
- First build a lightweight mental map of the relevant code.
- Look for existing implementations of similar models, utilities, datasets, training loops, evaluation code, and plotting helpers.
- Prefer adapting existing code over rewriting from scratch.

## Simplicity preference
This repository values:
- concise code
- clear logic
- minimal boilerplate
- minimal indirection
- minimal unnecessary abstraction

Avoid:
- over-engineered wrappers
- excessive config layers
- unnecessary generic utilities
- verbose test-like assertions inside training/research code
- enterprise-style robustness patterns unless explicitly requested

## Testing philosophy
- Do not write excessive tests or defensive checks.
- Prefer minimal verification only where it is directly useful.
- Do not add many asserts unless they protect an important invariant already relied on by the repository.
- Keep debugging and validation lightweight.

## Modification policy
- Preserve existing APIs unless the task requires changing them.
- Preserve directory structure unless there is a strong reason not to.
- Do not duplicate logic that already exists elsewhere.
- If extending code, do so in the most local and natural place.

## File-level memory / code map
Maintain a repository memory file at:

`__docs__/code_map.md`

This file should be updated whenever substantial repository understanding is built or new major code is added.

The file should contain concise summaries of:
- important models
- important utility functions
- training/evaluation entry points
- dataset loaders
- reusable helpers
- directory locations

Each entry should include:
- path
- object/function/class name
- short purpose
- notes on when it should be reused

Example format:

- `models/gnn.py`
  - `GNNModel`: main GNN backbone used for ...
  - Reuse when implementing new message-passing baselines.

- `utils/train.py`
  - `train_epoch(...)`: shared epoch-level training logic
  - Reuse instead of creating a new training loop.

## When adding new code
If you add a new reusable component:
- place it in the most natural existing directory
- update `__docs__/code_map.md`
- briefly note what existing code it relates to or replaces

## Response / work style
When completing a coding task:
- briefly mention which existing files/functions were reused
- keep edits targeted
- avoid unrelated refactors
- do not introduce new patterns unless necessary

## Priority order
1. Reuse existing code
2. Match repository style
3. Keep code concise and modular
4. Avoid unnecessary checks/validation
5. Update `__docs__/code_map.md` when useful