Code Structure & Modularity Guidelines

Always write modular and reusable code.

Avoid creating large monolithic files (e.g., 400+ lines). Split logic into well-named modules.

Extract reusable logic into separate functions or utility modules.

Reuse existing utility functions whenever possible instead of duplicating logic.

If similar logic may be needed again, proactively create a reusable utility function.

Prefer small, single-responsibility functions.

Prioritize concise, readable, and maintainable implementations.

Organize files by responsibility (e.g., utils, experiments, models)

For experiments and models add only the core logic to the main files and keep the logging and plotting files in utils and reuse exisitng ones

When creating a new file for a given type (model) try to match the style of existig models and write the code so that it can reuse their existing functions (without distorting the logic)

Goal: maintain clean architecture through modularization and component reuse.

Today 7:30 PM •
753 chars • 96 words