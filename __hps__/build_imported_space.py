"""
Reconstruct per-model HP search spaces from imported tuned configs.

Scans __hps__/imported/{model}/{dataset}/{src_tgt}/imported.yaml,
collects unique values per HP across all scenarios, and writes
__hps__/imported_space/{model}.yaml in the same format as __hps__/space/*.yaml.
"""

import os
import yaml
from collections import defaultdict
from pathlib import Path

IMPORTED_DIR = Path(__file__).parent / "imported"
OUTPUT_DIR = Path(__file__).parent / "imported_space"


def collect_spaces():
    # model -> hp_name -> set of values
    model_spaces = defaultdict(lambda: defaultdict(set))

    for model_dir in sorted(IMPORTED_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        model_name = model_dir.name

        for yaml_path in model_dir.rglob("imported.yaml"):
            with open(yaml_path) as f:
                config = yaml.safe_load(f)
            if not config:
                continue
            for key, val in config.items():
                model_spaces[model_name][key].add(val)

    return model_spaces


def format_value(v):
    """Keep numeric types; avoid quoting booleans/numbers in YAML output."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return v
    return v


def write_spaces(model_spaces):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_name, hp_dict in sorted(model_spaces.items()):
        space = {}
        for hp_name, values in hp_dict.items():
            sorted_vals = sorted(format_value(v) for v in values)
            space[hp_name] = sorted_vals

        out_path = OUTPUT_DIR / f"{model_name}.yaml"
        with open(out_path, "w") as f:
            f.write("model:\n")
            for hp_name, vals in space.items():
                repr_vals = []
                for v in vals:
                    if isinstance(v, bool):
                        repr_vals.append(str(v).lower())
                    elif isinstance(v, float):
                        repr_vals.append(str(v))
                    else:
                        repr_vals.append(str(v))
                f.write(f"  {hp_name}: [{', '.join(repr_vals)}]\n")

        print(f"  wrote {out_path.relative_to(IMPORTED_DIR.parent)}")


if __name__ == "__main__":
    spaces = collect_spaces()
    print(f"Found {len(spaces)} models")
    write_spaces(spaces)
    print("Done.")
