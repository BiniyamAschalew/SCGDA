from pathlib import Path
import sys
import types


def _bootstrap_repo_alias() -> None:
    repo_root = Path(__file__).resolve().parent
    learn_root = repo_root.parent
    learn_pkg = sys.modules.setdefault("Learn", types.ModuleType("Learn"))
    learn_pkg.__path__ = [str(learn_root)]

    clean_pkg = sys.modules.get("Learn.Clean_SCGDA")
    if clean_pkg is None:
        clean_pkg = types.ModuleType("Learn.Clean_SCGDA")
        sys.modules["Learn.Clean_SCGDA"] = clean_pkg
        setattr(learn_pkg, "Clean_SCGDA", clean_pkg)
    clean_pkg.__path__ = [str(repo_root)]


_bootstrap_repo_alias()

from  experiments.analysis.ugda_target_predictors.experiment import parse_args, run_experiment


if __name__ == "__main__":
    run_experiment(parse_args())
