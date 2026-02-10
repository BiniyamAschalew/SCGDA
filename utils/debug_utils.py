import pickle
import types

def save_state(scope, file_path = "__saved__/debug_state.pkl"):

    # Filter out unpickleable objects: modules, functions, methods, classes
    def is_pickleable(k, v):
        if k.startswith("__"):
            return False
        if isinstance(v, (types.ModuleType, types.FunctionType, types.MethodType, type)):
            return False
        # Try to pickle to catch any other unpickleable objects
        try:
            pickle.dumps(v)
            return True
        except (pickle.PicklingError, TypeError, AttributeError):
            return False

    state = {k: v for k, v in scope.items() if is_pickleable(k, v)}

    with open(file_path, "wb") as f:
        pickle.dump(state, f)


def load_state(file_path = "__saved__/debug_state.pkl"):
    with open(file_path, "rb") as f:
        state = pickle.load(f)

    # globals().update(state)
    return state