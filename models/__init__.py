import importlib
import re

def define_network(opt):
    opt = opt.copy()
    model_type = opt.pop('type')
    # Convert PascalCase to snake_case for filename convention
    snake_name = re.sub(r'(?<!^)(?=[A-Z])', '_', model_type).lower()
    
    try:
        module = importlib.import_module(f"models.{snake_name}")
    except ImportError:
        # Fallback: try importing from models directly if it's not in a submodule
        # or maybe the file name doesn't match the class name convention.
        # But for this project, we enforce the convention.
        raise ImportError(f"Could not find module 'models.{snake_name}' for model type '{model_type}'. Ensure file name matches class name (PascalCase -> snake_case).")
        
    model_cls = getattr(module, model_type, None)
    if model_cls is None:
        raise AttributeError(f"Module 'models.{snake_name}' has no class '{model_type}'.")
        
    return model_cls(**opt)
