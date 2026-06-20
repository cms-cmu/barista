try:
    import torch
    import numpy as np
    import sys
    safe_classes = []
    for module_name in ['numpy.core.multiarray', 'numpy._core.multiarray']:
        for attr in ['scalar', '_reconstruct']:
            safe_classes.append(type(attr, (), {'__module__': module_name}))
    for attr in dir(np):
        try:
            val = getattr(np, attr)
            if isinstance(val, type):
                safe_classes.append(val)
                module = val.__module__
                if module:
                    if module.startswith('numpy._core'):
                        legacy_module = module.replace('numpy._core', 'numpy.core')
                        safe_classes.append(type(val.__name__, (), {'__module__': legacy_module}))
                    elif module.startswith('numpy.core'):
                        new_module = module.replace('numpy.core', 'numpy._core')
                        safe_classes.append(type(val.__name__, (), {'__module__': new_module}))
        except Exception: pass
    if hasattr(np, 'dtypes'):
        for attr in dir(np.dtypes):
            try:
                val = getattr(np.dtypes, attr)
                if isinstance(val, type):
                    safe_classes.append(val)
                    module = val.__module__
                    if module:
                        if module.startswith('numpy._core'):
                            legacy_module = module.replace('numpy._core', 'numpy.core')
                            safe_classes.append(type(val.__name__, (), {'__module__': legacy_module}))
                        elif module.startswith('numpy.core'):
                            new_module = module.replace('numpy.core', 'numpy._core')
                            safe_classes.append(type(val.__name__, (), {'__module__': new_module}))
            except Exception: pass
    
    # Also allowlist python/standard types if needed for legacy pickles
    import uuid
    safe_classes.append(uuid.UUID)
    
    if safe_classes and hasattr(torch, 'serialization') and hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals(safe_classes)
except Exception:
    pass
