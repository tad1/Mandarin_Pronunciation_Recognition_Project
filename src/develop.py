import importlib
from operator import call
import sys
import inspect

def reload_module(module):
    """
    Reloads a module to ensure that the latest version is used.
    
    Args:
        module: The module to reload.
    """
    importlib.reload(module)

def reload_function(func):
    module_name = func.__module__
    func_name = func.__name__
    
    module = sys.modules.get(module_name)
    if module:
        module = importlib.reload(module)
    
    caller_frame = inspect.currentframe().f_back
    caller_globals = caller_frame.f_globals
    caller_locals = caller_frame.f_locals
    
    if func_name in caller_globals:
        caller_globals[func_name] = getattr(module, func_name)
    if func_name in caller_locals:
        caller_locals[func_name] = getattr(module, func_name)
    
    return getattr(module, func_name)