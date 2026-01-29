from typing import ParamSpec, Callable, TypeVar, Any, cast
from inspect import signature

P = ParamSpec("P")
T = TypeVar("T")


def Kwargs(cls: Callable[P, T]) -> Callable[P, dict[str, Any]]:
    """Type-safe kwargs builder that provides type hints and documentation for a class.
    
    Captures the signature of a class's __init__ method and returns a function
    that accepts the same parameters but returns a dict instead of an instance.
    
    Usage:
        OptimizerKwargs = Kwargs(torch.optim.Adam)
        optimizer_config = OptimizerKwargs(lr=1e-4, weight_decay=1e-4)
        optimizer = torch.optim.Adam(**optimizer_config)
        
    The type checker will validate that lr and weight_decay are valid
    parameters for torch.optim.Adam.__init__ and provide autocomplete.
    """
    def create_kwargs(*args: P.args, **kwargs: P.kwargs) -> dict[str, Any]:
        return dict(kwargs) if not args else {**dict(enumerate(args)), **kwargs}
    
    # Preserve the signature for better IDE support
    create_kwargs.__signature__ = signature(cls)  # type: ignore
    create_kwargs.__doc__ = cls.__doc__
    create_kwargs.__annotations__ = getattr(cls, '__annotations__', {})
    
    return create_kwargs