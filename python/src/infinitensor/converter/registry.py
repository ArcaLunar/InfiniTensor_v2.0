import inspect
from typing import Dict, Callable, Optional
import torch.nn as nn
import torch
from torch import fx


class ConverterRegistry:

    def __init__(self):
        # { "aten.matmul": { "default": fn, "out": fn, None: fn } }
        self._method_converters: Dict[str, Dict[Optional[str], Callable]] = {}

    def register(self, op_name: str, overload: Optional[str] = None):
        """Decorator: register method and function converters"""

        def decorator(func):
            self._method_converters.setdefault(op_name, {})[overload] = func
            return func

        return decorator

    def get_method_converter(
        self, op_name: str, overload: Optional[str] = None
    ) -> Optional[Callable]:
        """Get method and function converter"""
        if op_name in self._method_converters:
            table = self._method_converters[op_name]
            if overload:
                if overload in table:
                    return table[overload]
                else:
                    raise ValueError(f"Unsupported op.overload : {op_name}_{overload}")
            else:
                if None in table:
                    return table[None]
                else:
                    raise ValueError(f"Unsupported op.overload : {op_name}")
        else:
            raise ValueError(f"Unsupported op : {op_name}")

    def update(self, custom_converters: Dict):
        """Update converters
        Args:
            custom_converters:
            {
                (op_name, overload): converter
            }
        """
        for key, converter in custom_converters.items():
            if isinstance(key, tuple) and len(key) == 2:
                op_name, overload = key
                self._method_converters.setdefault(op_name, {})[overload] = converter
            if isinstance(key, str):
                self._method_converters[key] = converter
            else:
                raise TypeError(f"Invalid key type: {type(key)}")

    def clear(self):
        """Clear all converters"""
        self._method_converters.clear()

    def list_all_converters(self):
        """List all converters"""
        return {
            "methods": list(self._method_converters.keys()),
        }


# Global registry instance
registry = ConverterRegistry()
