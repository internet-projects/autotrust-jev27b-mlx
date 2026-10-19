"""AutoTrust-JEV-27B v1.1 on Apple Silicon (MLX): one-pass decisions with bidirectional prompt attention."""
from .engine import Jev27, load

__all__ = ["Jev27", "load"]
__version__ = "1.0.0"
