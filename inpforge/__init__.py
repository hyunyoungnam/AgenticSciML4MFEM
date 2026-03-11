"""
inpforge: Automated Abaqus .inp file generation for AI/ML training datasets.

This library provides multi-agent AI systems to automatically generate diverse,
validated Abaqus .inp files through intelligent mesh morphing. It eliminates
the tedious manual work of creating FEA datasets for machine learning research.

Example:
    >>> from inpforge import AbaqusManager, apply_morphing
    >>> manager = AbaqusManager("model.inp")
    >>> apply_morphing(manager, config_path="morphing.md", delta_r=0.5)
    >>> manager.write("output.inp")
"""

__version__ = "0.1.0"
__author__ = "Q. Jiang"

# Core API
from inpforge.parser import InpParser
from inpforge.manager import AbaqusManager
from inpforge.morphing import apply_morphing, MorphingContext
from inpforge.schema import HeavyData, LightData
from inpforge.validator import validate_model
from inpforge.writer import InpWriter
from inpforge.vtu_export import export_to_vtu

__all__ = [
    # Version info
    "__version__",
    "__author__",
    # Core classes
    "InpParser",
    "AbaqusManager",
    "MorphingContext",
    "HeavyData",
    "LightData",
    "InpWriter",
    # Functions
    "apply_morphing",
    "validate_model",
    "export_to_vtu",
]
