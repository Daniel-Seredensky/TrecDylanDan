"""
TREC-2025-DRAGUN Run Generation Package

This package contains scripts and utilities for generating TREC-2025-DRAGUN runs
from the pipeline output in the required format.
"""

from .generate_runs import generate_runs, generate_runs_async
from .convert_to_run_format import convert_pipeline_output_to_run, convert_file
from .test_run_format import validate_run_format, validate_runs_file

__all__ = [
    'generate_runs',
    'generate_runs_async', 
    'convert_pipeline_output_to_run',
    'convert_file',
    'validate_run_format',
    'validate_runs_file'
] 