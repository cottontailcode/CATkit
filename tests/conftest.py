"""Shared test configuration.

This file exists so that ``tests/`` is on ``sys.path`` (pytest "prepend" import
mode): the helper modules ``scientific``, ``synth`` and, for equivalence tests,
``compare`` can then be imported by plain name from any test directory.
"""
