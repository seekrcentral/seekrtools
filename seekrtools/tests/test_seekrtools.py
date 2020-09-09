"""
Unit and regression test for the seekrtools package.
"""

# Import package, test suite, and other packages as needed
import seekrtools
import pytest
import sys

def test_seekrtools_imported():
    """Sample test, will always pass so long as import statement worked"""
    assert "seekrtools" in sys.modules
