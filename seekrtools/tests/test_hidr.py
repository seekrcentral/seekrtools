"""
test_hidr.py
"""

import os
import pytest

import seekrtools.hidr.hidr as hidr

def test_catch_erroneous_destination():
    """
    
    """
    assert hidr.catch_erroneous_destination("1") == True
    assert hidr.catch_erroneous_destination("0") == True
    assert hidr.catch_erroneous_destination("any") == True
    assert hidr.catch_erroneous_destination("all") == False
    assert hidr.catch_erroneous_destination("gobblygook") == False
    return