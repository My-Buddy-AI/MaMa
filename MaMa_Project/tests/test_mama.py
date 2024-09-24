import pytest
from mama.mama_framework import MAMAFramework

def test_mama_process_query():
    mama = MAMAFramework()
    query = "This is a great movie!"
    result = mama.process_query(query)
    assert result in ["positive", "negative"]
