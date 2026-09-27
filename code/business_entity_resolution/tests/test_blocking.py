import pytest
from business_entity_resolution.blocking import CountryInvertedIndex

def test_blocking_index():
    index = CountryInvertedIndex()
    index.add_target_record("S2-100", "Moore Bitwise Inc", "337 Oakland Ave, Michigan City, IN")
    index.add_target_record("S3-200", "Chong's Hospitality LLC", "1884 Pine Meadow Court, Gurnee, IL")
    
    # Query with alias
    cands = index.query("moorebitwise.com", "0337 Oakland Avenue")
    assert "S2-100" in cands
    
    # Query unrelated
    cands_unrelated = index.query("Random Business Name", "999 Nowhere Street")
    assert len(cands_unrelated) == 0
