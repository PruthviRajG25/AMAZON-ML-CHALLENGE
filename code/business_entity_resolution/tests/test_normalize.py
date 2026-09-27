import pytest
from business_entity_resolution.normalize import (
    clean_name,
    canonical_name_key,
    token_signature_key,
    normalize_address,
    extract_address_keys
)

def test_clean_name_legal_suffixes():
    assert clean_name("Moore Bitwise Inc") == "moore bitwise"
    assert clean_name("Chong's Hospitality LLC") == "chong s hospitality"
    assert clean_name("Raj Investments LLP") == "raj investments"
    assert clean_name("Azamgarh Marketing Private Limited") == "azamgarh"

def test_clean_name_domain_extensions():
    assert clean_name("moorebitwise.com") == "moorebitwise"
    assert clean_name("chongshospitality.co.in") == "chongshospitality"

def test_canonical_name_key():
    assert canonical_name_key("Moore Bitwise Inc") == "moorebitwise"
    assert canonical_name_key("moorebitwise.com") == "moorebitwise"
    assert canonical_name_key("D & L Metro LLC") == "dlmetro"

def test_normalize_address_zero_stripping():
    addr = "0337 Oakland Avenue, Michigan City, IN"
    norm = normalize_address(addr)
    assert "337" in norm
    assert "0337" not in norm
    assert "avenue" in norm

def test_extract_address_keys():
    keys = extract_address_keys("85 Wayne Avenue, Ticonderoga, NY")
    assert "85_wayne" in keys
