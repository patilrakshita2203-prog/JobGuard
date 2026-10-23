from conftest import FAKE_TEXT, GENUINE_TEXT
from src.explainability.red_flags import detect_red_flags, red_flag_score


def codes(text):
    return {f["code"] for f in detect_red_flags(text)}


def test_scam_text_hits_the_big_flags():
    found = codes(FAKE_TEXT)
    assert {"upfront_payment", "guaranteed_outcome", "messaging_app_hiring"} <= found


def test_genuine_text_has_no_flags():
    assert detect_red_flags(GENUINE_TEXT) == []


def test_free_email_contact_is_flagged():
    assert "free_email_contact" in codes("Send your CV to hr.recruit@gmail.com today")


def test_hindi_keywords_are_detected():
    found = codes("पंजीकरण शुल्क 500 रुपये, तुरंत जॉइन करें, व्हाट्सएप पर संपर्क करें")
    assert {"upfront_payment", "messaging_app_hiring", "urgency_pressure"} <= found


def test_flags_are_sorted_worst_first_and_carry_evidence():
    flags = detect_red_flags(FAKE_TEXT)
    order = {"high": 0, "medium": 1, "low": 2}
    assert [order[f["severity"]] for f in flags] == sorted(order[f["severity"]] for f in flags)
    assert all(f["evidence"] for f in flags)


def test_score_is_capped_at_100():
    assert red_flag_score(detect_red_flags(FAKE_TEXT)) == 100
    assert red_flag_score([]) == 0


def test_none_and_empty_input_are_safe():
    assert detect_red_flags(None) == []
    assert detect_red_flags("") == []
