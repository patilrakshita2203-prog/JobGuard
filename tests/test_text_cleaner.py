from data_cleaning import TextCleaner


def test_cleaner_lowercases_and_strips_html():
    out = TextCleaner().clean("URGENT <b>Hiring</b>   NOW!!")
    assert "<b>" not in out
    assert out == out.lower()
    assert "urgent" in out


def test_cleaner_replaces_urls_and_emails():
    out = TextCleaner().clean("Apply at https://spam.example/x or mail me@x.com")
    assert "https" not in out
    assert "@" not in out


def test_cleaner_handles_empty_text():
    assert TextCleaner().clean("") == ""
