from src.explainability import company_verifier as cv


def setup_function():
    cv._CACHE.clear()


def test_domain_must_match_company_name():
    assert cv._domain_matches_company("https://www.infosys.com/about", "Infosys Technologies Ltd")
    assert cv._domain_matches_company("https://fastmoney.com", "FastMoney Solutions")
    assert not cv._domain_matches_company("https://random-blog.net/post", "FastMoney Solutions")


def test_aggregator_sites_do_not_count():
    assert not cv._domain_matches_company("https://www.linkedin.com/company/infosys", "Infosys")
    assert not cv._domain_matches_company("https://www.naukri.com/infosys-jobs", "Infosys")


def test_search_error_means_unknown_not_missing(monkeypatch):
    def boom(query):
        raise RuntimeError("rate limited")

    monkeypatch.setattr(cv, "_search", boom)
    assert cv.company_has_website("Acme Labs") is None


def test_empty_name_or_empty_results_are_unknown(monkeypatch):
    assert cv.company_has_website("") is None
    monkeypatch.setattr(cv, "_search", lambda q: [])
    assert cv.company_has_website("Acme Labs") is None


def test_match_and_no_match(monkeypatch):
    monkeypatch.setattr(cv, "_search", lambda q: [{"href": "https://acmelabs.io"}])
    assert cv.company_has_website("Acme Labs") is True
    cv._CACHE.clear()
    monkeypatch.setattr(cv, "_search", lambda q: [{"href": "https://unrelated.org"}])
    assert cv.company_has_website("Acme Labs") is False


def test_definitive_results_are_cached(monkeypatch):
    calls = []

    def fake(query):
        calls.append(query)
        return [{"href": "https://acmelabs.io"}]

    monkeypatch.setattr(cv, "_search", fake)
    cv.company_has_website("Acme Labs")
    cv.company_has_website("Acme Labs")
    assert len(calls) == 1
