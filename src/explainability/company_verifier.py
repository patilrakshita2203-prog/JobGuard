from duckduckgo_search import DDGS


def company_has_website(company_name):

    if company_name.strip() == "":
        return False

    try:
        with DDGS() as ddgs:

            results = list(
                ddgs.text(
                    f"{company_name} official website",
                    max_results=3
                )
            )

        for result in results:

            url = result.get("href", "").lower()

            if "linkedin.com" not in url:
                return True

        return False

    except Exception:
        return False