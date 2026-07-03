from src.explainability.company_verifier import company_has_website
def analyze_job(job):

    facts = {}

    # Salary
    facts["salary_disclosed"] = (
        job.salary_range.strip().lower() != "not disclosed"
        and job.salary_range.strip() != ""
    )

    # Company
    facts["company_available"] = (
        job.company.strip() != ""
    )
    facts["official_website_found"] = company_has_website(job.company)

    # Location
    facts["location_available"] = (
        job.location.strip() != ""
    )

    # Description
    facts["description_length"] = len(job.description.strip())

    # Experience
    facts["experience_available"] = (
        job.experience.strip() != ""
    )

    return facts