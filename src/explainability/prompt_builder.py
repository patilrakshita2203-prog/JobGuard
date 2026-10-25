def _website_state(value):
    if value is None:
        return "Could not be checked (do not assume either way)"
    return value


def _red_flag_lines(facts):
    flags = facts.get("red_flags") or []
    if not flags:
        return "None detected"
    return "; ".join(f"{f['label']} (\"{f['evidence']}\")" for f in flags)


def build_prompt(job, facts):

    prompt = f"""
You are an AI cybersecurity analyst.

Analyze the following job posting using ONLY the verified facts.

JOB DETAILS

Title: {job.title}

Company: {job.company}

Location: {job.location}

Platform: {job.source_platform}

RULE ENGINE FINDINGS

Salary Disclosed: {facts['salary_disclosed']}

Company Available: {facts['company_available']}

Official Website Found: {_website_state(facts['official_website_found'])}

Location Available: {facts['location_available']}

Experience Mentioned: {facts['experience_available']}

Description Length: {facts['description_length']} characters

Scam Red Flags Found In Text: {_red_flag_lines(facts)}

Instructions:

1. Do NOT invent facts.

2. Explain only what can be concluded from the Rule Engine findings.

3. Write exactly 3 professional bullet points.

4. End with one recommendation sentence.

"""

    return prompt