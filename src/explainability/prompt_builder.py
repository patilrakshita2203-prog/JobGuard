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

Official Website Found: {facts['official_website_found']}

Location Available: {facts['location_available']}

Experience Mentioned: {facts['experience_available']}

Description Length: {facts['description_length']} characters

Instructions:

1. Do NOT invent facts.

2. Explain only what can be concluded from the Rule Engine findings.

3. Write exactly 3 professional bullet points.

4. End with one recommendation sentence.

"""

    return prompt