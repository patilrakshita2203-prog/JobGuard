from src.explainability.knowledge_graph_realtime import RealTimeSkillKnowledgeGraph

kg = RealTimeSkillKnowledgeGraph()


def test_skills_match_whole_words_only():
    """Regression: 'hr' matched inside 'three', 'git' inside 'digital'."""
    assert not kg._mentions("hr", "we have three years of through experience")
    assert not kg._mentions("git", "digital marketing")
    assert kg._mentions("hr", "contact hr team")
    assert kg._mentions("ci/cd", "we use ci/cd pipelines")


def test_data_entry_job_asking_for_developer_skills_is_suspicious():
    result = kg.analyze_job_posting(
        "Data Entry Operator",
        "Required: Java, Python, Spring Boot, Django, Docker, Linux, Git, AWS",
    )
    assert result["role_detected"] == "data entry"
    assert result["is_suspicious"] is True
    assert result["skill_analysis"]["mismatched_skills"]


def test_matching_skills_are_not_suspicious():
    result = kg.analyze_job_posting("Data Entry Specialist", "Excel, Tally, typing speed, accuracy, ms office")
    assert result["is_suspicious"] is False
    assert result["skill_analysis"]["found_skills"]


def test_unknown_role_and_result_types():
    assert kg.analyze_job_posting("Astronaut", "x")["role_detected"] is None
    assert isinstance(kg.analyze_job_posting("Data Analyst", "sql excel")["is_suspicious"], bool)
    assert "mismatch_score" in kg.validate_skills_for_role("no such role", "text")
