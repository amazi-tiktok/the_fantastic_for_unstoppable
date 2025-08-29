from main import detect_policy_violation

def test_detect_policy_violation_on_toxic_text():
    cfg = {
        "activate_language_toxicity_detection": True,
        "toxicity_threshold": 0.8
    }
    result = detect_policy_violation(cfg, text="You are stupid.")
    assert isinstance(result, dict)
    assert "toxicity" in result

def test_detect_policy_violation_on_non_toxic_text():
    cfg = {
        "activate_language_toxicity_detection": True,
        "toxicity_threshold": 0.8
    }
    result = detect_policy_violation(cfg, text="Have a nice day!")
    assert isinstance(result, dict)
    assert "toxicity" not in result

def test_detect_policy_violation_disabled():
    cfg = {
        "activate_language_toxicity_detection": False,
        "toxicity_threshold": 0.8
    }
    result = detect_policy_violation(cfg, text="Any text")
    assert result is None

