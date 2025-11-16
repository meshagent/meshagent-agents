from meshagent.agents.config import RulesConfig


def test_parse_empty_text():
    config = RulesConfig.parse(text="")

    assert config.rules == []
    assert config.client_rules == {}


def test_parse_global_rules_only():
    text = """\
first rule
second rule
third rule"""

    config = RulesConfig.parse(text=text)

    assert config.rules == ["first rule", "second rule", "third rule"]
    assert config.client_rules == {}


def test_comments():
    text = """\
# first rule
second rule
third rule"""

    config = RulesConfig.parse(text=text)

    assert config.rules == ["second rule", "third rule"]
    assert config.client_rules == {}


def test_parse_single_client_rules_only():
    text = """\
[web]
can browse the web
can use cookies
"""

    config = RulesConfig.parse(text=text)

    assert config.rules == []
    assert config.client_rules == {"web": ["can browse the web", "can use cookies"]}


def test_parse_global_and_client_rules():
    text = """\
global rule 1
global rule 2
[mobile]
use push notifications
limit data usage
"""

    config = RulesConfig.parse(text=text)

    assert config.rules == ["global rule 1", "global rule 2"]
    assert config.client_rules == {
        "mobile": ["use push notifications", "limit data usage"]
    }


def test_parse_multiple_clients():
    text = """\
[web]
show rich ui
support mouse
[mobile]
support touch
battery saver
"""

    config = RulesConfig.parse(text=text)

    assert config.rules == []
    assert config.client_rules == {
        "web": ["show rich ui", "support mouse"],
        "mobile": ["support touch", "battery saver"],
    }


def test_parse_remembers_current_client_until_next_header():
    text = """\
[client_a]
rule 1
rule 2
[client_b]
rule 3
"""

    config = RulesConfig.parse(text=text)

    assert config.client_rules == {
        "client_a": ["rule 1", "rule 2"],
        "client_b": ["rule 3"],
    }


def test_parse_keeps_blank_lines_as_rules():
    # This matches the current implementation, which does *not* skip empty lines.
    text = """\
global 1
global 2
[web]
rule after blank
"""

    config = RulesConfig.parse(text=text)

    assert config.rules == ["global 1", "global 2"]
    assert config.client_rules == {"web": ["rule after blank"]}
