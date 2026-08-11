import sys
from pathlib import Path

import pytest

REQ_REVERSE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REQ_REVERSE))

FIXTURES = REQ_REVERSE / "tests" / "fixtures"


@pytest.fixture
def fixtures() -> Path:
    return FIXTURES


@pytest.fixture
def subject() -> Path:
    """Мини-репозиторий для E2E-прогонов и покрытия."""
    return FIXTURES / "subject"


@pytest.fixture
def tenacity_repo() -> Path:
    """Настоящие исходники tenacity 9.1.4 (шесть модулей прогона Kimi)."""
    return FIXTURES / "tenacity"


@pytest.fixture
def tenacity_rules() -> Path:
    """145 правил с дословными цитатами; у tenacity-R-301 цитата сокращена."""
    return FIXTURES / "tenacity_artifacts" / "rules"


@pytest.fixture
def canned() -> Path:
    return FIXTURES / "canned"
