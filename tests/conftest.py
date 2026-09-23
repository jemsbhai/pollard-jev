from datetime import datetime, timezone

import pytest

from pollard_jev.demo import example_request


@pytest.fixture
def now():
    return datetime.now(timezone.utc)


@pytest.fixture
def request_data(now):
    return example_request(now)
