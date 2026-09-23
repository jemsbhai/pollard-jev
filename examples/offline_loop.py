"""Run from the project root after editable installation."""

from datetime import datetime, timezone

from pollard_jev.demo import example_request
from pollard_jev.loop import DecisionLoop
from pollard_jev.providers.fixture import FixtureProvider

request = example_request(datetime.now(timezone.utc), request_id="example-robot-1")
with DecisionLoop(
    FixtureProvider(), max_requests=1,
    records_path="artifacts/example-decisions.jsonl",
    pollard_path="artifacts/example-pollard.db",
) as loop:
    record = loop.decide(request)
    print(record.model_dump_json(indent=2))
