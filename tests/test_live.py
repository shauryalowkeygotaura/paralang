import json
import urllib.error
import urllib.request

import pytest

from paralang.live import ALLOWED_ORIGINS, LiveFeed


@pytest.fixture
def feed():
    f = LiveFeed.start(port=0)          # 0 = any free port
    yield f
    f.stop()


def get(feed, path, origin=None, method="GET"):
    req = urllib.request.Request(f"http://127.0.0.1:{feed.port}{path}", method=method,
                                 headers={"Origin": origin} if origin else {})
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status, dict(r.headers), r.read()


def test_publish_end_snapshot(feed):
    feed.publish("c1", {"turn": 1, "tag": ""}, meta={"lead": "Rohan"})
    feed.publish("c1", {"turn": 2, "tag": "[tone: hesitant]"})
    feed.publish("c2", {"turn": 1, "tag": ""})
    feed.end("c2", {"tier": "warm"})
    status, _, body = get(feed, "/calls")
    calls = json.loads(body)["calls"]
    assert status == 200 and [c["id"] for c in calls] == ["c1", "c2"]       # live first
    assert len(calls[0]["turns"]) == 2 and calls[0]["meta"]["lead"] == "Rohan"
    assert calls[1]["status"] == "ended" and calls[1]["result"]["tier"] == "warm"


def test_cors_only_for_allowed_origins_and_pna_preflight(feed):
    ok = next(iter(ALLOWED_ORIGINS))
    _, h, _ = get(feed, "/calls", origin=ok)
    assert h.get("Access-Control-Allow-Origin") == ok
    _, h, _ = get(feed, "/calls", origin="https://evil.example")
    assert "Access-Control-Allow-Origin" not in h
    status, h, _ = get(feed, "/calls", origin=ok, method="OPTIONS")
    assert status == 204 and h.get("Access-Control-Allow-Private-Network") == "true"


def test_loopback_only():
    with pytest.raises(ValueError):
        LiveFeed.start(host="0.0.0.0")


def test_turn_event_is_small_and_json_safe():
    import numpy as np
    from paralang import Analyzer
    from paralang.live import turn_event
    t = np.arange(16000) / 16000
    res = Analyzer(acoustic=False).analyze_turn((0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32),
                                                16000, "haan ji " * 40, response_latency_s=3.0)
    ev = turn_event(res, coach="slow down", lang="hi")
    json.dumps(ev)
    assert len(ev["text"]) <= 80 and ev["coach"] == "slow down" and 0 <= ev["engagement"] <= 100


def test_dns_rebinding_blocked(feed):
    req = urllib.request.Request(f"http://127.0.0.1:{feed.port}/calls", headers={"Host": "evil.example:8765"})
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=5)
    assert e.value.code == 403


def test_odd_values_do_not_break_the_feed(feed):
    feed.publish("c9", {"turn": 1, "weird": object()})
    status, _, body = get(feed, "/calls")
    assert status == 200 and json.loads(body)["calls"][0]["id"] == "c9"
