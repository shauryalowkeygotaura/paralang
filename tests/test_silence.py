from types import SimpleNamespace

from paralang.silence import CLOSE, DEFAULT_S, MAX_S, MIN_S, NUDGE, SilencePolicy


def turns(*pairs):
    return SimpleNamespace(turns=[SimpleNamespace(text=t, features=SimpleNamespace(response_latency_s=l))
                                  for t, l in pairs])


def test_timeout_scales_with_this_caller():
    assert SilencePolicy(turns(("haan", 1.0))).timeout_s() == DEFAULT_S          # 1 turn: default
    assert SilencePolicy(turns(("a", 2.0), ("b", 2.4))).timeout_s() == 2.5 * 2.2  # thoughtful caller
    assert SilencePolicy(turns(("a", 0.3), ("b", 0.4))).timeout_s() == MIN_S      # fast caller: floor
    assert SilencePolicy(turns(("a", 6.0), ("b", 7.0))).timeout_s() == MAX_S      # never wait forever


def test_nudge_then_close_in_callers_language():
    p = SilencePolicy(turns(("Mera plan kab khatam hoga ji", 1.0)))
    first, second = p.on_idle(), p.on_idle()
    assert first.say == NUDGE["hi"] and not first.end_call
    assert second.say == CLOSE["hi"] and second.end_call
    e = SilencePolicy(turns(("Can you send me the details please", 1.0)))
    assert e.on_idle().say == NUDGE["en"]


def test_caller_speech_resets_strikes():
    p = SilencePolicy(turns(("hello there friend", 1.0)))
    p.on_idle()
    p.caller_spoke()
    assert not p.on_idle().end_call
