from paralang import Cue, DeliveryPlanner


def plan_reply(sentences, cues=None, wobble=False):
    p = DeliveryPlanner(wobble=wobble)
    return [p.plan(s, cues, reply_id="r1") for s in sentences]


def test_first_sentence_has_no_gap_later_ones_breathe():
    a, b, c = plan_reply(["Achha.", "Aapka plan kal expire ho raha hai?", "Main bata deti hoon."])
    assert a.pause_before_ms == 0
    assert b.pause_before_ms == 320          # after "."
    assert c.pause_before_ms == 380          # after "?" -> longer beat


def test_new_reply_resets_gap():
    p = DeliveryPlanner(wobble=False)
    p.plan("Haan ji.", reply_id="r1")
    assert p.plan("Theek hai.", reply_id="r2").pause_before_ms == 0


def test_hesitant_caller_gets_slower_softer_longer_gaps():
    calm = plan_reply(["Achha.", "Koi jaldi nahi hai."])
    soft = plan_reply(["Achha.", "Koi jaldi nahi hai."], [Cue("hesitant", 1.0)])
    assert soft[1].rate_pct < calm[1].rate_pct
    assert soft[1].volume_pct < calm[1].volume_pct
    assert soft[1].pause_before_ms > calm[1].pause_before_ms


def test_rushed_caller_gets_faster_tighter():
    calm = plan_reply(["Ok.", "Do baatein."])
    fast = plan_reply(["Ok.", "Do baatein."], [Cue("rushed", 1.0)])
    assert fast[1].rate_pct > calm[1].rate_pct and fast[1].pause_before_ms < calm[1].pause_before_ms


def test_agitated_caller_gets_calmer_lower():
    p = plan_reply(["Main samajh sakti hoon."], [Cue("agitated", 1.0)])[0]
    assert p.rate_pct < 0 and p.pitch_hz < 0 and p.volume_pct < 0


def test_strength_scales_effect():
    half = plan_reply(["x."], [Cue("rushed", 0.5)])[0]
    full = plan_reply(["x."], [Cue("rushed", 1.0)])[0]
    assert 0 < half.rate_pct < full.rate_pct


def test_questions_lift_pitch_and_wobble_varies_but_is_stable():
    q, s = plan_reply(["Aap busy hain?", "Aap busy hain."])
    assert q.pitch_hz > s.pitch_hz
    w = DeliveryPlanner()
    rates = {w.plan(t).rate_pct for t in ["Haan ji.", "Bilkul.", "Achha, samjha.", "Ek minute."]}
    assert len(rates) > 1                                  # not every sentence the same tempo
    assert DeliveryPlanner().plan("Haan ji.").rate_pct == DeliveryPlanner().plan("Haan ji.").rate_pct


def test_limits_and_edge_format():
    cues = [Cue(n, 1.0) for n in ("rushed", "barged_in", "flat")]
    p = DeliveryPlanner(base_rate_pct=20).plan("Haan?", cues)
    assert p.rate_pct <= 25
    e = p.edge()
    assert e["rate"].endswith("%") and e["pitch"].endswith("Hz") and e["rate"][0] in "+-"
