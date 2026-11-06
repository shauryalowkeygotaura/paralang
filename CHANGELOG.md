# Changelog

All notable changes to this project are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-11-06

First tagged release. Recent work:

- Samples: Hindi voice A/B clips and Orpheus default mix
- Sound bank: extra breath/sigh/yawn prompts (all 10 sounds now covered in both voices); cue docs and listening set
- sounds: narrow the onomatopoeia gate (no nested quantifiers, real words rejected)
- Fish-style free-form cues on every engine: direction resolver, 20 new emotions, sound bank, Fish engine
- Add MIT license and README
- Expressive mode: one-take stitching, emotion endings, direction stripping, bench
- Orpheus: skip keys whose org lacks terms instead of failing
- emotion_for: agitated always maps to calm, even when rushed ties it
- v0.3 #13: agent self-check + weekly tuning proposals
- v0.3 #12: local live-call feed for a supervisor view
- v0.3 #11: cue-triggered playbooks (JSON rules -> one-turn [coach: ...] note, once per call)
- v0.3 #10: emotion-arc report per call + weekly roll-up (moments paired with the agent's prior line)
- v0.3 #8: laughs, sighs, breaths (conservative heuristics)
- v0.3 #7: voicemail / IVR / hold-music detection
- v0.3 #2: end-of-turn telemetry (smart-turn v3 decides, paralang measures)
