"""Round 2: whole-paragraph Orpheus (one request) vs Aura with sentence gaps only."""
import re, sys
from pathlib import Path
import make_ab as m

OUT = Path(__file__).parent
ORPHEUS_ONE = ("[serious] There are four specific rocks in space, currently on a collision course with our planet. "
               "[dramatic] And none of them... are theoretical. "
               "[serious] One of them even has a name... Apophis.")

if "--orpheus" in sys.argv:
    for v in ("hannah", "troy", "diana", "daniel"):
        m.to_wav(m.orpheus(ORPHEUS_ONE, v), ".wav", OUT / f"E_orpheus_{v}_onecall.wav")
    print("E written")
else:
    # Aura: one clip per SENTENCE, untouched speed, paralang gaps between them.
    sents = re.findall(r"[^.!?]+[.!?]", m.PLAIN)
    import tempfile
    tmp = Path(tempfile.mkdtemp()); parts = []
    gaps = [0, 520, 450, 600]  # serious->dramatic gets the long beat
    for i, s in enumerate(sents):
        if i:
            g = tmp / f"g{i}.wav"; m.silence(gaps[i], g); parts.append(g)
        p = tmp / f"s{i}.wav"; m.to_wav(m.aura(s.strip()), ".mp3", p); parts.append(p)
    m.concat(parts, OUT / "F_aura_sentence_gaps.wav")
    print("F written")
