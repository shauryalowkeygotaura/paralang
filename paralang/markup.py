"""
markup.py -- add [emotion] / `|` beat / `*emphasis*` marks to plain text with
an LLM, so a 2,900-word script does not have to be tagged by hand.

    from paralang.markup import annotate
    marked = annotate(script_text)            # GROQ_API_KEY in the environment
    say_wav(marked, "out.wav", engine="orpheus")

    python -m paralang.markup script.txt > script.marked.txt

THE GUARD: a model asked to "add tags" will sometimes also "improve" the words.
A voiceover must say exactly what was written, so every paragraph is checked:
strip the marks from the reply and the words must equal the input's words. A
paragraph that fails is returned UNMARKED (plain text still speaks fine)
rather than with words the author never wrote.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

from .emotion import EMOTIONS, _TAG_RE, clean

URL = "https://api.groq.com/openai/v1/chat/completions"
# Tried in order; a retired (404), throttled (429) or failing model moves on.
MODELS = ("openai/gpt-oss-120b", "qwen/qwen3.8-27b", "openai/gpt-oss-20b")

PROMPT = """You mark up a voiceover script for a narrator. Return the SAME text with marks added. Never change, add, remove or reorder a single word or punctuation mark.

Marks:
- [emotion] before a sentence where the delivery should change. Allowed: {emotions}. Use few; most sentences inherit the previous one.
- | (a beat) inside a sentence, right before the payoff of a line, where a human narrator would hold for a moment. At most one per sentence, and only on lines that deserve weight.
- *word* around the one word that carries a line's punch. Rare: at most one per 3 sentences.

Choose emotions that fit the story, not the punctuation: danger, stakes and reveals are serious or dramatic, never excited or cheerful. Keep the text on one line, exactly as given apart from the marks. Start the text with an [emotion]. Output only the marked text, no explanation.

Example input:
Scientists once gave it a real chance of hitting us. They were wrong about the date. They were not wrong about the size.
Example output:
[serious] Scientists once gave it a real chance of hitting us. They were wrong about the date. [dramatic] They were not wrong | about the *size*.

Text:
{text}"""


def _words(text: str) -> list[str]:
    plain = clean(_TAG_RE.sub(" ", text))
    return re.findall(r"[a-z0-9']+", plain.lower())


def tidy(marked: str) -> str:
    """Drop marks that do nothing and normalise spacing.

    A beat at a sentence end duplicates the sentence gap the planner already
    adds, so it is removed; so are line breaks the model invents inside a
    paragraph, and `|` glued to a word."""
    t = re.sub(r"\s*\n\s*", " ", marked)
    t = re.sub(r"\s*\|\s*(?=[.!?…,;:]|$|\[)", " ", t)  # beat before punctuation/end/tag
    t = re.sub(r"([.!?…])\s*\|", r"\1", t)  # beat right after a sentence end
    t = re.sub(r"\s*\|\s*", " | ", t)
    # Restore a space the model swallowed ("small.It"); a digit after the dot
    # ("2.7") is a decimal and is left alone.
    t = re.sub(r"([.!?])(?=[A-Z\[])", r"\1 ", t)
    return re.sub(r"[ \t]{2,}", " ", t).strip()


def _call(paragraph: str, key: str) -> str | None:
    body = {"messages": [{"role": "user", "content": PROMPT.format(
                emotions=", ".join(EMOTIONS), text=paragraph)}],
            "temperature": 0.2}
    for model in MODELS:
        payload = dict(body, model=model,
                       max_tokens=min(4000, len(paragraph) // 2 + 900),
                       **({"reasoning_effort": "low"} if model.startswith("openai/") else {}))
        req = urllib.request.Request(URL, data=json.dumps(payload).encode(), method="POST", headers={
            "Authorization": f"Bearer {key}", "Content-Type": "application/json",
            "User-Agent": "paralang/0.2"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise RuntimeError(f"Groq rejected the key (HTTP {e.code})") from None
            if e.code == 429:
                time.sleep(float(e.headers.get("retry-after") or 6))
            continue
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            continue
        out = (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        if out.strip():
            return out.strip()
    return None


def annotate(text: str, key: str | None = None) -> str:
    """Return `text` with pacing marks. Paragraph by paragraph; any paragraph
    whose words came back changed is kept unmarked (see THE GUARD above)."""
    key = key or os.environ.get("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GROQ_API_KEY not set")
    out = []
    for para in re.split(r"\n\s*\n", text.strip()):
        para = para.strip()
        if not para:
            continue
        good = None
        for _ in range(2):  # sampling is not deterministic: one more try if the words drifted
            marked = _call(para, key)
            if marked and _words(marked) == _words(para):
                good = tidy(marked)
                break
        if good:
            out.append(good)
        else:
            print(f"[paralang.markup] kept unmarked (words changed or no reply): {para[:60]!r}",
                  file=sys.stderr)
            out.append(para)
    return "\n\n".join(out)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        raise SystemExit("usage: python -m paralang.markup <file.txt | -> ")
    text = sys.stdin.read() if argv[0] == "-" else open(argv[0], encoding="utf-8").read()
    print(annotate(text))


if __name__ == "__main__":
    main()
