"""
_smile_worker.py -- openSMILE voice-quality worker, run as its own PROCESS.

    python -m paralang._smile_worker      (started by analyzer._Shared, never by hand)

Why a separate process: openSMILE's Python wrapper (audinterface/pandas) holds
the GIL for ~150 ms per turn. In a thread it stalls whatever live call is on
its critical path (measured: 7 ms turns became 70-150 ms). A process has its
own GIL and runs at below-normal OS priority, so the live path always wins.

Why `-m` and not multiprocessing: Windows' spawn re-imports the parent's main
script in the child, so an agent script without a __main__ guard would re-run
its top-level code (e.g. dial a call twice). This module imports only paralang.

Protocol on stdin/stdout: 4-byte little-endian length + pickle, both ways.
Request (float32 ndarray, sample_rate) -> response dict (voice_quality output).
The parent is the only peer; stdout is reserved for the protocol, anything
else printed (openSMILE logs) is redirected to stderr.
"""
from __future__ import annotations

import os
import pickle
import struct
import sys


def main() -> int:
    proto_out = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())   # stray prints can't corrupt the protocol
    proto_in = sys.stdin.buffer

    from paralang.features import prewarm, voice_quality
    prewarm()                                            # the ~12 s cold load, done here
    proto_out.write(struct.pack("<I", 0))                # 0-length frame = "ready"
    proto_out.flush()

    while True:
        hdr = proto_in.read(4)
        if len(hdr) < 4:
            return 0                                     # parent closed the pipe
        (n,) = struct.unpack("<I", hdr)
        try:
            x, sr = pickle.loads(proto_in.read(n))
            res = voice_quality(x, sr)
        except Exception as e:  # noqa: BLE001 - one bad turn never kills the worker
            print(f"paralang worker: {type(e).__name__}: {e}", file=sys.stderr)
            res = {}
        data = pickle.dumps(res)
        proto_out.write(struct.pack("<I", len(data)) + data)
        proto_out.flush()


if __name__ == "__main__":
    sys.exit(main())
