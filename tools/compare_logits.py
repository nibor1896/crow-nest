"""Compare two [T][V] f32 logit dumps: per-position max_abs delta + argmax match.
Usage: python tools/compare_logits.py <gpu.f32> <ref.f32> [T]"""
import sys
import numpy as np

gpu = np.fromfile(sys.argv[1], dtype=np.float32)
ref = np.fromfile(sys.argv[2], dtype=np.float32)
T = int(sys.argv[3]) if len(sys.argv) > 3 else len(gpu) // len(ref) if len(gpu) == len(ref) else 0
assert len(gpu) == len(ref), f"size mismatch {len(gpu)} vs {len(ref)}"
V = 248320
T = len(gpu) // V
assert T * V == len(gpu), f"{len(gpu)} not a multiple of {V}"
gpu = gpu.reshape(T, V)
ref = ref.reshape(T, V)
ok = 0
for t in range(T):
    d = np.abs(gpu[t] - ref[t])
    g_am = int(np.argmax(gpu[t]))
    r_am = int(np.argmax(ref[t]))
    srt = np.sort(ref[t])
    ok += g_am == r_am
    print(f"pos {t:3d}: max_abs={d.max():.4e} argmax gpu={g_am} ref={r_am} match={g_am == r_am} margin={srt[-1]-srt[-2]:.4f}")
print(f"argmax {ok}/{T}, worst {np.abs(gpu-ref).max():.4e}")
