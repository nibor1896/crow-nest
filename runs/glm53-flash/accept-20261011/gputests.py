"""The ignored GPU tests of #215 #219 #217 #216, one after the other, hidden; output per test to gputests/<name>.log."""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "engine"))
NO_WINDOW = 0x08000000
TESTS = {
    "215": ["--lib", "glm5_devsampler_gpu"],
    "219": ["--lib", "glm5_answer_reuse_gpu"],
    "217a": ["--lib", "glm5_kda::tests_gpu::kda_gpu_a_cut_scan_is_the_scan"],
    "217b": ["--lib", "glm5_prefix_anchor_gpu"],
    "216": ["--lib", "glm5_int_gpu_the_small_prompt"],
}
os.makedirs(os.path.join(HERE, "gputests"), exist_ok=True)
for name in (sys.argv[1:] or TESTS):
    args = ["cargo", "test", "--release"] + TESTS[name] + ["--", "--ignored", "--nocapture", "--test-threads", "1"]
    t0 = time.time()
    with open(os.path.join(HERE, "gputests", name + ".log"), "w", encoding="utf-8", errors="replace") as out:
        rc = subprocess.run(args, cwd=ENGINE, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                            creationflags=NO_WINDOW).returncode
    print(name, "rc", rc, "s", round(time.time() - t0, 1), flush=True)
