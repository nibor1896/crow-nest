# decode_out

- Rule: runs are not committed here. These are the inputs and references the gates read, not output (E3, issue #43).

| file | what it is | which gate reads it |
|---|---|---|
| `decode_out/final4-t1-read-run0-crow.json` | greedy reference record of the ten-task gate with answers and measurements | identity gate B4/C1/E3/E8 |
| `decode_out/final4-t1b-read-lang-run0-crow.json` | greedy reference record of the ten-task gate with answers and measurements | identity gate B4/C1/E3/E8 |
| `decode_out/final4-t2-write-run0-crow.json` | greedy reference record of the ten-task gate with answers and measurements | identity gate B4/C1/E3/E8 |
| `decode_out/final4-t2b-write-refactor-run0-crow.json` | greedy reference record of the ten-task gate with answers and measurements | identity gate B4/C1/E3/E8 |
| `decode_out/final4-t3-debug-run0-crow.json` | greedy reference record of the ten-task gate with answers and measurements | identity gate B4/C1/E3/E8 |
| `decode_out/final4-t3b-debug-syn-run0-crow.json` | greedy reference record of the ten-task gate with answers and measurements | identity gate B4/C1/E3/E8 |
| `decode_out/final4-t4-prose-run0-crow.json` | greedy reference record of the ten-task gate with answers and measurements | identity gate B4/C1/E3/E8 |
| `decode_out/final4-t5-agent-run0-crow.json` | greedy reference record of the ten-task gate with answers and measurements | identity gate B4/C1/E3/E8 |
| `decode_out/final4-t6-reason-run0-crow.json` | greedy reference record of the ten-task gate with answers and measurements | identity gate B4/C1/E3/E8 |
| `decode_out/final4-t6b-reason-multi-run0-crow.json` | greedy reference record of the ten-task gate with answers and measurements | identity gate B4/C1/E3/E8 |
| `decode_out/parity-ids.json` | id input | parity 8 |
| `decode_out/real512-ids.json` | id input | parity 512 |
| `decode_out/t3-debug-1024-ids.json` | id input | parity 1024 |
| `decode_out/hotsets-M-longctx2100-n160.json` | sidecar | serve default |
| `decode_out/demo-ids.json` | id input | harness input |
| `decode_out/real64-ids.json` | id input | harness input |
| `decode_out/tentasks-all-ids.json` | id input | harness input |
| `decode_out/ten-tasks.json` | prompt set | harness input |
| `decode_out/ten-tasks-authored-t7-t10.json` | prompt source | harness input |

- Cap: 19 files kept at E3 (issue #43), now 20 with this README.
- The ten `final4-*-run0-crow.json` records stay tracked because they are the greedy answers the identity gate compares new runs against, not raw output of a run.
