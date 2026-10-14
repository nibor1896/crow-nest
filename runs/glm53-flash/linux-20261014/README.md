# linux-20261014

- Date: 2026-10-14
- Tickets: #230
- Content: the first GLM-5.3-Flash runs on Linux (RTX 5090, Arch Linux, driver 610.57.04), `glm5_run -n 16` with default switches and `CROW_STAGE_PAR=0` (the #169 smoke), container sha256 `9ce11456c68f…`
  - `smoke16-host.json`: `CROW_PINNED_ALLOC=host` at `dc2edbb`, ids = Windows smoke
  - `smoke16-fix.json`: default pinned allocation (`register`) with the whole-record register pieces, ids = Windows smoke
  - the default run before the fix panicked at row 34 with `CUDA_ERROR_INVALID_VALUE` (logs local)
- Paths: personal paths replaced by `<home>` (#231)
