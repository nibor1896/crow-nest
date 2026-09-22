# Handoff PM — frische Session, 2026-09-20 (~15:00)

Vorheriger Stand: HANDOFF-2026-09-20.md (morgens). Diese Datei ersetzt sie.

## Was heute abgeschlossen ist

### crow (Repo ~/Projects/crow) — main gepusht bis `6fb4b31`
- `185bce4` **#203 Fix 1**: Cut-Tool-Call → strukturiertes Ergebnis, Konversations-Reparatur,
  500-Sicherheitsnetz. 2070 Tests grün. Fixes 2–4 aus dem Issue OFFEN.
- `6fb4b31` **#204/#205-Fixes** (Subagent gebaut, reviewt, 2070 Tests, operating-point 9/9):
  - #205: `rollover_digest` spricht jetzt den Dialekt des Zuges (kein `enable_thinking:false`
    mehr — das brach den warmen Prefix bei Token 3 und kostete 2× 180k+ COLD-Prefill à 217/223 s,
    Timeout tot, Antwort verworfen). Timeout skaliert mit Prompt (835 tok/s gemessen, Teiler 700),
    Antwort-Cap max(400, 2000), `<think>` wird gewaschen. Caller durchgereicht (run_turn,
    crow.py, crow_gui.py).
  - #204: Pump-Guard. WICHTIG dabei gemessen: Auf WebKitGTK wirft `evaluate_js` NIE —
    pywebview loggt GError 601 nur (logger "pywebview"). Failure-Klassifikator an zwei Türen
    (Exception + Logging-Tap), EIN Reload-Versuch (load_html der gespeicherten Seite),
    dann Freeze mit genau einer stderr-Zeile, Transients auf 3 gedeckelt.
  - render_page (#175-Nachtrag): `--enable-unsafe-swiftshader --use-angle=swiftshader`
    (Chromium 144 hat SwiftShader-Fallback entfernt), `--v=0` → Seiten-Konsole wandert ins
    Tool-Ergebnis, byte-identische Captures → WARN. Gemessen: rAF lief unter Virtual-Zeit
    vorher NIE an (drei Varianten, byte-identische 92.027-B-Fangs) — deshalb auditete der
    Voxel-Agent "environment-blocked, page correct" auf kaputtem Spiegelbild.
- Issue-Kommentare: #203 (Fix-1-Referenz), #204 (heutiger Crash + Kernbefund), #205 (Fix).
- Ticket-Neu: crow-nest #68-Kommentar (Degenerations-Episode voxel: 334 Runden bis 181k,
  zwei Loop-Signaturen, Goal-Checker rief 8/8 trotz report scene_visible:false).

### crow-nest (Repo ~/Projects/crow-nest) — main LOKAL bis `e27f005`, NICHT gepusht
- `7586234` #82-Shutdown-Fix (SIGTERM/SIGINT-Watcher + ctx_hard_reset) — Stand von früh.
- `487b3ab` #82-Erweiterung: Watcher nimmt **SIGHUP + SIGQUIT**. Grund: der Fix-Run-serve
  starb 11:58 mit seinem systemd-Scope (kein SIGTERM) → vierter Leak. serve-Tests 90/90.
- `e27f005` Gate-clippy-Erwartung 1458 → 1459: TOOLDRIFT (arch-Update 07:22, neue Lints),
  gemessen an 7586234 im cleanen Worktree VOR meinem Edit — meine Commits bringen 0 neue
  Warnungen.

## #82-Beweisstand (Runde 1, 14:07, rebootet, CROW_DROP_DBG=1)

- Baseline 54 GiB / Refcount 4 → serve geladen (45,1 GiB Tier) → **SIGTERM**: Watcher-Zeile kam,
  volle Drop-Sequenz (Engine→ThreeStates→Residency→Weights→Ple→Params→Scratch, Allos
  2243→459), **Exit in 5 s**, Refcount zurück auf **4**, VRAM leer (901 MB).
- Die 45 GiB im "used" danach = dokumentierter Treiber-Pool (#15, lazy): **Ballon-Beweis** —
  36 GiB in 3 s gefaultet, Swap 0 → Pool wird unter Druck zurückgegeben. KEIN Leak.
- SIGHUP-Runde (volles Tier) fehlt noch: blockiert, weil zcode-Electron inzwischen UVM+~8 GiB
  VRAM hält → Loader rechnet konservativ (MemAvailable-Pfad) und verweigert die Config
  (manager.rs:203). Der HUP-Pfad ist identisch zum TERM-Pfad (nur die Signalnummer im Set).

## Was die NÄCHSTE Session machen muss (Reihenfolge)

0. **VOXEL-TEST KOMPLETT NEU STARTEN mit maximaler visueller Qualität** (robin will das;
   geplant für Mittwoch, ggf. mit Claude — der Stand ist vorbereitet):
   - Memory der gescheiterten Läufe ist GELÖSCHT (voxel_cnq3/.crow, voxel_cnq2/.crow).
   - Frisches Zielverzeichnis: `voxel_cnq4` — enthält NUR `reference.png` (robins
     Referenzbild: isometrische Voxel-Miniatur, winzige Voxel, extreme Detaildichte,
     warmes Studio-Licht) + `.crow/root.json` {"mode":"auto"} (YOLO vorbelegt).
   - Der Task-Prompt für CROW: `/home/nibor1896/Projects/localconf/prompt_crow_voxel_3.md`
     — Referenz-first (Schritt 0: read_image reference.png), 25k+ feine Voxel per
     Generator, Pflicht-Seh-Loop (5 Runden render_page → read_image → 3 Mängel → fix),
     bewiesene three.js-Inline-Kette aus voxel_cnq3 wiederverwenden.
   - Plattform-Seitig erledigt heute: Abbruch-Kappe 8192→16384 (CROW_MAX_TOKENS),
     append_file-Tool, render_page mit Konsolen-Zeilen + Identical-Capture-WARN
     (crow 6301e0e). Die Werkzeuge funktionieren — der Loop ist lauffähig.
   - Ablauf: crow-nest starten (Electron-GPU-Hinweis unten beachten), Crow öffnen,
     Prompt aus prompt_crow_voxel_3.md rein, YOLO.
1. **#82 fertig**: Gate muss GRÜN laufen — geht nur, wenn Electron kein GPU+UVM hält
   (frisch gebootet VOR dem Start schwerer GUI-Apps, ODER: zcode komplett schließen, in einem
   simplen Terminal `tools/gate-linux.sh /tmp/gate-82` laufen lassen, zcode wieder öffnen).
   Dann: `git push origin main` (3 Commits), Issue #82 schließen (Beweis: Runde 1 oben +
   Gate). HUP-volles-Tier-Beweis bei Gelegenheit nachholen (nächster Reboot, vor Electron).
2. **#80**: blindes pointwise Grading (checklist.json, nur answers.md), `report`, Doku-Verdict,
   ten-task-Runner-Files committen (`tools/ten-task-run.py`, `tools/test_ten_task_run.py`,
   `docs/ten-task-linux.md`, untracked). B-llama-32k als fehlende Kontrolle dokumentieren.
3. **#203**: Fixes 2–4 (Issue crow#203).
4. **#204** REST: der Push-Guard ist DRIN (6fb4b31) — im Issue schließen oder Rest
   (WebKit+NVIDIA-VRAM-Druck-Thematik) abtrennen.
5. Offen alt: crow-nest #65, #61/#62, #68 (Kommentar heute, unbearbeitet).

## Kritische Betriebs-Gefahren (heute blutig gelernt)

- **ZCode-Sandbox killt Hintergrundprozesse mit SIGKILL, wenn der Tool-Call endet.** Ein serve,
  der über Call-Grenzen leben soll, MUSS innerhalb eines einzigen Calls geladen+signalisiert+
  beendet werden. Der vierte Leak heute war genau das (SIGKILL meiner Sandbox, kein SIGHUP).
- **kein GPU-Kontakt beim frischen Boot, wenn der Nutzer testen will** — und Electron (zcode)
  greift irgendwann am Tag nach GPU+UVM; danach verweigert der Loader den Betriebnpunkt
  (konservativer MemAvailable-Pfad, manager.rs:203). Gate/serve früh nach Boot fahren.
- Die "[cache] COLD" Runden kommen von CROW-Prefix-Rewrites (Digest-Leg, Tail-Rewrites),
  NICHT vom Engine-Cache und NICHT von Bildern (13-Bild-Turns: WARM wie COLD, identische grids).
- `viewer2.html` in voxel_cnq2 ist ein abgebrochener Schreibvorgang (1208 B, mitten im CSS) —
  Distractor; `viewer.html` ist das Deliverable. Voxel-Seiten laufen (52 fps), aber WebKit+NVIDIA
  komponiert die GL-Ebene unsichtbar (SIGSEGV libnvidia-eglcore, #204) — Agent braucht die
  Konsole als Beweis, nicht nur Pixel (render_page liefert sie jetzt).

## Startbefehle (unverändert)

- llama-Arm: `python3 ~/.local/share/crow/tools/start-server.py flash-next-q2-k-xl` (8083)
- crow-nest: `crow-nest` (8099); Crow-Fenster: `crow --base-url http://127.0.0.1:8099/v1`
- Engine-Log: `~/.local/state/crow/logs/engine.log` · Gate: `tools/gate-linux.sh`
