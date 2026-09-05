# Ten-task expected answers & grading protocol

Fixierung der Kernanforderungen und der Auswerteregeln für das Ten-Task-Gate
(siehe `docs/ten-tasks.md`, spec §5.2). Die Kernanforderungen wurden aus den
eingefrorenen Prompts (`decode_out/ten-tasks.json`, run 0) abgeleitet; die
Vorbewertung in §3 kalibriert die Regeln an den vorhandenen Run-0-Antworten
(`decode_out/ref-run0-llama.json`, `decode_out/crow-run0-crow.json`) und ist
KEIN Messergebnis. Finale Urteile fallen nur über einen echten Messlauf.

Bewertet wird immer gegen den **eingefrorenen Prompt-Text**, nicht gegen die
aktuelle Arbeitskopie einer Quelldatei (für t1b: der im Prompt eingebettete
Quelltext ist ein älterer Snapshot von `engine/src/kernels.rs` — siehe
Widerspruch W1 in §4).

---

## 1. Bewertungsregeln (deterministisch)

**Pass** — alle Kernanforderungen des Tasks materiell erfüllt.
- Wording ist frei; Formatierung ist frei.
- Code wird **semantisch**, nicht textuell bewertet: gleiche Datenstrukturen,
  gleiche Kontrollflüsse, gleiche Randfälle — andere Bezeichner sind ok.
- Rechenwerte müssen exakt stimmen (bei t6b: jeder Zwischenwert).

**Partial** — Mehrheit der Kernanforderungen erfüllt, UND mindestens eine
Kernanforderung falsch ODER die Antwort bricht vor Vollständigkeit ab
(Abbruch, Token-Cutoff, Degeneration nach substantiellem Anfang).

**Fail** — Kernanforderungen überwiegend verletzt, ODER Token-Degeneration
(Wiederholungs-Loops, leere `<think>`-Zyklen, Rollen-Tags im Text,
sprach-Kontamination), ODER Fehlantwort auf die gestellte Frage.

Abbruch ist deterministisch ein Defizit: Was im Record nicht steht, ist
nicht geliefert — egal warum (auch bei `max_tokens`-Cutoff, siehe W2–W6).

**Konsistenzregel** — wo crow und llama Aussagen über überprüfbare Fakten
machen (Rechenwerte, Code-Semantik, Zitate aus dem Prompt-Material), müssen
sie übereinstimmen. Widersprechen sie sich, entscheidet die Sachlage
(Quelle im Prompt, nachrechenbar, Dateiinhalt) — nicht die Mehrheit und
nicht das Prestige der Referenz. Ein falsches Fakten-Statement ist falsch,
auch wenn nur eine Seite es macht.

**Eine Antwort, die die Prämissen der Frage korrekt zurückweist** („das
Gefragte existiert im Material nicht"), ist eine korrekte Antwort, wenn die
Prämisse tatsächlich falsch ist — sie ist eine Fehlantwort, wenn die Prämisse
im Material stimmt. Für t1b-Frage 2 ist der Fall in §2.7 und W1 dokumentiert.

---

## 2. Kernanforderungen je Task

### 2.1 t1-read — Tensor-Pfad Datei → Backend-Buffer

Der Prompt zeigt zwei Partial-Dateien (`llama-model-loader.cpp`,
`llama-mmap.cpp`). Die Frage ist eine Falle mit Substanz: der eigentliche
Kopierpfad ist **nicht** im gezeigten Code enthalten.

1. Benennt Stage 1 korrekt: der Loader-Konstruktor macht nur Metadaten —
   `gguf_init_from_file` (Header/KV), Aufbau von `weights_map` mit
   `llama_tensor_weight` (pro Tensor: Datei + Offset), `n_elements`/`n_bytes`;
   **kein Tensor-Byte wird gelesen**.
2. Benennt Stage 2 korrekt: `llama_file` (llama-mmap.cpp) stellt nur
   Byte-Primitiven (`read_raw`, `read_raw_at`, `read_aligned_chunk`,
   POSIX-mmap) bereit — sie „wissen nichts von Tensoren".
3. Sagt explizit, dass die eigentliche Kopie **nicht im gezeigten Code**
   passiert, und benennt, wo sie im Gesamtsystem passiert:
   `llama_model_loader::load_tensors` / `load_all_data` → Allokation des
   Backend-Buffers (`ggml_backend_buffer` / backend-spezifisch) → Kopie über
   die `read_raw`-Primitiven an die in `weights_map` verzeichneten Offsets
   (oder zero-copy über mmap).
4. Die Funktionsreihenfolge ist in sich stimmig
   (gguf_init_from_file → weights_map → Buffer-Allokation → read_raw/mmap).
5. Eine Antwort, die behauptet, der gezeigte Code selbst lese Tensor-Daten in
   einen Backend-Buffer, ist falsch (Kernanforderung 3 verletzt).

### 2.2 t2-write — LRU-Cache-Header

1. Einzelner, selbstständiger C++17-Header; `template<class K, class V>`;
   Includes beschränkt auf `<unordered_map>`, `<list>`, `<optional>`,
   `<cstddef>` (`<stdexcept>` für den Throw ist implizit der Kontrakt).
2. `explicit` Konstruktor mit capacity; `std::invalid_argument` bei 0.
3. `get(key) -> std::optional<V>`; befördert den Eintrag auf MRU.
4. `put(key, value)` insert-or-update; evictiert bei voller Capacity den
   LRU-Eintrag (update wächst nicht über die Capacity).
5. `erase(key) -> bool`, `size()`, `capacity()`, `clear()`.
6. O(1)-amortisierte Struktur: `unordered_map<K, list_iterator>` +
   `std::list` (splice/move ohne Knoten-Kopie); keine Iterator-Invalidierung
   für unberührte Einträge (Begründung oder korrekte Verwendung von splice).
7. `peek(key) -> std::optional<V>` **ohne** Recency-Änderung.
8. Unit-Tests als `assert()` in einem `main()`, abgedeckt: Eviction-Reihenfolge,
   update-does-not-grow, erase von LRU-head und -tail, Capacity-1-Randfall.
9. Kurze Design-Kommentare dort, wo nicht offensichtlich.

### 2.3 t3-debug — PowerShell/UTF-8-Harness

Gegebene Fakten im Prompt: `prompt.txt` ist pure ASCII (≤127), gültiges
UTF-8, 15831 Zeichen, als JSON eine einzige Zeile; `$prompt` ist
`System.String`, Länge 15831; die Zeichenkette `Snap-In` kommt in
`prompt.txt` **nicht** vor. Server-Fehler: JSON-Parsefehler „ill-formed
UTF-8 byte" bei Zeile 621, Spalte 126, im String
`"Dieses Windows PowerShell-Snap-In enthael…"`.

1. Verortet die Ursache **client-seitig zwischen dem geprüften String und
   dem Wire** — nicht in `prompt.txt`, nicht in `$prompt`, und nicht im
   serverseitigen Batch-Handler (die beigegebene `llama-batch`-Quelle ist ein
   Red-Herring: der JSON-Parsefehler schlägt zu, BEVOR irgendeine
   Batch-Validierung läuft; `llama_batch_allocr::init` wird nie erreicht).
2. Erklärt, warum beide „checks" saubere Ergebnisse liefern können: beide
   prüfen echte Invarianten (Datei ist ASCII/UTF-8; Variable ist der richtige
   String) — die Korruption entsteht erst danach, beim Senden (klassisch:
   Windows PowerShell 5.1 kodiert einen String-Body ohne `charset`-Angabe im
   `Content-Type` mit Legacy-Encoding/ANSI-Codepage statt UTF-8 neu).
3. Zieht den `Snap-In`-Schluss: der im Parsefehler zitierte Text steht nicht
   in `prompt.txt` ⇒ die Bytes, die der Server geparst hat, sind nicht (nur)
   der Inhalt von `prompt.txt`/des gebauten JSON-Körpers; Zeile 621 ist mit
   einem Einzeiler-JSON unvereinbar. Eine Antwort, die `Get-Content` die
   Datei korumpieren lässt oder den Server-Handler beschuldigt, widerspricht
   den gegebenen Fakten.
4. Kleinster korrekter Fix: Body explizit als UTF-8-Bytes senden, z. B.
   `-Body ([System.Text.Encoding]::UTF8.GetBytes($body))` mit
   `-ContentType 'application/json; charset=utf-8'` (Äquivalente ok).
5. Benennt zwei weitere Cmdlets mit derselben Default-Encoding-Falle
   (PS 5.1), z. B. `Invoke-WebRequest`, `Send-MailMessage`, `Set-Content`/
   `Out-File`/`Export-Csv` — zwei plausible genügen.

   Hinweis zur Determiniertheit: Der exakte Mechanismus, wie der deutsche
   PowerShell-Text (und ein 621-zeiliger Body) in den Request gelangt, ist
   aus den Prompt-Fakten allein nicht vollständig ableitbar — bewertet wird
   der prüfbare Kern (Punkte 1–5), nicht eine bestimmte Herleitung. Siehe
   auch §4 (W7).

### 2.4 t4-prose — Portierungs-Annahme (Deutsch, ≤ 400 Wörter)

1. Antwort auf Deutsch, Prosa, ≤ ~400 Wörter.
2. Benennt als wahrscheinlichste falsche Annahme (oder inhaltliches
   Äquivalent davon): dass der Slot-Cache plus die festen Komponenten
   **unterhalb der VRAM-Decke** der Karte bleibt, sodass die Slot-Kosten
   konstant bleiben (312,44 MiB/Slot) — der Text selbst warnt: „Above the
   card's limit the driver silently moves the excess into host memory and
   per-slot cost stops being constant."
3. Begründet, warum GERADE diese Annahme bei halbem Speicher bricht: der
   Spielraum ist bereits minimal (z. B. 593–924 MiB frei bei 62–64 Slots;
   6,51 GiB feste Komponenten; ~17,7–20,4 GiB Cache bei 58 Slots; 1,32 GiB
   KV bei 200k); halber VRAM verschiebt dieselbe Cache-Größe über die Decke,
   der Treiber lagert still in den Host-Speicher aus, das gemessene
   Kosten-/Hitrate-Modell kollabiert.
4. Verwendete Zahlen müssen zum Text passen (6,51 GiB; 312,44 MiB/Slot;
   58 Slots ≈ 20,43 GiB bzw. 17,70 GiB je nach Textstelle; 1,32 GiB KV;
   90,17 GiB Experten). Falsche Zahlen = falsche Kernanforderung.

### 2.5 t5-agent — CMake-Kommandozeile aus Transcript

1. Gibt EINE exakte Configure-Kommandozeile in ein **frisches** Build-
   Verzeichnis (`cmake -B <neu> …`), die nur den Server-Target baut:
   `-DGGML_CUDA=ON` und `-DLLAMA_BUILD_SERVER=ON`, plus „nur Server" entweder
   über `-DLLAMA_BUILD_TOOLS=OFF -DLLAMA_BUILD_EXAMPLES=OFF
   -DLLAMA_BUILD_TESTS=OFF` oder über einen `--target server`-Build-Schritt.
2. Nutzt `GGML_CUDA` als CUDA-Option (das Transcript zeigt, dass
   `LLAMA_CUBLAS`/`LLAMA_CUDA` als deprecated auf `GGML_CUDA` umleiten).
3. Listet Annahmen, die das Transcript NICHT bestätigt, mindestens: exakter
   Name des Server-Targets; Default/Abhängigkeit von `LLAMA_BUILD_COMMON` in
   einer frischen Configure; Verfügbarkeit/Pfad des CUDA-Toolkits; Generator
   und Build-Type. Mindestens zwei davon müssen genannt sein.
4. Erfindet keine Transcript-Bestätigungen. Insbesondere: die leeren
   `read_file`-Ergebnisse ab Offset 480 (Dateiende) bestätigen nichts und
   dürfen nicht als Bestätigung zitiert werden.

### 2.6 t6-reason — k-tes kleinstes DISTINCT in Range

1. Algorithmus: offline nach `r` (oder persistent/wavelet-Äquivalent) mit
   **Last-Occurrence-Markierung**: beim Vorkommen von Wert `v` an Position `i`
   wird `prev[v]` auf 0 gesetzt und `i` auf 1; `v` ist in `[l, r]` vorhanden
   gdw. seine letzte Occurrence ≤ r an Position ≥ l liegt. k-tes distinct =
   k-th gesetzte Position in Version `r`, Prüfung `≥ l`.
2. Datenstruktur präzise genug zum Implementieren (persistentes
   Segmentbaum-Array über Positionen mit Versionen je `r`, oder BIT offline
   + binäre Suche).
3. Komplexität angegeben UND begründen: O((n + q) log n) Zeit
   (bzw. O(q log² n) für die BIT-Variante), O(n log n) bzw. O(n + q)
   Speicher; innerhalb von 2 s für n, q = 2·10⁵.
4. Konkreter Fall n=8, a=[3,1,3,4,1,5,9,2] mit **exakten** Antworten und
   gezeigtem Zwischenzustand:
   - Query (1,5,2): Range [3,1,3,4,1], Distinct {1,3,4} → **3**
   - Query (2,7,3): Range [1,3,4,1,5,9], Distinct {1,3,4,5,9} → **4**
   - Query (4,4,2): Range [4], nur 1 Distinct < k → **-1**
5. Zwei Eingaben, die eine naive Implementierung brechen, mit Begründung
   (je zwei plausible genügen: All-Equal/heavy-duplicate-Array, das
   Occurrence-Zählung statt Distinct-Logik strandet; k größer als
   Distinct-Anzahl (-1-Pfad); alternierendes Muster mit maximalen
   Last-Occurrence-Updates; Ein-Element-Ranges).

### 2.7 t1b-read-lang — drei Fragen zum (eingefrorenen) kernels.rs

Normantworten gegen den **eingefrorenen Prompt-Quelltext**:

**Frage 1 (FP4-GEMV + zweistufige Skala).** Die Kernel sind
`gemv_fp4`, `gemv_fp4_b`, `gemv_fp4_ptrb` (im eingefrorenen Stand; die
aktuelle Datei kennt zusätzlich `gemv_fp4_bs`, das im Prompt nicht vorkommt —
beide Antworten gelten, Bewertung gegen den Prompt). Mechanismus muss
enthalten sein: Block = 36 Byte (4 UE4M3-Sub-Block-Skalen + 32 Byte
gepackte Nibbles = 64 Gewichte); pro Sub-Block
`s = ue4m3(blk[sb]) * gs` mit `gs = gs_ptr[0]` (globale Per-Tensor-Skala);
`part` = Summe über 16 Nibbles `e2m1(nib) · x[...]`; dann `acc += part * s`;
Block-Reduktion am Ende. Die zweistufige Skala tritt also als **Produkt pro
Sub-Block** vor der Akkumulation ein.

**Frage 2 (`__nanosleep`-Poll-Pfad).** Die korrekte Antwort ist die
**Prämissen-Zurückweisung**: Weder der eingefrorene Prompt-Quelltext noch die
aktuelle `engine/src/kernels.rs` enthält IRGENDEINE Poll-Schleife mit
`__nanosleep` (und auch keine „memop-based release"). Verifikation:
`grep __nanosleep` → 0 Treffer in beiden. Eine Antwort, die einen solchen
Pfad erfindet oder beschreibt, ist FALSCH; die korrekte Antwort stellt das
klar und benennt, was stattdessen da ist (block-lokale `__syncthreads()`-
Reduktionen, keine Host-Device-Polling-Synchronisation in dieser Datei).
Siehe Widerspruch W1.

**Frage 3 (Stride-Loop-Regel).** Der Stride-Loop
(`for (int i = threadIdx.x; i < limit; i += blockDim.x)`) hält EINE feste
Launch-Konfiguration für JEDE Laufzeitgröße korrekt. Der verhinderte
Korrektheits-Bug: bei einem Plain-Guard-Kernel
(`i = blockIdx.x*blockDim.x + threadIdx.x; if (i >= n) return;`) hängt die
Vollständigkeit davon ab, dass der Launch `ceil(n/blockDim)` Blöcke umfasst —
ein Launch, der aus einer falschen/älteren/geschätzten `n` gebaut wurde,
**überspringt stumm Elemente** (kein Fehler, falsches Ergebnis).
Konkretes Beispiel aus der Datei nennen: `gemv_bf16`/`gemv_bf16_b`
(`k_dim` ist Laufzeitwert, Loop `i < k_dim; i += blockDim.x`) oder
`rms_group` (`i < 2560; i += 256`); als Kontrast die Plain-Guard-Kernel
`dequant_fp4_flat`/`bf16_to_f32`, die nur korrekt sind, weil der Launch aus
`*n_p` dimensioniert wird.

### 2.8 t3b-debug-syn — Host/Device-Koordination

Die Frage verlangt sechs Teile; bewertet wird, ob alle sechs materiell
vorliegen:

1. Benennt einen konkreten Koordinations-Bug, der erst aus BEIDEN Snippets
   folgt: Der Konsument-Kernel ist VOR der ersten Submission gestartet und
   pollt `flags[slot]` (device-seitig, `volatile`, `<`), während der
   Publisher die HOST-Seite ist, die `r.flags[slot] = ++r.submitted` als
   schlichten, ungezäunten Store schreibt. Bug-Familie: Host→Device-
   Sichtbarkeit/Ordnung — ohne device-sichtbaren Speicher (mapped/pinned)
   plus Release-Semantik auf Host-Seite (bzw. System-Fence) ist der Publish
   für den Poller nicht garantiert sichtbar oder nicht geordnet gegenüber
   den `fill_payload`-Writes. (Eine kohärent argumentierte ABA-Variante über
   Slot-Wiederverwendung mit Start-Late-Consumer wird akzeptiert.)
2. Exakte Ereignissequenz (launch vor submit; init 0; poll 0; publish 1 …).
3. EINE konkrete 3-Schritt-Interleaving, die fehlgeht (Schritt-Reihenfolge
   mit beobachteten Werten, nicht eine Allgemeinplatz-Beschreibung).
4. Den Standard-Fix (mapped/pinned, device-sichtbares Flags-Array +
   Release-Store/`__threadfence_system`-Äquivalent bzw. Konsumenten erst
   nach der ersten Submission starten; doorbell/MemOp-Pattern).
5. Die `==`-Frage substantiell beantwortet: `==` allein fixt KEINE
   Sichtbarkeits-/Ordnungsursache; `==` ist nur korrekt, wenn der Konsument
   den exakten Wert nicht verpassen kann. Bei Slot-Wiederverwendung kann ein
   spät startender Konsument einen LÄNGEREN Sequenzwert sehen — `<` toleriert
   das, `==` hängt ewig.
6. Die Wrap-Around-Wendung: bei Wrap des Zählers kann `<` falsch
   weiterlaufen/hängen (neuer Wert numerisch < target, ABA), während `==`
   nur beim exakten Treffen freigibt — die Antwort muss die Abhängigkeit
   der Wahl von Wrap-/Wiederverwendungs-Annahmen begründen (jede in sich
   konsistente, begründete Richtung genügt).

### 2.9 t2b-write-refactor — Refactor mit Verhaltens-Contract

1. Benennt die Duplikation (zwei identische Feld-Parse-Blöcke) und den
   generischen Helfer mit Signatur, etwa
   `template<typename T> bool parse_field(const uint8_t* rec, size_t len,
   size_t& off, std::vector<T>& out, uint32_t expected_kind)` —
   parametrisiert auf Elementtyp `T` (`sizeof(T)`) und erwartetem `kind`.
2. Findet den echten Bug: im zweiten Block steht
   `if (!read_u32(kind_b) || !read_u32(kind_b)) return false;` — `count_b`
   wird NIE gelesen (zweites Mal wird `kind_b` überschrieben); Folge:
   initialisierter-Use/UB und eine Bounds-Prüfung gegen Müllwerte.
3. Benennt die latente Portabilitätsfalle: `memcpy` multi-Byte-Integer setzt
   Host-Endianness = Record-Layout (Little-Endian) voraus.
   (`off + (size_t)count*4`-Overflow auf 32-Bit-`size_t` gilt als
   zusätzliche Nennung, nicht als Ersatz.)
4. Behält das Verhalten EXAKT — inklusive des Bugs — ODER flaggt den Bug
   ausdrücklich als bewusste Abweichung mit Begründung („sagen, was man
   ändern würde und warum"); ein STILLER Fix verletzt den Contract, ein
   nicht erwähnter Bug verletzt Punkt 2.
5. Zeigt den refactored Code vollständig; die Out-Param-Reihenfolge
   (`a_out` vollständig vor Begin des `b`-Parsings) bleibt erhalten.

### 2.10 t6b-reason-multi — KV-Cache-Rechenkette (alle Zwischenwerte exakt)

1. **Step 1:** 12 × 2 × 256 × 1 B = **6.144 Byte/Token**.
2. **Step 2:** 262.144 × 6.144 = **1.610.612.736 Byte** = **1,5 GiB exakt**
   (2¹⁸ · 6 · 2¹⁰ = 6 · 2²⁸ = (6/4) · 2³⁰).
3. **Step 3:** Budget 3,0 GiB = 3.221.225.472 Byte; 3.221.225.472 ÷ 6.144 =
   **524.288 volle Token**, Division geht exakt auf:
   6.144 · 524.288 = (6 · 2¹⁰) · 2¹⁹ = 6 · 2²⁹ = 3 · 2³⁰. ✓
4. **Step 4:** BF16 = 2 Byte/Element → 12.288 Byte/Token;
   262.144 × 12.288 = **3.221.225.472 Byte** — exakt das 3,0-GiB-Budget;
   Overhead über FP8 = **+100 %**.
5. **Step 5:** 200.000 × 12.288 = **2.457.600.000 Byte ≈ 2,289 GiB**
   ≤ 3.221.225.472 Byte → **PASS**; Marge = 763.625.472 Byte =
   **728,25 MiB exakt** (763.625.472 / 2²⁰).

Alle fünf Zwischenwerte müssen erscheinen; ein einziger falscher Zwischenwert
macht die Kernanforderung falsch. (Verifiziert 2026-09-03, inkl. Binär-Beweis
der Exaktheit; die Marge 728,25 MiB ist ein Vielfaches von ¼ MiB.)

---

## 3. Vorbewertung Run 0 (Kalibrierung, kein Messergebnis)

Referenz-Prior aus `docs/ten-tasks.md`: llama 10/10. Bewertung erfolgt über
die Run-0-Records, wie sie existieren. `Pass*` = materiell korrekt im
sichtbaren Umfang; die Antwort ist an `max_tokens` gekappt, sodass
restliche Kernanforderungen im Record nicht nachgewiesen werden können
(siehe W2–W6). Crow-Antworten stammen vom unkalierten BASE-Container
(`converter/Qwen3.8-Flash-Next-CNQ4.5.cnq`); Degeneration ist ein
messbares Ergebnis, keine Anschuldigung.

| Task | llama run 0 | crow BASE run 0 | Begründung (Zitate aus den Records) |
|---|---|---|---|
| t1-read | Pass* | **Fail** | Llama benennt beide Stufen, den fehlenden Kopierpfad und `load_tensors`/`read_raw` (Kappung im Schlusssatz). Crow startet mit „What the code actually shows" und fällt sofort in leere `<think>`-Wiederholungen mit geleakten `user`/`assistant`-Tags — keine Funktion, keine Kopiestelle. |
| t2-write | Pass* | **Fail** | Llama liefert kontraktgetreue Struktur (unordered_map + list + splice, capacity-0-Throw), wird aber in `erase()` an max_tokens=640 gekappt — Tests fehlen im Record (W2). Crow ist reine Token-Degeneration: hunderte ```` ```cpp ````-Fragmente, kein Code, keine Tests. |
| t3-debug | Pass* | **Fail** | Llama trifft den Kern („the string being sent to the server is **not** the content of prompt.txt", PS-ANSI/Encoding), wird aber vor Fix und Cmdlets gekappt. Crow beginnt richtig („character encoding mismatch … system's default ANSI codepage"), degeneriert dann in „CP12522,2222…" — zwei-Checks-Erklärung, `Snap-In`-Schluss, Fix und Cmdlets fehlen. |
| t4-prose | Pass | **Fail** | Llama benennt die VRAM-Ceiling-/Slot-Kosten-Annahme, zitiert die Textwarnung („driver silently moves the excess into host memory") mit passenden Zahlen. Crow hat eine brauchbare These (Konstanz der Slot-Kosten), aber korrupte Fakten (KV-Cache „132 GiB" statt 1,32 GiB) und versackt im Wiederholungs-Loop „plus der KV-Cache (132 GiB) plus der Host-Tier (132 GiB) …". |
| t5-agent | Pass | **Fail** | Llama liefert eine gültige frische-Dir-Kommandozeile (`-DGGML_CUDA=ON -DLLAMA_BUILD_SERVER=ON`, Rest OFF) plus sechs unbestätigte Annahmen. Crow verheddert sich in einem Think-Loop („the file is named ‚CMakeLists.txt' not ‚CMakeLists.txt'" ×20) — keine Kommandozeile, keine Annahmenliste. |
| t6-reason | Pass* | **Fail** | Llama leitet Last-Occurrence + Offline-nach-r korrekt her, wird aber vor Struktur-Feinspezifikation und konkretem Fall gekappt (W3). Crow hat den richtigen Ansatz (`prev[i] < l` korrekt), degeneriert dann in zehnfach „We need to find the k-th distinct value." — konkreter Fall (3/4/-1), Beweis und zwei Brecher-Eingaben fehlen. |
| t1b-read-lang | Pass | **Fail** | Llama beantwortet Q1 korrekt (drei Kernel, `s = ue4m3(blk[sb]) * gs`, `acc += part * s`) und weist Q2 korrekt als Fehlprämisse zurück („There is **no** polling loop using `__nanosleep` in the provided source code"). Crow ist in Q1 materiell korrekt (beste Crow-Antwort des Satzes — Kernelnamen und Skalen-Mechanismus stimmen, Zitat-Konstanten aber zu Lücken degeneriert: „sb * 1 + j", „blk[4 + (idx >> )]"), bricht in Q2 nach „Hmm. Let me" ab; Q3 fehlt. 1 von 3 Fragen = keine Mehrheit. |
| t3b-debug-syn | Pass* | **Fail** | Llama verortet die Familie richtig (Host-Store ohne Fence/atomics gegenüber device-volatile-Poller, Launch vor Submission), wird aber vor Bug-Finalisierung, Interleaving, Fix und `==`/Wrap gekappt (W5). Crow gibt eine TOCTOU-Geste mit Fehlzuordnung („proceed to execute the `fill_payload`" — das ist host-seitig) und kippt dann in Kontamination: wiederholte chinesische Kontext-Dialoge („这是一个关于多轮对话中上下文管理的问题"), keine Interleaving, kein Fix, kein `==`/Wrap. |
| t2b-write-refactor | Pass* | **Fail** | Llama findet beide Defekte präzise (doppeltes `read_u32(kind_b)` → `count_b` nie gelesen; Endianness-Hazard) und behandelt den Fix/Contract-Konflikt explizit, wird aber vor dem vollständigen refactored Code gekappt (W6). Crow ist Degeneration: „Let me analyze the code carefully." ×20 mit leeren `<think>`-Blöcken — keine Duplikat-Analyse, kein Bug, kein Code. |
| t6b-reason-multi | Pass* | **Fail** | Llama rechnet Step 1 und 2 exakt (6.144 B/Token; 1.610.612.736 B), wird mitten in Step 2 an max_tokens=512 gekappt — Steps 3–5 fehlen im Record (W4). Crows Step 1 ist rechnerisch falsch: „12 * 2 * 256 * 1 = 24 * 256 = **25,048**" (korrekt: 6.144), danach Loop-Degeneration („Let me compute: 262,144 * 25,048" ×6). |

Zwischenstand Run 0 (Vorbewertung): llama 10 × Pass (davon 7 mit
Token-Cutoff-Vorbehalt), crow BASE **0/10** — 9 klare Degenerations-Fails,
t1b als Teil-Credit (Q1 korrekt), aber ebenfalls Fail. Konsistenzregel: keine
Task-Paare, in denen beiden Engines überprüfbare Fakten widersprechen;
der eine relevante Fall (t1b Q2) entscheidet die Sachlage zugunsten von
llama (W1).

---

## 4. Dokumentierte Prompt-/Erwartungs-Widersprüche

Diese Punkte werden dokumentiert, nicht still entschieden:

- **W1 — t1b Frage 2 hat eine falsche Prämisse.** Die Frage behauptet eine
  `__nanosleep`-Poll-Schleife („next to the memop-based release") — weder der
  eingefrorene Prompt-Quelltext noch die aktuelle `engine/src/kernels.rs`
  enthält `__nanosleep` oder eine MemOp-Release (je 0 Treffer). Korrekte
  Antwort ist die Prämissen-Zurückweisung (so llama). ZUSÄTZLICH ist der
  eingebettete Quelltext ein älterer Snapshot der Datei (4 Diff-Stellen:
  `gemv_fp4_bs` fehlt im Prompt; `acc_combo` steht im Prompt noch in der
  alten, verloren schreibenden Variante; Kernel-Namensliste weicht ab).
  Bewertung muss gegen den eingefrorenen Text laufen — die Aufgaben-notiz
  „formuliere die `__nanosleep`-Poll-Pfad-Normantwort" ist damit selbst
  gegen die Quelle widerlegt.
- **W2 — t2-write: `max_tokens=640` macht einen vollständigen Pass
  unmöglich.** Der Kontrakt verlangt Header + Tests in `main()`; die
  llama-Referenz wird in `erase()` gekappt, bevor `peek` und alle Tests
  erscheinen. Nach den hiesigen Regeln kann bei diesem Budget KEINE Engine
  t2-write strict-Passen. Either `max_tokens` anheben oder Kontrakt kürzen —
  Entscheidung vor dem nächsten Messlauf, nicht beim Bewerten.
- **W3 — t6-reason: `max_tokens=640` vs. fünf geforderte Teile**
  (Herleitung + Beweis + Struktur + durchgerechneter Fall + zwei Brecher-
  Eingaben). Die Referenz bricht in der Herleitung ab; der konkrete Fall
  (3/4/-1) erscheint nirgends.
- **W4 — t6b-reason-multi: `max_tokens=512` vs. Fünf-Schritt-Kette mit
  allen Zwischenwerten.** Die Referenz endet mitten in Step 2; Steps 3–5
  (524.288-Token-Beweis, BF16, 200k-Floor) fehlen im Record.
- **W5 — t3b-debug-syn: `max_tokens=640` vs. sechs geforderte Teile.**
  Die Referenz endet vor der Festlegung auf den Bug; Interleaving, Fix und
  `==`/Wrap fehlen.
- **W6 — t2b-write-refactor: `max_tokens=768` vs. „Show the refactored code
  complete".** Die Referenz kappt nach Teil 2; Teil 3 (Code) fehlt. Zudem
  eine in-Prompt-Spanne: „Keep the observable behaviour EXACTLY identical"
  kollidiert mit dem existierenden echten Bug (doppeltes `read_u32(kind_b)`)
  — auflösbar über „finden, nicht still fixen, Änderung benennen", aber ein
  stummer Fix bzw. verschweigen muss jeweils als Verstoß zählen.
- **W7 — t3-debug ist in sich unterbestimmt.** Aus den Prompt-Fakten (ASCII-
  Datei, korrekte Variable, `Snap-In` abwesend, Einzeiler-JSON vs. Fehler in
  Zeile 621) folgt sicher nur: die Korruption liegt zwischen geprüftem String
  und Wire, und der gesendete Body ist nicht das gebaute JSON. Der exakte
  Mechanismus (wie deutsche PowerShell-Texte in den Body gerieten) ist nicht
  herleitbar; Bewertung nur über den prüfbaren Kern (§2.3, Punkte 1–5).
- **W8 — Namensdrift in `docs/ten-tasks.md`.** Die Fixierung sagt „Prompts
  live in `parity-prompts.json`; expected-answer notes in
  `parity-expected.md`" — die Ten-Task-Prompts liegen tatsächlich in
  `decode_out/ten-tasks.json`, die Expected-Notes in dieser Datei
  (`docs/ten-task-expected.md`). Dokumentiert als Drift, nicht umbenannt.

---

## 5. Finale Abnahme

**Finale Abnahme je Task: robin** — Datum wird beim echten Messlauf eingetragen.

| # | Task | Abnahme (robin) | Datum |
|---|------|-----------------|-------|
| 1 | t1-read | ☐ | |
| 2 | t2-write | ☐ | |
| 3 | t3-debug | ☐ | |
| 4 | t4-prose | ☐ | |
| 5 | t5-agent | ☐ | |
| 6 | t6-reason | ☐ | |
| 7 | t1b-read-lang | ☐ | |
| 8 | t3b-debug-syn | ☐ | |
| 9 | t2b-write-refactor | ☐ | |
| 10 | t6b-reason-multi | ☐ | |


---

## Rev2 der Serie (2026-09-03, nach diesem Protokoll — bearbeitet vom Hauptagenten)

Die Befunde W1–W8 sind eingearbeitet:
- **Budgets angehoben** (t1-read/t3-debug/t4-prose/t1b-read-lang/t6b 1024, t2-write/t3b/t2b 1280, t5-agent/t6-reason 1536) — vollständige Pässe sind damit möglich; die Pass*-Vorbehalt-Liste (W2–W6) gilt für die Run-0-Historie, nicht für kommende Messungen.
- **t1b-Frage 2 ersetzt** (W1, falsche Prämisse __nanosleep) — neue Frage: `gemv_fp4_bs`-Stride-Kontrakt (Ausgabeindex-Formel, Aufrufer-Layout, Fehlverhalten bei falschem Stride).
- **t1b-Material aktualisiert:** embeddet jetzt den Post-Fix-`kernels.rs` (inkl. `gemv_fp4_bs`, deterministisches `acc_combo`, Row-Offset-fix im `gemv_fp4_ptrb`).
- **t6b-Marge korrigiert:** exakt 728,25 MiB (763.625.472 B).
- W8 bleibt als Doku-Hinweis: maßgeblich sind `decode_out/ten-tasks.json` (Serie) und diese Datei.

Die Vorbewertung oben (crow-BASE 0/10) bezieht sich auf Run 0 mit Rev1 — sie bleibt als Kalibrierungsbeispiel und Degenerations-Beleg gültig; die kommende Messung läuft mit Rev2 auf dem neuen Container.


---

## Rev3 der Bewertungsregeln (2026-09-03, fable gate — zwei Asymmetrien geschlossen)

**Kernanforderung 0 (alle Tasks): Antwortsprache = Promptsprache.**
- t4-prose ist auf Deutsch gestellt (die Forderung steht im Prompt selbst, nicht nur im Protokoll), die übrigen neun auf Englisch.
- Konsistent falsche Sprache bei korrektem Inhalt: max **Partial** (kann Kernanforderung 1 nicht ersetzen).
- Sprachwechsel INNERHALB einer Antwort (z. B. deutsch → chinesische Dialogfragmente): **Fail**.

**EOS-Regel (alle Tasks):** Die Antwort endet am ersten EOS (248046/248044); alles danach im Rohtrace zählt nicht in die Bewertung — der Rohtrace bleibt trotzdem vollständig im Record. Beide Arme stoppen symmetrisch am EOS (llama-server macht das per Default; der Crow-Arm hat den Stop jetzt im Harness, `stopped_eos` steht in der note jedes Messwerts).

**Harness-Symmetrie (beide Fixes, fable gate):** Beide Arme bekommen ab jetzt DENSELBEN Token-Stream — Chat-Template mit `enable_thinking:false` auch für Crow (`tools/tokenize_ids.py --chat`). Die Run-0-Artefakte des Crow-Arms (Rollen-Tags, leere Think-Zyklen im Antworttext) waren das Symptom der rohen Prompts und sind mit Rev3-Messungen nicht mehr vergleichbar; Run 0 bleibt als Kalibrierungsbeleg der Degeneration, nicht als Qualitätsmessung.
