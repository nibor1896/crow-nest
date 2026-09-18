# quality probe - C1-crow-dense-bf16

- date 2026-09-18, generated at repo commit `22ec6b4`, prompt set version 1
- endpoint `http://127.0.0.1:8099`, engine `crow`, model `Qwen3.8-Flash-Next-CNQ4.5-M`
- row temperature 1.0, top_p 0.95, top_k 20, presence_penalty 0.0, min_p 0.0, max_tokens 2600
- thinking: absent (crow); seeds [1201, 1202, 1203]; 36 generations, 0 failed, 845 s wall

## the arm in one table

| metric | value |
|---|---|
| non-word rate DE per 1000 words, per generation | 22.06 (6.81 to 52.38) |
| non-word rate DE, loanwords the EN dictionary knows removed | 14.84 (5.11 to 29.59) |
| non-word rate DE, pooled over 16039 words | 19.39 |
| non-word rate EN per 1000 words, per generation | 6.29 (3.52 to 8.20) |
| non-word rate EN, loanwords the DE dictionary knows removed | 5.83 (3.52 to 8.20) |
| non-word rate EN, pooled over 9442 words | 6.14 |
| exact literals reproduced (share of literals) | 0.940 (0.600 to 1.000) |
| exact literals reproduced (share of demanded occurrences) | 0.978 (0.833 to 1.000) |
| near-miss literal kinds seen | 0 |
| JSON: whole answer a valid document / shape ok | 4 / 4 of 6 |
| distinct-word ratio | 0.536 (0.400 to 0.695) |
| longest immediate repeat run | 1.8 (1.0 to 5.0), max 5 |
| generations with foreign-script characters | 1 of 24 (2 chars) |
| words per answer | 714 (18 to 1446) |
| answers stopped at max_tokens | 0 of 36 |

## per prompt

| prompt | lang | seed | words | finish | non-word/1000 | flagged | literals | json | longest repeat | foreign |
|---|---|---|---|---|---|---|---|---|---|---|
| de-prose-plakat | de | 1201 | 934 | stop | 7.49 | 7/934 | - | - | 1x1 | 0 |
| de-prose-plakat | de | 1202 | 1168 | stop | 13.70 | 16/1168 | - | - | 1x1 | 0 |
| de-prose-plakat | de | 1203 | 1145 | stop | 12.23 | 14/1145 | - | - | 2x1 | 0 |
| de-prose-speicher | de | 1201 | 1158 | stop | 19.00 | 22/1158 | - | - | 2x1 | 0 |
| de-prose-speicher | de | 1202 | 1027 | stop | 33.11 | 34/1027 | - | - | 2x1 | 0 |
| de-prose-speicher | de | 1203 | 924 | stop | 33.55 | 31/924 | - | - | 2x1 | 0 |
| de-prose-buchbinderei | de | 1201 | 1049 | stop | 13.35 | 14/1049 | - | - | 2x1 | 0 |
| de-prose-buchbinderei | de | 1202 | 1184 | stop | 11.82 | 14/1184 | - | - | 2x1 | 0 |
| de-prose-buchbinderei | de | 1203 | 1171 | stop | 8.54 | 10/1171 | - | - | 2x1 | 2 |
| de-prose-wasserwerk | de | 1201 | 980 | stop | 30.61 | 30/980 | - | - | 2x1 | 0 |
| de-prose-wasserwerk | de | 1202 | 1244 | stop | 18.49 | 23/1244 | - | - | 2x1 | 0 |
| de-prose-wasserwerk | de | 1203 | 883 | stop | 27.18 | 24/883 | - | - | 1x1 | 0 |
| en-prose-foundry | en | 1201 | 1446 | stop | 6.22 | 9/1446 | - | - | 2x1 | 0 |
| en-prose-foundry | en | 1202 | 1220 | stop | 8.20 | 10/1220 | - | - | 2x1 | 0 |
| en-prose-foundry | en | 1203 | 1311 | stop | 6.86 | 9/1311 | - | - | 2x1 | 0 |
| en-prose-archive | en | 1201 | 1421 | stop | 3.52 | 5/1421 | - | - | 2x1 | 0 |
| en-prose-archive | en | 1202 | 1394 | stop | 5.02 | 7/1394 | - | - | 2x1 | 0 |
| en-prose-archive | en | 1203 | 1175 | stop | 6.81 | 8/1175 | - | - | 2x1 | 0 |
| lit-farbpalette | de | 1201 | 579 | stop | 12.09 | 7/579 | 7/7 (1.00) | - | 1x1 | - |
| lit-farbpalette | de | 1202 | 587 | stop | 6.81 | 4/587 | 7/7 (1.00) | - | 1x1 | - |
| lit-farbpalette | de | 1203 | 526 | stop | 28.52 | 15/526 | 6/7 (0.97) | - | 1x1 | - |
| lit-releasenote | en | 1201 | 504 | stop | 7.94 | 4/504 | 5/5 (1.00) | - | 1x1 | - |
| lit-releasenote | en | 1202 | 441 | stop | 4.54 | 2/441 | 3/5 (0.83) | - | 1x1 | - |
| lit-releasenote | en | 1203 | 530 | stop | 7.55 | 4/530 | 5/5 (1.00) | - | 1x1 | - |
| json-tensorplan | en | 1201 | 25 | stop | - | - | 5/5 (1.00) | ok | 5x3 | - |
| json-tensorplan | en | 1202 | 25 | stop | - | - | 5/5 (1.00) | ok | 5x3 | - |
| json-tensorplan | en | 1203 | 18 | stop | - | - | 5/5 (1.00) | ok | 5x2 | - |
| json-schritte | de | 1201 | 39 | stop | - | - | - | fragment | 2x1 | - |
| json-schritte | de | 1202 | 61 | stop | - | - | - | fragment | 1x1 | - |
| json-schritte | de | 1203 | 72 | stop | - | - | - | ok | 1x1 | - |
| de-agent-umzug | de | 1201 | 255 | stop | 11.76 | 3/255 | - | - | 1x1 | 0 |
| de-agent-umzug | de | 1202 | 287 | stop | 20.91 | 6/287 | - | - | 1x1 | 0 |
| de-agent-umzug | de | 1203 | 247 | stop | 16.19 | 4/247 | - | - | 1x1 | 0 |
| de-agent-fehlersuche | de | 1201 | 340 | stop | 50.00 | 17/340 | - | - | 1x1 | 0 |
| de-agent-fehlersuche | de | 1202 | 210 | stop | 52.38 | 11/210 | - | - | 1x1 | 0 |
| de-agent-fehlersuche | de | 1203 | 141 | stop | 35.46 | 5/141 | - | - | 1x1 | 0 |

## flagged words, with context

| lang | word | note | prompt | seed | context |
|---|---|---|---|---|---|
| de | `Abgleichung` |  | de-agent-umzug | 1202 | sortieren. Der Vorgang ist beendet, wenn die Abgleichung zwischen alter Standortakte und neuer Positi |
| de | `Actions` | loanword | lit-farbpalette | 1203 | t den Blick des Lesers zu kritischen Call-to-Actions. #FFB347 sollte stets in Kombination mit #05 |
| de | `Allroundern` |  | de-prose-buchbinderei | 1202 | elevanz, da der Bedarf an hochqualifizierten Allroundern durch den Bedarf an operativ tätigen Maschin |
| de | `Altersung` |  | de-prose-wasserwerk | 1202 | m ist die Instandhaltung und die allmähliche Altersung des Netzes. Alt werden Leitungen nicht allei |
| de | `Alterungsprozeesse` |  | de-prose-buchbinderei | 1201 | nes Fabrikats, was die Fehlerquellen und die Alterungsprozeesse des Einbandes grundlegend veränderte.  Diese |
| de | `Anachronie` |  | de-prose-plakat | 1202 | ischen Antiqua-Schnitten, kann eine unnötige Anachronie vermeiden, doch die direkte Imitation histor |
| de | `Aquiferen` |  | de-prose-wasserwerk | 1203 | Brunnen, die das Grundwasser aus gefilterten Aquiferen fördern, während andere Standorte auf Talspe |
| de | `Architekturs` |  | de-prose-speicher | 1201 | n.  Um diese Schranke zu verschieben, nutzen Architekturs eine mehrstufige Speicherhierarchie, die auf |
| de | `Armaturarm` |  | de-prose-wasserwerk | 1202 | g, denn auf dem Weg vom Hochbehälter bis zum Armaturarm kann das Wasser durch biofilmische Prozesse  |
| de | `Asset` | loanword | de-prose-wasserwerk | 1201 | eparaturregime hin zu einer vorausschauenden Asset-Management-Strategie, bei der die restliche  |
| de | `Asymmetie` |  | de-prose-plakat | 1202 |  zeitgenössischen Praxis hat sich dabei eine Asymmetie als erfolgreich erwiesen, bei der das Bild n |
| de | `Beprobungen` |  | de-prose-wasserwerk | 1202 | gische Überwachung erfolgt durch regelmäßige Beprobungen an strategischen Punkten im Netz, etwa an En |
| de | `Betrachers` |  | de-prose-plakat | 1202 | rrativ vorgibt und die Erwartungshaltung des Betrachers steuert. Für Sammlungen, deren Fokus auf Obj |
| de | `Betrachterinteresse` |  | de-prose-plakat | 1202 | arbe der Fläche eine Stimmung setzt, die das Betrachterinteresse weckt. In dieser Distanz verschwinden die De |
| de | `Bevölkerungs` |  | de-prose-wasserwerk | 1201 | llt für eine mittelgroße Stadt, die in ihrem Bevölkerungs- und Industriebestand einem stetigen Wandel  |
| de | `Bewachungrung` |  | de-prose-wasserwerk | 1202 | Netzes unterliegt jedoch einer der ständigen Bewachungrung, denn auf dem Weg vom Hochbehälter bis zum A |
| de | `Bibliothekspur` |  | de-prose-buchbinderei | 1201 | urch neuartige Pappstoffe wie die sogenannte Bibliothekspur oder später das Pergamin ersetzt. Diese neue |
| de | `Blackwell` | loanword | de-prose-speicher | 1202 |  wie die von AMD RDNA oder NVIDIA Ampere bis Blackwell nutzen daher sehr große, mehrstufige Cache-S |
| de | `Blocker` | loanword | de-agent-fehlersuche | 1203 |  und deutet auf einen internen Deadlock oder Blocker hin. 5. Untersuche die Firewall-Regeln und R |
| de | `Blocking` | loanword | de-prose-speicher | 1201 | sind. Dies kann geschehen, durch Tiling oder Blocking, bei dem ein großer Berechnungsvorgang in vi |
| de | `Bohrkernchern` |  | de-prose-wasserwerk | 1202 | rosionskinematik, oft durch die Entnahme von Bohrkernchern aus verdächtigen Leitungsabschnitten und die |
| de | `Bouncing` | loanword | de-prose-speicher | 1203 | nomen auf, das als Cache-Thrash oder Storage-Bouncing bekannt ist. In diesem Zustand werden Daten, |
| de | `Bound` | loanword | de-prose-speicher | 1201 | nn, so läuft der Code im sogenannten Compute-Bound-Bereich, bei dem die Rechenwerke voll ausgel |
| de | `Buchblocke` |  | de-prose-buchbinderei | 1203 |  Toleranz gegenüber Materialabweichungen der Buchblocke und den Vorlieben des Kunden drastisch reduz |
| de | `Buffer` | loanword | de-prose-speicher | 1203 | ung, da die Speicherhierarchie effizient als Buffer zwischen langsamerem RAM und schnelleren Cac |
| de | `Buslastung` |  | de-prose-speicher | 1202 | -Treffer-Quote, bei der die GPU-Interconnect-Buslastung minimal bleibt und die Rechenkerne praktisch |
| de | `Cacheschwelle` |  | de-prose-speicher | 1203 | n, dass die Arbeitsmenge stets unterhalb der Cacheschwelle bleibt, sodass die teuren Speicherzyklen der |
| de | `Carbonatisierung` |  | de-prose-wasserwerk | 1203 |  eine progressive Festigkeitsminderung durch Carbonatisierung erfahren. Die Ursachen für die Alterung sind |
| de | `Chlormethode` |  | de-prose-wasserwerk | 1202 | r die Dosierung von Chlor erfolgt, wobei die Chlormethode den Vorteil bietet, dass sie eine gewisse Ke |
| de | `Compute` | loanword | de-prose-speicher | 1201 | alten kann, so läuft der Code im sogenannten Compute-Bound-Bereich, bei dem die Rechenwerke voll  |
| de | `Computing` | loanword | de-prose-speicher | 1202 | on Echtzeit-Rendering und wissenschaftlichem Computing legt. |
| de | `Constant` | loanword | de-prose-speicher | 1201 | h noch einen L3-Cache oder einen sogenannten Constant-Cache, der speziell für Daten optimiert ist, |
| de | `Core` | loanword | de-prose-speicher | 1203 | Cache, der oft direkt in der Nähe der Shader-Core liegt, kann in wenigen Taktzyklen erfolgen,  |
| de | `Cores` | loanword | de-prose-speicher | 1201 | der L2-Cache, der häufig von mehreren Shader-Cores gemeinsam genutzt wird und als Zwischenpuffe |
| de | `Cyan` | loanword | de-prose-plakat | 1203 | aive Verwendung von gesättigtem Magenta oder Cyan könnte den historischen Bezug zerstören und  |
| de | `Deadlock` | loanword | de-agent-fehlersuche | 1202 |  `poll` oder Dateizugriffen bestätigen einen Deadlock oder eine Ressourcen-Verfügbarkeitsproblemat |
| de | `Degenerationssymptome` |  | de-prose-wasserwerk | 1201 | , die kritischen Schwachpunkte bilden. Diese Degenerationssymptome werden oft schleichend und über Jahre hinweg |
| de | `Degradationsmechanismen` |  | de-prose-wasserwerk | 1203 | gsten Jahrhunderts, unterliegen spezifischen Degradationsmechanismen. Graugussleitungen neigen zur Graphitisierun |
| de | `Designvorgaben` |  | lit-farbpalette | 1202 |  technisch exakte Umsetzung der festgelegten Designvorgaben. Die vorliegende Datei unter /srv/plakat/202 |
| de | `Distanzund` |  | de-prose-plakat | 1202 | rrespondiert.  Schließlich ist die Frage der Distanzund Nahwirkung die alles übergreifende Perspekti |
| de | `Distanzz` |  | lit-farbpalette | 1201 | D7FFE0 extrem hoch ist und auch aus weiterer Distanzz noch klar erkennbar bleibt. Wichtig beim Dru |
| de | `Druckerhöhungsanlagen` |  | de-prose-wasserwerk | 1201 | Druck wird dabei durch ein Zusammenspiel von Druckerhöhungsanlagen, strategisch platzierten Hochbehältern zur P |
| de | `Druckminderungsventile` |  | de-prose-wasserwerk | 1202 | nsteuerungen und in kritischen Bereichen auf Druckminderungsventile. Der Druck muss überall hoch genug sein, das |
| de | `Druckminderzonen` |  | de-prose-wasserwerk | 1201 | henunterschied der Topografie erzeugt, wobei Druckminderzonen vermieden werden müssen, da ein zu niedriger |
| de | `Durchlichtung` |  | lit-farbpalette | 1203 | 505 sollte im Druck so tief sein, dass keine Durchlichtung des Papieres sichtbar wird, obwohl die Farbe |
| de | `Endpoint` | loanword | de-agent-fehlersuche | 1201 | len HTTP- oder API-Request gegen den lokalen Endpoint mit `curl -v` aus, um zu sehen, ob der Diens |
| de | `Endwasserqualtit` |  | de-prose-wasserwerk | 1203 | h verankerten Überwachung, die nicht nur die Endwasserqualtit im Hausanschluss, sondern auch die Prozessst |
| de | `Engines` | loanword | de-prose-speicher | 1203 | rchie entscheidet. Moderne APIs und Laufzeit-Engines versuchen, dieses Problem durch Tile-basiert |
| de | `Exceptions` | loanword | de-agent-fehlersuche | 1202 | -f`, ob der Dienst in den Systemlogs Fehler, Exceptions oder Absturz-Meldungen um Mitternacht zeigt; |
| de | `Execution` | loanword | de-prose-speicher | 1203 | ene Präfetching-Mechanismen und Out-of-Order Execution, um Daten vorherzusagen, bevor sie explizit  |
| de | `Fadennahtmaschine` |  | de-prose-buchbinderei | 1201 | e ersten mechanischen Apparate, wie etwa die Fadennahtmaschine, welche die bis dahin mühsame und und zeitra |
| de | `Faktur` |  | de-prose-wasserwerk | 1201 |  signifikanten Anstieg des sogenannten Nicht-Faktur-Wassers, also der Differenz zwischen produzi |
| de | `Fakturier` |  | de-prose-wasserwerk | 1203 | chneten Wassermenge. Diese sogenannten Nicht-Fakturier-Wasserverluste sind ein primärer Indikator f |
| de | `Falzarten` |  | de-prose-buchbinderei | 1202 | schiedenen Garne und um die Kalkulierung der Falzarten verblasste in den Hintergrund, während das V |
| de | `Farbtonalitäten` |  | de-prose-plakat | 1202 | berfläche und die psychologische Wirkung der Farbtonalitäten betrifft. Für ein Plakat der Kunstgewerbe-Sa |
| de | `Faults` | loanword | de-agent-fehlersuche | 1201 | pezifischen Fehlermeldungen wie Segmentation Faults oder Out-of-Memory-Kills zu suchen. Das Auff |
| de | `Fayencetellers` |  | de-prose-plakat | 1203 | äßige Ausleuchtung, die die Farbigkeit eines Fayencetellers authentisch wiederliefert. Die Entscheidung  |
| de | `Feinstfiltration` |  | de-prose-wasserwerk | 1202 | en. In vielen Fällen schließt sich eineadsor Feinstfiltration über Sandfilter an, die auch organische Trüb |
| de | `Festigkeitsminderung` |  | de-prose-wasserwerk | 1203 | , während Asbestzementrohre eine progressive Festigkeitsminderung durch Carbonatisierung erfahren. Die Ursache |
| de | `Feuchtewerte` |  | de-agent-umzug | 1203 | hritt ist erledigt, wenn die Temperatur- und Feuchtewerte stabil im Sollbereich liegen. 10. Dokumentie |
| de | `Findmittel` |  | de-agent-umzug | 1201 |  8. Aktualisiere die Suchinstrumente und die Findmittel im Archivverwaltungssystem auf die neue Loka |
| de | `Flag` | loanword | de-prose-plakat | 1203 | lakat eine grafische Signatur, ein visuelles Flag. Hier zählt die grobe Form, die Farbkombinat |
| de | `Flankierkontraste` |  | lit-farbpalette | 1203 | kt neben #1B3A5F ohne Trennung, da sonst die Flankierkontraste schwach werden. Der Akzent #FFB347 ist das l |
| de | `Footprint` | loanword | de-prose-speicher | 1201 | ße einer Arbeitsmenge, im Fachjargon oft als Footprint bezeichnet, spielt in diesem Kontext die ent |
| de | `Gemaßte` |  | de-prose-buchbinderei | 1202 | auf eine bloße Verpackung. Der Stolz auf das Gemaßte, auf den geraden Rücken und den festgeschlos |
| de | `Gewinnaufzehren` |  | lit-farbpalette | 1203 |  bei 4800 Stück die Nachbesserungskosten den Gewinnaufzehren würden. |
| de | `Git` | loanword | de-agent-fehlersuche | 1202 |  bekannten funktionierenden Version oder dem Git-Status; Unterschiede in Parametern wie Threa |
| de | `Grammatur` |  | de-prose-plakat | 1202 | wertigen Bilderdruckpapiere erfolgen, dessen Grammatur und Oberfläche die Haptik des Plakats unters |
| de | `Graphitisierung` |  | de-prose-wasserwerk | 1203 | onsmechanismen. Graugussleitungen neigen zur Graphitisierung und Versprödung durch Korrosionsprozesse von |
| de | `Graugussleitungen` |  | de-prose-wasserwerk | 1203 | liegen spezifischen Degradationsmechanismen. Graugussleitungen neigen zur Graphitisierung und Versprödung d |
| de | `Großverbünde` |  | de-prose-wasserwerk | 1202 |  Resourcen verfügt, noch einfach auf externe Großverbünde zurückgreifen kann, ohne die lokale Autonomi |
| de | `Gutachtenakzept` |  | de-agent-umzug | 1201 | t gilt als erfolgreich, wenn ein technisches Gutachtenakzept für die Lagerbedingungen vorliegt. 3. Bereit |
| de | `Handstichung` |  | de-prose-buchbinderei | 1201 | e die bis dahin mühsame und und zeitraubende Handstichung durch die Buchhefte ersetzte, oder die Präge |
| de | `Hauptpeicher` |  | de-prose-speicher | 1203 | ttdessen werden diese Daten einmalig aus dem Hauptpeicher geladen und in den nahen Cache-Ebenen abgele |
| de | `Hexcodes` |  | lit-farbpalette | 1202 | B347 muss leuchten. So entsteht aus den vier Hexcodes ein Gesamtkunstwerk, das die 4800 Stück wert |
| de | `Highlighter` | loanword | lit-farbpalette | 1201 | estimmte Schlüsselbotschaften oder grafische Highlighter zu markieren, die vom Betrachter sofort wahr |
| de | `Hintergrun` |  | lit-farbpalette | 1201 | us dem dunklen Rahmen #1B3A5F und dem hellen Hintergrun #D7FFE0 erzeugt eine klare Gliederung, währe |
| de | `Infrastrukturmaterialien` |  | de-prose-wasserwerk | 1203 |  Werkstoffe und Bauarten ist der Verfall von Infrastrukturmaterialien eine unvermeidbare physikalische Realität, d |
| de | `Infrastrukturüberwachung` |  | de-prose-wasserwerk | 1201 | r verändern.  Parallel zu dieser technischen Infrastrukturüberwachung steht die rigorose Qualitätsüberwachung als  |
| de | `Inspektionssysteme` |  | de-prose-wasserwerk | 1203 | dienen georadarkampagnen oder videobasierete Inspektionssysteme dazu, den inneren Zustand der Rohrwand und d |
| de | `Instabilitätsfehler` |  | de-agent-fehlersuche | 1202 | p steckt; häufige Neustarts bestätigen einen Instabilitätsfehler während der Initialisierung.  4. Analyse die |
| de | `Instandhaltungsprogramm` |  | de-prose-wasserwerk | 1202 | wirken, wendet die Stadt auf ein präventives Instandhaltungsprogramm an, welches die Restlebensdauer jeder Leitun |
| de | `Interconnect` | loanword | de-prose-speicher | 1202 | m hohen Cache-Treffer-Quote, bei der die GPU-Interconnect-Buslastung minimal bleibt und die Rechenkern |
| de | `Jugendstilschale` |  | de-prose-plakat | 1203 | Objekt der Sammlung – vielleicht eine üppige Jugendstilschale oder eine strenge Biedermeier Kommode – isol |
| de | `Kalkulierung` |  | de-prose-buchbinderei | 1202 | nschaften der verschiedenen Garne und um die Kalkulierung der Falzarten verblasste in den Hintergrund, |
| de | `Kaskadenfehler` |  | de-agent-fehlersuche | 1201 | lschlagen der Abhängigkeiten bestätigt einen Kaskadenfehler, während gesunde Abhängigkeiten den lokalen  |
| de | `Kibibytes` |  | de-prose-speicher | 1201 | itektur und Implementierung zwischen wenigen Kibibytes bis mehreren hundert Kibibytes groß sein kan |
| de | `Kills` | loanword | de-agent-fehlersuche | 1201 | n wie Segmentation Faults oder Out-of-Memory-Kills zu suchen. Das Auffinden eines wiederkehrend |
| de | `Kommanditunternehmungen` |  | de-prose-buchbinderei | 1201 | isierung als Aktiengesellschaften oder große Kommanditunternehmungen emporwuchsen, sahen im Buch zunächst das gei |
| de | `Konsistenzprotokoll` |  | de-prose-speicher | 1201 | treffenden Block gesperrt sein kann, je nach Konsistenzprotokoll. Diese Latenz ist nicht statisch, sondern hä |
| de | `Korrelationssverfahren` |  | de-prose-wasserwerk | 1203 | ren Leckagen werden in der Praxis akustische Korrelationssverfahren eingesetzt, bei denen hochsensible Sensoren  |
| de | `Korrosionskinematik` |  | de-prose-wasserwerk | 1202 | rt, doch nur die systematische Erfassung der Korrosionskinematik, oft durch die Entnahme von Bohrkernchern au |
| de | `Korrosionsneigung` |  | de-prose-wasserwerk | 1201 | sondere Stahlleitungen mit ihrer natürlichen Korrosionsneigung und ältere Gusseisenrohre, die aufgrund ihre |
| de | `Korrosionsprodukten` |  | de-prose-wasserwerk | 1201 | a korrosive Vorgänge und die Freisetzung von Korrosionsprodukten die ionische Zusammensetzung des Wassers mes |
| de | `Korrosionsvorgänge` |  | de-prose-wasserwerk | 1202 | schnitt erreicht hat. Hinzu kommen chemische Korrosionsvorgänge bei kupfernen Leitungen, welche durch das Wa |
| de | `Kostenvorgaben` |  | de-prose-buchbinderei | 1203 | eativer Spielraum durch Produktionspläne und Kostenvorgaben beschnitten war. Der Zugang zu exklusivem Lu |
| de | `Layoutplan` |  | de-agent-umzug | 1202 | au. Die Prüfung ist bestanden, wenn der neue Layoutplan alle bisherigen Bestände unter Berücksichtig |
| de | `Leak` | loanword | de-agent-fehlersuche | 1201 | reichen des Limit bestätigt einen Ressourcen-Leak, während niedrige Werte auf andere Blockaden |
| de | `Lederverarbeitformung` |  | de-prose-buchbinderei | 1201 |  handgetriebene Goldpresse oder die komplexe Lederverarbeitformung, wurden zum Kernbestand einer neu definierte |
| de | `Lehrlingswesen` |  | de-prose-buchbinderei | 1203 | renzierung führte. Während das traditionelle Lehrlingswesen in den Handwerkskammerbetrieben bestehen bli |
| de | `Leistungsausbeute` |  | de-prose-speicher | 1202 | z zu einem spürbaren Zeitverzug auf, der die Leistungsausbeute begrenzen kann. Die Technik der Speicherhier |
| de | `Leitfähigkeitswerte` |  | de-prose-wasserwerk | 1201 | und der kontinuierlichen Überwachung von der Leitfähigkeitswerte basieren, da korrosive Vorgänge und die Frei |
| de | `Lesbarkeitsproblem` |  | de-prose-plakat | 1202 | dabei den Reflexionsgrad reduzieren, der das Lesbarkeitsproblem unter wechselnden Lichtverhältnissen verbess |
| de | `Line` | loanword | de-prose-speicher | 1202 | uster stark eingeschränkt werden. Jede Cache-Line, die geladen wird, enthält neben den tatsäch |
| de | `Linening` |  | de-prose-wasserwerk | 1203 |  über die innere Auskleidung, das sogenannte Linening, bis hin zur umfassenden Neutrassierung reic |
| de | `Logs` | loanword | de-agent-fehlersuche | 1201 | Fehlers bestätigt die Ursache, während leere Logs auf ein Problem außerhalb des Prozesses hind |
| de | `Lokalitätsprinzip` |  | de-prose-speicher | 1203 | le Prinzip dieser Hierarchie basiert auf dem Lokalitätsprinzip, das besagt, dass aufeinanderfolgende Datenz |
| de | `Loop` | loanword | de-agent-fehlersuche | 1201 | ienst aktiv ist oder im Absturzzyklus (Crash Loop) vorliegt. Ein aktiver Dienst widerlegt eine |
| de | `Manganentfernung` |  | de-prose-wasserwerk | 1201 | e Verfahren wie die Belüftung zur Eisen- und Manganentfernung sowie die Aktivkohlefilterung zur Eliminieru |
| de | `Marktbarkeit` |  | de-prose-buchbinderei | 1201 | ert wurde, das den Schutz des Bandes und die Marktbarkeit im Regal zu gewährleisten hatte. Der Einband |
| de | `Marktgängigkeit` |  | de-prose-buchbinderei | 1202 | es Einbandes unterwarf sich den Gesetzen der Marktgängigkeit und der Haltbarkeit im Transportwesen. Die V |
| de | `Maschiene` |  | de-prose-buchbinderei | 1202 | e Zeit, in der sich das Verhältnis von Hand, Maschiene und Wert neu ordnete. Der Konflikt zwischen  |
| de | `Massenproduzierbarkeit` |  | de-prose-plakat | 1202 | hunderts wider, in dem sich die industrielle Massenproduzierbarkeit und die handwerkliche Meisterschaft gegenübe |
| de | `Microservices` |  | de-agent-fehlersuche | 1201 | en des Dienstes, wie Datenbanken oder andere Microservices, um festzustellen, ob diese überlastet sind  |
| de | `Mintgrün` |  | lit-farbpalette | 1203 | und in #D7FFE0, ein sehr helles, fast weißes Mintgrün, das Ruhe, Frische und Klarheit ausstrahlt.  |
| de | `Misses` | loanword | de-prose-speicher | 1201 | inimum reduzieren, doch in der Praxis treten Misses auf, weil die Datenmenge nicht in den Cache  |
| de | `Nachbesserungskosten` |  | lit-farbpalette | 1203 | v3.2.1 sind untersagt, da bei 4800 Stück die Nachbesserungskosten den Gewinnaufzehren würden. |
| de | `Nahwirkung` |  | de-prose-plakat | 1202 | t.  Schließlich ist die Frage der Distanzund Nahwirkung die alles übergreifende Perspektive, die all |
| de | `Netzwerkreichtweite` |  | de-agent-fehlersuche | 1201 | onfigurationsfehler hinweisen.  4. Teste die Netzwerkreichtweite und Latenz zum Dienst mit `ping` und `tracer |
| de | `Nouveau` |  | de-prose-plakat | 1203 | fließende Linie die Aufbruchstimmung des Art Nouveau anklingen lässt.  Eng verzahnt mit der Bilda |
| de | `On` | loanword | de-prose-speicher | 1203 | fordert eine längere Signallaufzeit über den On-Chip-Bus, was zu einer deutlich höheren Verz |
| de | `Papieres` |  | lit-farbpalette | 1203 | k so tief sein, dass keine Durchlichtung des Papieres sichtbar wird, obwohl die Farbe #050505 bei  |
| de | `Pastellen` |  | de-prose-plakat | 1202 | Verwendung von erdfarbenen Tönen, gedämpften Pastellen oder klassischen Schwarz-Weiß-Druckakzenten  |
| de | `Patinas` | loanword | de-prose-plakat | 1203 | rlagen chemischen Prozessen, die spezifische Patinas und Sättigungen erzeugten. Eine naive Verwen |
| de | `Pergamin` |  | de-prose-buchbinderei | 1201 | ie sogenannte Bibliothekspur oder später das Pergamin ersetzt. Diese neuen Materialien waren masch |
| de | `Prefetching` |  | de-prose-speicher | 1201 | auf bestimmte Speicherbl zu minimieren. Auch Prefetching-Techniken, die Daten vorausschauend aus dem  |
| de | `Prägepresse` |  | de-prose-buchbinderei | 1203 | r angehende Geselle lernte nun, wie man eine Prägepresse justierte, welche Temperatur für ein bestimm |
| de | `Prägepressen` |  | de-prose-buchbinderei | 1203 |  den 1850er Jahren sowie die Entwicklung der Prägepressen mit erhitzten Stempeln markierten den Beginn |
| de | `Punzen` |  | de-prose-buchbinderei | 1201 | e Leder und der aufwendig mit Blattgold oder Punzen verzierte Vollledereinband wurden zunehmend  |
| de | `Qualitäts` |  | de-prose-wasserwerk | 1201 | wobei der Fokus immer auf der Prävention der Qualitäts- und Druckverluste liegen muss, da die nacht |
| de | `Raytracing` |  | de-prose-speicher | 1202 | ometriedaten und aufwendige Berechnungen wie Raytracing erforderlich sind, rechtzeitig an die Berech |
| de | `Reclametten` |  | de-prose-buchbinderei | 1202 | hänomen der „Volksausgaben“ und der billigen Reclametten war in dieser Form nur durch die Überwindung |
| de | `Rendering` | loanword | de-prose-speicher | 1202 | rückgreifen muss. Ist die Arbeitsmenge eines Rendering- oder Compute-Kernels kleiner als die Kapazi |
| de | `Reproduktionsgrößeß` |  | de-prose-plakat | 1201 | t und Klarheit besitzt, um auch in kleinerer Reproduktionsgrößeß erkennbar zu bleiben, ohne dabei an Detailsc |
| de | `Request` | loanword | de-agent-fehlersuche | 1201 | et.  5. Führe einen manuellen HTTP- oder API-Request gegen den lokalen Endpoint mit `curl -v` aus |
| de | `Requests` | loanword | de-agent-fehlersuche | 1201 | um Ressourcenengpässe zu identifizieren, die Requests verlangsamen. Eine hohe Auslastung bestätigt |
| de | `Restaurierungsbands` |  | de-prose-buchbinderei | 1201 |  handwerklichen Sektor, der sich im Feld des Restaurierungsbands, des Künstlerbandes und des Luxusleders vert |
| de | `Ruckheftmaschinen` |  | de-prose-buchbinderei | 1202 | inen, insbesondere der Falzmaschinen und der Ruckheftmaschinen, beschleunigte den Arbeitsprozess um ein Meh |
| de | `Satzspiegsels` |  | de-prose-plakat | 1203 | rch ihre Proportionen und die Sorgfalt ihres Satzspiegsels eine historische Tiefe andeutet, ohne die Fo |
| de | `Setzweite` |  | de-prose-plakat | 1201 | tändlich in die Gesamtstruktur einfügen. Die Setzweite, die Laufweite und die Zeilenabstände sind d |
| de | `Shader` |  | de-prose-speicher | 1201 | er ist der L2-Cache, der häufig von mehreren Shader-Cores gemeinsam genutzt wird und als Zwische |
| de | `Sollbereich` |  | de-agent-umzug | 1203 | n die Temperatur- und Feuchtewerte stabil im Sollbereich liegen. 10. Dokumentiere den Umzug abschließ |
| de | `Sorgfaltigkeit` |  | de-prose-buchbinderei | 1203 | urch die präzise Mechanisierung die manuelle Sorgfaltigkeit in der Langlebigkeit der Struktur, da die Kl |
| de | `Stucksatzes` |  | de-prose-plakat | 1203 | es Seitenlicht, das die Reliefstruktur eines Stucksatzes offenbart, oder durch eine gleichmäßige Ausl |
| de | `Suchmaskern` |  | de-agent-umzug | 1202 | . Aktualisiere alle digitalen Findmittel und Suchmaskern auf die neue Ortungslogik. Das System ist pr |
| de | `Synecdoche` |  | de-prose-plakat | 1202 | . Die Bildauswahl fungiert hier als eine Art Synecdoche, bei der das einzelne Detail für das Ganze s |
| de | `Systemlogs` |  | de-agent-fehlersuche | 1202 | fe mit `journalctl -f`, ob der Dienst in den Systemlogs Fehler, Exceptions oder Absturz-Meldungen um |
| de | `Thread` | loanword | de-agent-fehlersuche | 1202 | m Git-Status; Unterschiede in Parametern wie Thread-Pools oder Timeout-Werten bestätigen eine fe |
| de | `Threads` | loanword | de-prose-speicher | 1202 | lasten optimiert sind, in denen Tausende von Threads gleichzeitig auf dieselben Datenblöcke zugre |
| de | `Tierhäutes` |  | de-prose-buchbinderei | 1201 | sich nicht mehr an den natürlichen Wuchs des Tierhäutes anpassen musste, sondern standardisierten ge |
| de | `Timeout` | loanword | de-agent-fehlersuche | 1201 | stätigt die Grundfunktionalität, während ein Timeout auf hängende Worker-Prozesse hindeutet.  6.  |
| de | `Timeouts` | loanword | de-agent-fehlersuche | 1203 | nstname>`), um nach wiederkehrenden Fehlern, Timeouts oder OOM-Kills zu suchen; eine Meldung über  |
| de | `Typohierarchie` |  | de-prose-plakat | 1201 | te des Bildes, die Farbflächen und die grobe Typohierarchie ins Spiel. Die Silhouette muss so deutlich s |
| de | `Verfügbarkeitsproblematik` |  | de-agent-fehlersuche | 1202 | stätigen einen Deadlock oder eine Ressourcen-Verfügbarkeitsproblematik. |
| de | `Verpcke` |  | de-agent-umzug | 1203 | rtlichkeiten schriftlich festgelegt sind. 5. Verpcke die archivarischen Güter sachgerecht und ver |
| de | `Vollschwarzes` |  | lit-farbpalette | 1203 | ibt. Die Wahl von #050505 statt eines reinen Vollschwarzes verhindert, dass der Text auf dem hellen Gru |
| de | `Vorreihstellung` |  | de-prose-buchbinderei | 1202 |  Das Handwerk verlor seine gesellschaftliche Vorreihstellung und den sozialen Zusammenhalt der Zunft. Die |
| de | `Werkstoffeigenschaften` |  | de-prose-buchbinderei | 1203 | tion erforderte eine umfassende Kenntnis der Werkstoffeigenschaften und eine hohe manuelle Geschick, wobei jeder |
| de | `Wärmeeigkeit` |  | lit-farbpalette | 1201 | CMYK-Set gedruckt werden, um die spezifische Wärmeeigkeit des Orange zu treffen. Die Wiederholung dies |
| de | `Zeitout` |  | de-agent-fehlersuche | 1203 | auf der Port-Ebene mit `curl` oder `nc`; ein Zeitout beim lokalen Test widerlegt Netzwerkinfrastr |
| de | `abgrenz` |  | de-prose-plakat | 1201 | espondieren, sodass die Farbschichten sauber abgrenz und die Details des Bildes nicht im Sumpf de |
| de | `anleihen` |  | de-prose-plakat | 1203 | ie Formensprache der ausgewählten Bildmotive anleihen: Die geometrische Strenge einer Bauhaus-nahe |
| de | `baulichenlichen` |  | de-agent-umzug | 1202 | sherigen Bestände unter Berücksichtigung der baulichenlichen Gegebenheiten aufnehmen kann.  3. Entwickle  |
| de | `biedermeierischen` |  | de-prose-plakat | 1203 | schlossene Form kann die Geschlossenheit der biedermeierischen Welt symbolisieren, während eine dynamische, |
| de | `cyan` | loanword | lit-farbpalette | 1203 |  ist.#1B3A5F darf niemals zu violett oder zu cyan-lastig erscheinen, da dies die Autorität des |
| de | `derunter` |  | de-agent-umzug | 1201 | hnet sind. 4. Führe den physischen Transport derunter strikter Aufsicht und Dokumentation der Kett |
| de | `farbstimulierender` |  | lit-farbpalette | 1203 | , ob #D7FFE0 tatsächlich als neutraler, aber farbstimulierender Grundton erscheint. Es ist kritisch, dass di |
| de | `festgeschlossenen` |  | de-prose-buchbinderei | 1202 |  das Gemaßte, auf den geraden Rücken und den festgeschlossenen Band, wendete sich gegen die kalte Symmetrie |
| de | `festiger` |  | de-prose-buchbinderei | 1202 | d Buchblock durch Leimmaschinen und Klammern festiger.  Die wichtigste Errungenschaft lag in der D |
| de | `glanzende` |  | de-prose-plakat | 1201 | ratsam, matte Papierqualitäten zu wählen, da glanzende Oberflächen die historisch getragene Haptik  |
| de | `hochintensiver` |  | de-prose-buchbinderei | 1203 | rn die graduale Mechanisierung spezifischer, hochintensiver Einzelschritte. Das Schneiden des Buchblocke |
| de | `imitierbar` |  | de-prose-buchbinderei | 1201 | en, die durch die Maschine nicht vollständig imitierbar waren, wie etwa die handgetriebene Goldpress |
| de | `jugendstilistischen` |  | de-prose-plakat | 1203 | h ausbalancierte für die reformerischen oder jugendstilistischen. Die Fluchtlinien, die durch die Kanten der  |
| de | `kriticl` |  | lit-farbpalette | 1201 | iederholung dieser Werte in den Profilen ist kriticl, denn nur so wird das Endresultat der 4800 E |
| de | `kuratorische` |  | de-prose-plakat | 1203 | Vielmehr fungiert das Plakat als eine erste, kuratorische These, die den Betrachter bereits vor dem Be |
| de | `kühlige` |  | lit-farbpalette | 1202 | D7FFE0 definiert, welcher eine extrem helle, kühlige Grün-Türkis-Nuance darstellt. Dieser Farbton |
| de | `lastig` |  | lit-farbpalette | 1203 | #1B3A5F darf niemals zu violett oder zu cyan-lastig erscheinen, da dies die Autorität des Plakat |
| de | `naturgewachsenen` |  | de-prose-buchbinderei | 1202 | Nachahmung vermittelte. Die Werkstoffe waren naturgewachsenen Ursprungs, das Pergament stammte von Weide u |
| de | `pastellig` |  | de-prose-plakat | 1201 | hrte, die heute oft als gedämpft, erdig oder pastellig wahrgenommen wird. Für das moderne Plakat be |
| de | `pastellige` |  | lit-farbpalette | 1201 | ch seine hohe Helligkeit und die kühle, fast pastellige Note erzeugt #D7FFE0 den nötigen Kontrast zu |
| de | `produktionsreif` |  | de-agent-umzug | 1202 | rn auf die neue Ortungslogik. Das System ist produktionsreif, wenn Testnutzer alle Bestände mit der alten |
| de | `pt` | loanword | lit-farbpalette | 1202 | 1.svg liegt die Dicke der Linien bei exakt 2 pt, was bei einer Auflage von 4800 Stück präzis |
| de | `systemd` |  | de-agent-fehlersuche | 1202 | stname>   grep Restart`, ob der Dienst durch systemd wiederholt gestartet wird oder in einem Cras |
| de | `tracen` |  | de-agent-fehlersuche | 1202 | emnahe Aufrufe bei den hängenden Anfragen zu tracen; Blockierungen in `futex`, `poll` oder Datei |
| de | `verlagsgesteuerten` |  | de-prose-buchbinderei | 1203 |  zur die eines ausführenden Gliedes in einer verlagsgesteuerten Kette. Auch die soziale Stellung des Handwer |
| de | `versiehe` |  | de-agent-umzug | 1203 | cke die archivarischen Güter sachgerecht und versiehe sie mit eindeutigen Standortcodes. Der Vorga |
| de | `vorkonzipierten` |  | de-prose-buchbinderei | 1203 | n und seine Fertigkeiten in den Dienst einer vorkonzipierten Idee zu stellen, was seine Rolle vom Schöpfe |
| de | `wirkung` |  | de-prose-plakat | 1201 | s das Plakat als statisches, museales Schild wirkung, ohne dass dabei die Informationshierarchie  |
| en | `APIs` | loanword | lit-releasenote | 1203 | ct stable loading behavior, unchanged public APIs, and improved alignment with the intended ga |
| en | `acidification` |  | en-prose-archive | 1202 |  that migrate to the documents, accelerating acidification. Instead, archives utilize acid-free, lignin |
| en | `backpropagation` |  | lit-releasenote | 1201 | dresses subtle gradients that emerged during backpropagation tests, particularly when dealing with sparse |
| en | `bitwise` |  | lit-releasenote | 1203 | de loading logic. However, users who compare bitwise outputs between release-2026-09-18a and prio |
| en | `ceilinged` |  | en-prose-foundry | 1201 | e of the workers who navigated its dim, high-ceilinged expanse. In the early twentieth century, the |
| en | `checksums` |  | lit-releasenote | 1202 | the standard metadata headers and validation checksums that accompany all release-2026-09-18a deliv |
| en | `cockling` |  | en-prose-archive | 1201 | that must be done with extreme care to avoid cockling or cracking. If ink is soluble, any water-ba |
| en | `convolutional` |  | lit-releasenote | 1201 | objective of this update is to stabilize the convolutional layers within the neural network architectur |
| en | `de` | loanword | en-prose-foundry | 1201 | he casting was extracted from the sand. This de-molding process revealed the raw piece, stil |
| en | `deacidification` |  | en-prose-archive | 1203 | time, eventually turning to dust. While mass deacidification processes exist, they are resource-intensive |
| en | `depolymerize` |  | en-prose-archive | 1202 | c environment causes the cellulose chains to depolymerize, turning the paper brittle and yellowed unti |
| en | `draught` |  | en-prose-foundry | 1203 | charge of coke, limestone, and pig iron. The draught was often assisted by mechanical blowers, th |
| en | `elementwise` |  | lit-releasenote | 1203 | seline, since the change remains a scalar or elementwise configuration update rather than a shape exp |
| en | `embrittling` |  | en-prose-archive | 1203 | nds in lignin and cellulose, fading inks and embrittling sheets. Dust and atmospheric pollutants, suc |
| en | `evidential` |  | en-prose-archive | 1201 |  routine correspondence that have no lasting evidential or informational value. Holding onto this cl |
| en | `gelatinin` |  | en-prose-archive | 1203 | spores and silverfish find sustenance in the gelatinin sizes and proteins within the paper. Pest co |
| en | `grey` |  | en-prose-foundry | 1201 |  the black scale and sand away to reveal the grey, dull iron beneath. The final gate remnants  |
| en | `hydrostatic` |  | en-prose-foundry | 1202 | r crack. Often, the casting was taken to the hydrostatic pressure test stand if it was a pressure ves |
| en | `hygroscopic` |  | en-prose-archive | 1202 | ellulose, the primary component of paper, is hygroscopic, meaning it absorbs and and releases moistur |
| en | `kraft` | loanword | en-prose-archive | 1203 | selves. Folders must be made from unbleached kraft paper or cotton rag, avoiding glossy coating |
| en | `logits` |  | lit-releasenote | 1201 | ring team has reduced the variance in output logits by approximately fourteen percent, as measur |
| en | `melter` |  | en-prose-foundry | 1201 | ing from the spout and the experience of the melter, who knew from the viscosity and the light o |
| en | `misruns` |  | en-prose-foundry | 1201 |  metal had flowed together but not fused, or misruns where the metal had stalled before filling t |
| en | `mould` |  | en-prose-foundry | 1203 | , a void waiting to be filled with fire. The mould was closed, aligned with dowel pins, and str |
| en | `moulding` |  | en-prose-foundry | 1203 |  compounds to resist the abrasive contact of moulding sand. The pattern maker was not merely a car |
| en | `oversized` |  | en-prose-archive | 1201 |  to bend or fold under their own weight. For oversized maps or ledgers, which cannot be folded with |
| en | `plasticizers` |  | en-prose-archive | 1203 | ften disastrous, leaching lignin, sulfur, or plasticizers into the records they are meant to protect.  |
| en | `pourer` |  | en-prose-foundry | 1201 | phed movement of strength and precision. The pourer, a massive man with a steady hand and nerves |
| en | `pourers` |  | en-prose-foundry | 1202 |  sand, awaited the arrival of the metal. The pourers, clad in heavy leather aprons and face shiel |
| en | `rammers` |  | en-prose-foundry | 1201 | round the pattern with heavy wooden or metal rammers, packing it tight enough to withstand the hy |
| en | `roadmap` |  | en-prose-archive | 1201 | ory that describes the holdings, providing a roadmap for the researcher. It is not a simple index |
| en | `sintered` |  | en-prose-foundry | 1202 | urface of the casting was covered in a hard, sintered skin of sand, known as burn-on, which had to |
| en | `smoothnessness` |  | en-prose-foundry | 1202 | cooling iron. The wood was planed to a silky smoothnessness, its edges rounded to where required, and it |
| en | `sprue` |  | en-prose-foundry | 1201 | r had to time the flow exactly, ensuring the sprue was filled to the correct level to account f |
| en | `sprues` |  | en-prose-foundry | 1202 |  its earthen prison. The excess material—the sprues, runners, and gates—was chipped away with ha |
| en | `taphole` |  | en-prose-foundry | 1203 |  and the consistency of the slag through the taphole. If the iron was too cold, it would refuse t |
| en | `thefinding` |  | en-prose-archive | 1203 | The utility of a holding depends entirely on thefinding aids, which are the intellectual maps that a |
| en | `thethe` |  | en-prose-foundry | 1203 | of the pattern was calculated to account for thethe thermal contraction of the cooling iron, a a |
| en | `toto` |  | lit-releasenote | 1201 | ch has undergone comprehensive recalculation toto eliminate previous rounding errors that occa |
| en | `tuyeres` |  | en-prose-foundry | 1201 | rom the top, while air was blown in from the tuyeres near the bottom to create a fierce combustio |

## near-miss literals

None: every literal that appeared at all appeared exactly.

