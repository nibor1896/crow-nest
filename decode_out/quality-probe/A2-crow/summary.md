# quality probe - A2-crow

- date 2026-09-18, generated at repo commit `c1cf95c`, prompt set version 1
- scored at commit `2bc92f2` on 2026-09-18 (`--rescore`, from the stored texts)
- endpoint `http://127.0.0.1:8099`, engine `crow`, model `Qwen3.8-Flash-Next-CNQ4.5-M`
- row temperature 1.0, top_p 0.95, top_k 20, presence_penalty 0.0, min_p 0.0, max_tokens 2600
- thinking: absent (crow); seeds [1201, 1202, 1203]; 36 generations, 0 failed, 694 s wall

## the arm in one table

| metric | value |
|---|---|
| non-word rate DE per 1000 words, per generation | 20.90 (9.01 to 54.88) |
| non-word rate DE, loanwords the EN dictionary knows removed | 15.22 (8.69 to 33.73) |
| non-word rate DE, pooled over 15327 words | 20.49 |
| non-word rate EN per 1000 words, per generation | 8.59 (3.74 to 14.93) |
| non-word rate EN, loanwords the DE dictionary knows removed | 8.43 (3.74 to 14.93) |
| non-word rate EN, pooled over 9690 words | 7.74 |
| exact literals reproduced (share of literals) | 0.984 (0.857 to 1.000) |
| exact literals reproduced (share of demanded occurrences) | 0.996 (0.966 to 1.000) |
| near-miss literal kinds seen | 3 |
| JSON: whole answer a valid document / shape ok | 5 / 4 of 6 |
| distinct-word ratio | 0.535 (0.400 to 0.748) |
| longest immediate repeat run | 1.7 (0.0 to 5.0), max 5 |
| generations with foreign-script characters | 1 of 24 (2 chars) |
| words per answer | 703 (0 to 1443) |
| answers stopped at max_tokens | 0 of 36 |

## per prompt

| prompt | lang | seed | words | finish | non-word/1000 | flagged | literals | json | longest repeat | foreign |
|---|---|---|---|---|---|---|---|---|---|---|
| de-prose-plakat | de | 1201 | 766 | stop | 18.28 | 14/766 | - | - | 2x1 | 0 |
| de-prose-plakat | de | 1202 | 1137 | stop | 15.83 | 18/1137 | - | - | 2x1 | 0 |
| de-prose-plakat | de | 1203 | 1021 | stop | 17.63 | 18/1021 | - | - | 2x1 | 2 |
| de-prose-speicher | de | 1201 | 921 | stop | 22.80 | 21/921 | - | - | 2x1 | 0 |
| de-prose-speicher | de | 1202 | 966 | stop | 23.81 | 23/966 | - | - | 1x1 | 0 |
| de-prose-speicher | de | 1203 | 933 | stop | 30.01 | 28/933 | - | - | 2x1 | 0 |
| de-prose-buchbinderei | de | 1201 | 948 | stop | 10.55 | 10/948 | - | - | 2x1 | 0 |
| de-prose-buchbinderei | de | 1202 | 1203 | stop | 14.13 | 17/1203 | - | - | 1x1 | 0 |
| de-prose-buchbinderei | de | 1203 | 932 | stop | 19.31 | 18/932 | - | - | 2x1 | 0 |
| de-prose-wasserwerk | de | 1201 | 1194 | stop | 21.78 | 26/1194 | - | - | 1x1 | 0 |
| de-prose-wasserwerk | de | 1202 | 1186 | stop | 33.73 | 40/1186 | - | - | 2x1 | 0 |
| de-prose-wasserwerk | de | 1203 | 991 | stop | 16.15 | 16/991 | - | - | 1x1 | 0 |
| en-prose-foundry | en | 1201 | 1280 | stop | 7.81 | 10/1280 | - | - | 2x1 | 0 |
| en-prose-foundry | en | 1202 | 1443 | stop | 9.01 | 13/1443 | - | - | 2x1 | 0 |
| en-prose-foundry | en | 1203 | 1426 | stop | 9.12 | 13/1426 | - | - | 2x1 | 0 |
| en-prose-archive | en | 1201 | 1269 | stop | 4.73 | 6/1269 | - | - | 2x1 | 0 |
| en-prose-archive | en | 1202 | 1299 | stop | 6.16 | 8/1299 | - | - | 2x1 | 0 |
| en-prose-archive | en | 1203 | 1338 | stop | 3.74 | 5/1338 | - | - | 2x1 | 0 |
| lit-farbpalette | de | 1201 | 671 | stop | 14.90 | 10/671 | 7/7 (1.00) | - | 2x1 | - |
| lit-farbpalette | de | 1202 | 624 | stop | 16.03 | 10/624 | 7/7 (1.00) | - | 2x1 | - |
| lit-farbpalette | de | 1203 | 478 | stop | 18.83 | 9/478 | 6/7 (0.97) | - | 1x1 | - |
| lit-releasenote | en | 1201 | 552 | stop | 14.49 | 8/552 | 5/5 (1.00) | - | 2x1 | - |
| lit-releasenote | en | 1202 | 547 | stop | 7.31 | 4/547 | 5/5 (1.00) | - | 1x1 | - |
| lit-releasenote | en | 1203 | 536 | stop | 14.93 | 8/536 | 5/5 (1.00) | - | 1x1 | - |
| json-tensorplan | en | 1201 | 0 | stop | - | - | 5/5 (1.00) | ok | 0x0 | - |
| json-tensorplan | en | 1202 | 25 | stop | - | - | 5/5 (1.00) | ok | 5x3 | - |
| json-tensorplan | en | 1203 | 25 | stop | - | - | 5/5 (1.00) | fragment | 5x3 | - |
| json-schritte | de | 1201 | 81 | stop | - | - | - | ok | 2x7 | - |
| json-schritte | de | 1202 | 67 | stop | - | - | - | shape | 1x1 | - |
| json-schritte | de | 1203 | 90 | stop | - | - | - | ok | 1x1 | - |
| de-agent-umzug | de | 1201 | 289 | stop | 13.84 | 4/289 | - | - | 1x1 | 0 |
| de-agent-umzug | de | 1202 | 259 | stop | 15.44 | 4/259 | - | - | 1x1 | 0 |
| de-agent-umzug | de | 1203 | 267 | stop | 22.47 | 6/267 | - | - | 1x1 | 0 |
| de-agent-fehlersuche | de | 1201 | 102 | stop | 29.41 | 3/102 | - | - | 1x1 | 0 |
| de-agent-fehlersuche | de | 1202 | 328 | stop | 54.88 | 18/328 | - | - | 2x1 | 0 |
| de-agent-fehlersuche | de | 1203 | 111 | stop | 9.01 | 1/111 | - | - | 1x1 | 0 |

## flagged words, with context

| lang | word | note | prompt | seed | context |
|---|---|---|---|---|---|
| de | `Abgleichung` |  | de-agent-umzug | 1201 | vorliegen. 7. Eingangsqualitätskontrolle und Abgleichung der transportierten Bestände mit der ursprün |
| de | `Abschliessend` |  | de-prose-plakat | 1203 | hter unmittelbar nahe gebracht werden muss.  Abschliessend ist die Frage der lesenden Distanz zu erör.  |
| de | `Absichticht` |  | de-prose-plakat | 1201 | ahl im Druck vollendet diese kompositorische Absichticht, indem sie die materielle und geistige Aura  |
| de | `Adressierungs` |  | de-prose-speicher | 1201 | rbursts zu verschmelzen, um den Overhead des Adressierungs- und Protokollübertrags zu amortisieren.  Ei |
| de | `Akzentrutelement` |  | lit-farbpalette | 1202 |  Ruhe des Hintergrund #D7FFE0 zu stören. Das Akzentrutelement #FFB347 schließlich fungiert als aktives Ste |
| de | `Alitierung` |  | de-prose-wasserwerk | 1202 |  altert dabei nicht gleichmäßig, sondern die Alitierung folgt einer Mischung aus chemischen, physika |
| de | `Altersungsprozesses` |  | de-prose-wasserwerk | 1201 | ch mechanische Belastungen. Die Ursachen des Altersungsprozesses sind vielschichtig: Bei metallischen Leitung |
| de | `Anlagemachines` |  | de-prose-buchbinderei | 1203 | chkompetenzen für die Bedienung spezifischer Anlagemachines zurückgedrängt. In den Meisterschulen wurde  |
| de | `Antiquas` |  | de-prose-plakat | 1203 | eknüpfte, humanistische Grotesken oder feine Antiquas mit klaren Serifen, welche die Lesbarkeit au |
| de | `Aquiferen` |  | de-prose-wasserwerk | 1202 |  jedoch stets eine sorgfältige Erkundung der Aquiferen durch hydrogeologische Modellierungen und la |
| de | `Archivguttypen` |  | de-agent-umzug | 1203 |  genehmigt und die Standorte für spezifische Archivguttypen definiert sind.  3. Organisiere spezialisier |
| de | `Arts` | loanword | de-prose-buchbinderei | 1203 | iriert durch englische Einflüsseusse wie den Arts and Crafts, suchte in den letzten Jahrzehnte |
| de | `Aufbereitenwerk` |  | de-prose-wasserwerk | 1202 | ebe, da das Wasser auf seinem langen Weg vom Aufbereitenwerk bis zum letzten Entnahmeventil im Gebäude de |
| de | `Aufstauwasser` |  | de-prose-wasserwerk | 1201 | nfache Desinfektion ausreicht, erfordert das Aufstauwasser aus offenen Oberflächenflächen aufgrund mögl |
| de | `Ausgestaltungskunst` |  | de-prose-buchbinderei | 1202 | andwerklicher Souveränität und individueller Ausgestaltungskunst verloren ging, war beträchtlich. Der Leser s |
| de | `Ausstellungs` |  | de-prose-plakat | 1202 | o gewählt sein, dass die Hauptzeile, oft der Ausstellungs- oder Sammlungsname, als massiver Textkörper |
| de | `Balancing` | loanword | de-prose-plakat | 1201 | esen ist. Es bedarf daher einer sorgfältigen Balancing-Übung, in der der Text nicht als Störfaktor  |
| de | `Bandwidth` | loanword | de-prose-speicher | 1202 |  Phase des "Capacity Bound" oder später des "Bandwidth Bound" dominiert der Zugriff auf weiter entf |
| de | `Batches` | loanword | de-prose-speicher | 1201 | herzugriffe nicht mehr isoliert, sondern als Batches oder Ströme organisiert sind. Die Hardware-C |
| de | `Berufstand` |  | de-prose-buchbinderei | 1202 | etrieben drastisch ab. Es entstand ein neuer Berufstand, der zwar technisch versiert, aber häufig en |
| de | `Beschnittung` |  | de-prose-buchbinderei | 1203 | den gesamten Prozess von der Heftung bis zur Beschnittung persönlich überwachte. Die Vorstellung, dass |
| de | `Betrachers` |  | lit-farbpalette | 1203 | arbton ist bewusst gewählt, um die Augen des Betrachers nicht zu ermüden, sondern einen weiten, einl |
| de | `Betrachterauge` |  | de-prose-plakat | 1202 | in seiner konzentrierten Essenz enthält. Das Betrachterauge wird auf die texturale Oberfläche gelenkt, u |
| de | `Betrachterraum` |  | lit-farbpalette | 1201 |  Wahl von #D7FFE0 als Fläche dient dazu, den Betrachterraum atmend zu gestalten, indem sie eine leichte, |
| de | `Bibliothekwesen` |  | de-prose-buchbinderei | 1202 | r die moderne Lesekultur und das öffentliche Bibliothekwesen war. Das handgebundene Buch der Elite, ein O |
| de | `Bilduung` |  | de-prose-wasserwerk | 1202 | be von orthophosphatenhaltigen Reagenzen zur Bilduung einer schützenden Kalkschicht oder durch die |
| de | `Bound` | loanword | de-prose-speicher | 1202 | t. Dies entspricht idealerweise dem "Compute Bound"-Szenario, wo die Rechenkapazität die Grenze |
| de | `Buchbindergewerbe` |  | de-prose-buchbinderei | 1202 |  die Zahl der ausgebildeten Arbeitskräfte im Buchbindergewerbe insgesamt stark anstieg, sank die Tiefe des  |
| de | `Burst` | loanword | de-prose-speicher | 1203 | e strikte Sequenzierung oder ein optimiertes Burst-Lesen bevorzugt. Die Architektur muss daher  |
| de | `Capacity` | loanword | de-prose-speicher | 1202 | ich der lokalen Caches. In dieser Phase des "Capacity Bound" oder später des "Bandwidth Bound" dom |
| de | `Chromatiksystem` |  | lit-farbpalette | 1202 | em sehr spezifischen und bewusst reduzierten Chromatiksystem getragen, das aus exakt vier definierten Far |
| de | `Coalescing` | loanword | de-prose-speicher | 1203 | egation von Anfragen, die in der Fachsprache Coalescing genannt wird, können mehrere einzelne Lesevo |
| de | `Compute` | loanword | de-prose-speicher | 1202 |  fungiert. Dies entspricht idealerweise dem "Compute Bound"-Szenario, wo die Rechenkapazität die  |
| de | `Computing` | loanword | de-prose-speicher | 1202 |  die primäre Limitation heutiger heterogener Computing-Systeme gilt. Während die Anzahl der Rechene |
| de | `Crafts` | loanword | de-prose-buchbinderei | 1203 | rch englische Einflüsseusse wie den Arts and Crafts, suchte in den letzten Jahrzehnten des Jahrh |
| de | `Cyan` | loanword | lit-farbpalette | 1201 | len leicht zur Verschiebung in ein zu kaltes Cyan neigt, was die Harmonie mit #1B3A5F zerstöre |
| de | `Datenzugr` |  | de-prose-speicher | 1202 | end.  Letztlich bestimmt die Architektur der Datenzugr einen fundamentalen Unterschied zwischen CPU |
| de | `Detailverwoblung` |  | de-prose-plakat | 1201 | en Jahrhunderts oft in der Überfülle und der Detailverwoblung ihren Reiz finden, das Plakat aber im Gegent |
| de | `Docker` | loanword | de-agent-fehlersuche | 1202 | reproduzierbare Umgebung her (z.B. lokal via Docker), um das Verhalten mit synthetischen Anfrage |
| de | `Druckminderungsventilen` |  | de-prose-wasserwerk | 1201 | ckhaltung erfolgt daher durch ein System aus Druckminderungsventilen und geregelten Pumpstationen, die den Druck  |
| de | `Druckproofs` |  | lit-farbpalette | 1203 | bkodierung sicherstellt. Bei der Prüfung der Druckproofs ist stets die in der Datei hinterlegte Versi |
| de | `Drucksteigerungsmaßnahmen` |  | de-prose-wasserwerk | 1201 | Kernbereiche gewahrt bleibt, ohne dass teure Drucksteigerungsmaßnahmen in unterversorgten Sektoren den Gesamtenergi |
| de | `Durchdrung` |  | de-prose-wasserwerk | 1201 | umfasst. Nur durch diese tiefe, kontextuelle Durchdrung des Netzes kann die Abwägung zwischen dem wi |
| de | `Durchfluß` |  | de-prose-wasserwerk | 1202 | die spezifische Oberfläche im Verhältnis zum Durchfluß, was die Adhäsionskräfte für Biofilmkeime un |
| de | `Durchflußkapazität` |  | de-prose-wasserwerk | 1202 |  Rohren zu lokalisieren und gleichzeitig die Durchflußkapazität für den morgendlichen Spitzenverbrauch vorzu |
| de | `Durchflußmessungen` |  | de-prose-wasserwerk | 1202 | rch eine Kombination aus direkten Druck- und Durchflußmessungen im Feld, die durch eine statistische Analyse |
| de | `Durchsatzbedarf` |  | de-prose-speicher | 1202 | mengen mit hoher Latenz-Toleranz, aber hohem Durchsatzbedarf handhaben. Die hierarchische Speicherschicht |
| de | `Einflüsseusse` |  | de-prose-buchbinderei | 1203 | ng der Buchkunst, inspiriert durch englische Einflüsseusse wie den Arts and Crafts, suchte in den letzt |
| de | `Endverteilnetz` |  | de-prose-wasserwerk | 1203 | die nicht mehr von den üblichen Verlusten im Endverteilnetz, den sogenannten unkontrollierten Leckagen,  |
| de | `Eutrifizierungen` |  | de-prose-wasserwerk | 1201 | ffenen Oberflächenflächen aufgrund möglicher Eutrifizierungen und saisonal bedingter Trübebelastungen ein  |
| de | `Execution` | loanword | de-prose-speicher | 1201 | nnvoll auszulasten. Dieser als „Out-of-Order Execution“ oder Thread-Switching bekannte Mechanismus  |
| de | `Expositionsfläche` |  | de-prose-plakat | 1201 | ueller Vorhang, der sich vor die eigentliche Expositionsfläche legt, und muss daher so komponiert sein, das |
| de | `FLOPs` |  | de-prose-speicher | 1202 |  hängt daher weniger von der reinen Zahl der FLOPs ab, als von der intelligenten Strukturierung |
| de | `Falzapparat` |  | de-prose-buchbinderei | 1202 | nderei, als spezialisierte Werkzeuge wie der Falzapparat, der Heftstich und die Deckelmontage-Maschin |
| de | `Farbigkeiten` |  | de-prose-plakat | 1201 | sferiert. Statt der grellen, kontrastreichen Farbigkeiten, die die zeitgenössische Reklame kennzeichne |
| de | `Findmittel` |  | de-agent-umzug | 1203 |  keine Mängel aufweist.  9. Aktualisiere die Findmittel und digitale Kataloge, um die neuen Standort |
| de | `Fittings` | loanword | de-prose-wasserwerk | 1203 |  aus den alten Legierungen der Armaturen und Fittings stammen, auf eine fortschreitende Degradatio |
| de | `Flockungs` |  | de-prose-wasserwerk | 1202 | ufe häufig durch eine chemisch-physikalische Flockungs- und Sandfiltrationsstufe ergänzt werden, um |
| de | `Frakturschriften` |  | de-prose-plakat | 1203 | ung erzeugen. Die Verwendung von überladenen Frakturschriften, wie sie etwa in der deutschen Buchkunst des |
| de | `GPUs` |  | de-prose-speicher | 1201 | kollen versehen sind, ist der L2-Speicher in GPUs für den Durchsatz von großen Datenströmen op |
| de | `Gesellenschaftsrechte` |  | de-prose-buchbinderei | 1203 | fung basierte, geriet durch die Anhebung der Gesellenschaftsrechte und die neuen Anforderungen an das Maschinen |
| de | `Graphenberechnungen` |  | de-prose-speicher | 1203 | i irregularen Datenstrukturen oder komplexen Graphenberechnungen der Fall ist, kollabiert die Leistung, da di |
| de | `Graphikarchitekturen` |  | de-prose-speicher | 1202 | er hinaus existieren in den meisten modernen Graphikarchitekturen explizit nutzbare, aber auch implizit genutz |
| de | `Graphikkernel` |  | de-prose-speicher | 1203 | king Sets. Die Größe der Datenmenge, die ein Graphikkernel im aktiven Rechenzyklus verarbeitet, bestimm |
| de | `Graubild` |  | de-prose-plakat | 1202 |  Details auf die Distanz zu einem unlesbaren Graubild, während zu weite Läufe den Zusammenhalt der |
| de | `Gravurspuren` |  | de-prose-plakat | 1202 | er silbern getriebenen Kanne, der die feinen Gravurspuren unter Streiflicht zeigt, oder der Querschnit |
| de | `Grunddrucktension` |  | de-prose-wasserwerk | 1202 | htdruckkurven ergänzt wird. Eine Abnahme der Grunddrucktension in einem bestimmten Sektor, gepaart mit eine |
| de | `Hadernpapierpapier` |  | de-prose-buchbinderei | 1202 | prüche der Kundschaft: Schwerer Bütten- oder Hadernpapierpapier bildete den Untergrund, während die Bezüge a |
| de | `Handwerkwississ` |  | de-prose-buchbinderei | 1202 |  sank die Tiefe des individuell vermittelten Handwerkwississ in vielen Betrieben drastisch ab. Es entstan |
| de | `Hexcodes` |  | lit-farbpalette | 1202 | Fassung v3.2.1 begründet. Nur wenn alle vier Hexcodes im Verhältniss zueinander gehalten werden, f |
| de | `Hierarchieeb` |  | de-prose-speicher | 1203 |  von weiteren Anfragen in die nächsttieferen Hierarchieeb auslöst, bis hinab in den GDDR-Controller. D |
| de | `High` | loanword | de-prose-speicher | 1203 | idendsten Architekturmerkmale im Bereich der High-Performance-Computing-Architekturen dar. Um  |
| de | `Hochbehältters` |  | de-prose-wasserwerk | 1203 |  entkeimt werden, bevor es in die Stufen des Hochbehältters oder in die Pumpenstufen der Mittelzone eing |
| de | `Hochleistungscomputings` |  | de-prose-speicher | 1201 | echenplattform gewandelt, die im Zentrum des Hochleistungscomputings steht. Diese Entwicklung bringt jedoch eine  |
| de | `Indizium` |  | de-prose-wasserwerk | 1203 | dem Verteilnetz zurückfließt, ist ein klares Indizium dafür, dass die Korrosion an der inneren Roh |
| de | `Infrastruturen` |  | de-prose-wasserwerk | 1202 | ngebäude, Gewerbebetriebe und die kommunalen Infrastruturen wie Schwimmbäder oder Brandlöschbrunnen vers |
| de | `Initialle` |  | de-prose-buchbinderei | 1201 | unte Decke, die kleine, vom Meister gesetzte Initialle – all dies, was das Buch zum handgewerkliche |
| de | `Interposer` |  | de-prose-speicher | 1202 | on HBM-Modulen, liegt physisch auf denselben Interposer wie der GPU-Chip, was zwar die Signallaufweg |
| de | `Jahrunderte` |  | de-prose-buchbinderei | 1202 | h es verlor die Monopolstellung, die es über Jahrunderte innehatte. Die Maschine übernahm den Anteil  |
| de | `Jugendstilsprache` |  | de-prose-plakat | 1203 | e, organische Kurvatur zeigt, wie sie in der Jugendstilsprache oder der Naturalisten-Ästhetik des späten Ne |
| de | `Karthärte` |  | de-prose-wasserwerk | 1202 |  Sandfiltrationsstufe ergänzt werden, um die Karthärte auf ein trinkwassertaugliches Niveau abzusen |
| de | `Kilometmetern` |  | de-prose-wasserwerk | 1202 | m über eine Gesamtlänge von mehreren hundert Kilometmetern erstreckt und dabei sowohl die einzelnen Woh |
| de | `Kohärenzprotokollen` |  | de-prose-speicher | 1201 | e Caches stark verkleinert und mit komplexer Kohärenzprotokollen versehen sind, ist der L2-Speicher in GPUs f |
| de | `Konfig` |  | de-agent-fehlersuche | 1202 | ung auf, was externe Faktoren (Netzwerklast, Konfig) als Ursache ausschließt.  8. Überprüfe die  |
| de | `Kontaminationsrisiken` |  | de-prose-wasserwerk | 1201 | ems abbildet. Ein zu niedriger Druck kann zu Kontaminationsrisiken führen, indem durch Undichtigkeiten an Verbi |
| de | `Kriechverhalten` |  | de-prose-wasserwerk | 1201 | ung durch zyklische Druckbelastungen und das Kriechverhalten unter Langzeitlast zu einer Versteifung und  |
| de | `Kuratierung` |  | de-prose-plakat | 1202 | nalisiert bereits im Plakat die Qualität der Kuratierung.  Die Farbwahl im Druck ist technisch und ps |
| de | `Langzeitlast` |  | de-prose-wasserwerk | 1201 | uckbelastungen und das Kriechverhalten unter Langzeitlast zu einer Versteifung und schließlich zur Spr |
| de | `Leaks` | loanword | de-agent-fehlersuche | 1202 | p` oder `smem` zur Identifikation von Memory-Leaks oder Swap-Überlast.    *Bestätigung*: Der RA |
| de | `Leckageortung` |  | de-prose-wasserwerk | 1203 | atische Überwachung, die über die klassische Leckageortung hinausgeht. Sie prüft nicht nur die hydrauli |
| de | `Leimeimente` |  | de-prose-buchbinderei | 1203 | g von Zelluloseleimen und späterer, härterer Leimeimente, die bei niedrigeren Temperaturen abliefenen |
| de | `Leimgerät` |  | de-prose-buchbinderei | 1202 | eue Facharbeiter an der Falzmaschine oder am Leimgerät wurde nicht mehr als ganzheitlicher Handwerk |
| de | `Leimung` |  | de-prose-buchbinderei | 1203 | rallel dazu entwickelte sich die Technik der Leimung und der Kaschierung weiter, wobei chemische  |
| de | `Lesebarkeit` |  | lit-farbpalette | 1203 |  als Informationsträger, da es die necessary Lesebarkeit bei verschiedenen Betrachtungsabständen sich |
| de | `Lock` | loanword | de-agent-fehlersuche | 1202 | : Der Dienst nutzt z.B. eine global geteilte Lock-Variable, die nach der ersten Anfrage nicht  |
| de | `Logeinträge` |  | de-agent-fehlersuche | 1203 | instabilen Last bestätigt. 2. Untersuche die Logeinträge im System-Journal seit heute Morgen mit `jou |
| de | `Logs` | loanword | de-agent-fehlersuche | 1202 | hen würde.  2. Durchsuche die System-Journal-Logs der letzten Stunden (`journalctl -u meindien |
| de | `Lpc` |  | lit-farbpalette | 1201 | ten scharen bleiben, wenn die Raster auf 175 Lpc für die Akzente eingestellt werden, was im R |
| de | `Manganpartikel` |  | de-prose-wasserwerk | 1201 | eicht. Wenn sich im Leitungsnetz Eisen- oder Manganpartikel, also sogenannte Rostflocken oder Sedimente, |
| de | `Mangans` |  | de-prose-wasserwerk | 1203 | rn im Wasserwerk. Ein Anstieg des Eisens und Mangans im aufbereiteten Wasser, das aus dem Verteil |
| de | `Maroquin` |  | de-prose-buchbinderei | 1202 | den Untergrund, während die Bezüge aus edlem Maroquin, feinem Kalbsleder, bunten Musterseiden oder |
| de | `Maschinisierung` |  | de-prose-buchbinderei | 1202 | er bibliophilen Erfahrung.  Der Übergang zur Maschinisierung, der sich über mehrere Dekaden hinweg vollzo |
| de | `Membrantechniken` |  | de-prose-wasserwerk | 1201 | lter oder in moderneren Anlagen direkt durch Membrantechniken, wobei die Vorfilterung oft durch Aktivkohle |
| de | `Mitbestimmer` |  | de-prose-buchbinderei | 1201 | num einer Auflage. Der Buchbinder, der einst Mitbestimmer des ästhetischen Wertes war, wurde zum ausfü |
| de | `Multiprocessors` | loanword | de-prose-speicher | 1201 | führungseinheiten, der sogenannten Streaming Multiprocessors, exponentiell angestiegen ist und die theore |
| de | `Musterseiden` |  | de-prose-buchbinderei | 1202 | us edlem Maroquin, feinem Kalbsleder, bunten Musterseiden oder teilsweise aus imprägniertem Leinen bes |
| de | `Nouveau` |  | de-prose-plakat | 1201 |  von den organisch fließenden Linien der Art Nouveau hin zu den statischeren, serifenbetontenen L |
| de | `Occupancy` | loanword | de-prose-speicher | 1201 | n, die berechnet werden soll. Die sogenannte Occupancy, also die Anzahl der gleichzeitig aktiven Th |
| de | `Overhead` | loanword | de-prose-speicher | 1201 | teren Speicherbursts zu verschmelzen, um den Overhead des Adressierungs- und Protokollübertrags zu |
| de | `Overprint` | loanword | lit-farbpalette | 1201 | Die Fassung v3.2.1 enthält daher spezifische Overprint-Einstellungen, die die Überlagerungen dieser |
| de | `Patine` | loanword | de-prose-plakat | 1203 | Gold der Messingbeschläge, das Kupferrot der Patine, die erdigen Brauntöne des Holzes und die kü |
| de | `Plakatkommposition` |  | de-prose-plakat | 1202 | , ist das eigentliche Architekturproblem der Plakatkommposition. In der klassischen Werbung der Epoche, die  |
| de | `Portnummer` |  | de-agent-fehlersuche | 1201 | m festzustellen, ob der Dienst die erwartete Portnummer hört, etwa über `ss -tunp   grep dienstname` |
| de | `Prefetching` |  | de-prose-speicher | 1203 | r komplexe Strategien zur Prädiktion und zum Prefetching, dem Vorladen von Daten, anwenden. Dabei ent |
| de | `Prepress` |  | lit-farbpalette | 1201 | alstarken #FFB347 muss auch in der digitalen Prepress-Ansicht kontrolliert werden, um sicherzustel |
| de | `Pressenscheck` |  | lit-farbpalette | 1201 | ieser vier Töne regeln. Es ist ratsam, einen Pressenscheck zu drucken, da die Toleranzen für #050505 se |
| de | `Qualitätseinstupfe` |  | de-prose-wasserwerk | 1201 | leibt, ohne dass die Wasserverluste oder die Qualitätseinstupfe so weit fortschreiten, dass die Gesundheit d |
| de | `Random` | loanword | de-prose-speicher | 1201 | uf den sehr viel langsameren VRAM, den Video-Random-Access-Memory, reduziert. Die Bandbreite die |
| de | `Rastervierung` |  | lit-farbpalette | 1202 |  sondern eine Notwendigkeit für die optische Rastervierung im Offsetdruck. Wenn die Schrift in #050505  |
| de | `Rastung` |  | lit-farbpalette | 1202 | E0 als Hintergrund muss vollflächig und ohne Rastung abgemischt auslaufen, damit die Fläche ruhig |
| de | `Reflektionseigenschaften` |  | lit-farbpalette | 1203 | nem oder ungestrichenem Papier verändert die Reflektionseigenschaften der vier genannten Hexcodes erheblich. Das m |
| de | `Regalkapazität` |  | de-agent-umzug | 1201 | ns veranlassen, um sicherzustellen, dass die Regalkapazität und die klimatischen Bedingungen den archivi |
| de | `Regalplan` |  | de-agent-umzug | 1202 | m neuen Gebäude gemäß dem vorher definierten Regalplan. Die Aufgabe ist erledigt, wenn alle Kisten  |
| de | `Reproduktionsprozesses` |  | de-prose-buchbinderei | 1201 | doppelten Wesen – als Teil des industriellen Reproduktionsprozesses und als Hüter des handwerklichen Restes – la |
| de | `Reproduktionstreue` |  | lit-farbpalette | 1202 | egt. Für eine Auflage von 4800 Stück ist die Reproduktionstreue des Hintergrund #D7FFE0, der Schrift #050505 |
| de | `Retry` | loanword | de-agent-fehlersuche | 1202 | stente Werte, insbesondere bei Timeout- oder Retry-Einstellungen, die die zweite Anfrage betref |
| de | `Sammlungsname` |  | de-prose-plakat | 1202 | s die Hauptzeile, oft der Ausstellungs- oder Sammlungsname, als massiver Textkörper lesbar bleibt, auch |
| de | `Sammlungsnatur` |  | de-prose-plakat | 1202 | ntation ist, sondern eine Interpretation der Sammlungsnatur. Es muss die Spannung zwischen dem industrie |
| de | `Schriftgrösse` |  | de-prose-plakat | 1203 | n die Flächen und die grobe Komposition. Die Schriftgrösse des Titels muss so bemessen sein, dass er de |
| de | `Sepia` | loanword | de-prose-plakat | 1202 | das Objektdetail, sei es in Schwarz-Weiß, in Sepia oder in reduzierten Farbpressen erscheint, a |
| de | `Sickerwässern` |  | de-prose-wasserwerk | 1203 | on Pestizidrückständen oder nitratbelasteten Sickerwässern aus landwirtschaftlich intensiv genutzten Ge |
| de | `Signum` |  | de-prose-buchbinderei | 1201 | ormierten, aber auch zur identitätstiftenden Signum einer Auflage. Der Buchbinder, der einst Mit |
| de | `State` | loanword | de-agent-fehlersuche | 1202 | is die zweite Anfrage eintrifft, was auf ein State-Problem im Prozess hinweist.  10. Dokumentie |
| de | `Strasse` |  | de-prose-plakat | 1203 |  den Text für die schnelle Erfassung auf der Strasse zu langsam machen würde. Entscheidend ist je |
| de | `Swap` | loanword | de-agent-fehlersuche | 1202 | em` zur Identifikation von Memory-Leaks oder Swap-Überlast.    *Bestätigung*: Der RAM-/Swap-Ve |
| de | `Thread` | loanword | de-prose-speicher | 1201 | ads im Registerbereich ab, dass er, wenn ein Thread auf eine Speicheranfrage wartet und blockier |
| de | `Threads` | loanword | de-prose-speicher | 1201 | uniger plant so viele unabhängige Faden oder Threads im Registerbereich ab, dass er, wenn ein Thr |
| de | `Timeout` | loanword | de-agent-fehlersuche | 1202 | pezifische Fehlermeldungen (z.B. OOM-Killer, Timeout) direkt vor dem Absturz hin.  3. Untersuche  |
| de | `Transaktions` |  | de-agent-fehlersuche | 1202 |  nicht-idempotente Operationen oder fehlende Transaktions-Behandlung bei wiederholten Anfragen.    *Be |
| de | `Uferfiltrationsbrunnen` |  | de-prose-wasserwerk | 1203 | eist durch eine Kombination von Tiefbrunnen, Uferfiltrationsbrunnen in der Nähe von Flüssen und vereinzelten Que |
| de | `Umschaffung` |  | de-prose-buchbinderei | 1201 | e Zerstörung der Form, sondern auch als ihre Umschaffung zu deuten ist. |
| de | `Uniformbuch` |  | de-prose-buchbinderei | 1203 | it der Lieferzeiten erhöhte. Das sogenannte „Uniformbuch“, dessen Einband in großen Mengen maschinell |
| de | `Verhältniss` |  | lit-farbpalette | 1202 | .1 begründet. Nur wenn alle vier Hexcodes im Verhältniss zueinander gehalten werden, funktioniert die |
| de | `Verleimung` |  | de-prose-buchbinderei | 1201 | rucktes Papier, erforderte eine gleichmäßige Verleimung und ein präzises Zusammentreffen der Falzung |
| de | `Verpakkung` |  | de-agent-umzug | 1203 | erten Haftungsregeln unterzeichnet sind.  4. Verpakkung aller Materialien in stoßfeste, klimastabile |
| de | `Verrohungsprozesse` |  | de-prose-buchbinderei | 1203 | in Gegentrend, der sich als Reaktion auf die Verrohungsprozesse lesen lässt. Die Bewegung der Buchkunst, ins |
| de | `Verteilnetz` |  | de-prose-wasserwerk | 1203 | Mangans im aufbereiteten Wasser, das aus dem Verteilnetz zurückfließt, ist ein klares Indizium dafür, |
| de | `Weiss` | loanword | de-prose-plakat | 1203 | orgen. Eine zu starke, Reduktion auf Schwarz-Weiss wäre zwar elegant, verliert aber die sinnlic |
| de | `Working` | loanword | de-prose-speicher | 1203 | Rolle der Arbeitsmenge, also des sogenannten Working Sets. Die Größe der Datenmenge, die ein Grap |
| de | `Zelluloseleimen` |  | de-prose-buchbinderei | 1203 |  zentrale Rolle spielten. Die Einführung von Zelluloseleimen und späterer, härterer Leimeimente, die bei  |
| de | `Zierbinderkunst` |  | de-prose-buchbinderei | 1203 | aschinentabellen Vorrang vor der klassischen Zierbinderkunst erhielt. Dies bedeutete jedoch nicht das vol |
| de | `Zuerdest` |  | lit-farbpalette | 1201 |  sind mehrere Faktoren zwingend zu beachten. Zuerdest muss die Korrespondenz zwischen den digitale |
| de | `Zustandprotokolls` |  | de-agent-umzug | 1202 | entarisiere sämtliche Bestände inklusive des Zustandprotokolls, wobei jeder Gegenstand eine eindeutige Iden |
| de | `Zweidimensionalität` |  | de-prose-plakat | 1201 | rielle und geistige Aura der Sammlung in die Zweidimensionalität des Papiers transferiert. Statt der grellen, |
| de | `activating` | loanword | de-agent-fehlersuche | 1202 | atus zeigt wiederholte Neustarts oder einen „activating“-Zustand, während „active“ bei Erfolg stehen |
| de | `active` | loanword | de-agent-fehlersuche | 1201 | dienstname --since today -n 50`; ein Status „active“ bei unverändertem Protokoll ohne Fehlermeld |
| de | `archivgerechten` |  | de-agent-umzug | 1202 |  als bestanden, wenn die Werte innerhalb der archivgerechten Normbereiche stabil liegen.  9. Führe eine S |
| de | `archivischen` |  | de-agent-umzug | 1201 | pazität und die klimatischen Bedingungen den archivischen Normen entsprechen. Die Fertigstellung ist d |
| de | `archivkritischen` |  | de-agent-umzug | 1203 |  erfüllt, wenn die Sensordaten innerhalb der archivkritischen Toleranzbereiche stabil liegen.  8. Führe ei |
| de | `asbestzementierten` |  | de-prose-wasserwerk | 1203 | Netze bestehen überwiegend aus Grauguss, aus asbestzementierten Faserzementrohren oder aus frühen Stahlausfü |
| de | `atmosphärdichten` |  | de-prose-plakat | 1201 | undanzen, um das Plakat zu einem kohärenten, atmosphärdichten und funktional präzisen Artefakt zu formen,  |
| de | `auflage` |  | lit-farbpalette | 1201 | 50505 ein kritischer Faktor, da es bei einer auflage von 4800 Exemplaren zu ungleichmäßigem Tinte |
| de | `blockhafte` |  | de-prose-plakat | 1202 | iftsprache der Zeit abhebt. Eine zu massive, blockhafte Groteske, wie sie später in der Moderner auf |
| de | `contaminieren` |  | lit-farbpalette | 1202 | halten und nicht durch den Rahmen #1B3A5F zu contaminieren.  Die Summe der Elemente ergibt eine Balance |
| de | `differenziierte` |  | de-prose-plakat | 1203 | l zu studieren, muss das Plakat eine zweite, differenziierte Lesestufe bieten. Hier entfaltet sich der De |
| de | `entsättigtes` |  | lit-farbpalette | 1202 | hmens übernimmt der Ton #1B3A5F, ein tiefes, entsättigtes Marineblau, das der Komposition eine feste,  |
| de | `erör` |  | de-prose-plakat | 1203 | essend ist die Frage der lesenden Distanz zu erör. Das Plakat muss zweifichtig funktionieren.  |
| de | `essentialen` |  | de-prose-plakat | 1203 | esucher, die es anspricht. Der Textblock der essentialen Infos, etwa der Ausstellertitel, der Datum u |
| de | `failed` | loanword | de-agent-fehlersuche | 1201 | inen abgestürzten Prozess aus, während eine „failed“-Meldung auf einen internen Fehler hinweist. |
| de | `festgelegenen` |  | de-agent-umzug | 1201 | ung der Dokumente im neuen Magazin gemäß dem festgelegenen Ordnungssystem. Der Schritt ist fertig, wenn |
| de | `ganzheit` |  | de-prose-buchbinderei | 1201 | g, war die Einmaligkeit des Objektes und die ganzheit innige Verbindung zwischen dem Herstellungsa |
| de | `gehoren` |  | de-prose-plakat | 1203 | nen zur zweiten und dritten Ebene der Lesung gehoren.  Das Verhältnis von Fläche und Schrift ist  |
| de | `glanzig` |  | lit-farbpalette | 1201 |  da die Oberfläche mit #050505 Text nicht zu glanzig oder zu stumpf wirken darf. Die Balance zwis |
| de | `handgewerklichen` |  | de-prose-buchbinderei | 1201 | etzte Initialle – all dies, was das Buch zum handgewerklichen Artefakt erhob, trat zugunsten der reprodukt |
| de | `heikleses` |  | de-prose-plakat | 1201 |  Räume, ist bei dieser Aufgabe ein besonders heikleses Problem, da die Kunstgewerbe des neunzehnten |
| de | `hinwegleuchtet` |  | lit-farbpalette | 1202 |  Ton #D7FFE0 und den stabilen Rahmen #1B3A5F hinwegleuchtet.  Diese Farbtetralogie ist in der Fassung v3 |
| de | `identitätstiftenden` |  | de-prose-buchbinderei | 1201 | sband wurde zum uniformierten, aber auch zur identitätstiftenden Signum einer Auflage. Der Buchbinder, der ei |
| de | `ingenieurstechnische` |  | de-prose-wasserwerk | 1203 | llt für eine mittelgroße Stadt eine komplexe ingenieurstechnische und hydrogeologische Herausforderung dar, di |
| de | `klimastabile` |  | de-agent-umzug | 1203 | . Verpakkung aller Materialien in stoßfeste, klimastabile Behälter und versiegere sie zur Nachverfolgu |
| de | `makrofotografisches` |  | de-prose-plakat | 1202 | ntimität. Stattdessen empfiehlt es sich, ein makrofotografisches oder zeichnerisches Detail eines einzelnen O |
| de | `modemeißig` |  | de-prose-plakat | 1201 | sten Abwägung herausfordert. Eine zu rigide, modemeißig gesetzte Schrift kann die historische Anmut  |
| de | `monokasualistische` |  | de-prose-plakat | 1202 | tellungsdaten, kann eine feinere, vielleicht monokasualistische Groteske als komplementäre Informationsschri |
| de | `necessary` | loanword | lit-farbpalette | 1203 | es Plakats als Informationsträger, da es die necessary Lesebarkeit bei verschiedenen Betrachtungsab |
| de | `papiernass` |  | de-prose-buchbinderei | 1201 | undene Buch der Vormoderne auf hochwertiges, papiernass geleimtes oder ledergebundenes Einbände gese |
| de | `serifenbetontenen` |  | de-prose-plakat | 1201 | ien der Art Nouveau hin zu den statischeren, serifenbetontenen Lettern der Buchkunst, offenbart sich ein we |
| de | `signalstarken` |  | lit-farbpalette | 1201 | Balance zwischen dem ruhigen #1B3A5F und dem signalstarken #FFB347 muss auch in der digitalen Prepress- |
| de | `silbernene` |  | de-prose-plakat | 1201 | portieren, die in der Fotografie oft nur als silbernene Reflexion oder stumpfe Glasur erscheint. Es  |
| de | `stoßgedämpft` |  | de-agent-umzug | 1202 |  Kategorien, wobei fragile Stücke zusätzlich stoßgedämpft verpackt werden. Die Richtigkeit ist geprüft |
| de | `ungestrichenem` |  | lit-farbpalette | 1203 | ben können.  Der Druck auf gestrichenem oder ungestrichenem Papier verändert die Reflektionseigenschafte |
| de | `versiegere` |  | de-agent-umzug | 1203 | lien in stoßfeste, klimastabile Behälter und versiegere sie zur Nachverfolgung    - Die Aufgabe gilt |
| en | `acidification` |  | en-prose-archive | 1201 | idize, releasing acids that catalyze further acidification, causing the paper to turn brown and crumble |
| en | `amidst` |  | en-prose-foundry | 1202 | atched the required dimensions. It was here, amidst the scent of sawdust and linseed oil, that t |
| en | `checksums` |  | lit-releasenote | 1201 | al. The tag release-2026-09-18a provides the checksums necessary to validate the container path mod |
| en | `codebase` |  | lit-releasenote | 1201 | distinct identifier for this snapshot of the codebase and model weights. The build 0xB7A31F is the |
| en | `contiguently` |  | lit-releasenote | 1201 | del, which is held separately, can be mapped contiguently against the memory blocks defined here in th |
| en | `convolutional` |  | lit-releasenote | 1201 | itical component for the state space model's convolutional front-end. The ssm_conv1d_alpha tensor now e |
| en | `curation` |  | en-prose-archive | 1201 | , the intellectual access, the philosophical curation, and the physical restoration of the holding |
| en | `deaccession` |  | en-prose-archive | 1202 | ith no unique data, are often candidates for deaccession. This destruction is not undertaken lightly; |
| en | `deacidified` |  | en-prose-archive | 1201 | eed to be re-encapsulated or, in rare cases, deacidified, though the latter is a complex and expensiv |
| en | `deserialization` |  | lit-releasenote | 1203 | on kernels. It is crucial to verify that the deserialization process handles these 248320 records efficie |
| en | `discretization` |  | lit-releasenote | 1203 | ional convolutional operations preceding the discretization step in the state-space modeling blocks rema |
| en | `earthearth` |  | en-prose-foundry | 1201 |  the pour. The pouring floor, a long, rammed-earthearth trench lined with heat-resistant bricks, was |
| en | `embrittlement` |  | en-prose-archive | 1202 | dic bonds in the cellulose chain, leading to embrittlement and eventual disintegration. This phenomenon |
| en | `evidentially` |  | en-prose-archive | 1202 | ks not only what is interesting, but what is evidentially significant for the long-term understanding  |
| en | `glycosidic` |  | en-prose-archive | 1202 | ghly acidic. Acid hydrolysis breaks down the glycosidic bonds in the cellulose chain, leading to emb |
| en | `granularized` |  | lit-releasenote | 1201 | ised set of coefficients that allow for more granularized state transitions within the linear attentio |
| en | `grey` |  | en-prose-foundry | 1201 | turning the workers’ skin a uniform shade of grey. The cleaning was not just cosmetic; it reve |
| en | `hygroscopic` |  | en-prose-archive | 1202 |  temperature and relative humidity. Paper is hygroscopic, meaning it constantly absorbs and releases  |
| en | `interpretable` |  | en-prose-archive | 1203 |  that the information remains accessible and interpretable.  However, the preservation of a collection  |
| en | `melter` |  | en-prose-foundry | 1201 | entity, requiring constant monitoring by the melter. The melter, often the founder himself, watc |
| en | `microclimate` |  | en-prose-archive | 1203 |  line of defense, ensuring that the internal microclimate remains distinct from the chaotic exterior.  |
| en | `misruns` |  | en-prose-foundry | 1202 | too violent eroded the sand cores and caused misruns. The floor was a trellis of green sand, awai |
| en | `mould` |  | en-prose-foundry | 1203 | ed too tightly, the gases trapped within the mould could not escape, leading to blowholes and v |
| en | `moulder` |  | en-prose-foundry | 1202 | al shock of of molten iron. The skill of the moulder lay in the tactile sense of moisture and com |
| en | `moulders` |  | en-prose-foundry | 1203 | emporary vessel for industrial creation. The moulders worked in pairs, men with broad chests and f |
| en | `moulding` |  | en-prose-foundry | 1203 | lished to perfection, they were moved to the moulding floor, the heart of the foundry’s daily rhyt |
| en | `moulds` |  | en-prose-foundry | 1203 | the sand and then to the cleaning floor. The moulds were broken, smashed apart by men with heavy |
| en | `ofof` |  | en-prose-foundry | 1201 |  the next heat. In this cycle, from the wood ofof the pattern to the final inspection, the sma |
| en | `oversized` |  | en-prose-foundry | 1202 | s as it cools, meaning the pattern had to be oversized in precise mathematical proportions to ensur |
| en | `oversizes` |  | en-prose-foundry | 1203 | and so he built his patterns with deliberate oversizes, leaving allowances for the contraction of c |
| en | `pourer` |  | en-prose-foundry | 1201 | n rows, their tops open to the air. The lead pourer, a a figure of immense responsibility, moved |
| en | `pre` |  | en-prose-foundry | 1201 | the small iron foundry stood as a bastion of pre-industrial manufacturing precision, a place  |
| en | `rammers` |  | en-prose-foundry | 1202 | rked the sand against the pattern using hand rammers and pneumatic jolters, layers of earth compa |
| en | `recalibrated` |  | lit-releasenote | 1203 | ation drift in earlier beta builds, has been recalibrated to ensure deterministic behavior across dist |
| en | `recalibration` |  | lit-releasenote | 1202 |  the ssm_conv1d_alpha tensor has undergone a recalibration that aligns with the requirements set out fo |
| en | `reproducibility` |  | lit-releasenote | 1202 |  maintain this exact structure to ensure the reproducibility promised by the release-2026-09-18a tag.  Fr |
| en | `respirable` |  | en-prose-foundry | 1202 | s, a task that filled the the air with fine, respirable dust. The castings, still retaining signific |
| en | `roadmap` |  | lit-releasenote | 1203 | extualizes this release within the quarterly roadmap for the Flash-Next initiative. While the pri |
| en | `sprue` |  | en-prose-foundry | 1201 | ed in a bright, blinding ribbon into the the sprue and runners of of the molds. The sound was a |
| en | `stdarkened` |  | en-prose-foundry | 1203 | d manual craftsmanship blurred into the soot-stdarkened skin and calloused hands of the men who work |
| en | `theshrinkage` |  | en-prose-foundry | 1202 | hese patterns. Each curve had to account for theshrinkage, the phenomenon by which molten metal contra |
| en | `thethe` |  | en-prose-foundry | 1201 | act of prophetic engineering, accounting for thethe shrinkage of iron as it cooled, a variable t |
| en | `transmittals` |  | en-prose-archive | 1203 | d. Routine duplicates, purely administrative transmittals that contain no unique information, and obso |
| en | `tuyeres` |  | en-prose-foundry | 1201 | roaring fans that sent blasts of air through tuyeres into the belly of the shaft. The cupola was  |
| en | `uncompromised` |  | lit-releasenote | 1201 | ash-next/CNQ4.5-M-00002-of-00003.cnq remains uncompromised despite the high throughput of the processin |
| en | `unglamorous` |  | en-prose-archive | 1202 | vation, but rather a symphony of meticulous, unglamorous decisions made over decades and sometimes ce |

## near-miss literals

| prompt | seed | demanded | seen | distance | count |
|---|---|---|---|---|---|
| lit-farbpalette | 1201 | `/srv/plakat/2026/vorlage-v3.2.1.svg` | `/srv/plakat/2026/vorlage-v3.211.svg` | 1 | 1 |
| lit-farbpalette | 1201 | `/srv/plakat/2026/vorlage-v3.2.1.svg` | `/srv/plakat/2026/vorlage-v3.210.svg` | 2 | 1 |
| lit-farbpalette | 1201 | `v3.2.1` | `v3.2` | 2 | 1 |

