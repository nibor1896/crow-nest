# quality probe - C0-crow-control

- date 2026-09-18, generated at repo commit `22ec6b4`, prompt set version 1
- endpoint `http://127.0.0.1:8099`, engine `crow`, model `Qwen3.8-Flash-Next-CNQ4.5-M`
- row temperature 1.0, top_p 0.95, top_k 20, presence_penalty 0.0, min_p 0.0, max_tokens 2600
- thinking: absent (crow); seeds [1201, 1202, 1203]; 36 generations, 0 failed, 838 s wall

## the arm in one table

| metric | value |
|---|---|
| non-word rate DE per 1000 words, per generation | 23.20 (5.25 to 57.47) |
| non-word rate DE, loanwords the EN dictionary knows removed | 15.95 (5.25 to 35.03) |
| non-word rate DE, pooled over 15507 words | 20.57 |
| non-word rate EN per 1000 words, per generation | 6.90 (3.90 to 11.95) |
| non-word rate EN, loanwords the DE dictionary knows removed | 6.90 (3.90 to 11.95) |
| non-word rate EN, pooled over 9511 words | 7.25 |
| exact literals reproduced (share of literals) | 0.895 (0.200 to 1.000) |
| exact literals reproduced (share of demanded occurrences) | 0.955 (0.667 to 1.000) |
| near-miss literal kinds seen | 4 |
| JSON: whole answer a valid document / shape ok | 4 / 4 of 6 |
| distinct-word ratio | 0.532 (0.400 to 0.701) |
| longest immediate repeat run | 1.9 (1.0 to 5.0), max 5 |
| generations with foreign-script characters | 4 of 24 (8 chars) |
| words per answer | 702 (25 to 1501) |
| answers stopped at max_tokens | 0 of 36 |

## per prompt

| prompt | lang | seed | words | finish | non-word/1000 | flagged | literals | json | longest repeat | foreign |
|---|---|---|---|---|---|---|---|---|---|---|
| de-prose-plakat | de | 1201 | 940 | stop | 23.40 | 22/940 | - | - | 2x1 | 0 |
| de-prose-plakat | de | 1202 | 1004 | stop | 13.94 | 14/1004 | - | - | 1x1 | 0 |
| de-prose-plakat | de | 1203 | 1058 | stop | 6.62 | 7/1058 | - | - | 2x1 | 0 |
| de-prose-speicher | de | 1201 | 841 | stop | 24.97 | 21/841 | - | - | 1x1 | 0 |
| de-prose-speicher | de | 1202 | 765 | stop | 30.07 | 23/765 | - | - | 2x1 | 2 |
| de-prose-speicher | de | 1203 | 1106 | stop | 25.32 | 28/1106 | - | - | 2x1 | 2 |
| de-prose-buchbinderei | de | 1201 | 1115 | stop | 14.35 | 16/1115 | - | - | 2x1 | 0 |
| de-prose-buchbinderei | de | 1202 | 1214 | stop | 32.13 | 39/1214 | - | - | 2x1 | 0 |
| de-prose-buchbinderei | de | 1203 | 1094 | stop | 20.11 | 22/1094 | - | - | 2x1 | 2 |
| de-prose-wasserwerk | de | 1201 | 1049 | stop | 9.53 | 10/1049 | - | - | 2x1 | 0 |
| de-prose-wasserwerk | de | 1202 | 1052 | stop | 18.06 | 19/1052 | - | - | 2x1 | 0 |
| de-prose-wasserwerk | de | 1203 | 1096 | stop | 23.72 | 26/1096 | - | - | 2x1 | 0 |
| en-prose-foundry | en | 1201 | 1483 | stop | 10.11 | 15/1483 | - | - | 2x1 | 0 |
| en-prose-foundry | en | 1202 | 1501 | stop | 5.33 | 8/1501 | - | - | 2x1 | 0 |
| en-prose-foundry | en | 1203 | 1423 | stop | 11.95 | 17/1423 | - | - | 1x1 | 0 |
| en-prose-archive | en | 1201 | 1222 | stop | 5.73 | 7/1222 | - | - | 2x1 | 0 |
| en-prose-archive | en | 1202 | 1361 | stop | 8.08 | 11/1361 | - | - | 2x2 | 0 |
| en-prose-archive | en | 1203 | 1278 | stop | 3.91 | 5/1278 | - | - | 1x1 | 0 |
| lit-farbpalette | de | 1201 | 531 | stop | 18.83 | 10/531 | 6/7 (0.93) | - | 2x1 | - |
| lit-farbpalette | de | 1202 | 571 | stop | 5.25 | 3/571 | 7/7 (1.00) | - | 1x1 | - |
| lit-farbpalette | de | 1203 | 505 | stop | 7.92 | 4/505 | 7/7 (1.00) | - | 1x1 | - |
| lit-releasenote | en | 1201 | 511 | stop | 3.91 | 2/511 | 5/5 (1.00) | - | 1x1 | - |
| lit-releasenote | en | 1202 | 219 | stop | 9.13 | 2/219 | 1/5 (0.67) | - | 2x1 | - |
| lit-releasenote | en | 1203 | 513 | stop | 3.90 | 2/513 | 5/5 (1.00) | - | 1x1 | - |
| json-tensorplan | en | 1201 | 25 | stop | - | - | 5/5 (1.00) | ok | 5x3 | - |
| json-tensorplan | en | 1202 | 25 | stop | - | - | 5/5 (1.00) | fragment | 5x3 | - |
| json-tensorplan | en | 1203 | 25 | stop | - | - | 5/5 (1.00) | ok | 5x3 | - |
| json-schritte | de | 1201 | 34 | stop | - | - | - | fragment | 1x1 | - |
| json-schritte | de | 1202 | 64 | stop | - | - | - | ok | 1x1 | - |
| json-schritte | de | 1203 | 85 | stop | - | - | - | ok | 1x1 | - |
| de-agent-umzug | de | 1201 | 398 | stop | 25.13 | 10/398 | - | - | 2x1 | 0 |
| de-agent-umzug | de | 1202 | 263 | stop | 15.21 | 4/263 | - | - | 2x1 | 2 |
| de-agent-umzug | de | 1203 | 314 | stop | 44.59 | 14/314 | - | - | 1x1 | 0 |
| de-agent-fehlersuche | de | 1201 | 193 | stop | 41.45 | 8/193 | - | - | 1x1 | 0 |
| de-agent-fehlersuche | de | 1202 | 137 | stop | 29.20 | 4/137 | - | - | 1x1 | 0 |
| de-agent-fehlersuche | de | 1203 | 261 | stop | 57.47 | 15/261 | - | - | 2x1 | 0 |

## flagged words, with context

| lang | word | note | prompt | seed | context |
|---|---|---|---|---|---|
| de | `Abnahmeprotokk` |  | de-agent-umzug | 1203 | fung ist erfolgreich, wenn ein schriftliches Abnahmeprotokk durch den Facility Manager des neuen Gebäude |
| de | `Adernungen` |  | de-prose-plakat | 1201 | en Verläufe der Emailmalerei oder die feinen Adernungen des Holzes überlagern und zerstören.  Schlie |
| de | `Anwendungs` |  | de-agent-fehlersuche | 1203 | e oder dem Versionsverlauf.  6. Suche in den Anwendungs- oder Debug-Logs nach Mustern, die auf eine  |
| de | `Anwendungsebenebene` |  | de-agent-fehlersuche | 1201 | Anfragen bestätigt ein Problem unterhalb der Anwendungsebenebene, während explizit geloggte Timeouts oder Exc |
| de | `Aquiferen` |  | de-prose-wasserwerk | 1202 | Jahren können die Wasserführungswerte in den Aquiferen anwachsen, während trockene Sommerphasen zu  |
| de | `Ausb` |  | de-prose-wasserwerk | 1201 | erdende, korrodierte Innenwand und die durch Ausb die Querschnittsreduktion der Rohre stark zu |
| de | `Auslagenschaufenstern` |  | de-prose-plakat | 1202 | chen Raum, an Litfaßäulen, Aushängen oder in Auslagenschaufenstern betrachtet, und muss zunächst als ein zusamm |
| de | `Bandwidth` | loanword | de-prose-speicher | 1201 | rzuheben sind die On-Die-Caches und die High-Bandwidth Memory Stacks, die in den Chipverbund integr |
| de | `Baststoff` |  | de-prose-buchbinderei | 1201 | rielle Eignung voraussetzte: der sogenannte „Baststoff-Leinwand-Bindes“ oder gar der direkt auf den |
| de | `Batching` | loanword | de-prose-speicher | 1201 | ser Umstand führt zu einem Phänomen, das als Batching oder Chunking bezeichnet wird. Anstatt zu ve |
| de | `Beschleunigerchip` |  | de-prose-speicher | 1202 |  Form von GDDR-Modulen, und dem eigentlichen Beschleunigerchip die entscheidende Hürde dar, welche die Leis |
| de | `Beschleunigerhaltung` |  | de-prose-speicher | 1201 | o der Datenmenge, die im aktiven Zustand der Beschleunigerhaltung bleiben, ihre entscheidende Rolle. In vielen |
| de | `Betracheraufmerksamkeit` |  | lit-farbpalette | 1203 | hl der Farbpalette die primäre Steuerung der Betracheraufmerksamkeit und der hierarchischen Struktur bestimmt. In |
| de | `Betrachterlebnis` |  | lit-farbpalette | 1202 | che und funktionale Rolle übernimmt, die das Betrachterlebnis der Besucher steuern soll.  Der Hintergrund  |
| de | `Betrachtes` |  | de-prose-plakat | 1201 | Kraft der Leerfläche nutzen, um dem Auge des Betrachtes Ruhe und Orientierung zu bieten. Zu dicht ge |
| de | `Bindes` |  | de-prose-buchbinderei | 1201 | ussetzte: der sogenannte „Baststoff-Leinwand-Bindes“ oder gar der direkt auf den Block geklebte  |
| de | `Blackwell` | loanword | de-prose-speicher | 1202 | ekturen, wie sie etwa in der NVIDIA Ada oder Blackwell Serie zu finden sind, kann der Zugriff auf d |
| de | `Brown` | loanword | de-prose-buchbinderei | 1202 | sierter Bindemaschinen, wie der von David H. Brown konzipierten automatischen Binde- und Heftsy |
| de | `Buchbindergesellenvolkes` |  | de-prose-buchbinderei | 1201 | en.  Die Auswirkungen auf die Ausbildung des Buchbindergesellenvolkes waren tiefgreifend und widersprüchlich. Bis  |
| de | `Buckram` | loanword | de-prose-buchbinderei | 1203 | rgrund. Der sogenannte Verlagsband, oft als „Buckram-Bindung“ bezeichnet, wurde zum Normfall. Hie |
| de | `Buffers` | loanword | de-prose-speicher | 1202 | radig strukturiert sind. Texturdaten, Vertex-Buffers oder Feature-Karten in Inferenz-Modellen wer |
| de | `Cause` | loanword | de-agent-fehlersuche | 1203 | ierten Beobachtungen eine Hypothese zur Root-Cause und dokumentiere diese, um den Fix gezielt i |
| de | `Chain` | loanword | de-agent-umzug | 1201 | de-Tagging versehen sind und ein lückenloses Chain-of-Cust-Logistik-Dokument für die gesamte Pa |
| de | `Chunking` | loanword | de-prose-speicher | 1201 | hrt zu einem Phänomen, das als Batching oder Chunking bezeichnet wird. Anstatt zu versuchen, eine  |
| de | `Coalescing` | loanword | de-prose-speicher | 1202 | e komplexe Pipeline der Speichermanager, die Coalescing-Prozesse und die physische Strecke der PCB-L |
| de | `Compute` | loanword | de-prose-speicher | 1203 | n sogenannten Streaming Multiprocessors oder Compute Units integriert sind. Diese lokalen Caches  |
| de | `Computing` | loanword | de-prose-speicher | 1201 | von Anwendungen für hochleistungsgerechnetes Computing. |
| de | `Cust` |  | de-agent-umzug | 1201 | g versehen sind und ein lückenloses Chain-of-Cust-Logistik-Dokument für die gesamte Partie vor |
| de | `Daseaseins` |  | de-prose-wasserwerk | 1202 | dern bildet das zentrale Element städtischer Daseaseins. Die Komplexität dieses Systems ergibt sich  |
| de | `Datemitem` |  | de-prose-speicher | 1203 | ischen dem Absenden einer Anfrage nach einem Datemitem und dem Eintreffen der gewünschten Informati |
| de | `Deadlock` | loanword | de-agent-fehlersuche | 1203 | auf eine blockierende Ressourcensperre, eine Deadlock-Situation oder eine fehlgeschlagene Abhängig |
| de | `Debug` | loanword | de-agent-fehlersuche | 1203 | nsverlauf.  6. Suche in den Anwendungs- oder Debug-Logs nach Mustern, die auf eine blockierende |
| de | `Deep` | loanword | de-prose-speicher | 1201 | le. In vielen Workloads, insbesondere in der Deep Learning Inference oder in Simulationsszenen |
| de | `Dekore` |  | de-prose-buchbinderei | 1202 | e Auslagerung der aufwendigen handwerklichen Dekore aus dem Produktionsprozess in den Bereich de |
| de | `Druckereigewerbe` |  | de-prose-buchbinderei | 1201 | ei die entscheidenden Impulse jedoch aus dem Druckereigewerbe zu kommen schienen, denn die Verlage, um die |
| de | `Druckerhöher` |  | de-prose-wasserwerk | 1203 | erlaufbecken oder moderne, frequenzgeregelte Druckerhöher stabilisiert wird. Um die statischen Drücke  |
| de | `Druckfarbenwahl` |  | de-prose-plakat | 1201 |  Pigmenten und Materialien besteht, muss die Druckfarbenwahl eine doppelte Aufgabe bewältigen: Sie muss d |
| de | `Durchflußkapazität` |  | de-prose-wasserwerk | 1203 | , sogenannten Verkrustungen, führen, die die Durchflußkapazität der Röhren systematisch mindern. Die Filterg |
| de | `Einbandung` |  | de-prose-buchbinderei | 1201 | röße herzustellen, auf denen die maschinelle Einbandung dann in kurzer Taktzeit erfolgen konnte. Par |
| de | `Einlagerungsnummer` |  | de-agent-umzug | 1203 | nahme durch den zuständigen Archivar und die Einlagerungsnummer im neuen System bestätigt wurde.  7. Überprü |
| de | `Einlagerungszone` |  | de-agent-umzug | 1201 | ystematisch und bringe sie in die definierte Einlagerungszone des neuen Gebäudes. Der Schritt ist abgeschl |
| de | `End` | loanword | de-prose-speicher | 1202 | tte betrachtet werden muss. In modernen High-End-GPU-Architekturen, wie sie etwa in der NVIDI |
| de | `Erhart` |  | de-prose-plakat | 1203 | , wie etwa eine ausgewählte Bronze von Georg Erhart oder ein charakteristisches Gefäß aus der Ma |
| de | `Exceptions` | loanword | de-agent-fehlersuche | 1201 | ene, während explizit geloggte Timeouts oder Exceptions auf einen internen Bug hinweisen.  2. Unters |
| de | `Execution` | loanword | de-prose-speicher | 1203 | gen. Diese Technik, bekannt als Out-of-Order Execution oder multithreading innerhalb der Grafikproz |
| de | `FLOPs` |  | de-prose-speicher | 1202 | wohl die Shader-Kerne theoretisch noch viele FLOPs pro Sekunde berechnen könnten.  Die Ingenieu |
| de | `Facility` | loanword | de-agent-umzug | 1203 | n ein schriftliches Abnahmeprotokk durch den Facility Manager des neuen Gebäudes vorliegt.  3. Ste |
| de | `Fadennähmaschine` |  | de-prose-buchbinderei | 1202 | eschwindigkeit bestimmte. Die Einführung der Fadennähmaschine durch die Familie Singer und die parallele E |
| de | `Fadenzufuhr` |  | de-prose-buchbinderei | 1202 | Logik der Metallnocken und der automatischen Fadenzufuhr ausgelagert wurde. Dieser Übergang, der sich |
| de | `Farb` |  | de-prose-wasserwerk | 1202 | genau lokalisieren. Auch die Beobachtung der Farb- und Trübungswerte im Netz, welche durch die |
| de | `Faults` | loanword | de-agent-fehlersuche | 1203 | hlermeldungen wie Timeouts oder Segmentation Faults vorliegen.  2. Untersuche die Systemressourc |
| de | `Filtrierprozeß` |  | de-prose-wasserwerk | 1203 | tion zu befähden, damit sie im nachfolgenden Filtrierprozeß durch Sandfilter zurückbehalten werden könne |
| de | `Filtrierstufen` |  | de-prose-wasserwerk | 1202 | n und der damit einhergehenden Belastung der Filtrierstufen der Aufbereitung zutage tritt, bietet einen  |
| de | `Findmittel` |  | de-agent-umzug | 1201 | t und bestätigt wurden.  8. Aktualisiere die Findmittel und Katalogeinträge, um die neuen physischen |
| de | `Fingertippen` |  | de-prose-buchbinderei | 1202 | agerte das handwerkliche Wissen, das sich im Fingertippen und im Blick für die Materialwiderstände aus |
| de | `Firewallblockade` |  | de-agent-fehlersuche | 1201 | ler in den relevanten Ketten widerlegen eine Firewallblockade, während sprunghafte Anstiege in DROP- oder  |
| de | `Flußläufen` |  | de-prose-wasserwerk | 1203 | lächenwasserbeeinflussung in näherer Nähe zu Flußläufen. Die Gewinnung erfordert dabei eine präzise  |
| de | `Gegentheil` |  | de-prose-buchbinderei | 1202 | ium der Bildung zu schwächen, sondern es, im Gegentheil, durch die Demokratisierung des Zugangs in e |
| de | `Gerüstkörper` |  | de-prose-wasserwerk | 1202 | zurückbleibt ein kohlenstoffreicher, weicher Gerüstkörper, der zwar die Form des Rohres vorläufig bewa |
| de | `Gesellenordnung` |  | de-prose-buchbinderei | 1202 | tergabe des Wissens im Rahmen der Zunft- und Gesellenordnung, durch das Lernen am Objekt, durch die Beoba |
| de | `Gesellenschaften` |  | de-prose-buchbinderei | 1203 | e Monopolstellung zu verlieren. Zwar blieben Gesellenschaften und Innungen bis tief ins neunzehndige Jahrh |
| de | `Glatheit` |  | de-prose-buchbinderei | 1203 | schwanden zugunsten einer kalten, normierten Glatheit. Der Verlust war folglich nicht nur ein mate |
| de | `Graphikanwendungen` |  | de-prose-speicher | 1202 | t auf der Beobachtung, dass Datenzugriffe in Graphikanwendungen und maschinellen Lernprozessen oft nicht zuf |
| de | `Graphikbeschleuniger` |  | de-prose-speicher | 1202 | stem zu limitieren. In den frühen Jahren der Graphikbeschleuniger lag der Engpass tatsächlich in der Rechenges |
| de | `Graphitisierung` |  | de-prose-wasserwerk | 1202 | efüges zerstört. Hinzu kommt der Prozess der Graphitisierung, bei dem sich der eisenreiche Anteil des Gef |
| de | `Handfalzbe` |  | de-prose-buchbinderei | 1202 | terialität begriff, den Falz präzise mit dem Handfalzbe glättete, die Lagen mit einer sorgfältig aus |
| de | `Hexcode` |  | lit-farbpalette | 1201 | t grünlich anmutige Ton, definiert durch den Hexcode #D7FFE0, verhindert die optische Ermüdung, d |
| de | `Hexcodes` |  | lit-farbpalette | 1202 | Plakaten jeder einzelne Satz mit den exakten Hexcodes #D7FFE0, #050505, #1B3A5F und #FFB347 kontro |
| de | `High` | loanword | de-prose-speicher | 1201 | hervorzuheben sind die On-Die-Caches und die High-Bandwidth Memory Stacks, die in den Chipverb |
| de | `Hinführungselemente` |  | lit-farbpalette | 1201 | dem der warme Ton #FFB347 gezielt platzierte Hinführungselemente belebt, ohne den Gesamtcharakter zu zerstöre |
| de | `Hintergrgrundes` |  | lit-farbpalette | 1201 | namik, die sich in der exakten Anwendung des Hintergrgrundes #D7FFE0, der Schriftfarbe #050505, des Rahme |
| de | `Holzschliffmühle` |  | de-prose-buchbinderei | 1203 | herstellungsprozesses. Mit der Erfindung der Holzschliffmühle und der späteren Zellstoffherstellung senkte |
| de | `Indikatorkeime` |  | de-prose-wasserwerk | 1202 | t, wobei besonders die Spurenbildner und die Indikatorkeime für fäkale Belastungen eine strengere Überwa |
| de | `Inference` | loanword | de-prose-speicher | 1201 | Workloads, insbesondere in der Deep Learning Inference oder in Simulationsszenen, lassen sich die A |
| de | `Informationstransportierung` |  | de-prose-plakat | 1202 | schen Konzept dient. Im Gegensatz zur reinen Informationstransportierung, wie sie in digitalen Medien oder zeitgenöss |
| de | `Instananzen` |  | de-prose-buchbinderei | 1202 | hrer neuen Rolle als industriell-ökonomische Instananzen, drängten auf die Vereinheitlichung des Einb |
| de | `Instandhaltungregimes` |  | de-prose-wasserwerk | 1203 |  hydraulischer Gesetze sowie eines rigorosen Instandhaltungregimes, um die öffentliche Gesundheit zu schützen u |
| de | `Interconnect` | loanword | de-prose-speicher | 1202 | st, während die physischen Limitierungen der Interconnect-Strukturen und der externen Speicherzugriffe |
| de | `Kalksteinhülle` |  | de-prose-wasserwerk | 1203 | r so genannten "Rostbläschen", die unter der Kalksteinhülle am Eisenrohr selbst entstehen.  Das Bemerken |
| de | `Kaschierpapier` |  | de-prose-buchbinderei | 1203 | at zugunsten von gepressten Pappen, billigem Kaschierpapier und industriell verarbeiteten Geweben in den |
| de | `Klassizismen` |  | de-prose-plakat | 1201 | Pedanterie zu verfallen. Eine allzu strengen Klassizismen der Bleisatz-Schriftarten könnte die Plakatk |
| de | `Kleinnbuchstaben` |  | de-prose-plakat | 1201 |  und eine klare Unterscheidung der Groß- und Kleinnbuchstaben eine hohe Lesbarkeit auf Distanz sichern. Di |
| de | `Klimadatenlogger` |  | de-agent-umzug | 1201 | m. Dieser Schritt gilt als beendet, wenn die Klimadatenlogger im neuen Gebäude aktiv sind und die Sollwert |
| de | `Kohärenzprotokollen` |  | de-prose-speicher | 1201 | einen komplexen Ablauf aus Adressumwandlung, Kohärenzprotokollen und eventuell Prefetch-Mechanismen. In der P |
| de | `Learned` | loanword | de-agent-umzug | 1203 | icht, der Abweichungen, Probleme und Lessons Learned dokumentiert. Der Prozess ist beendet, wenn  |
| de | `Learning` | loanword | de-prose-speicher | 1201 | n vielen Workloads, insbesondere in der Deep Learning Inference oder in Simulationsszenen, lassen  |
| de | `Leckageraten` |  | de-prose-wasserwerk | 1203 | ser zu protokollieren – sowie am Anstieg der Leckageraten und der Druckverluste über den Tag- und Nach |
| de | `Leckströmkomponente` |  | de-prose-wasserwerk | 1202 |  Energie einsparen, indem sie die sogenannte Leckströmkomponente minimieren, die bei starren Pumpsystemen unv |
| de | `Lesebarkeit` |  | de-prose-plakat | 1201 | n Entscheidungen in der Frage der zweifachen Lesebarkeit, die jedes Plakat im öffentlichen Raum bewäl |
| de | `Lessons` | loanword | de-agent-umzug | 1203 | hlussbericht, der Abweichungen, Probleme und Lessons Learned dokumentiert. Der Prozess ist beende |
| de | `Letzlich` |  | de-prose-plakat | 1202 |  Auge für die Rasterung der Halbtonflächen.  Letzlich ist die Erfolgskriterien eines solchen Plaka |
| de | `Litfassade` |  | de-prose-plakat | 1203 | s der Entfernung, wo das Plakat oft an einer Litfassade oder an einer Bahnsteigung wahrgenommen wird |
| de | `Litfaßäulen` |  | de-prose-plakat | 1202 | en? Das Plakat wird im öffentlichen Raum, an Litfaßäulen, Aushängen oder in Auslagenschaufenstern bet |
| de | `Logeinträge` |  | de-agent-fehlersuche | 1203 | 1. Prüfe den Status und die die letzten Logeinträge des betroffenen Dienstes mit `systemctl stat |
| de | `Logistikketten` |  | de-agent-umzug | 1203 | klare Transportzeitfenster und routebasierte Logistikketten, um Unterbrechungen im Archivbetrieb zu mini |
| de | `Logs` | loanword | de-agent-fehlersuche | 1202 | ürzlich abgestürzt ist oder Neustarts in den Logs vermerkt sind. Diese Beobachtung bestätigt o |
| de | `Manganentfernung` |  | de-prose-wasserwerk | 1201 | t der Fokus auf der Belüftung zur Eisen- und Manganentfernung. Durch das Einbringen von Luft wird das gelö |
| de | `Matschigkeit` |  | de-prose-plakat | 1202 | bei der Annäherung nicht zu einer unscharfen Matschigkeit verläuft, die den Betrachter enttäuscht. Die |
| de | `Misses` | loanword | de-prose-speicher | 1203 | s die hohen Datenströme nicht zu übermäßigen Misses in den Caches führen.  Moderne Grafikprozess |
| de | `Multiprocessors` | loanword | de-prose-speicher | 1203 | lle Caches, die in den sogenannten Streaming Multiprocessors oder Compute Units integriert sind. Diese lo |
| de | `Nachbarnicht` |  | lit-farbpalette | 1201 | dem Hintergrund #D7FFE0 darf durch zu dunkle Nachbarnicht verzerrt werden, wobei der Akzent #FFB347 al |
| de | `Nachfragekosten` |  | de-prose-buchbinderei | 1201 | hienen, denn die Verlage, um die gestiegenen Nachfragekosten zu decken und die Auflagen zu erweitern, suc |
| de | `Nahtfadens` |  | de-prose-buchbinderei | 1203 | abe von Wissen über die richtige Führung des Nahtfadens oder die Anwendung von Werkzeugen regelte, b |
| de | `Normfall` |  | de-prose-buchbinderei | 1203 |  als „Buckram-Bindung“ bezeichnet, wurde zum Normfall. Hierin zeigt sich die wachsende Trennung zw |
| de | `On` | loanword | de-prose-speicher | 1201 | en ergänzt. Besonders hervorzuheben sind die On-Die-Caches und die High-Bandwidth Memory Sta |
| de | `Pigging` | loanword | de-prose-wasserwerk | 1203 |  Kalkstein – gemessen durch den sogenannten "Pigging"-Test, bei dem eine Kugel durch eine isolier |
| de | `Plakatdesign` |  | de-prose-plakat | 1203 | steuert. Der weiße oder neutrale Raum ist im Plakatdesign keineswegs als Unvollständigkeit oder als Ma |
| de | `Plakatkartei` |  | de-prose-plakat | 1201 | izismen der Bleisatz-Schriftarten könnte die Plakatkartei zu schwerfällig und leblos wirken lassen, wä |
| de | `Prefetch` |  | de-prose-speicher | 1201 | mwandlung, Kohärenzprotokollen und eventuell Prefetch-Mechanismen. In der Praxis zeigt sich, dass  |
| de | `Pressenarbeiten` |  | de-prose-buchbinderei | 1201 | esten, etwa bei Sonderausgaben, bibliophilen Pressenarbeiten oder repräsentativen Werken, einen letzten H |
| de | `Primärnetz` |  | de-prose-wasserwerk | 1203 | ngungen. Es glied sich typischerweise in ein Primärnetz, welches die Hauptpumpstationen mit den Druc |
| de | `Qualtität` |  | de-prose-plakat | 1201 | s der Lesbarkeit ist der Maßstab, an dem die Qualtität der Plakate Gestaltung im Bereich des Kunstg |
| de | `Querschnittfläche` |  | de-prose-wasserwerk | 1201 | rrosion und externe chemische Angriffe seine Querschnittfläche verliert. Diesem Alterungsprozess, dem so ge |
| de | `Querschnittsreduktion` |  | de-prose-wasserwerk | 1201 | korrodierte Innenwand und die durch Ausb die Querschnittsreduktion der Rohre stark zugenommen hat. Ein zweites  |
| de | `Random` | loanword | de-prose-speicher | 1203 | en. Der Hauptspeicher, meist als dynamischer Random Access Memory ausgeführt, bietet enorme Kapa |
| de | `Rasterisierungseinheiten` |  | de-prose-speicher | 1203 | erationen pro Sekunde oder in der Anzahl der Rasterisierungseinheiten. Diese Metrik, die sich an der Fähigkeit ori |
| de | `Redundanzsystem` |  | de-prose-wasserwerk | 1202 | und fortgeschrittener Technik dient hier als Redundanzsystem, das die Versorgungssicherheit bei Schwankun |
| de | `Regalkapazitäten` |  | de-agent-umzug | 1201 | drissplan des Systems konsistent ist und die Regalkapazitäten visuell kontrolliert und bestätigt wurden.   |
| de | `Request` | loanword | de-agent-fehlersuche | 1203 | e`, um detailliertere Informationen über den Request-Flow und die interne Verarbeitung der Anfrag |
| de | `Respektiertheit` |  | de-prose-plakat | 1201 | eine visuelle Synthese zwischen historischer Respektiertheit und zeitgenössischer Wirksamkeit erfordert.  |
| de | `Rezeptionsverhalten` |  | de-prose-plakat | 1201 | ektuelle Disposition für das gesamte spätere Rezeptionsverhalten fest. Bei der Bildauswahl für das Kunstgewer |
| de | `Root` | loanword | de-agent-fehlersuche | 1203 | kumulierten Beobachtungen eine Hypothese zur Root-Cause und dokumentiere diese, um den Fix gez |
| de | `Rundbogenfalzern` |  | de-prose-buchbinderei | 1203 |  von Heftmaschinen und für die Bedienung von Rundbogenfalzern. Die Ausbildungsordnungen, die sich in den s |
| de | `Räumle` |  | de-agent-umzug | 1203 | e bei der Migration festgestellt wurden.  9. Räumle das alte Magazin und entsorge oder recycle n |
| de | `Scan` | loanword | de-agent-umzug | 1202 | len Einheiten versiegelt und mit eindeutigen Scan-Codes gekennzeichnet sind.  4. Führe den phy |
| de | `Serifensetzung` |  | de-prose-plakat | 1201 | egt oft in einer Nuance, die die historische Serifensetzung in einen modernen, luftigen Satzgewand überf |
| de | `Shader` |  | de-prose-speicher | 1202 | ächlich in der Rechengeschwindigkeit der der Shader-Kerne selbst, doch mit der dramatischen Skal |
| de | `Siebverfahren` |  | de-prose-wasserwerk | 1203 | rung liegt. Hier werden typischerweise grobe Siebverfahren zur Beseitigung von organischen Trübstoffen  |
| de | `Sollbereich` |  | de-agent-umzug | 1202 | chlossen, wenn die Sensoren stabile Werte im Sollbereich melden.  10. Dokumentiere die Umzüge in eine |
| de | `Spezifikationsblättern` |  | de-prose-speicher | 1203 | ine Leistungszahl der Recheneinheiten in den Spezifikationsblättern der Hersteller nur ein Teil der Wahrheit ist |
| de | `Staging` | loanword | de-agent-fehlersuche | 1203 | umentiere diese, um den Fix gezielt in einer Staging-Umgebung verifizieren zu können. |
| de | `Strukturprincip` |  | de-prose-plakat | 1202 |  daher kein Nebensache, sondern das tragende Strukturprincip. Die Schrift darf die Bildfläche nicht als s |
| de | `Substratmaterialien` |  | lit-farbpalette | 1203 | 1B3A5F und #FFB347 auf den unterschiedlichen Substratmaterialien der Serie v3.2.1 reproduzierbar bleiben. Es  |
| de | `Swap` | loanword | de-agent-fehlersuche | 1201 | ozess durch fehlenden Arbeitsspeicher, volle Swap-Partition oder CPU-Sättigung blockiert wird. |
| de | `Tagging` | loanword | de-agent-umzug | 1201 | ich, wenn alle Kisten mit eindeutigenBarcode-Tagging versehen sind und ein lückenloses Chain-of-C |
| de | `Timeout` | loanword | de-agent-fehlersuche | 1203 | f Limits für gleichzeitige Verbindungen oder Timeout-Einstellungen, durch Vergleich mit einer fun |
| de | `Timeouts` | loanword | de-agent-fehlersuche | 1201 | wendungsebenebene, während explizit geloggte Timeouts oder Exceptions auf einen internen Bug hinwe |
| de | `Titeline` |  | lit-farbpalette | 1201 | wichtigsten call-to-action-Bereiche oder die Titeline zieht, indem der warme Ton #FFB347 gezielt p |
| de | `Tracerstoffen` |  | de-prose-wasserwerk | 1203 | kustische Korrelation oder die Injektion von Tracerstoffen in die Leitungen, nicht mehr an isolierten,  |
| de | `Traffics` | loanword | de-agent-fehlersuche | 1201 | Warteschlangen ohne parallele Steigerung des Traffics bestätigt einen Fehler in der Freigabe von D |
| de | `Transaktiosabschlüsse` |  | de-agent-fehlersuche | 1202 | n, auf blockierte Verbindungen oder fehlende Transaktiosabschlüsse. Diese Beobachtungen klären, ob die Ursache  |
| de | `Transförmung` |  | de-prose-buchbinderei | 1203 | erlust und dem Gewinn, die jede industrielle Transförmung aufwirft. Was an handwerklicher Qualität gin |
| de | `Transportk` |  | de-agent-umzug | 1203 | e Verpackung, Zwischenlagerung und spezielle Transportk vor. Der Prozess ist abgeschlossen, wenn die |
| de | `Trübungswerte` |  | de-prose-wasserwerk | 1202 | lisieren. Auch die Beobachtung der Farb- und Trübungswerte im Netz, welche durch die Absonderung von Ro |
| de | `Vasenobjekt` |  | de-prose-plakat | 1202 | ine filigrane Silberkanne oder ein gläsernes Vasenobjekt, im Großformat darzustellen. Diese Vergrößer |
| de | `Verpake` |  | de-agent-umzug | 1202 | systeme auf Belastbarkeit getestet sind.  3. Verpake die empfindlichen und digitalisierten Materi |
| de | `Verschleißtendenzen` |  | de-prose-wasserwerk | 1202 | opie, chemische Reaktionen und physikalische Verschleißtendenzen. Die Komplexität des Systems, von der geolog |
| de | `Warehouse` | loanword | de-agent-umzug | 1201 | schlossen, wenn alle physischen Einheiten im Warehouse-Management-System (WMS) als „im Magazin eing |
| de | `Werkens` |  | de-prose-buchbinderei | 1201 | nn durchformtes Ding, in dem der Prozess des Werkens in der Materialität noch ablesbar war. Der E |
| de | `Werkers` |  | de-prose-buchbinderei | 1201 | er Einbande die individuelle Handschrift des Werkers verloren, die in der handwerklichen Epoche j |
| de | `Working` | loanword | de-prose-speicher | 1202 |  die Größe der Arbeitsmenge, der sogenannten Working Set, eine zentrale Rolle, die oft unterschät |
| de | `Zeite` |  | de-prose-speicher | 1203 |  Die Bandbreite, also der Datendurchsatz pro Zeite unit, der über die den Systembus und die int |
| de | `Zerrütung` |  | de-prose-wasserwerk | 1202 | n Leckagen und damit auf die fortschreitende Zerrütung der Rohrsubstanz hin. Moderne Methoden der N |
| de | `Zufalligkeit` |  | de-prose-buchbinderei | 1201 | rkstoffwahl erreicht, das der handwerklichen Zufalligkeit überlegen war, wo es um den reinen Zweck der |
| de | `action` | loanword | lit-farbpalette | 1201 |  Publikum direkt auf die wichtigsten call-to-action-Bereiche oder die Titeline zieht, indem der  |
| de | `altingesessenen` |  | de-prose-buchbinderei | 1203 | e die pädagogischen Wege zur Erlernung eines altingesessenen Gewerbes fundamental transformierte. Zu Begi |
| de | `anglo` |  | de-prose-buchbinderei | 1203 | Häuser in Leipzig, Berlin und später auch im anglo-amerikanischen Raum, begannen, den Einband n |
| de | `archivischen` |  | de-agent-umzug | 1203 | digitale und gedruckte Inventarliste von der archivischen Fachabteilung freigegeben wurde.  2. Identif |
| de | `bikochrome` |  | de-prose-plakat | 1202 | ch die Beschränkung auf eine monochrome oder bikochrome Palette, die mit dem Druckverfahren korrespo |
| de | `biozidfreie` |  | de-prose-wasserwerk | 1201 |  die Qualität des Endproduktes erhält. Diese biozidfreie Aufbereitung stellt sicher, dass der Restchl |
| de | `bleienden` |  | de-prose-wasserwerk | 1201 | rung eingesetzte Blei, wobei bei alten, noch bleienden Leitungen aus der Zeit vor der Verbotsverord |
| de | `call` | loanword | lit-farbpalette | 1201 |  der das Publikum direkt auf die wichtigsten call-to-action-Bereiche oder die Titeline zieht,  |
| de | `detailierter` |  | de-prose-plakat | 1201 | Abwägung zwischen repräsentativer Breite und detailierter Schärfe geboten. Es gilt, jenes einzelne Obj |
| de | `dmesg` |  | de-agent-fehlersuche | 1202 | shistorie und Systemmeldungen mit dem Befehl dmesg sowie dem Status des Dienstes via systemctl, |
| de | `durchformtes` |  | de-prose-buchbinderei | 1201 | rkeit des Buches als ein durch Hand und Sinn durchformtes Ding, in dem der Prozess des Werkens in der  |
| de | `eindeutigenBarcode` |  | de-agent-umzug | 1201 | eitung ist erfolgreich, wenn alle Kisten mit eindeutigenBarcode-Tagging versehen sind und ein lückenloses Ch |
| de | `energierlicher` |  | lit-farbpalette | 1201 | dient der Akzent in #FFB347, der als warmer, energierlicher Lichtpunkt fungiert, der das Publikum direkt |
| de | `entindividualisiert` |  | de-prose-buchbinderei | 1203 | kompetenz. Der Lernweg wurde standardisiert, entindividualisiert und immer stärker an die Bedürfnisse der Fab |
| de | `equally` | loanword | de-prose-plakat | 1202 |  der Typografie, welche als sekundäres, aber equally gewichtiges Element die visuelle Hierarchie  |
| de | `formalelle` |  | de-agent-umzug | 1203 | der Leitung zur Kenntnisnahme vorlag und die formalelle Abnahme des Umzugs erfolgte. |
| de | `geloggte` |  | de-agent-fehlersuche | 1201 | lb der Anwendungsebenebene, während explizit geloggte Timeouts oder Exceptions auf einen internen  |
| de | `geometrischere` |  | de-prose-plakat | 1203 | erkbunds zuzuordnen sind, kann eine klarere, geometrischere Grotesk, die den handwerklichen Geist der Ze |
| de | `hängengehalten` |  | de-agent-fehlersuche | 1201 | orfen oder in einer Tabelle mit voller Größe hängengehalten werden. Widerspruchslose Paketzähler in den  |
| de | `ingenieurstechnischen` |  | de-prose-wasserwerk | 1201 | inzugsgebiet umfassen, eine der komplexesten ingenieurstechnischen Herausforderungen im kommunalen Bereich dar. |
| de | `konsentiert` |  | de-agent-umzug | 1203 | aufplan mit allen beteiligten Dienstleistern konsentiert wurde.  5. Bereite die besonders empfindlich |
| de | `konsistiert` |  | de-agent-umzug | 1202 | ald alle neuen Lagerorte im Datenbank-System konsistiert und synchronisiert sind.  9. Prüfe und aktiv |
| de | `kostenspieligen` |  | de-prose-buchbinderei | 1201 | , suchten nach Methoden, die den Einband als kostenspieligen Faktor eliminieren konnten. Der entscheidend |
| de | `kuratorischen` |  | de-prose-plakat | 1202 | ondern als erste sinnliche Begegnung mit dem kuratorischen Konzept dient. Im Gegensatz zur reinen Infor |
| de | `kuratorischer` |  | de-prose-plakat | 1203 | ivs ist keine ästhetische Laune, sondern ein kuratorischer Akt, der die gesamte Ausstellung auf ein ein |
| de | `läßt` |  | de-prose-plakat | 1202 | ekts wahrt, indem es dem Betrachter die Zeit läßt, die zur kontemplativen Wahrnehmung nöthig i |
| de | `neoklassizischen` |  | de-prose-plakat | 1203 | hnittstelle bewegen zwischen der Strenge der neoklassizischen Tradition, die der Epoche inhärent ist, und  |
| de | `nöthig` |  | de-prose-plakat | 1202 | eit läßt, die zur kontemplativen Wahrnehmung nöthig ist, statt die Information zu erzwingen. Die |
| de | `of` | loanword | de-agent-umzug | 1201 | ging versehen sind und ein lückenloses Chain-of-Cust-Logistik-Dokument für die gesamte Parti |
| de | `pastellgrünes` |  | lit-farbpalette | 1202 | urch den Ton #D7FFE0 definiert, ein sanftes, pastellgrünes Tinten, das nicht als bloße Fläche fungiert, |
| de | `prozessuellen` |  | de-prose-buchbinderei | 1202 | s, was wir heute als Buchform bezeichnen, im prozessuellen Vollzug der Arbeitshand erst zum Sein bracht |
| de | `rauerwerdende` |  | de-prose-wasserwerk | 1201 | sen, dass die hydraulische Reibung durch die rauerwerdende, korrodierte Innenwand und die durch Ausb di |
| de | `reduzierbaren` |  | de-prose-buchbinderei | 1202 | konstitutiven Kern, den unmittelbaren, nicht-reduzierbaren Zugriff auf das Material, und erhielt im Geg |
| de | `systemctl` |  | de-agent-fehlersuche | 1202 | fehl dmesg sowie dem Status des Dienstes via systemctl, um festzustellen, ob der Dienst kürzlich ab |
| de | `to` | loanword | lit-farbpalette | 1201 | das Publikum direkt auf die wichtigsten call-to-action-Bereiche oder die Titeline zieht, ind |
| de | `unabschließbaren` |  | de-prose-buchbinderei | 1202 | -praktisches Problem, noch in einer offenen, unabschließbaren, dialektischen Spannung, zur handwerklichen  |
| de | `untergestützten` |  | de-prose-wasserwerk | 1201 | betreiber mit automatischen Druckreglern und untergestützten Pumpwerken, die die hydraulischen Widerstand |
| de | `vernachläßigt` |  | de-prose-plakat | 1203 | nkert wird. Ein Plakat, das diese Schichtung vernachläßigt, bleibt entweder unlesbar aus der Ferne oder |
| de | `vertrauenserweckenden` |  | lit-farbpalette | 1201 |  soll.  Der Rahmen, umgesetzt in dem tiefen, vertrauenserweckenden Blau #1B3A5F, dient nicht nur als dekorative |
| de | `überbeansicht` |  | de-prose-wasserwerk | 1201 |  sonst durch zu hohe Lasten an Schwebstoffen überbeansicht werden würde.  Die eigentliche Aufbereitung  |
| en | `Aspergillus` |  | en-prose-archive | 1202 | wever, are molds, particularly species like *Aspergillus* and *Fusarium*. These fungi thrive in damp, |
| en | `Fusarium` |  | en-prose-archive | 1202 | particularly species like *Aspergillus* and *Fusarium*. These fungi thrive in damp, stagnant air,  |
| en | `Overpacked` |  | en-prose-archive | 1201 | t allow for proper breathing within the box. Overpacked boxes trap heat and moisture, creating ideal |
| en | `aa` |  | en-prose-foundry | 1201 |  through the heavy timber doors was to cross aa threshold into a world where physics was not |
| en | `acidification` |  | en-prose-archive | 1201 | collection in a matter of days. More subtly, acidification is the silent killer of modern paper records |
| en | `amidst` |  | en-prose-foundry | 1203 | nd water. The foreman, a figure of authority amidst the swirling dust, oversaw the ramming of th |
| en | `changelog` |  | lit-releasenote | 1202 | 09-18a)  This document serves as the primary changelog and operational summary for the CNQ4.5-M mod |
| en | `checksums` |  | lit-releasenote | 1202 | ting these weights should verify their local checksums against the artifacts described herein befor |
| en | `cockling` |  | en-prose-archive | 1202 | act, causing mechanical stress that leads to cockling, cracking, and eventual disintegration. Ther |
| en | `convolutional` |  | lit-releasenote | 1201 |  that our refactoring efforts to isolate the convolutional alpha parameters from the general weight str |
| en | `councillors` |  | en-prose-archive | 1201 |  preserved, while the draft circulated among councillors for comment is discarded, recognizing that t |
| en | `curation` |  | en-prose-archive | 1201 | . A two-hundred-year horizon requires strict curation. Municipal records are voluminous, often con |
| en | `deacidification` |  | en-prose-archive | 1202 | , this is a mass event of destruction. While deacidification technologies exist, they are costly and slow |
| en | `entropic` |  | en-prose-archive | 1201 | ory, engineered over centuries to resist the entropic pull of time. To ensure that the ledgers, co |
| en | `evidential` |  | en-prose-archive | 1203 | ty of the archive. Archivists must weigh the evidential value of records against the costs of preser |
| en | `fireclay` |  | en-prose-foundry | 1201 |  breath. This shaft of brickwork, lined with fireclay and crowned with a hood of rusted sheet meta |
| en | `frass` |  | en-prose-archive | 1202 | e: small, irregular holes or the presence of frass, the granular waste they leave behind. Worse |
| en | `glycosidic` |  | en-prose-archive | 1203 | ow process by which hydrogen ions attack the glycosidic bonds in cellulose. Much of the paper produc |
| en | `grey` |  | en-prose-foundry | 1202 | . The laborers, their faces coated in a fine grey dust that seemed to penetrate the pores of t |
| en | `hemicellulose` |  | en-prose-archive | 1202 | le environment. Over decades, the lignin and hemicellulose in these papers break down, releasing acidic |
| en | `hygroscopic` |  | en-prose-archive | 1202 | n understanding has revealed that paper is a hygroscopic material, meaning it absorbs and releases mo |
| en | `liquidus` |  | en-prose-foundry | 1203 |  the iron reached the correct temperature, a liquidus consistency that flowed like water yet carri |
| en | `moldable` |  | en-prose-foundry | 1201 |  This sand was not merely a filler; it was a moldable memory, a substance that could be pressed in |
| en | `mould` |  | en-prose-foundry | 1201 | , building the upper and lower halves of the mould as separate entities that had to align with  |
| en | `moulder` |  | en-prose-foundry | 1201 | d held there only as long as the will of the moulder sustained it. The challenge of the sand was  |
| en | `moulders` |  | en-prose-foundry | 1201 | ar truth of of the process was revealed. The moulders, men whose forearms were roped with muscle a |
| en | `moulding` |  | en-prose-foundry | 1201 |  finally emerged into the harsh world of the moulding floor, it would not shed sand or warp in the |
| en | `moulds` |  | en-prose-foundry | 1201 | e sand between their fingers. They built the moulds in layers, packing the the sand tightly arou |
| en | `ofof` |  | en-prose-foundry | 1201 | hose eyes possessed the discerning sharpness ofof a jeweler, worked with a singular devotion t |
| en | `oversized` |  | en-prose-foundry | 1202 | it cooled, so he built the patterns slightly oversized, adding invisible allowances that would only |
| en | `paystubs` |  | en-prose-archive | 1203 | , low-value documentation, such as duplicate paystubs, routine maintenance logs, and mass-produced |
| en | `pourers` |  | en-prose-foundry | 1201 | lds, each a cavity waiting to be filled. The pourers, the most courageous members of the team, st |
| en | `pre` |  | en-prose-foundry | 1203 | on, descending slowly through the shaft, was pre-heated by the ascending column of hot gases, |
| en | `predation` |  | en-prose-archive | 1202 | lders, from the vigilance against biological predation to the intellectual rigor of the finding aid |
| en | `rammers` |  | en-prose-foundry | 1201 |  sand tightly around the patterns using hand rammers, driving home the grain until it sang with a |
| en | `sharding` |  | lit-releasenote | 1201 | iated with the second part of our three-part sharding strategy.  Central to the validation efforts |
| en | `shranked` |  | en-prose-foundry | 1202 | f molten metal. The craftsman knew that iron shranked as it cooled, so he built the patterns sligh |
| en | `sprues` |  | en-prose-foundry | 1202 |  signaled the timing. The men approached the sprues, the entry channels of the molds, and tipped |
| en | `thermocouples` |  | en-prose-foundry | 1203 | ng zone, gauged the state of the bath not by thermocouples, but by the sight of the tapping hole and th |
| en | `thetag` |  | lit-releasenote | 1203 | inst the production requirements defined for thetag release-2026-09-18a.  At the core of the str |
| en | `thethe` |  | en-prose-archive | 1203 | ough to prevent crushing under the weight of thethe stacking systems, which can be immense in a  |
| en | `tuyeres` |  | en-prose-foundry | 1202 | nto the top, while the blast of air from the tuyeres below provided the oxygen necessary for the  |
| en | `uncoated` |  | en-prose-archive | 1202 | ermanent, can be problematic if the metal is uncoated, as rust can stain and perforate the paper.  |
| en | `underpacked` |  | en-prose-archive | 1201 | ting ideal conditions for mold growth, while underpacked boxes allow documents to shift and abrade ag |
| en | `uninterruptable` |  | en-prose-archive | 1202 | loy triple-redundant HVAC systems, backed by uninterruptable power supplies, ensuring that even during a  |

## near-miss literals

| prompt | seed | demanded | seen | distance | count |
|---|---|---|---|---|---|
| lit-farbpalette | 1201 | `/srv/plakat/2026/vorlage-v3.2.1.svg` | `/srv/plakat/2026/vorlage-v3.2/1.svg` | 1 | 3 |
| lit-farbpalette | 1201 | `v3.2.1` | `v3.2/1` | 1 | 2 |
| lit-farbpalette | 1201 | `#1B3A5F` | `#1B3A5F/` | 1 | 1 |
| lit-farbpalette | 1201 | `#D7FFE0` | `#D7FFE0/` | 1 | 1 |

