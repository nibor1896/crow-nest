# quality probe - B1-llama

- date 2026-09-18, generated at repo commit `c1cf95c`, prompt set version 1
- scored at commit `2bc92f2` on 2026-09-18 (`--rescore`, from the stored texts)
- endpoint `http://127.0.0.1:8083`, engine `llama`, model `Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf`
- row temperature 1.0, top_p 0.95, top_k 20, presence_penalty 0.0, min_p 0.0, max_tokens 2600
- thinking: none (llama); seeds [1201, 1202, 1203]; 36 generations, 0 failed, 929 s wall

## the arm in one table

| metric | value |
|---|---|
| non-word rate DE per 1000 words, per generation | 16.46 (3.37 to 54.19) |
| non-word rate DE, loanwords the EN dictionary knows removed | 8.69 (3.37 to 16.35) |
| non-word rate DE, pooled over 16497 words | 14.37 |
| non-word rate EN per 1000 words, per generation | 8.56 (1.50 to 16.47) |
| non-word rate EN, loanwords the DE dictionary knows removed | 8.20 (1.50 to 16.47) |
| non-word rate EN, pooled over 9632 words | 8.72 |
| exact literals reproduced (share of literals) | 1.000 (1.000 to 1.000) |
| exact literals reproduced (share of demanded occurrences) | 1.000 (1.000 to 1.000) |
| near-miss literal kinds seen | 0 |
| JSON: whole answer a valid document / shape ok | 6 / 6 of 6 |
| distinct-word ratio | 0.546 (0.455 to 0.900) |
| longest immediate repeat run | 1.2 (0.0 to 2.0), max 2 |
| generations with foreign-script characters | 2 of 24 (5 chars) |
| words per answer | 729 (0 to 1456) |
| answers stopped at max_tokens | 0 of 36 |

## per prompt

| prompt | lang | seed | words | finish | non-word/1000 | flagged | literals | json | longest repeat | foreign |
|---|---|---|---|---|---|---|---|---|---|---|
| de-prose-plakat | de | 1201 | 1114 | stop | 8.98 | 10/1114 | - | - | 2x1 | 0 |
| de-prose-plakat | de | 1202 | 945 | stop | 8.47 | 8/945 | - | - | 2x1 | 0 |
| de-prose-plakat | de | 1203 | 1052 | stop | 9.51 | 10/1052 | - | - | 2x1 | 2 |
| de-prose-speicher | de | 1201 | 1043 | stop | 19.18 | 20/1043 | - | - | 1x1 | 0 |
| de-prose-speicher | de | 1202 | 1081 | stop | 13.88 | 15/1081 | - | - | 1x1 | 3 |
| de-prose-speicher | de | 1203 | 895 | stop | 15.64 | 14/895 | - | - | 1x1 | 0 |
| de-prose-buchbinderei | de | 1201 | 920 | stop | 9.78 | 9/920 | - | - | 2x1 | 0 |
| de-prose-buchbinderei | de | 1202 | 1176 | stop | 12.76 | 15/1176 | - | - | 2x1 | 0 |
| de-prose-buchbinderei | de | 1203 | 1086 | stop | 9.21 | 10/1086 | - | - | 1x1 | 0 |
| de-prose-wasserwerk | de | 1201 | 1132 | stop | 12.37 | 14/1132 | - | - | 2x1 | 0 |
| de-prose-wasserwerk | de | 1202 | 1110 | stop | 14.41 | 16/1110 | - | - | 1x1 | 0 |
| de-prose-wasserwerk | de | 1203 | 989 | stop | 11.12 | 11/989 | - | - | 1x1 | 0 |
| en-prose-foundry | en | 1201 | 1191 | stop | 14.27 | 17/1191 | - | - | 1x1 | 0 |
| en-prose-foundry | en | 1202 | 1275 | stop | 16.47 | 21/1275 | - | - | 1x1 | 0 |
| en-prose-foundry | en | 1203 | 1334 | stop | 11.24 | 15/1334 | - | - | 1x1 | 0 |
| en-prose-archive | en | 1201 | 1259 | stop | 3.97 | 5/1259 | - | - | 1x1 | 0 |
| en-prose-archive | en | 1202 | 1456 | stop | 3.43 | 5/1456 | - | - | 1x1 | 0 |
| en-prose-archive | en | 1203 | 1234 | stop | 6.48 | 8/1234 | - | - | 1x1 | 0 |
| lit-farbpalette | de | 1201 | 672 | stop | 10.42 | 7/672 | 7/7 (1.00) | - | 2x1 | - |
| lit-farbpalette | de | 1202 | 595 | stop | 5.04 | 3/595 | 7/7 (1.00) | - | 1x1 | - |
| lit-farbpalette | de | 1203 | 619 | stop | 9.69 | 6/619 | 7/7 (1.00) | - | 2x1 | - |
| lit-releasenote | en | 1201 | 600 | stop | 8.33 | 5/600 | 5/5 (1.00) | - | 1x1 | - |
| lit-releasenote | en | 1202 | 666 | stop | 1.50 | 1/666 | 5/5 (1.00) | - | 2x1 | - |
| lit-releasenote | en | 1203 | 617 | stop | 11.35 | 7/617 | 5/5 (1.00) | - | 1x1 | - |
| json-tensorplan | en | 1201 | 0 | stop | - | - | 5/5 (1.00) | ok | 0x0 | - |
| json-tensorplan | en | 1202 | 0 | stop | - | - | 5/5 (1.00) | ok | 0x0 | - |
| json-tensorplan | en | 1203 | 0 | stop | - | - | 5/5 (1.00) | ok | 0x0 | - |
| json-schritte | de | 1201 | 58 | stop | - | - | - | ok | 1x1 | - |
| json-schritte | de | 1202 | 62 | stop | - | - | - | ok | 1x1 | - |
| json-schritte | de | 1203 | 10 | stop | - | - | - | ok | 1x1 | - |
| de-agent-umzug | de | 1201 | 297 | stop | 3.37 | 1/297 | - | - | 1x1 | 0 |
| de-agent-umzug | de | 1202 | 343 | stop | 8.75 | 3/343 | - | - | 1x1 | 0 |
| de-agent-umzug | de | 1203 | 217 | stop | 4.61 | 1/217 | - | - | 1x1 | 0 |
| de-agent-fehlersuche | de | 1201 | 406 | stop | 54.19 | 22/406 | - | - | 1x1 | 0 |
| de-agent-fehlersuche | de | 1202 | 367 | stop | 51.77 | 19/367 | - | - | 1x1 | 0 |
| de-agent-fehlersuche | de | 1203 | 438 | stop | 52.51 | 23/438 | - | - | 1x1 | 0 |

## flagged words, with context

| lang | word | note | prompt | seed | context |
|---|---|---|---|---|---|
| de | `Actions` | loanword | lit-farbpalette | 1203 | t die Aufmerksamkeit auf spezifische Call-to-Actions oder wichtige Datenpunkte. In der Datei /srv |
| de | `Allkompetenz` |  | de-prose-buchbinderei | 1203 | inen wesentlichen Teil seiner traditionellen Allkompetenz. Die klassische Lehrlingsausbildung, die meh |
| de | `Anfragefrequenz` |  | de-agent-fehlersuche | 1203 | esten Sie, ob das Problem bei einer erhöhten Anfragefrequenz (Stress-Test) linear zunimmt oder ob es durc |
| de | `Aquifers` | loanword | de-prose-wasserwerk | 1201 | tnahmemengen die natürliche Regeneration des Aquifers nicht übersteigen und das ökologische Gleich |
| de | `Arts` | loanword | de-prose-buchbinderei | 1202 | e William Morris und später die Künstler des Arts and Crafts Movement oder der Wiener Werkstät |
| de | `Ausführer` |  | de-prose-buchbinderei | 1201 | stand. Der Buchbinder wurde vom Schöpfer zum Ausführer, seine kreative Autonomie zugunsten der indu |
| de | `Auslastungsspitze` |  | de-agent-fehlersuche | 1203 |  Spitzen zu identifizieren. Eine korrelierte Auslastungsspitze zum Zeitpunkt des ersten Ausfalls bestätigt  |
| de | `Balancer` |  | de-agent-fehlersuche | 1203 |   7. Prüfen Sie die Firewall-Regeln und Load-Balancer-Einstellungen auf aggressive Timeout-Konfigu |
| de | `Balancern` |  | de-agent-fehlersuche | 1202 | one Routing-Konfiguration (ARP/MAC) bei Load-Balancern oft zu Paketverlusten führt. Das Problem ver |
| de | `Bandwidth` | loanword | de-prose-speicher | 1201 |  auch spezialisierte Speicherformen wie High Bandwidth Memory (HBM) eine zunehmend wichtige Rolle i |
| de | `Bodoni` |  | de-prose-plakat | 1202 | er Kontraststärke, inspiriert von Didot oder Bodoni, kann die Eleganz und den Luxus des Kunstgew |
| de | `Broway` |  | de-prose-wasserwerk | 1202 | e sich lösen und das Wasser trüben, was als „Broway“ oder Braunwasser bezeichnet wird. Gleichzei |
| de | `Buckram` | loanword | de-prose-buchbinderei | 1201 | ehmend durch textile Überzugsmaterialien wie Buckram, Leinen oder Moiré verdrängt. Diese Stoffe w |
| de | `Cachespeicherkapazität` |  | de-prose-speicher | 1201 |  Überschreitet eine Anwendung die verfügbare Cachespeicherkapazität, kollabiert die Leistung, weil die hohen Lat |
| de | `Chargenunterschiede` |  | lit-farbpalette | 1202 |  Exemplare hinweg erfordert. Es dürfen keine Chargenunterschiede entstehen, weshalb die Kalibrierung der Masc |
| de | `Cloth` | loanword | de-prose-buchbinderei | 1203 | olutionierte den Einband. Diese sogenannten „Cloth-bound“-Ausgaben waren robuster, haltbarer un |
| de | `Computing` | loanword | de-prose-speicher | 1202 | zu speisen. Die Zukunft der High-Performance-Computing-Architekturen wird daher nicht nur von der R |
| de | `Condition` | loanword | de-agent-fehlersuche | 1203 | ährend ein zufälliges Auftreten auf ein Race-Condition-Problem hindeutet.  10. Implementieren Sie k |
| de | `Counter` | loanword | de-agent-fehlersuche | 1202 | Netzwerkinterface, auf Link-Flaps oder Error-Counter mit `ethtool -S <Interface>`. Steigende Fehl |
| de | `Crafts` | loanword | de-prose-buchbinderei | 1202 |  Morris und später die Künstler des Arts and Crafts Movement oder der Wiener Werkstätte reagiert |
| de | `Daseinsvorsorge` |  | de-prose-wasserwerk | 1201 | rukturellen Aufgaben der modernen kommunalen Daseinsvorsorge dar. Es handelt sich dabei um ein hochsensib |
| de | `Debates` | loanword | de-agent-fehlersuche | 1203 | Backend-Abhängigkeiten des Dienstes auf Lock-Debates oder veraltete Verbindungspools, die sich na |
| de | `Defunct` | loanword | de-agent-fehlersuche | 1203 | gelmäßig abgebrochen werden oder im Zustand „Defunct“ verbleiben. Diese Beobachtung bestätigt ein |
| de | `Didone` |  | de-prose-plakat | 1203 | zwischen Haaren und Strichen erinnern an die Didone-Schriften, die im frühen neunzehnten Jahrhun |
| de | `Didot` |  | de-prose-plakat | 1202 | art mit hoher Kontraststärke, inspiriert von Didot oder Bodoni, kann die Eleganz und den Luxus  |
| de | `Druckerhöhungsanlagen` |  | de-prose-wasserwerk | 1202 |  vorhanden sein, der durch Hochbehälter oder Druckerhöhungsanlagen erzeugt wird. In einer mittelgroßen Stadt mi |
| de | `Einbanddecken` |  | de-prose-buchbinderei | 1201 |  die das traditionelle hölzerne Material der Einbanddecken ersetzten. Holz war sperrig, schwer und in d |
| de | `Einlagerungsplan` |  | de-agent-umzug | 1202 | tt ist abgeschlossen, wenn ein detaillierter Einlagerungsplan mit exakten Koordinaten für alle Einheiten v |
| de | `Emile` | loanword | de-prose-buchbinderei | 1202 | der des frühen zwanzigsten Jahrhunderts, wie Emile Ruse oder die Mitglieder der verschiedenen K |
| de | `Error` | loanword | de-agent-fehlersuche | 1201 | en Systems, wie z. B. den Apache- oder Nginx-Error-Logs, um zu sehen, ob der Reverse Proxy den  |
| de | `Exception` | loanword | de-agent-fehlersuche | 1202 | wie "failed", "oom-kill" oder wiederkehrende Exception-Meldungen im Log.  2. Überprüfe die Systemau |
| de | `Exceptions` | loanword | de-agent-fehlersuche | 1201 | lctl`, um nach spezifischen Fehlermeldungen, Exceptions oder Abbruchsignalen des Dienstes zu suchen. |
| de | `Fadenheftens` |  | de-prose-buchbinderei | 1201 | etzten allmählich die mühsame Handarbeit des Fadenheftens und des Kleisterschöpfens. Besonders in den  |
| de | `Fadenrücken` |  | de-prose-buchbinderei | 1202 | indigkeit hatte ihren Preis. Der handgenähte Fadenrücken, das Synonym für Langlebigkeit und Wertigkei |
| de | `Falzbogen` |  | de-prose-buchbinderei | 1202 | rtigung. Zuvor wurden Bücher von Hand in den Falzbogen geheftet, was bei hoher Auflage unmöglich wa |
| de | `Faults` | loanword | de-agent-fehlersuche | 1203 | dungen wie „Out of Memory“ oder Segmentation Faults während der Ausfallzeiten. Das Finden solche |
| de | `Flaps` | loanword | de-agent-fehlersuche | 1202 | insbesondere das Netzwerkinterface, auf Link-Flaps oder Error-Counter mit `ethtool -S <Interfac |
| de | `Framerate` |  | de-prose-speicher | 1201 | e Texel häufiger nachgeladen werden, was die Framerate bricht, da die GPU nicht so schnell zeichnen |
| de | `Frequently` | loanword | de-prose-speicher | 1203 | Policies) wie Least Recently Used oder Least Frequently Used beeinflusst wird. Die Latenzen, die bei |
| de | `Färbeprozesse` |  | de-prose-plakat | 1201 | e tiefe, satter Farben, die durch aufwendige Färbeprozesse erzeugt wurden, von tiefen Indigos bis zu sc |
| de | `Gate` | loanword | de-prose-speicher | 1202 |  Wege zurücklegen müssen und potenziell mehr Gate-Verzögerungen durchlaufen. Der L2-Cache fung |
| de | `Gebäudeinnersten` |  | de-prose-wasserwerk | 1202 |  und endet erst an der letzten Zapfstelle im Gebäudeinnersten, doch der Weg dorthin ist gesäumt von techni |
| de | `Graphikalgorithmus` |  | de-prose-speicher | 1201 |  kann. Solange die aktiven Datenmengen eines Graphikalgorithmus kleiner sind als die Kapazität des vorhanden |
| de | `Graphikanwendungen` |  | de-prose-speicher | 1201 | m Erliegen bringen. Daher ist das Design von Graphikanwendungen und Rendering-Algorithmen immer auch ein Opt |
| de | `Handshakes` | loanword | de-agent-fehlersuche | 1201 | in Netzwerkproblem, während erfolgreiche TCP-Handshakes den Netzwerkpfad als funktionsfähig ausschli |
| de | `Hexcode` |  | lit-farbpalette | 1201 | rfahrung legt der Hintergrund, der durch den Hexcode #D7FFE0 definiert wird. Diese sehr helle, fa |
| de | `Hexcodes` |  | lit-farbpalette | 1203 | leiden. Daher ist die strikte Einhaltung der Hexcodes für den Erfolg der Kampagne unerlässlich. Di |
| de | `High` | loanword | de-prose-speicher | 1201 | ielen auch spezialisierte Speicherformen wie High Bandwidth Memory (HBM) eine zunehmend wichti |
| de | `Hochgeschwindigkeits` |  | de-prose-speicher | 1202 | ptspeicher zugreifen, der typischerweise aus Hochgeschwindigkeits-GDDR-Arbeitsspeicher besteht. Dieser Schritt |
| de | `Industrialästhetik` |  | de-prose-buchbinderei | 1203 | derte, um der Entfremdung und der schlechten Industrialästhetik zu entgegnen.  Die Frage, was in diesem Proz |
| de | `Infrastrukturalterung` |  | de-prose-wasserwerk | 1202 | orgungssicherheit auch angesichts wachsender Infrastrukturalterung und veränderter klimatischer Bedingungen lan |
| de | `Infrastrukturleistung` |  | de-prose-wasserwerk | 1202 | ner mittelgroßen Stadt ist eine hochkomplexe Infrastrukturleistung, die weit über die bloße Verteilung eines Le |
| de | `Instandhaltungsstrategien` |  | de-prose-wasserwerk | 1201 | erwachung, präzise Datenanalyse und gezielte Instandhaltungsstrategien im Zaum gehalten werden kann. Eine Stadt, di |
| de | `Interposer` |  | de-prose-speicher | 1201 | ssor platziert, sondern direkt auf demselben Interposer, also einer Art Zwischenschicht, direkt nebe |
| de | `Kleisterschöpfens` |  | de-prose-buchbinderei | 1201 |  mühsame Handarbeit des Fadenheftens und des Kleisterschöpfens. Besonders in den Jahren um die Jahrhundertw |
| de | `Kohärenzprotokolle` |  | de-prose-speicher | 1203 |  3D-stapelten Speicher (HBM) und neuer Cache-Kohärenzprotokolle bringen, doch das grundlegende physikalische |
| de | `Konfigurations` |  | de-agent-fehlersuche | 1201 | nach dem Start bestätigt einen fundamentalen Konfigurations- oder Codefehler, während ein späteres Auftr |
| de | `Korretheit` |  | de-prose-plakat | 1203 | iftart wird auf ihre Eleganz und historische Korretheit untersucht, und die Lesbarkeit der kleineren |
| de | `Korrosionspartikel` |  | de-prose-wasserwerk | 1203 | wird das Netz regelmäßig gespült, um lockere Korrosionspartikel aus den Rohren zu entfernen. Der Austausch d |
| de | `Korrosionsprodukte` |  | de-prose-wasserwerk | 1202 | rjüngen sich die Rohre von innen durch diese Korrosionsprodukte, was den Durchflussquerschnitt reduziert und |
| de | `Kryptosporidien` |  | de-prose-wasserwerk | 1201 | andlung, um auch chlorresistente Erreger wie Kryptosporidien zu eliminieren, ohne dass übermäßige Nebenpr |
| de | `Layern` |  | lit-farbpalette | 1201 | B3A5F und #FFB347 nicht nur in den digitalen Layern stimmen, sondern auch in den Profilen für de |
| de | `Leaks` | loanword | de-agent-fehlersuche | 1201 | d Verbindungen mit `lsof` oder `netstat`, um Leaks oder eine erschöpfte Verbindungsliste zu erk |
| de | `Leckagehistorie` |  | de-prose-wasserwerk | 1201 |  die Materialalter, Druckbelastung, previous Leckagehistorie und die Kritikalität des versorgten Bereichs |
| de | `Leckageraten` |  | de-prose-wasserwerk | 1202 | iefer gelegenen Stadtteilen kann zu erhöhten Leckageraten führen, da die Rohre mechanischen Stress erf |
| de | `Legrain` |  | de-prose-buchbinderei | 1201 | es Kunstwerk verstanden. Künstler wie Pierre Legrain in Frankreich oder die Mitglieder der Verein |
| de | `Limitwerte` |  | de-agent-fehlersuche | 1202 | eout`) zu identifizieren. Eine Anpassung der Limitwerte in der Konfiguration und ein anschließender  |
| de | `Load` | loanword | de-agent-fehlersuche | 1202 | ynchrone Routing-Konfiguration (ARP/MAC) bei Load-Balancern oft zu Paketverlusten führt. Das P |
| de | `Lock` | loanword | de-agent-fehlersuche | 1203 | oder Backend-Abhängigkeiten des Dienstes auf Lock-Debates oder veraltete Verbindungspools, die |
| de | `Logs` | loanword | de-agent-fehlersuche | 1201 | us dies widerlegt.  2. Untersuche die System-Logs der letzten Stunde mit `journalctl`, um nach |
| de | `Loopback` |  | de-agent-fehlersuche | 1201 | isolierten Test des Dienstes auf der lokalen Loopback-Schnittstelle durch, um externe Netzwerkgerä |
| de | `Magazinpforten` |  | de-agent-umzug | 1202 |  übereinstimmt.  10. Schließen Sie die neuen Magazinpforten wieder sicher ab und schalten Sie die Klima- |
| de | `Maps` | loanword | de-prose-speicher | 1201 | igten Datenmengen rapide an. Wenn die Textur-Maps zu groß sind, um in den Texture-Cache zu pas |
| de | `Marroquin` |  | de-prose-buchbinderei | 1202 | ominierten naturbelassene Materialien. Rotes Marroquin, grünes Saffian, gepresstes Kalbsleder und m |
| de | `Membrantechnologien` |  | de-prose-wasserwerk | 1201 | e sowie, in fortschrittlichen Anlagen, durch Membrantechnologien. Ein entscheidender Aspekt der Aufbereitung  |
| de | `Misses` | loanword | de-prose-speicher | 1201 | hlich benötigt wird. Diese sogenannten Cache-Misses sind extrem kostenintensiv, da sie den Proze |
| de | `Movement` | loanword | de-prose-buchbinderei | 1202 |  und später die Künstler des Arts and Crafts Movement oder der Wiener Werkstätte reagierten auf di |
| de | `Nachfluszahlen` |  | de-prose-wasserwerk | 1203 | len, dass die Entnahmemengen die natürlichen Nachfluszahlen nicht übersteigen und die Ökosysteme der Ein |
| de | `Nachfüllstrategien` |  | de-prose-speicher | 1203 | sches System, das ständig durch Zugriffe und Nachfüllstrategien (Replacement Policies) wie Least Recently Us |
| de | `Nat` | loanword | de-agent-fehlersuche | 1202 |  Untersuche die Firewall-Regeln und iptables-Nat-Richtlinien, da eine asynchrone Routing-Konf |
| de | `Netzesdrucks` |  | de-prose-wasserwerk | 1202 | erqualität und einer bewussten Steuerung des Netzesdrucks, kann eine mittelgroße Stadt ihre Versorgung |
| de | `Nginx` |  | de-agent-fehlersuche | 1201 | gebenden Systems, wie z. B. den Apache- oder Nginx-Error-Logs, um zu sehen, ob der Reverse Prox |
| de | `Nichtabgerechnetes` |  | de-prose-wasserwerk | 1202 | gen, die im sogenannten „Non-Revenue-Water“ (Nichtabgerechnetes Wasser) zum Ausdruck kommen. Wenn die Differ |
| de | `Non` | loanword | de-prose-wasserwerk | 1202 | ohrschäden und Leckagen, die im sogenannten „Non-Revenue-Water“ (Nichtabgerechnetes Wasser) z |
| de | `Nouveau` |  | de-prose-plakat | 1203 | um die Anleihen des Jugendstils oder des Art Nouveau zu spiegeln. Wichtig ist jedoch, dass die Sc |
| de | `Offsetpapier` |  | de-prose-plakat | 1201 | sei es auf hochwertigem, leicht texturiertem Offsetpapier oder in der aufwendigeren lithografischen Ma |
| de | `Pantone` |  | de-prose-plakat | 1202 |  dominant wirken. Oft empfiehlt es sich, mit Pantone-Farben zu arbeiten, die eine bessere Kontrol |
| de | `Payload` | loanword | de-agent-fehlersuche | 1203 | einen konkreten Trigger wie eine spezifische Payload oder einen Header ausgelöst wird. Ein reprod |
| de | `Pilotzonen` |  | de-prose-wasserwerk | 1201 | sungen an strategischen Punkten, sogenannten Pilotzonen, liefern Ingenieuren in Echtzeit Daten darüb |
| de | `Plakatdesigns` |  | de-prose-plakat | 1202 | mt. In der klassischen Kompositionslehre des Plakatdesigns gilt oft der Grundsatz, dass das Bild die Ha |
| de | `Policies` | loanword | de-prose-speicher | 1203 | Zugriffe und Nachfüllstrategien (Replacement Policies) wie Least Recently Used oder Least Frequent |
| de | `Prefetching` |  | de-prose-speicher | 1202 | griffe hingegen können durch Mechanismen wie Prefetching, also das vorausschauende Laden von Daten, o |
| de | `Pricken` |  | de-prose-buchbinderei | 1203 | fasste und alles von der Vergoldung über das Pricken und Rollen bis zur komplexen Lederverarbeitu |
| de | `Priorisierungslisten` |  | de-prose-wasserwerk | 1203 | Störungsdaten. Auf Basis dieser Daten werden Priorisierungslisten für die Erneuerung erstellt, wobei nicht ein |
| de | `Processing` | loanword | de-prose-speicher | 1201 | mationen, Beleuchtungsberechnungen oder Post-Processing-Effekte zu berechnen.  Sobald die Arbeitsmen |
| de | `Proof` | loanword | lit-farbpalette | 1203 | ckprozess bei 4800 Stück mit einem digitalen Proof zu starten. Die Datei /srv/plakat/2026/vorla |
| de | `Purgierungen` |  | de-prose-plakat | 1201 | den, von tiefen Indigos bis zu schmerzhaften Purgierungen. Ein Plakat, das diese Farben in CMYK-Vierfa |
| de | `Recently` | loanword | de-prose-speicher | 1203 | lstrategien (Replacement Policies) wie Least Recently Used oder Least Frequently Used beeinflusst  |
| de | `Relining` | loanword | de-prose-wasserwerk | 1201 | ehen neuer Kunststoffrohre in das alte Rohr (Relining) oder durch partielle Auswechslungen. Diese  |
| de | `Rendering` | loanword | de-prose-speicher | 1201 | er ist das Design von Graphikanwendungen und Rendering-Algorithmen immer auch ein Optimieren der Da |
| de | `Replacement` | loanword | de-prose-speicher | 1203 | ändig durch Zugriffe und Nachfüllstrategien (Replacement Policies) wie Least Recently Used oder Least |
| de | `Ressourcenleak` |  | de-agent-fehlersuche | 1201 | hne korrespondierende Anfragen bestätigt ein Ressourcenleak, während stabile Zahlen diese Möglichkeit wi |
| de | `Revenue` | loanword | de-prose-wasserwerk | 1202 | chäden und Leckagen, die im sogenannten „Non-Revenue-Water“ (Nichtabgerechnetes Wasser) zum Ausdr |
| de | `Rohrwandungen` |  | de-prose-wasserwerk | 1202 | chäden belasten die Rohrverbindungen und die Rohrwandungen selbst. Diese mechanischen Spannungen führen |
| de | `Rostversalzung` |  | de-prose-wasserwerk | 1203 |  Eine wesentliche Ursache ist die sogenannte Rostversalzung, bei der sich im Inneren der Metallrohre übe |
| de | `Ruse` | loanword | de-prose-buchbinderei | 1202 | s frühen zwanzigsten Jahrhunderts, wie Emile Ruse oder die Mitglieder der verschiedenen Kunstg |
| de | `Saffian` |  | de-prose-buchbinderei | 1202 | lassene Materialien. Rotes Marroquin, grünes Saffian, gepresstes Kalbsleder und marmorisierte Pap |
| de | `Sans` | loanword | de-prose-plakat | 1201 | ung bietet. Ein zu strenger, modernistischer Sans-Serif-Kurs kann im Widerspruch zur barocken  |
| de | `Schrifttypeen` |  | de-prose-plakat | 1202 | en Schriftarten an. Eine reine Übernahme von Schrifttypeen aus dem 19. Jahrhundert könnte jedoch im heu |
| de | `Serif` | loanword | de-prose-plakat | 1201 | ietet. Ein zu strenger, modernistischer Sans-Serif-Kurs kann im Widerspruch zur barocken oder j |
| de | `Serifenschrifttypus` |  | de-prose-plakat | 1201 | ravität ausstrahlt, mit einem humanistischen Serifenschrifttypus, der an die typografischen Traditionen der J |
| de | `Shader` |  | de-prose-speicher | 1201 | rend die Rechengeschwindigkeit der einzelnen Shader-Kerne und Tensor-Einheiten in den vergangene |
| de | `Shadern` |  | de-prose-speicher | 1203 |  Programme ab. Grafikprogramme, insbesondere Shadern, weisen oft hohe Datenlokalität auf, was bed |
| de | `Shared` | loanword | de-prose-speicher | 1202 |  den Kernen, befinden sich die L1-Caches und Shared Memory-Bereiche. Diese Speicher sind so konz |
| de | `Siliziumfläche` |  | de-prose-speicher | 1202 | ie sind extrem schnell, da sie auf derselben Siliziumfläche wie die Arithmetik-Logik-Einheiten liegen un |
| de | `Skaling` |  | de-agent-fehlersuche | 1201 | h erst bei erhöhtem Lastniveau bestätigt ein Skaling- oder Thread-Pool-Problem, während Fehler au |
| de | `Sollbereich` |  | de-agent-umzug | 1203 | eratur- und Luftfeuchtigkeitswerte stabil im Sollbereich liegen.  5. Aufbau des neuen Ordnungssystems |
| de | `Thread` | loanword | de-agent-fehlersuche | 1201 | öhtem Lastniveau bestätigt ein Skaling- oder Thread-Pool-Problem, während Fehler auch bei Einzel |
| de | `Threads` | loanword | de-prose-speicher | 1202 | ese Verzögerung kritisch, da er Tausende von Threads gleichzeitig ausführen kann. Wenn diese Thre |
| de | `Tiling` | loanword | de-prose-speicher | 1203 | ierarchieebenen zu minimieren. Techniken wie Tiling, bei denen große Probleme in kleinere, cache |
| de | `Timeout` | loanword | de-agent-fehlersuche | 1203 | Latenz misst. Ein Anstieg der Latenz vor dem Timeout bestätigt eine Leistungsproblem oder Blockad |
| de | `Totwasserzonen` |  | de-prose-wasserwerk | 1201 | kennen. Besonders gefährlich sind sogenannte Totwasserzonen, also Leitungsabschnitte, die wenig oder gar |
| de | `Uferfiltratquellen` |  | de-prose-wasserwerk | 1201 | ielen Fällen speisen Grundwasserbrunnen oder Uferfiltratquellen das städtische System, wobei die geologische |
| de | `Unified` | loanword | de-prose-speicher | 1203 | len Speichertransparenztechniken, wie sie in Unified Memory-Konzepten vorkommen, eine wachsende R |
| de | `Used` | loanword | de-prose-speicher | 1203 | en (Replacement Policies) wie Least Recently Used oder Least Frequently Used beeinflusst wird. |
| de | `Vacui` |  | de-prose-plakat | 1203 | vor der leeren Fläche, das sogenannte Horror Vacui, oft eine treibende Kraft in der Dekorations |
| de | `Vergoldungslinien` |  | de-prose-buchbinderei | 1202 | , Leder zu zuschneiden, Faden zu spinnen und Vergoldungslinien mit der Feder zu ziehen. Doch ab den 1860er  |
| de | `Vergoldungsstrich` |  | de-prose-buchbinderei | 1202 | ers über das Nähen der Bogen bis zum letzten Vergoldungsstrich in den Händen eines einzigen Meisters oder e |
| de | `Verlegejahr` |  | de-prose-wasserwerk | 1203 | . Diese Modelle berücksichtigen Rohmaterial, Verlegejahr, Bodenbeschaffenheit und historische Störung |
| de | `Verteilleitungen` |  | de-prose-wasserwerk | 1203 | lknotenpunkten führt, von denen aus kleinere Verteilleitungen in die Wohngebiete abzweigen. Die Dimensioni |
| de | `Verteilnetz` |  | de-prose-wasserwerk | 1201 | olge, also die Verunreinigung des Wassers im Verteilnetz nach der Aufbereitung, frühzeitig zu erkenne |
| de | `Verteilnetzes` |  | de-prose-wasserwerk | 1201 | tung und der hydraulischen Eigenschaften des Verteilnetzes. Die Alterung des Netzes ist ein unvermeidli |
| de | `Vollschwarzes` |  | lit-farbpalette | 1201 |  Wahl von #050505 anstelle eines klassischen Vollschwarzes verhindert, dass das Bild zu hart oder techn |
| de | `Wiederkehrrate` |  | de-prose-speicher | 1202 | aten behält, von denen ein statistisch hoher Wiederkehrrate ausgegangen wird. Wenn eine Anfrage für Date |
| de | `active` | loanword | de-agent-fehlersuche | 1201 | m festzustellen, ob er im regulären Zustand „active“ oder in einer Fehlerphase wie „activating“  |
| de | `and` | loanword | de-prose-buchbinderei | 1202 | liam Morris und später die Künstler des Arts and Crafts Movement oder der Wiener Werkstätte r |
| de | `bereitgestapelt` |  | de-agent-umzug | 1201 |  verpackt, etikettiert und für den Transport bereitgestapelt sind. 4. Organisiere den physischen Transpor |
| de | `bound` | loanword | de-prose-buchbinderei | 1203 | nierte den Einband. Diese sogenannten „Cloth-bound“-Ausgaben waren robuster, haltbarer und deut |
| de | `cache` | loanword | de-prose-speicher | 1203 | iling, bei denen große Probleme in kleinere, cache-freundliche Blöcke zerlegt werden, sind gäng |
| de | `corrosiven` |  | de-prose-wasserwerk | 1203 | ten, während ein jüngerer Anschluss in einem corrosiven Boden mit hohem Grundwasserspiegel dringend  |
| de | `hinausschreibt` |  | de-prose-speicher | 1202 | hladen, während er gleichzeitig andere Daten hinausschreibt oder verdrängt. In diesem Szenario bricht di |
| de | `hochparallelen` |  | de-prose-speicher | 1202 | den Mikrosekundenbereich andauern. Für einen hochparallelen Grafikprozessor ist diese Verzögerung kritis |
| de | `hyperrealistischen` |  | de-prose-plakat | 1203 | bar zu machen, ohne dabei in den Bereich der hyperrealistischen Überzeichnung zu verfallen, die den historis |
| de | `ingenieurstechnischen` |  | de-prose-buchbinderei | 1203 | rfahren lag. Die Buchbinderei wurde zu einer ingenieurstechnischen Disziplin, die Physik, Chemie und Design zus |
| de | `jugendstilistischen` |  | de-prose-plakat | 1201 | f-Kurs kann im Widerspruch zur barocken oder jugendstilistischen Formensprache der Objekte stehen, während ei |
| de | `kerningiert` |  | de-prose-plakat | 1201 |  Designs. Ist die Schrift zu klein, schlecht kerningiert oder kontrastarm, wird der Leser frustriert  |
| de | `kuratorische` |  | de-prose-plakat | 1201 | ines Termins hinausgeht und stattdessen eine kuratorische Verdichtung in visueller Form erfordert. In  |
| de | `neonhaft` |  | lit-farbpalette | 1203 | dass die Farbe #FFB347 nicht zu satt oder zu neonhaft erscheint, da dies im direkten Vergleich zum |
| de | `ornamentalisierten` |  | de-prose-plakat | 1203 | le Kälte ausstrahlen, die im Widerspruch zur ornamentalisierten Welt des historischen Plakats stehen. Stattd |
| de | `paramounter` |  | lit-farbpalette | 1201 | hohen Auflage von 4800 Exemplaren ist es von paramounter Wichtigkeit, dass die Sättigung von #FFB347  |
| de | `precipitiert` |  | de-prose-wasserwerk | 1203 |  Metalle wie Eisen und Mangan aus dem Wasser precipitiert werden, indem Sauerstoff zugeführt wird, was |
| de | `repetitivrre` |  | de-prose-buchbinderei | 1201 | pezialisierten sich die Arbeiter auf wenige, repetitivrre Schritte, sodass das Verständnis für den ges |
| de | `sandhaltige` |  | de-prose-wasserwerk | 1203 | rklärbecken entfernt, bevor das Wasser durch sandhaltige Filtermedien geleitet wird, um feineren Schm |
| de | `silhouettenartige` |  | de-prose-plakat | 1203 |  Auto oder aus dem Zugfenster, zählt nur die silhouettenartige Erkennbarkeit und der hohe Kontrast. Das Mot |
| de | `texturalen` |  | de-prose-plakat | 1202 | len, die seine Silhouette betont, jedoch die texturalen Details wie die Maserung des Holzes oder die |
| de | `texturiertem` |  | de-prose-plakat | 1201 | zess selbst, sei es auf hochwertigem, leicht texturiertem Offsetpapier oder in der aufwendigeren litho |
| de | `tiefergehende` |  | de-prose-plakat | 1202 | en anregt, und bei der zweiten Sichtung eine tiefergehende Geschichte erzählt, die die Feinheiten des h |
| de | `to` | loanword | lit-farbpalette | 1203 | enkt die Aufmerksamkeit auf spezifische Call-to-Actions oder wichtige Datenpunkte. In der Da |
| de | `ungestrichenem` |  | lit-farbpalette | 1203 | s, um sicherzustellen, dass #FFB347 auch auf ungestrichenem Papier seine Leuchtkraft bewahrt. Bei einer  |
| de | `ungestrichenen` |  | lit-farbpalette | 1202 | abgrenzen, ohne zu bluten, was besonders auf ungestrichenen Papieren eine Herausforderung darstellt. Der |
| de | `unikales` |  | de-prose-buchbinderei | 1203 | Buchbinder als Künstler, der jedes Stück als unikales Meisterwerk behandelte, trat in den Hintergr |
| de | `vibrant` | loanword | de-prose-plakat | 1203 | ten Beige- und Brauntöne verliert, würde das vibrant lebendige Charakter dieser Epoche verfehlen. |
| de | `vollelektrische` |  | de-prose-buchbinderei | 1201 | twende beschleunigte sich diese Tendenz, als vollelektrische Maschinen in die Werkstätten Einzug hielten, |
| de | `vorzuverorten` |  | de-prose-plakat | 1202 | die haptische Qualität des Originals visuell vorzuverorten.  Eng verbunden mit der Bildauswahl ist die  |
| de | `Überzugsmaterialien` |  | de-prose-buchbinderei | 1201 |  Einbände war, wurde zunehmend durch textile Überzugsmaterialien wie Buckram, Leinen oder Moiré verdrängt. Di |
| en | `Deaccessioning` |  | en-prose-archive | 1202 | tanding the development of the municipality. Deaccessioning is the formal removal of records from the ar |
| en | `Hollinger` |  | en-prose-archive | 1203 | ven and nine. The most common housing is the Hollinger box, a sturdy container made from buffered p |
| en | `Moulders` |  | en-prose-foundry | 1203 | ld take weeks for the highest quality molds. Moulders, working in teams, would pack the sand aroun |
| en | `acidification` |  | en-prose-archive | 1203 | small tears and physical punctures to severe acidification and mold growth. The most common repair tech |
| en | `amidst` |  | en-prose-foundry | 1202 |  floor was a theater of coordinated movement amidst the roar of the blast furnace and the hiss o |
| en | `auditable` |  | lit-releasenote | 1203 |  ensures that the deployment is reliable and auditable. The integrity of the data in models/flash-n |
| en | `booklice` |  | en-prose-archive | 1201 | threats such as mold spores, silverfish, and booklice pose immediate dangers to collections. These |
| en | `checksums` |  | lit-releasenote | 1201 | ed manually, as doing so will invalidate the checksums associated with build 0xB7A31F.  The most si |
| en | `codebase` |  | lit-releasenote | 1201 |  confident that they are utilizing the exact codebase that passed our extensive validation suite,  |
| en | `convolutional` |  | lit-releasenote | 1201 | ensor governs the scaling factors within the convolutional layers of the state-space model, directly in |
| en | `deaccession` |  | en-prose-archive | 1201 |  the story of the community. The decision to deaccession is made in consultation with legal authoriti |
| en | `deaccessioning` |  | en-prose-archive | 1202 |  difficult decisions regarding appraisal and deaccessioning, the processes that determine what is kept a |
| en | `deacidification` |  | en-prose-archive | 1203 | ification, more intensive treatments such as deacidification baths or encapsulation in polyester sleeves  |
| en | `duplicative` |  | en-prose-archive | 1201 | elming volume of paperwork, much of which is duplicative, transient, or legally insignificant. Preser |
| en | `evidential` |  | en-prose-archive | 1202 |  order of the records is preserved if it has evidential value, or creating a new order if the origin |
| en | `evidentiary` |  | en-prose-archive | 1201 |  or legal value. This involves assessing the evidentiary significance, informational content, and leg |
| en | `fireclay` |  | en-prose-foundry | 1201 | ion. Ladles, massive iron vessels lined with fireclay, were filled with molten iron and carried by |
| en | `grey` |  | en-prose-foundry | 1201 |  sand was removed, revealing the rough, dark grey surface of the iron casting. This was the fi |
| en | `hardcoded` |  | lit-releasenote | 1201 | 003.cnq. Developers who previously relied on hardcoded overrides for ssm_conv1d_alpha should remove |
| en | `hydrostatic` |  | en-prose-foundry | 1202 | , such as engine blocks or pressure vessels, hydrostatic testing might be employed, where the casting |
| en | `hygroscopic` |  | en-prose-archive | 1203 |  of temperature and humidity, for paper is a hygroscopic material that absorbs moisture from the air, |
| en | `ingates` |  | en-prose-foundry | 1202 | ting moulds. The liquid iron flowed into the ingates, the channels carved into the sand that led  |
| en | `microclimate` |  | en-prose-foundry | 1201 | ingeing hair and drying out skin, creating a microclimate of extreme temperature that dictated the pac |
| en | `mould` |  | en-prose-foundry | 1201 | pressure of the liquid metal would burst the mould; if it was not permeable enough, the trapped |
| en | `moulder` |  | en-prose-foundry | 1203 | cally bonded sands, meaning the skill of the moulder was paramount. They relied on touch and inst |
| en | `moulders` |  | en-prose-foundry | 1201 | ld cause blowholes, ruining the casting. The moulders, working in shifts around the clock, packed  |
| en | `moulding` |  | en-prose-foundry | 1201 | the realm of geology and chemistry, into the moulding floor, where the earth itself was prepared t |
| en | `moulds` |  | en-prose-foundry | 1201 | d by crane or moved on tracks to the waiting moulds. The pourers, wearing heavy leathers and gog |
| en | `oversized` |  | en-prose-foundry | 1202 |  meaning the wooden model had to be slightly oversized. They incorporated draft angles, subtle tape |
| en | `patternmaker` |  | en-prose-foundry | 1203 | of how iron would contract as it cooled. The patternmaker had to account for this shrinkage, building  |
| en | `pourers` |  | en-prose-foundry | 1201 | r moved on tracks to the waiting moulds. The pourers, wearing heavy leathers and goggles, approac |
| en | `radiographic` |  | en-prose-foundry | 1203 | tion or, in some advanced shops, rudimentary radiographic methods using X-rays, though this was rare i |
| en | `rammers` |  | en-prose-foundry | 1203 | pattern in flasks, ramming it down with iron rammers until the density was perfect. Too loose, an |
| en | `recalibration` |  | lit-releasenote | 1203 |  in high-throughput scenarios, prompting the recalibration of ssm_conv1d_alpha. The new values provide  |
| en | `reproducibility` |  | lit-releasenote | 1203 | d under tag release-2026-09-18a to guarantee reproducibility and traceability across all distributed node |
| en | `sharding` |  | lit-releasenote | 1202 | note that this container is part of a larger sharding strategy, but for the purposes of this updat |
| en | `sprueing` |  | en-prose-foundry | 1203 | y compacted, often using a technique called "sprueing" to cut channels that would allow the molten |
| en | `sprues` |  | en-prose-foundry | 1201 |  chippers, and grinding stones to remove the sprues and risers. Sparks flew in showers as abrasi |
| en | `suboptimal` | loanword | lit-releasenote | 1201 | the previous parameter initialization led to suboptimal convergence rates in high-dimensional contex |
| en | `subseries` |  | en-prose-archive | 1202 | sed by the archivists. It identifies series, subseries, and containers, allowing a researcher to tr |
| en | `tuyeres` |  | en-prose-foundry | 1203 | hile air was blasted from the bottom through tuyeres, creating a combustion zone that reached tem |
| en | `underfilling` |  | en-prose-archive | 1202 | e paper and makes retrieval difficult, while underfilling allows the documents to collapse and fold in |
| en | `unglamorous` |  | en-prose-archive | 1203 | hose who come after. This continuous effort, unglamorous and often invisible to the public eye, forms |
| en | `workability` |  | en-prose-foundry | 1202 | was often pine or mahogany, selected for its workability and resistance to warping under the humidity |

## near-miss literals

None: every literal that appeared at all appeared exactly.

