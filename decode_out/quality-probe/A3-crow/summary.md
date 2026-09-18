# quality probe - A3-crow

- date 2026-09-18, repo commit `c1cf95c`, prompt set version 1
- endpoint `http://127.0.0.1:8099`, engine `crow`, model `Qwen3.8-Flash-Next-CNQ4.5-M`
- row temperature 1.0, top_p 0.95, top_k 20, presence_penalty 0.0, min_p 0.0, max_tokens 2600
- thinking: absent (crow); seeds [4401, 4402, 4403]; 36 generations, 0 failed, 665 s wall

## the arm in one table

| metric | value |
|---|---|
| non-word rate DE per 1000 words, per generation | 24.58 (9.33 to 72.54) |
| non-word rate DE, loanwords the EN dictionary knows removed | 18.35 (9.33 to 38.96) |
| non-word rate DE, pooled over 15308 words | 21.49 |
| non-word rate EN per 1000 words, per generation | 9.83 (0.00 to 18.72) |
| non-word rate EN, loanwords the DE dictionary knows removed | 9.48 (0.00 to 15.60) |
| non-word rate EN, pooled over 9064 words | 10.48 |
| exact literals reproduced (share of literals) | 0.898 (0.571 to 1.000) |
| exact literals reproduced (share of demanded occurrences) | 0.959 (0.793 to 1.000) |
| near-miss literal kinds seen | 3 |
| JSON: whole answer a valid document / shape ok | 4 / 4 of 6 |
| distinct-word ratio | 0.524 (0.400 to 0.805) |
| longest immediate repeat run | 1.9 (1.0 to 5.0), max 5 |
| generations with foreign-script characters | 0 of 24 (0 chars) |
| words per answer | 685 (25 to 1548) |
| answers stopped at max_tokens | 0 of 36 |

## per prompt

| prompt | lang | seed | words | finish | non-word/1000 | flagged | literals | json | longest repeat | foreign |
|---|---|---|---|---|---|---|---|---|---|---|
| de-prose-plakat | de | 4401 | 965 | stop | 9.33 | 9/965 | - | - | 2x1 | 0 |
| de-prose-plakat | de | 4402 | 1095 | stop | 15.53 | 17/1095 | - | - | 2x1 | 0 |
| de-prose-plakat | de | 4403 | 853 | stop | 12.90 | 11/853 | - | - | 2x1 | 0 |
| de-prose-speicher | de | 4401 | 956 | stop | 15.69 | 15/956 | - | - | 2x1 | 0 |
| de-prose-speicher | de | 4402 | 1041 | stop | 43.23 | 45/1041 | - | - | 2x1 | 0 |
| de-prose-speicher | de | 4403 | 965 | stop | 27.98 | 27/965 | - | - | 2x1 | 0 |
| de-prose-buchbinderei | de | 4401 | 1042 | stop | 13.44 | 14/1042 | - | - | 2x2 | 0 |
| de-prose-buchbinderei | de | 4402 | 975 | stop | 30.77 | 30/975 | - | - | 2x1 | 0 |
| de-prose-buchbinderei | de | 4403 | 954 | stop | 13.63 | 13/954 | - | - | 2x1 | 0 |
| de-prose-wasserwerk | de | 4401 | 1068 | stop | 21.54 | 23/1068 | - | - | 2x1 | 0 |
| de-prose-wasserwerk | de | 4402 | 1053 | stop | 16.14 | 17/1053 | - | - | 2x1 | 0 |
| de-prose-wasserwerk | de | 4403 | 1091 | stop | 21.08 | 23/1091 | - | - | 1x1 | 0 |
| en-prose-foundry | en | 4401 | 1282 | stop | 18.72 | 24/1282 | - | - | 2x1 | 0 |
| en-prose-foundry | en | 4402 | 1342 | stop | 13.41 | 18/1342 | - | - | 2x1 | 0 |
| en-prose-foundry | en | 4403 | 1548 | stop | 11.63 | 18/1548 | - | - | 2x1 | 0 |
| en-prose-archive | en | 4401 | 1196 | stop | 9.20 | 11/1196 | - | - | 2x1 | 0 |
| en-prose-archive | en | 4402 | 1250 | stop | 1.60 | 2/1250 | - | - | 2x1 | 0 |
| en-prose-archive | en | 4403 | 1100 | stop | 9.09 | 10/1100 | - | - | 1x1 | 0 |
| lit-farbpalette | de | 4401 | 593 | stop | 16.86 | 10/593 | 4/7 (0.90) | - | 2x1 | - |
| lit-farbpalette | de | 4402 | 588 | stop | 28.91 | 17/588 | 5/7 (0.79) | - | 1x1 | - |
| lit-farbpalette | de | 4403 | 573 | stop | 19.20 | 11/573 | 7/7 (1.00) | - | 1x1 | - |
| lit-releasenote | en | 4401 | 488 | stop | 14.34 | 7/488 | 5/5 (1.00) | - | 1x1 | - |
| lit-releasenote | en | 4402 | 478 | stop | 10.46 | 5/478 | 4/5 (0.94) | - | 2x1 | - |
| lit-releasenote | en | 4403 | 380 | stop | 0.00 | 0/380 | 5/5 (1.00) | - | 1x1 | - |
| json-tensorplan | en | 4401 | 25 | stop | - | - | 5/5 (1.00) | ok | 5x3 | - |
| json-tensorplan | en | 4402 | 25 | stop | - | - | 5/5 (1.00) | ok | 5x3 | - |
| json-tensorplan | en | 4403 | 25 | stop | - | - | 5/5 (1.00) | ok | 5x3 | - |
| json-schritte | de | 4401 | 87 | stop | - | - | - | fragment | 1x1 | - |
| json-schritte | de | 4402 | 64 | stop | - | - | - | ok | 1x1 | - |
| json-schritte | de | 4403 | 74 | stop | - | - | - | fragment | 1x1 | - |
| de-agent-umzug | de | 4401 | 77 | stop | 38.96 | 3/77 | - | - | 1x1 | 0 |
| de-agent-umzug | de | 4402 | 350 | stop | 28.57 | 10/350 | - | - | 2x1 | 0 |
| de-agent-umzug | de | 4403 | 349 | stop | 11.46 | 4/349 | - | - | 1x1 | 0 |
| de-agent-fehlersuche | de | 4401 | 214 | stop | 23.36 | 5/214 | - | - | 1x1 | 0 |
| de-agent-fehlersuche | de | 4402 | 193 | stop | 72.54 | 14/193 | - | - | 1x1 | 0 |
| de-agent-fehlersuche | de | 4403 | 313 | stop | 35.14 | 11/313 | - | - | 1x1 | 0 |

## flagged words, with context

| lang | word | note | prompt | seed | context |
|---|---|---|---|---|---|
| de | `ALUs` |  | de-prose-speicher | 4402 | unmittelbare Verfügbarkeit der Daten für die ALUs und Tensor-Cores ermöglicht. Diese Architekt |
| de | `Abgleichungsgesetz` |  | de-prose-wasserwerk | 4401 | notonischer Prozess, sondern ein dynamisches Abgleichungsgesetz zwischen biologischer Sicherheit, Geschmack  |
| de | `Ablesbarkeit` |  | lit-farbpalette | 4403 |  Kolorit gewählt. Im Zentrum steht die klare Ablesbarkeit, die durch den Kontrast von tiefstem Schwarz |
| de | `Adressierungs` |  | de-prose-speicher | 4402 | ame Nutzung von Ressourcen zur Reduktion von Adressierungs-Overheads optimiert ist, muss man die Archit |
| de | `Anwendungs` |  | de-agent-fehlersuche | 4403 | tigt Netzwerkinstabilität. 4. Untersuche die Anwendungs-logs des Dienstes auf wiederkehrende Pattern |
| de | `Appreturapparat` |  | de-prose-buchbinderei | 4402 | ewegung ersetzte. Parallel dazu entstand der Appreturapparat, der den Buchblock glättete, die Kapitale fe |
| de | `Auschnitt` |  | de-agent-fehlersuche | 4401 |  `systemctl status <dienstname>` und dem Log-Auschnitt `journalctl -n 50`, um festzustellen, ob der |
| de | `Ausrufzeichen` |  | de-prose-plakat | 4401 | ehen im Flanieren, muss es als ein visuelles Ausrufzeichen fungieren, das die Aufmerksamkeit des Passan |
| de | `Bandbreiteschranke` |  | de-prose-speicher | 4403 | er zentrale Mechanismus zur Verschiebung der Bandbreiteschranke ist die sogenannte Prefetching-Architektur,  |
| de | `Bandedtaste` |  | de-prose-speicher | 4403 | trem schnellen, aber kleinsten Registern und Bandedtaste-Speichern über mehrere Zwischenspeicherstufe |
| de | `Bandwidth` | loanword | de-prose-speicher | 4403 | eichers, oft in Form von HBM-Schichten (High Bandwidth Memory), bestimmt dabei, wie viele parallele |
| de | `Bebrütungsgrad` |  | de-prose-wasserwerk | 4403 | kstoff und an Gesamt-Keimzahlen an den 37 °C-Bebrütungsgrad, im Blick behalten. Eine besondere methodisc |
| de | `Bedeitung` |  | de-prose-plakat | 4402 | nstgewerbliche Sammlung, wo die Objekte ihre Bedeitung erst durch ihre Nutzung und ihren Kontext en |
| de | `Berecher` |  | de-prose-speicher | 4401 | rden, dass sie die Daten aufnimmt, bevor der Berecher sie wirklich braucht. Dies wird durch Hardwa |
| de | `Beschleunigerarchitektur` |  | de-prose-speicher | 4402 | n kann, und damit die Effizienz der gesamten Beschleunigerarchitektur bestimmt. |
| de | `Beschleunigerarchitekturen` |  | de-prose-speicher | 4402 | iefgreifende Konsequenzen für das Design von Beschleunigerarchitekturen und die Effizienz von Grafik-Workloads, wobe |
| de | `Bestimmungsgöße` |  | de-prose-speicher | 4402 | st die Latenz für den L1-Zugriff die primäre Bestimmungsgöße. Ist die Arbeitsmenge jedoch so beschaffen,  |
| de | `Betrachtes` |  | de-prose-plakat | 4401 |  Erinnerung an die Objekte im Gedächtnis des Betrachtes so fest haftet, wie die Tinten auf dem Papie |
| de | `Bewußtsein` |  | de-prose-buchbinderei | 4403 | ine letzte Form der Wertschöpfung sahen, das Bewußtsein für die gebrochene Tradition erhalten. Die G |
| de | `Bezugmaterial` |  | de-prose-buchbinderei | 4402 | z von Prägedrucktechniken. Der Druck auf den Bezugmaterial, sei es nun mit Prägefarben oder später im R |
| de | `Bindereiprodukt` |  | de-prose-buchbinderei | 4401 | d zum Verlagsprodukt, nicht mehr als nur zur Bindereiprodukt. Der Einband diente nun dem Schutz des Block |
| de | `Bindungs` |  | de-agent-fehlersuche | 4402 |  oder mehrfach belegtes Socket bestätigt ein Bindungs- oder Konfigurationsproblem. 4. Überwache di |
| de | `Bitumenauskleidung` |  | de-prose-wasserwerk | 4403 |  Eisenleitungen mit einer Zementmörtel- oder Bitumenauskleidung, durch die jahrelange Kontakt mit dem chemis |
| de | `Blaus` |  | lit-farbpalette | 4401 | erleiht. Er bricht die Dominanz des kühligen Blaus auf und setzt energische Punkte auf das Blat |
| de | `Blockierregeln` |  | de-agent-fehlersuche | 4403 | erkehr selektiv blockieren; das Fehlen neuer Blockierregeln widerlegt diese Theorie. 7. Teste den Dienst |
| de | `Buchbinderkartons` |  | de-prose-buchbinderei | 4402 | ränderten, steifen Papieren, den sogenannten Buchbinderkartons, sowie der Einsatz von Prägedrucktechniken.  |
| de | `Buchbinderleinenrücken` |  | de-prose-buchbinderei | 4402 | arbeitete Leder wurde in der Masse durch den Buchbinderleinenrücken und vor allem durch den sogenannten Halblein |
| de | `Buchr` |  | de-prose-buchbinderei | 4402 | lock glättete, die Kapitale festigte und den Buchr mit Leim beschichtete, sowie die Pressen, di |
| de | `Caching` | loanword | de-prose-speicher | 4402 | . Durch die Vorhersage (Prefetching) und das Caching von benachbarten Speicherblöcken wird die ef |
| de | `Changes` | loanword | de-agent-fehlersuche | 4403 | tion. 9. Analysiere die letzten Updates oder Changes (Code, Config, System), um einen zeitlichen  |
| de | `Competition` | loanword | de-prose-speicher | 4402 | Warp-Gruppen um den verfügbaren Platz. Diese Competition erzeugt eine spezifische Form der Latenz, di |
| de | `Compute` | loanword | de-prose-speicher | 4402 |  latenzarme Speicherstufe, die direkt an die Compute-Units gekoppelt ist. Die Latenz für einen Zu |
| de | `Concurrent` | loanword | de-agent-fehlersuche | 4402 | erbindungs-Pools. Eine zu niedrige Anzahl an Concurrent-Worker bestätigt, dass der zweite Thread feh |
| de | `Config` |  | de-agent-fehlersuche | 4403 | iere die letzten Updates oder Changes (Code, Config, System), um einen zeitlichen Zusammenhang m |
| de | `Cores` | loanword | de-prose-speicher | 4403 | o der Streaming-Multiprozessoren oder Tensor-Cores, einem exponentiellen Wachstum unterlag, sta |
| de | `Cyan` | loanword | de-prose-plakat | 4403 | inen Primärfarbigkeit bedeutet. Statt reiner Cyan- oder Magenta-Flächen sollte der Einsatz von |
| de | `Deadlock` | loanword | de-agent-fehlersuche | 4402 |  dass der zweite Thread fehlt oder sofort im Deadlock hängt. |
| de | `Debug` | loanword | de-agent-fehlersuche | 4401 | st.  6. Erhöhe die LogLevel des Dienstes auf Debug-Stufe, falls die Konfiguration dies erlaubt, |
| de | `Degradationsprozesse` |  | de-prose-wasserwerk | 4401 | keine einzelnen Fehler, sondern kumulierende Degradationsprozesse, die die Integrität des Systems von innen he |
| de | `Dosisierung` |  | de-prose-wasserwerk | 4403 | iesem Grunde ist die räumliche und zeitliche Dosisierung der Entnahme aus mehreren, geologisch unters |
| de | `Druckboosterstufen` |  | de-prose-wasserwerk | 4402 | sche Belastungswechsel durch den Betrieb von Druckboosterstufen. Diese Mikro-Vibrationen und Druckspitzen fü |
| de | `Druckminderstationen` |  | de-prose-wasserwerk | 4401 | ührt. Die Steuerung dieses Drucks, oft durch Druckminderstationen in kritischen Abschnitten, ist daher die uns |
| de | `Druckvorgaben` |  | lit-farbpalette | 4403 | e spezifische Nuance, die als #D7FFE0 in den Druckvorgaben firmiert, verhindert die Ermüdung des Auges  |
| de | `Durchlassgeschwindigkeit` |  | de-prose-wasserwerk | 4401 | Substanzen und Keime absorbiert. Doch wo die Durchlassgeschwindigkeit erhöht werden muss, um wachsende Abnahmen zu |
| de | `Durchsatzleistung` |  | de-prose-speicher | 4403 |  L2-Speicher – kann die theoretisch maximale Durchsatzleistung des internen Speicherbusses ausgenutzt werde |
| de | `Durchschüssung` |  | de-prose-plakat | 4403 | ge verkürzt und den Rhythmus durch eine enge Durchschüssung verdichtet, und einem modernen, luftigeren R |
| de | `Durchsätze` |  | de-prose-speicher | 4401 | n und ein Fluch zugleich. Sie erlaubt enorme Durchsätze, wenn die Daten im richtigen Moment am richt |
| de | `Durchzeichnung` |  | lit-farbpalette | 4403 |  der 4800 Exemplare ist auf die gleichmäßige Durchzeichnung zu achten. Der Rahmen in #1B3A5F darf nicht  |
| de | `End` | loanword | de-prose-speicher | 4401 | verfügbarkeit betrachten. Ein aktueller High-End-GPU enthält Tausende von Kernen, die gleichz |
| de | `Enteisenung` |  | de-prose-wasserwerk | 4402 | er aus den Brunnen meist nur einer einfachen Enteisenung und Entmanganung sowie einer UV-Entkeimungun |
| de | `Entkeimungung` |  | de-prose-wasserwerk | 4402 |  Enteisenung und Entmanganung sowie einer UV-Entkeimungung bedarf, muss das aus Oberflächenquellen stam |
| de | `Entmanganung` |  | de-prose-wasserwerk | 4402 | en meist nur einer einfachen Enteisenung und Entmanganung sowie einer UV-Entkeimungung bedarf, muss da |
| de | `Errung` |  | de-prose-buchbinderei | 4403 | lichen Einzigartigkeit war der Preis für die Errung einer neuen sozialen Reichweite. Die Buchbin |
| de | `Facturing` |  | de-prose-wasserwerk | 4402 | des, leckiges Netz ist die sogenannte „Nicht-Facturing-Balance“, also die Diskrepanz zwischen dem i |
| de | `Fadenheftapparat` |  | de-prose-buchbinderei | 4402 | Heftmaschine und später der vollautomatische Fadenheftapparat, der das mühselige manuelle Steppen durch ei |
| de | `Fadenstich` |  | de-prose-buchbinderei | 4402 |  wuchsen die Auflagenhöhen, die der manuelle Fadenstich, das manuelle Nähen der Bogen sowie das manu |
| de | `Fadenzuführung` |  | de-prose-buchbinderei | 4402 | pen durch eine präzise, von Spulen gespeiste Fadenzuführung und Nadelbewegung ersetzte. Parallel dazu en |
| de | `Farbübergangänge` |  | lit-farbpalette | 4403 | srv/plakat/2026/vorlage-v3.2.1.svg, dass die Farbübergangänge klar definiert bleiben müssen. Die Fassung v |
| de | `Fazitierend` |  | de-prose-speicher | 4401 | leme in kleinere, cache-kompatiblene Teile.  Fazitierend lässt sich feststellen, dass die Speicherhie |
| de | `Fe` | loanword | de-prose-wasserwerk | 4403 | d, wodurch gelöste Reduktanten, insbesondere Fe(II)-Ionen, in die unlösliche, fällbare Fe(II |
| de | `Fehlercodeinträge` |  | de-agent-fehlersuche | 4402 | ge, die den Turnus verfehlen, oder explizite Fehlercodeinträge bestätigen, dass der Prozess die Anfragen ni |
| de | `Filtrationsgeschwindigkeit` |  | de-prose-wasserwerk | 4401 | er intensiven Vorbehandlung bedürfen, um die Filtrationsgeschwindigkeit im richtigen Maß zu halten. Eine wachsende R |
| de | `Filtrationszeiträume` |  | de-prose-wasserwerk | 4403 | igkeit im Untergrund verändert und damit die Filtrationszeiträume, in denen das Wasser aufsteigend aufbereitet |
| de | `Findmittel` |  | de-agent-umzug | 4402 | t fertig, wenn die Suchabfragen im digitalen Findmittel exakt die neuen Lokationen anzeigen.  8. Übe |
| de | `Fluchtigen` |  | de-prose-plakat | 4402 | ndwerklichkeit, die Materialität, die man im Fluchtigen nicht sieht, aus. Ein gelungenes Plakat vere |
| de | `Gehil` |  | de-prose-buchbinderei | 4402 | as der Buchbinder in der Masseproduktion zum Gehil einer Verlagskalküle wurde, der den Einband  |
| de | `Gelbtonigkeit` |  | lit-farbpalette | 4403 | d darf bei der Auflage von 4800 nicht in die Gelbtonigkeit abdriften, wie es bei ungleichmäßig trocknen |
| de | `Geschwungenheit` |  | de-prose-plakat | 4402 | dustrialisierten Lettern oder die organische Geschwungenheit der Jugendstil-Anfänge – ohne dabei in pasti |
| de | `Gesellenstand` |  | de-prose-buchbinderei | 4402 | eg zum Meister führte über die Lehrzeit, den Gesellenstand und die Meisterprüfung, die das Beherrschen  |
| de | `Geselleprüfung` |  | de-prose-buchbinderei | 4402 | e Entwicklung, die das Verhältnis von Lehre, Geselleprüfung und Meisterstück neu bestimmte. Bis in die z |
| de | `Handwerklichkeit` |  | de-prose-plakat | 4402 |  lesbar sein und das historische Detail, die Handwerklichkeit, die Materialität, die man im Fluchtigen nic |
| de | `Hexcode` |  | lit-farbpalette | 4402 | te Bildgeschehen ruht, wobei der spezifische Hexcode #D7FFE0 die visuelle Last des Auges minimier |
| de | `Hexcodes` |  | lit-farbpalette | 4402 | glich, da die spezifische Mischunsg der vier Hexcodes #D7FFE0, #050505, #1B3A5F und #FFB347 im CMY |
| de | `Hiding` | loanword | de-prose-speicher | 4403 | t aktivieren. Dieses Verfahren, als „Latency Hiding“ oder Latenzunterdrückung bekannt, wandelt d |
| de | `Hierarchy` | loanword | lit-farbpalette | 4401 | t in der Fassung v3.2.1 als kritisch für die Hierarchy markiert und muss bei der Gesamtauflage von  |
| de | `High` | loanword | de-prose-speicher | 4401 | Datenverfügbarkeit betrachten. Ein aktueller High-End-GPU enthält Tausende von Kernen, die gle |
| de | `Historisten` |  | de-prose-plakat | 4401 | e in die Nostalgie zurückzufallen, die viele Historisten vernebelten. Die Anordnung der Buchstaben mu |
| de | `Hochbehälers` |  | de-prose-wasserwerk | 4401 | ßendes Gleichgewicht, das durch die Lage des Hochbehälers über dem Niveau der Endverbraucher, die Höhe |
| de | `Infrastrukturökonomie` |  | de-prose-wasserwerk | 4402 | tlichen Gesundheit, der Stadtplanung und der Infrastrukturökonomie berührt. Im Gegensatz zu Metropolen mit ausg |
| de | `Instruction` | loanword | de-prose-speicher | 4403 | n Anforderungen der SIMD-Architektur (Single Instruction, Multiple Data) zugeschnitten sind. Hier kön |
| de | `Kaiserreichszeit` |  | de-prose-buchbinderei | 4403 | in spezifisch deutsches Phänomen der zweiten Kaiserreichszeit und der Weimarer Frühzeit, verschmolz das Dr |
| de | `Kalbleder` |  | de-prose-buchbinderei | 4401 | m Leinen, halbledernem Pergament oder ganzem Kalbleder, war teuer, aber auch ein Statussymbol. Als  |
| de | `Klimatisierungsparameter` |  | de-agent-umzug | 4402 |  die Technik funktionsfähig getestet und die Klimatisierungsparameter auf den Sollwert eingestellt sind.  7. Verze |
| de | `Kontrasteffentsteht` |  | lit-farbpalette | 4401 | läche abzuheben, ohne dass ein „flackernder“ Kontrasteffentsteht. Diese Nuance ist in der Vorlage /srv/plakat |
| de | `Korrosionsstellen` |  | de-prose-wasserwerk | 4402 | d Gussleitungen bilden sich an den Wandungen Korrosionsstellen, die den effektiven Querschnitt der und dami |
| de | `Krustenschicht` |  | de-prose-wasserwerk | 4401 | eren im Wasser gelösten Stoffen, eine innere Krustenschicht, die den freien Querschnitt der Rohr verengt |
| de | `Kunströffmaterialien` |  | de-prose-wasserwerk | 4401 | sstoffe wahrnehmbar werden. Bei den modernen Kunströffmaterialien wie Polyethylen (PE) oder Polyvinylchlorid ( |
| de | `Latency` | loanword | de-prose-speicher | 4403 | chen-Takt aktivieren. Dieses Verfahren, als „Latency Hiding“ oder Latenzunterdrückung bekannt, wa |
| de | `Lebensmittelproduzieranlagen` |  | de-prose-wasserwerk | 4401 | auch sensible Zentren wie Krankenhäuser oder Lebensmittelproduzieranlagen gleichermaßen sicher versorgen. Der Druck im |
| de | `Lecksuchungsarbeiten` |  | de-prose-wasserwerk | 4401 | erlusten in den Abschnitten, ein Anstieg der Lecksuchungsarbeiten und ein sinkender Druck in den peripheren, t |
| de | `Leimwerkzeug` |  | de-prose-buchbinderei | 4401 | eftung, die durch den Faden und die Nadel im Leimwerkzeug erfolgte, wich allmählich der Draht- oder Fa |
| de | `Leitungsburchbrüchen` |  | de-prose-wasserwerk | 4401 | etzführung ist die zunehmende Häufigkeit von Leitungsburchbrüchen, die sich von den gewöhnlichen, durch äußere |
| de | `Lesungsbereiche` |  | lit-farbpalette | 4401 | ge-v3.2.1.svg als Standardtextfarbe für alle Lesungsbereiche festgelegt und trägt damit in der Fassung v3 |
| de | `Litfaß` |  | de-prose-plakat | 4402 | U-Bahn-Station, an der Hauswand oder auf dem Litfaß der Ausstellungshalle hängt, muss es durch e |
| de | `LogLevel` |  | de-agent-fehlersuche | 4401 | gigkeiten den Fehler auslöst.  6. Erhöhe die LogLevel des Dienstes auf Debug-Stufe, falls die Konf |
| de | `Logistikkette` |  | de-agent-umzug | 4402 | sten Behältern verpackt sind.  3. Stelle die Logistikkette durch die Beauftragung von spezialisierten U |
| de | `Logzeilen` |  | de-agent-fehlersuche | 4402 | lbaren Absturz. 2. Untersuche die letzten 50 Logzeilen im Systemjournal (`journalctl -u <dienst> -n |
| de | `Lokation` |  | de-prose-plakat | 4402 | e bloße Ankündigung eines Termins oder einer Lokation, denn sie fungiert als die erste und oft ent |
| de | `Lokationen` |  | de-agent-umzug | 4402 | agen im digitalen Findmittel exakt die neuen Lokationen anzeigen.  8. Überprüfe die Vollständigkeit  |
| de | `Malereiausstellung` |  | de-prose-plakat | 4403 |  zurückgreifen, die für die Werbung zu einer Malereiausstellung typisch ist, da die kunstgewerblichen Objekt |
| de | `Maschinisierung` |  | de-prose-buchbinderei | 4402 | ierten, um die Seiten glatt zu halten. Diese Maschinisierung war jedoch kein plötzlicher Bruch, vielmehr  |
| de | `Materialitäten` |  | de-prose-plakat | 4402 | uge, zwischen der haptischen Wahrnehmung der Materialitäten und der kognitiven Aufnahme der historischen |
| de | `Megal` |  | de-prose-wasserwerk | 4401 |  Gegensatz zu den weitverzweigten Netzen der Megal cities oder den autarken Kleinstädten, in de |
| de | `Meisterhänd` |  | de-prose-buchbinderei | 4403 | ptik nicht mehr die Spur einer individuellen Meisterhänd, sondern die glatte, einheitliche Oberfläche |
| de | `Membrantechnologie` |  | de-prose-wasserwerk | 4402 | fernen. Ein zentraler Aspekt ist hierbei die Membrantechnologie, die zunehmend eingesetzt wird, um eine höhe |
| de | `Mineralisierungsniveau` |  | de-prose-wasserwerk | 4403 | tschland chemisch bereits ein sehr niedriges Mineralisierungsniveau aufweist und dadurch als besonders weich ers |
| de | `Mint` | loanword | lit-farbpalette | 4402 | niert wird. Dieses sanfte, beinahe neongrüne Mint-Ton dient nicht nur als passive Fläche, sond |
| de | `Mischunsg` |  | lit-farbpalette | 4402 | stufen ist nicht möglich, da die spezifische Mischunsg der vier Hexcodes #D7FFE0, #050505, #1B3A5F  |
| de | `Nachfragerung` |  | de-prose-wasserwerk | 4402 | häufig im Spannungsfeld zwischen zunehmender Nachfragerung und den Grenzen ihrer bestehenden technische |
| de | `Nahansicht` |  | de-prose-plakat | 4402 | n haben, die sich zwischen der Fern- und der Nahansicht ergibt. Aus der Entfernung, wenn das Plakat  |
| de | `Ockerrot` |  | de-prose-plakat | 4401 | e Profil der Sammlung trägt, etwa ein tiefes Ockerrot für die Romantik der Form oder ein stumpfes  |
| de | `On` | loanword | de-prose-speicher | 4401 | ie Größe des L2-Caches oder gar der gesamten On-D Die-Cache-Kapazität übersteigt, bricht die |
| de | `Ornamentischen` |  | de-prose-plakat | 4401 | ordnung der Buchstaben muss den Rhythmus des Ornamentischen nachvollziehen; wo das Bild dicht und verwor |
| de | `Overheads` | loanword | de-prose-speicher | 4402 | sondern auch die Kontrolllogik und Protokoll-Overheads der Systeminterconnects nach sich. Diese Lat |
| de | `Pantone` |  | lit-farbpalette | 4402 | um Problem werden können. Die Verwendung von Pantone-äquivalenten in den Druckvorstufen ist nicht |
| de | `Papierweiß` |  | lit-farbpalette | 4403 | cken könnte. Die Fläche #D7FFE0 muss auf dem Papierweiß neutral erscheinen und darf bei der Auflage  |
| de | `Patterns` | loanword | de-agent-fehlersuche | 4403 | endungs-logs des Dienstes auf wiederkehrende Patterns bei den nicht beantworteten Anfragen, wie et |
| de | `Pigmentanteil` |  | lit-farbpalette | 4402 | #1B3A5F auftritt. Der Akzent #FFB347, dessen Pigmentanteil im Druck sehr hoch liegt, neigt bei falscher |
| de | `Prefetching` |  | de-prose-speicher | 4401 | n und Software-Optimierungen, etwa durch das Prefetching, unterstützt. Der Prozessor, oder vielmehr d |
| de | `Primärfarbigkeit` |  | de-prose-plakat | 4403 | taldruck eine bewusste Abkehr von der reinen Primärfarbigkeit bedeutet. Statt reiner Cyan- oder Magenta-Fl |
| de | `Prozeß` |  | de-prose-buchbinderei | 4401 |  mechanische Stanzmaschinen. Doch war dieser Prozeß kein einfaches Ersetzen, denn die Maschine s |
| de | `Prozeßsschritte` |  | de-prose-wasserwerk | 4403 | r die Haushaltszwecke sicherzustellen. Diese Prozeßsschritte finden in modernen Zentren der Trinkwasserau |
| de | `Prunksilber` |  | de-prose-plakat | 4401 | ngenden Zunftwesen. Ein Plakat, das etwa ein Prunksilber mit filigraner Arbeit zeigt, muss daher nich |
| de | `Prägemustern` |  | de-prose-buchbinderei | 4403 | Buch wurde an den Kanten, den Falzen und den Prägemustern erkannt. Mit dem Aufkommen der industriellen |
| de | `Prägetierung` |  | de-prose-buchbinderei | 4403 | ge für den modernen Einband. Die mechanische Prägetierung von Buchstäbchen und ornamentale Blinddrücke |
| de | `Querschnittsverlust` |  | de-prose-wasserwerk | 4402 | in starkes Indiz für eine Leckage oder einen Querschnittsverlust durch Korrosion. Gleichzeitig muss die Überw |
| de | `Rahmenzierung` |  | lit-farbpalette | 4402 | die Harmonie mit #050505 aufbreicht. Für die Rahmenzierung muss der Drucker sicherstellen, dass die Kan |
| de | `Rahmungselement` |  | lit-farbpalette | 4402 | nd scharf und kontrastreich erscheinen.  Das Rahmungselement, ein tiefes, ernstes Dunkelblau mit dem Code |
| de | `Rebootet` |  | de-agent-fehlersuche | 4403 | unterstützt eine Regression als Ursache. 10. Rebootet den Server oder den betroffenen Dienst im Wa |
| de | `Reduktanten` |  | de-prose-wasserwerk | 4403 | uftsauerstoff gebracht wird, wodurch gelöste Reduktanten, insbesondere Fe(II)-Ionen, in die unlöslich |
| de | `Regalplatz` |  | de-prose-buchbinderei | 4401 | n dem Schutz des Blocks, der Werbung und dem Regalplatz. Dies führte zur Entdeckung neuer, kostengün |
| de | `Remineralisation` |  | de-prose-wasserwerk | 4402 | ise passieren lässt oder durch kontrollierte Remineralisation wieder in ein physiologisch ausgewogenes Ver |
| de | `Rendering` | loanword | de-prose-speicher | 4401 | ten Datensätze von typischen Anwendungen wie Rendering oder maschinellem Lernen zu fassen. Der exte |
| de | `Reproduktionsqualität` |  | de-prose-plakat | 4402 |  lebendigen Alltagskultur. Dabei ist auf die Reproduktionsqualität zu achten; die fotografische oder illustrati |
| de | `Schriftpanels` |  | lit-farbpalette | 4401 | 347 Akzente zu halten. Die tiefe des #050505 Schriftpanels muss die Ruhe des #D7FFE0 Raums stabilisiere |
| de | `Sehweiseweise` |  | de-prose-plakat | 4403 | der Typografie spiegelt sich die historische Sehweiseweise wider, ohne dabei in die Nostalgie einer blo |
| de | `Sendungsliste` |  | de-agent-umzug | 4403 | tändig abgeschlossen zu betrachten, wenn die Sendungsliste am Zielpunkt durchgehend bestätigt wurde.  6 |
| de | `Shared` | loanword | de-prose-speicher | 4402 | elcher bei vielen modernen Architekturen als Shared-Memory-Element fungiert, das von allen Reche |
| de | `Sicherheits` |  | de-agent-umzug | 4402 | en der Regalsysteme und die Installation der Sicherheits-technik her. Der Schritt ist abgeschlossen,  |
| de | `Stanzmaschinen` |  | de-prose-buchbinderei | 4401 | r Draht- oder Fadenheftung durch mechanische Stanzmaschinen. Doch war dieser Prozeß kein einfaches Erset |
| de | `Systemd` |  | de-agent-fehlersuche | 4401 | ), um zu beobachten, ob der Fehler auch ohne Systemd-Kontext und Hintergrund-Prozesse auftritt. |
| de | `Systemlogs` |  | de-agent-fehlersuche | 4403 | ienst-Status und die letzten Einträge in den Systemlogs, um festzustellen, ob der Prozess noch läuft |
| de | `Thread` | loanword | de-prose-speicher | 4403 | e von parallelen Threads ausführen. Wenn ein Thread auf die Antwort eines L3- oder externen Spei |
| de | `Threads` | loanword | de-prose-speicher | 4403 | ers effektiv, da sie Tausende von parallelen Threads ausführen. Wenn ein Thread auf die Antwort e |
| de | `Tiling` | loanword | de-prose-speicher | 4401 | iederverwendung maximieren, etwa durch Block-Tiling oder durch die Aufteilung großer Probleme in |
| de | `Timeout` | loanword | de-agent-fehlersuche | 4402 | sdatei auf Einstellungen für Worker-Threads, Timeout-Werte oder Verbindungs-Pools. Eine zu niedri |
| de | `Timeouts` | loanword | de-agent-fehlersuche | 4402 | ), um nach spezifischen Fehlermeldungen oder Timeouts zu suchen. Regelmäßige Einträge, die den Tur |
| de | `Transportk` |  | de-agent-umzug | 4401 |  Bestandsliste werden Gefährdungspotenziale, Transportk Prioritäten für besonders schutzbedürftige o |
| de | `Transportlayer` |  | de-agent-fehlersuche | 4403 |  `curl`, um festzustellen, ob Pakete auf dem Transportlayer verloren gehen; eine hohe Paketverlustquote  |
| de | `Trübenwerte` |  | de-prose-wasserwerk | 4403 | ch vielmehr in einem langsamem Ansteigen der Trübenwerte in den entnommenen Proben, die in den Netzen |
| de | `Trübungsstoffe` |  | de-prose-wasserwerk | 4402 | asser aufwändigere Verfahren durchlaufen, um Trübungsstoffe, organische Fraktion und potenzielle Pathoge |
| de | `Unwiederbringbarkeit` |  | de-prose-buchbinderei | 4401 | an Handwerk verlor, war das Einzelstück, die Unwiederbringbarkeit, der Glanz des Besonderen. Das, was erwarb,  |
| de | `Variantentyp` |  | de-prose-speicher | 4401 | auf der Karte, meist ein sehr schneller GDDR-Variantentyp, bietet zwar mehr Kapazität, seine Bandbreit |
| de | `Veranderung` |  | de-prose-buchbinderei | 4403 | ntraler Aspekt dieser Entwicklung lag in der Veranderung der Werkstoffe und ihrer Verarbeitungsweise. |
| de | `Verbindungs` |  | de-agent-fehlersuche | 4402 | ungen für Worker-Threads, Timeout-Werte oder Verbindungs-Pools. Eine zu niedrige Anzahl an Concurrent |
| de | `Versicherungspolizzen` |  | de-agent-umzug | 4402 | ienstleistern vertraglich vereinbart und die Versicherungspolizzen unterschrieben sind.  4. Führe einen Testlau |
| de | `Verzeichnisere` |  | de-agent-umzug | 4402 | meter auf den Sollwert eingestellt sind.  7. Verzeichnisere die Bestände im neuen Räum, indem die alten  |
| de | `Viridian` |  | de-prose-plakat | 4401 |  für die Romantik der Form oder ein stumpfes Viridian für die beginnende Moderne. Die Farbe muss s |
| de | `Widerklang` |  | lit-farbpalette | 4401 | ones für die Konturierung steht in bewusstem Widerklang zur warmen Grundstimmung, erzeugt eine Spann |
| de | `Worker` | loanword | de-agent-fehlersuche | 4402 | ie Konfigurationsdatei auf Einstellungen für Worker-Threads, Timeout-Werte oder Verbindungs-Pool |
| de | `Workloads` | loanword | de-prose-speicher | 4402 | erarchitekturen und die Effizienz von Grafik-Workloads, wobei die Überwindung dieser Schranke eine  |
| de | `ZellstoffPapier` |  | de-prose-buchbinderei | 4403 |  Die Entdeckung und industrielle Nutzung von ZellstoffPapier, die Entwicklung des leimgebundenen Pappdeck |
| de | `Zweidimensionalität` |  | de-prose-plakat | 4401 | s die Dreidimensionalität des Objekts in die Zweidimensionalität der Fläche bannen, wobei die Perspektive so  |
| de | `aufbreicht` |  | lit-farbpalette | 4402 | alten ist, da sonst die Harmonie mit #050505 aufbreicht. Für die Rahmenzierung muss der Drucker sich |
| de | `aurahafte` |  | de-prose-buchbinderei | 4401 | assenware benötigt wurde, andererseits seine aurahafte Einzigartigkeit, getragen durch den handgebu |
| de | `ausgewogenenes` |  | de-prose-plakat | 4402 | tfläche den Raum für die Interpretation. Ein ausgewogenenes Verhältnis, das etwa den Bildmotiv und den T |
| de | `bakterielleische` |  | de-prose-wasserwerk | 4403 | blematisch erwiesen, da das aus Nitrat durch bakterielleische Denitrifikation im Untergrund gebildete Ammo |
| de | `begannte` |  | de-prose-buchbinderei | 4403 | r die Praxis am Werk weitergegeben wurde, so begannte sich die formale Berufsausbildung zu institu |
| de | `bewaisen` |  | lit-farbpalette | 4402 | die Authentizität gegenüber dem Kurator zu w bewaisen.  Bei einer Auflage von 4800 Exemplaren darf |
| de | `cache` | loanword | de-prose-speicher | 4401 |  die Aufteilung großer Probleme in kleinere, cache-kompatiblene Teile.  Fazitierend lässt sich  |
| de | `distribuierte` |  | de-agent-fehlersuche | 4403 | n Instanz deutet auf eine spezifische, nicht-distribuierte Ursache auf dem ersten Rechner hin. 8. Überp |
| de | `einladene` |  | lit-farbpalette | 4401 | se spezifische Nuance wurde gewählt, um eine einladene, luftige Grundstimmung zu erzeugen, die den  |
| de | `failed` | loanword | de-agent-fehlersuche | 4402 | ei gemeldet ist. Ein Status „inactive“ oder „failed“ widerlegt die Hypothese eines stabilen Betr |
| de | `fällbare` |  | de-prose-wasserwerk | 4403 | nsbesondere Fe(II)-Ionen, in die unlösliche, fällbare Fe(III)-Form überführt und an Sandfiltern zu |
| de | `gefaltene` |  | de-prose-plakat | 4403 |  patinierte Bronze einer Uhrgehäuse oder die gefaltene Seide eines Wandbehanges reproduziert wird,  |
| de | `gleichbleiben` |  | lit-farbpalette | 4403 | Kontrastwerte müssen bei den 4800 Exemplaren gleichbleiben; die Qualität der Druckplatten muss so hoch  |
| de | `gleichf` |  | de-prose-buchbinderei | 4401 | cht mehr als Einzelstück galten, sonbern als gleichf Massenprodukt. Damit wurde der Einband zum V |
| de | `gliediert` |  | lit-farbpalette | 4403 | ert wurde. Dieses tiefe, marineblaue Element gliediert das Blatt visuell und bündelt die einzelnen  |
| de | `halbledernem` |  | de-prose-buchbinderei | 4401 |  klassische Einband, meist aus rohem Leinen, halbledernem Pergament oder ganzem Kalbleder, war teuer,  |
| de | `herabzubrechen` |  | de-prose-plakat | 4401 | f ein rechteckiges Feld von begrenzter Größe herabzubrechen hat, ohne dass dabei die charakteristische S |
| de | `hochparallelen` |  | de-prose-speicher | 4401 | chenarchitektur, insbesondere im Bereich der hochparallelen Verarbeitung, hat eine fundamentale Verschie |
| de | `hände` |  | de-agent-umzug | 4403 |  vollständig, säubere die Räumlichkeiten und hände sie in einem Protokoll an die Nachfolgeverwa |
| de | `informationsebenen` |  | lit-farbpalette | 4402 | elle Trennlinie und ordnet die verschiedenen informationsebenen, während sie gleichzeitig eine Brücke zwisch |
| de | `inkl` |  | de-agent-umzug | 4401 |  Objekte festgelegt und ein Transportkonzept inkl. .  der benötigten Spezialverpackungen erste |
| de | `kupferstichige` |  | de-prose-plakat | 4402 | ine feingliedrige Serifenschrift, die an die kupferstichige Reproduktion früher Kataloge erinnert, sofer |
| de | `kühligen` |  | lit-farbpalette | 4401 | Dynamik verleiht. Er bricht die Dominanz des kühligen Blaus auf und setzt energische Punkte auf da |
| de | `leimgebundenen` |  | de-prose-buchbinderei | 4403 | ung von ZellstoffPapier, die Entwicklung des leimgebundenen Pappdeckels und die Vorproduktion von Gewebe |
| de | `logs` | loanword | de-agent-fehlersuche | 4403 | rkinstabilität. 4. Untersuche die Anwendungs-logs des Dienstes auf wiederkehrende Patterns bei |
| de | `mechanischenischen` |  | de-prose-buchbinderei | 4401 | oßen Wechsel von handwerklichen Techniken zu mechanischenischen Abläufen hinausging. In der Mitte des neunze |
| de | `neuntehnte` |  | de-prose-plakat | 4403 | mes suggeriert.  Bei der Bildauswahl für das neuntehnte Jahrhundert gilt es, die spezifische Komplex |
| de | `neuntehnten` |  | de-prose-plakat | 4403 | ten. Die Schriftarten, die für die Anmut der neuntehnten Jahrhunderts charakteristisch sind, tragen i |
| de | `nuance` | loanword | de-prose-plakat | 4402 | ren. Vielmehr empfiehlt sich eine gedämpfte, nuance-reiche Palette, die an den Papierfarbton alt |
| de | `offsettechnischen` |  | de-prose-plakat | 4403 | leisatzes erinnern, müssen im digitalen oder offsettechnischen Druck so gesetzt werden, dass ihre charakter |
| de | `protokollistisch` |  | de-agent-umzug | 4402 | stand abtransportiert und die Gebäudeleerung protokollistisch festgehalten wird. Der Vorgang ist abgeschlo |
| de | `riß` |  | de-prose-buchbinderei | 4403 |  Maschine die Kontrolle schrittweise an sich riß.  Ein zentraler Aspekt dieser Entwicklung la |
| de | `schwärzig` |  | lit-farbpalette | 4403 | ex. Das Grün #D7FFE0 ist im CMYK-Modell eher schwärzig und lässt den blauen Kanal kaum aus, was die |
| de | `serifenischen` |  | de-prose-plakat | 4403 | derts charakteristisch sind, tragen in ihren serifenischen Übergängen und ihrer variablen Strichstärke  |
| de | `signalhafte` |  | lit-farbpalette | 4403 |  der Akzent #FFB347. Hier kommt die wärmere, signalhafte Komponente zur Geltung. #FFB347 wird punktue |
| de | `sonbern` |  | de-prose-buchbinderei | 4401 | ände, die nicht mehr als Einzelstück galten, sonbern als gleichf Massenprodukt. Damit wurde der E |
| de | `technik` |  | de-agent-umzug | 4402 | systeme und die Installation der Sicherheits-technik her. Der Schritt ist abgeschlossen, wenn die |
| de | `texturale` |  | de-prose-plakat | 4403 | dern durch ihre räumliche Anordnung und ihre texturale Dichte bereits vor dem Betreten des Ausstell |
| de | `timeouts` | loanword | de-agent-fehlersuche | 4403 | i den nicht beantworteten Anfragen, wie etwa timeouts oder fehlende Antworten; das Fehlen von Eint |
| de | `transportfeste` |  | de-agent-umzug | 4403 | gilen Bestände sowie digitale Datenträger in transportfeste, beschriftete Container. Der Zustand ist err |
| de | `umverpackt` |  | de-agent-umzug | 4402 | ervatorisch gesichert, in neue Archivkartons umverpackt und mit eindeutigen Umzugsetiketten versehen |
| de | `ungestrichenem` |  | lit-farbpalette | 4403 | ärksten auf das Trägermaterial reagiert. Auf ungestrichenem Recyclingpapier, das oft mit Plakaten dieser |
| de | `unänderlich` |  | lit-farbpalette | 4401 | ge-v3.2.1.svg – enthält alle diese Werte als unänderlich, sodass die Fassung v3.2.1 als allein maßgeb |
| de | `zartlich` |  | lit-farbpalette | 4401 | rund, der in dem sanften, beinahe schon fast zartlich wirkenden Ton #D7FFE0 gehalten ist. Diese sp |
| en | `acidification` |  | en-prose-archive | 4401 | slow oxidation of paper is retarded, and the acidification that turns cheap, wood-pulp newsprint into d |
| en | `amidst` |  | en-prose-foundry | 4402 | ogether. It was a moment of profound silence amidst the surrounding roar, a breath held in antic |
| en | `andeach` |  | en-prose-archive | 4403 |  maintenance, updating with each acquisition andeach loss, ensuring that the finding aid does not |
| en | `andsulfur` |  | en-prose-archive | 4403 | istent, filtering out atmospheric pollutants andsulfur dioxide and nitrous oxides that act as corro |
| en | `backends` |  | lit-releasenote | 4402 |  loss when moving between different hardware backends. The previous version of the release, also t |
| en | `booklice` |  | en-prose-archive | 4401 | into ghosts. Insects, such as silverfish and booklice, seek out the starch in bindings and the glu |
| en | `cardinality` |  | lit-releasenote | 4401 | proximation; rather, it represents the exact cardinality of the dataset segment included in this shar |
| en | `checksums` |  | lit-releasenote | 4401 | egrity of build 0xB7A31F is verified through checksums and structural validators. |
| en | `chromophoric` |  | en-prose-archive | 4401 | adiation, is a potent bleach that breaks the chromophoric bonds of ink and pigment, causing text to fa |
| en | `clamour` |  | en-prose-foundry | 4401 | ieter, more refined space separated from the clamour of the moulding floors by a heavy wooden doo |
| en | `clays` |  | en-prose-foundry | 4402 | a carefully engineered composite, mixed with clays and binders to achieve the perfect balance b |
| en | `codebase` |  | lit-releasenote | 4402 | ical reference point for this version of the codebase. By adhering to this tag, operators can ensu |
| en | `compressibility` |  | en-prose-foundry | 4403 |  sand needed to possess the right balance of compressibility and porosity, properties that seem contradic |
| en | `convolutional` |  | lit-releasenote | 4401 | lays a pivotal role in the state space model convolutional layers, specifically handling the alpha para |
| en | `de` | loanword | en-prose-foundry | 4401 |  on the size of the piece, the castings were de-shacked from the moulds. This was the work o |
| en | `deaccession` |  | en-prose-archive | 4403 | other, more durable sources. This process of deaccession is not an act of destruction but a refinemen |
| en | `deserialization` |  | lit-releasenote | 4401 | serialization for build 0xB7A31F could cause deserialization failures. Therefore, the integrity of build  |
| en | `dimensionality` |  | lit-releasenote | 4402 | the training pipelines, which expect a fixed dimensionality. The tag release-2026-09-18a mandates this c |
| en | `duplicative` |  | en-prose-archive | 4401 |  are only destroyed if they are demonstrably duplicative or administrative to a degree that serves no |
| en | `evidential` |  | en-prose-archive | 4401 | gic of survival. This involves assessing the evidential weight of the paper. A routine purchase orde |
| en | `evidentiary` |  | en-prose-archive | 4403 | administrative notices, which hold no unique evidentiary value, are weeded out alongside routine invo |
| en | `failer` |  | en-prose-foundry | 4401 |  the boundary between the functional and the failer. |
| en | `fibered` |  | en-prose-archive | 4401 |  Japanese tissue paper, a thin, strong, long-fibered paper that is chemically neutral and physica |
| en | `fireclay` |  | en-prose-foundry | 4401 | nstructed of steel and lined with refractory fireclay, stood ready to devour a charge of scrap iro |
| en | `grey` |  | en-prose-foundry | 4402 | re brushing or grinding, revealing the cold, grey surface of the iron.  Finally, the finished  |
| en | `hemicellulose` |  | en-prose-archive | 4402 | r is, at its core, a composite of cellulose, hemicellulose, and lignin, materials that are inherently p |
| en | `hygroscopic` |  | en-prose-archive | 4402 | his stability is critical because paper is a hygroscopic material, meaning it absorbs and releases wa |
| en | `ingate` |  | en-prose-foundry | 4402 |  cut into the sand, guided the flow from the ingate, the entry point, into the cavity of the mou |
| en | `isreflects` |  | en-prose-archive | 4401 |  the fold. This meticulous choice of housing isreflects the understanding that the environment of su |
| en | `misruns` |  | en-prose-foundry | 4403 | He looked for cold shuts, for blowholes, for misruns, and for shrinkage defects. He checked for c |
| en | `mould` |  | en-prose-foundry | 4401 | e pattern from the sand without breaking the mould. These patterns were the silent architects o |
| en | `moulders` |  | en-prose-foundry | 4401 | ed entirely on the the clay content, and the moulders spent as much time tempering their sand as t |
| en | `moulding` |  | en-prose-foundry | 4401 | ined space separated from the clamour of the moulding floors by a heavy wooden door. Here, the pat |
| en | `moulds` |  | en-prose-foundry | 4401 | tempering their sand as they did forming the moulds. Too much water caused steam holes and bloat |
| en | `nitrous` |  | en-prose-archive | 4403 | atmospheric pollutants andsulfur dioxide and nitrous oxides that act as corrosive agents on acid- |
| en | `ofof` |  | en-prose-foundry | 4402 | mered with a dull, orange light. The tapping ofof the furnace was a dangerous and precise ritu |
| en | `ofsizing` |  | en-prose-archive | 4401 | s with mechanical wood pulp and an abundance ofsizing agents and fillers that slowly hydrolyze the |
| en | `orthe` |  | en-prose-archive | 4403 | . The finding aid, the descriptive inventory orthe calendar, is the intellectual map that trans |
| en | `papermakers` |  | en-prose-archive | 4401 | hemselves. Before the mid-twentieth century, papermakers replaced rags with mechanical wood pulp and  |
| en | `pourer` |  | en-prose-foundry | 4401 | e workshop floor to the prepared moulds. The pourer, a usually a senior man with nerves of steel |
| en | `pre` |  | en-prose-archive | 4403 | eenth-century waterworks or the minutes of a pre-war council meeting remain legible to a hist |
| en | `prefetching` |  | lit-releasenote | 4402 |  feature that that facilitates better memory prefetching behavior during model execution.  Operators  |
| en | `rammer` |  | en-prose-foundry | 4401 | ey packed the sand around the pattern with a rammer, striking the face of the sand until it achi |
| en | `rammers` |  | en-prose-foundry | 4402 | rvers.  The ramming was performed using hand rammers, heavy steel tools with broad, circular head |
| en | `recalibrated` |  | lit-releasenote | 4401 | and values within ssm_conv1d_alpha have been recalibrated to align with the broader architectural opti |
| en | `sprues` |  | en-prose-foundry | 4403 | lid. The metal flowed through the gates, the sprues, and the runners, navigating the channels cu |
| en | `theit` |  | en-prose-archive | 4401 | e control and conservation necessary to keep theit alive. Therefore, the archivist acts as a cu |
| en | `thethe` |  | en-prose-archive | 4403 | es, feeding on the gelatin sizing agents and thethe starch pastes that once held these documents |
| en | `tuyeres` |  | en-prose-foundry | 4402 | A jet of blast air was forced into the lower tuyeres, igniting the coke and creating a roaring, i |
| en | `unforgivingly` |  | en-prose-foundry | 4403 | poorly joined seam, would be faithfully, and unforgivingly, multiplied in the final metal form, a testa |
| en | `unindexed` |  | en-prose-archive | 4401 |  guides the researcher through the forest of unindexed boxes. In a long-term view, where the origin |
| en | `unshielded` |  | en-prose-archive | 4403 | ght, particularly ultraviolet radiation from unshielded skylights or fluorescent tubes, causes photo |
| en | `validators` |  | lit-releasenote | 4401 | is verified through checksums and structural validators. |

## near-miss literals

| prompt | seed | demanded | seen | distance | count |
|---|---|---|---|---|---|
| lit-farbpalette | 4401 | `/srv/plakat/2026/vorlage-v3.2.1.svg` | `/srv/plakat/2026/vorlage-v3.201.svg` | 1 | 1 |
| lit-farbpalette | 4402 | `/srv/plakat/2026/vorlage-v3.2.1.svg` | `/srv/plakat/2026/vorlage-v3.201.svg` | 1 | 4 |
| lit-farbpalette | 4402 | `v3.2.1` | `v3.201` | 1 | 4 |

