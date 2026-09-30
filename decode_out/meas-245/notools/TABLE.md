A rounds 36, cut by the budget 31 (pairs), uncut 5

| prompt | seed | A kind | B kind | A finish | B finish | A s | B s | A first 80 | B first 80 |
|---|---|---|---|---|---|---|---|---|---|
| de-prose-plakat | 0 | call | call | tool_calls | tool_calls | 52.0 | 47.2 | write_file | run_command |
| de-prose-plakat | 1 | answer | call | stop | tool_calls | 45.2 | 44.8 | Das Plakat für eine Ausstellung kunstgewerblicher Sammlungen des neunzehnten Jah | write_file |
| de-prose-plakat | 2 | call | call | tool_calls | tool_calls | 44.5 | 42.2 | write_file | write_file |
| de-prose-speicher | 0 | answer | answer | stop | stop | 57.6 | 47.7 | Der Fachtext zum Thema Speicherhierarchie moderner Grafikprozessoren:  ---  Wer  | Der Fachtext ist unten. Wortzählung: rund 990 Wörter, innerhalb des geforderten  |
| de-prose-speicher | 1 | answer | call | stop | tool_calls | 43.4 | 37.0 | Die Speicherhierarchie moderner Grafikprozessoren  Ein moderner Grafikbeschleuni | write_file |
| de-prose-speicher | 2 | answer | call | stop | tool_calls | 47.8 | 41.9 | Grafikprozessoren sind seit einigen Jahren zu den leistungsfähigsten Rechenarchi | write_file |
| de-prose-buchbinderei | 0 | answer | call | stop | tool_calls | 58.5 | 40.5 | **Der Weg des Einbands: Zur Geschichte der Buchbinderei zwischen 1850 und 1930** | run_command |
| de-prose-buchbinderei | 1 | call | call | tool_calls | tool_calls | 16.5 | 17.0 | web_search | goal_set |
| de-prose-buchbinderei | 2 | call | call | tool_calls | tool_calls | 61.3 | 42.6 | write_file | write_file |
| de-prose-wasserwerk | 0 | call | call | tool_calls | tool_calls | 42.8 | 14.8 | write_file | list_dir |
| de-prose-wasserwerk | 1 | call | call | tool_calls | tool_calls | 42.0 | 52.5 | run_command | write_file |
| de-prose-wasserwerk | 2 | answer | call | stop | tool_calls | 47.9 | 44.5 | Die Wasserversorgung einer mittelgroßen Stadt ist eines der komplexesten technis | write_file |
| en-prose-foundry | 0 | answer | answer | stop | stop | 43.3 | 43.4 | A small grey iron foundry in the first decades of the twentieth century turned o | A small iron foundry of the early twentieth century ran, in its essentials, on t |
| en-prose-foundry | 1 | answer | call | stop | tool_calls | 34.6 | 47.8 | The day in a small iron foundry began before the cupola was lit, in the pattern  | run_command |
| en-prose-foundry | 2 | answer | answer | stop | stop | 41.9 | 39.5 | In a small iron foundry at the beginning of the twentieth century, the pattern s | A small iron foundry at the beginning of the twentieth century was a place where |
| en-prose-archive | 0 | answer | call | stop | tool_calls | 47.2 | 16.9 | The first act of keeping paper alive for two centuries is the building, and the  | run_command |
| en-prose-archive | 1 | answer | call | stop | tool_calls | 40.5 | 17.6 | The first decision a municipal archive makes about its paper is architectural, b | run_command |
| en-prose-archive | 2 | answer | answer | stop | stop | 47.2 | 39.6 | # Municipal Paper Preservation: Two Centuries of Legibility  A municipal archive | The question of how a municipal archive keeps its paper legible for two hundred  |
| lit-farbpalette | 0 | answer | answer | stop | stop | 39.7 | 34.9 | Für das Ausstellungsplakat sind vier Farben festgelegt, die einander nicht im We | Hier ist der Text – anschließend ein paar Stichpunkte zur Selbstkontrolle der Vo |
| lit-farbpalette | 1 | call | call | tool_calls | tool_calls | 14.9 | 14.4 | goal_set | list_dir, run_command |
| lit-farbpalette | 2 | answer | call | stop | tool_calls | 37.7 | 14.0 | Der Hintergrundton #D7FFE0 ist die stillste und zugleich die arbeitssamste der v | read_file |
| lit-releasenote | 0 | answer | answer | stop | stop | 33.7 | 27.7 | This release note documents build 0xB7A31F, carrying the tag release-2026-09-18a | CNQ4.5-M release notes, build 0xB7A31F  This note describes build 0xB7A31F, publ |
| lit-releasenote | 1 | answer | answer | stop | stop | 22.9 | 26.3 | Release note for CNQ4.5-M  CNQ4.5-M, build 0xB7A31F, ships under the tag release | Release note — CNQ4.5-M, build 0xB7A31F, tag release-2026-09-18a  CNQ4.5-M, buil |
| lit-releasenote | 2 | answer | call | stop | tool_calls | 22.2 | 27.4 | The CNQ4.5-M release lands as build 0xB7A31F, tagged release-2026-09-18a, and it | run_command |
| json-schritte | 1 | answer | call | stop | tool_calls | 16.2 | 13.8 | {   "vorhaben": "Umzug eines Archivs in ein neues Magazin",   "dauer_tage": 6,   | goal_set |
| json-schritte | 2 | call | answer | tool_calls | stop | 18.1 | 16.9 | run_command | {"vorhaben":"Umzug des Archivs in das neue Magazin","dauer_tage":14,"fertig":fal |
| de-agent-umzug | 1 | answer | answer | stop | stop | 22.8 | 22.3 | 1. Erstellen Sie eine vollständige Inventarliste aller Objekte, Kisten und Medie | 1. Erstellen Sie eine vollständige Inventarliste aller Objekte, Kisten und Medie |
| de-agent-umzug | 2 | answer | answer | stop | stop | 22.1 | 22.2 | 1. Der Archivar erfasst den gesamten Bestand mit eindeutigen Inventarnummern und | 1. Der Archivar erfasst den gesamten Bestand mit einer eindeutigen Inventarnumme |
| de-agent-fehlersuche | 0 | call | answer | tool_calls | stop | 39.4 | 30.3 | read_file | 1. Reproduziere das Muster quantitativ: Führe z. B. `curl -s -o /dev/null -w "%{ |
| de-agent-fehlersuche | 1 | answer | call | stop | tool_calls | 32.9 | 17.9 | 1. Reproduziere das Muster systematisch: Schicke in einer Schleife 20–30 Anfrage | goal_step, goal_step, goal_step, goal_step, goal_step, goal_step |
| de-agent-fehlersuche | 2 | answer | answer | stop | stop | 29.6 | 28.0 | 1. Prüfe, ob der Dienst heute Morgen neu gestartet wurde oder ob ein geplantes J | 1. Prüfe, ob der Dienst heute Morgen neu gestartet wurde oder ob ein geplanter J |

| arm | pairs | empty | mid-word | answer | call | finish length | s/round median |
|---|---|---|---|---|---|---|---|
| SA1024 | 31 | 0 | 0 | 22 | 9 | 0 | 40.5 |
| SB1024 | 31 | 0 | 0 | 12 | 19 | 0 | 30.3 |

discordant: B called where A wrote the answer 12, A called where B wrote it 2
PREREG Amendment 3 rule: {'a_empty': True, 'b_midword': True, 'c_length': True, 'd_calls': False, 'e_wall_clock': True} -> harm measured: B is not proposed as the global constant
