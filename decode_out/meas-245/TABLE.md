| arm | K | rounds ok | no call (95 % CI) | corrupt calls | schema-wrong | budget closed | reasoning chunks median [min-max] | s/round median [min-max] |
|---|---|---|---|---|---|---|---|---|
| A1024 | 10 | 9/9 | 0 (0.0, 0.299) | 0 | 0 | 6 | 1042 [241-1042] | 19.9 [7.1-21.6] |
| B2048 | 10 | 9/9 | 0 (0.0, 0.299) | 0 | 0 | 4 | 1604 [241-2066] | 29.9 [6.9-36.7] |
| A1024 | 45 | 9/9 | 1 (0.02, 0.435) | 0 | 0 | 9 | 1042 [1042-1042] | 21.5 [20.6-31.1] |
| B2048 | 45 | 9/9 | 0 (0.0, 0.299) | 0 | 0 | 5 | 2066 [1674-2066] | 42.7 [34.7-46.4] |

| arm | rounds | failed | no call (95 % CI) | corrupt | schema-wrong | budget closed | s/round median |
|---|---|---|---|---|---|---|---|
| A1024 | 18 | 0 | 1 (0.01, 0.258) | 0 | 0 | 15 | 20.9 |
| B2048 | 18 | 0 | 0 (0.0, 0.176) | 0 | 0 | 9 | 36.35 |

PREREG decision rule: {'a_no_call': True, 'b_corrupt': True, 'c_wall_clock': False} -> 1024 stays
