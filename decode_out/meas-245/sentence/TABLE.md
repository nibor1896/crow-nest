| arm | K | rounds ok | no call (95 % CI) | corrupt calls | schema-wrong | budget closed | reasoning chunks median [min-max] | s/round median [min-max] |
|---|---|---|---|---|---|---|---|---|
| SA1024 | 10 | 33/33 | 0 (0.0, 0.104) | 0 | 0 | 19 | 1042 [162-1042] | 19.1 [5.6-24.5] |
| SB1024 | 10 | 33/33 | 0 (0.0, 0.104) | 0 | 0 | 19 | 1038 [162-1038] | 18.6 [5.6-21.5] |
| SA1024 | 45 | 33/33 | 4 (0.048, 0.273) | 0 | 0 | 30 | 1042 [296-1042] | 22.0 [6.5-29.9] |
| SB1024 | 45 | 33/33 | 0 (0.0, 0.104) | 0 | 0 | 30 | 1038 [296-1038] | 21.2 [6.6-26.6] |

| arm | rounds | failed | no call (95 % CI) | corrupt | schema-wrong | budget closed | s/round median |
|---|---|---|---|---|---|---|---|
| SA1024 | 66 | 0 | 4 (0.024, 0.146) | 0 | 0 | 49 | 20.6 |
| SB1024 | 66 | 0 | 0 (0.0, 0.055) | 0 | 0 | 49 | 20.0 |

closed rounds only: no call A 4/49, B 0/49, one-sided Fisher p (B fewer) 0.0587

PREREG decision rule: {'a_fewer_no_call': True, 'b_corrupt_schema': True, 'c_wall_clock': True} -> sentence B is PROPOSED (robin decides; one constant for every model)

determinism vs the budget series' A1024: 18 of 18 rounds identical (finish, calls, reasoning chunks)
