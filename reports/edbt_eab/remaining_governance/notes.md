# Working Notes

This file records evidence discovered during implementation. It intentionally begins before protocol freezing or experiment execution.

## Pre-run inventory

- Five local natural datasets are present and bound by the existing v2 train-fitted-category freeze: Bank Marketing, Lending Club, BTS Flights, Chicago Food, and NYC311.
- Natural retained shapes range from 16 to 140 features; every case contains both legitimate and boundary-invalid fields.
- The frozen 5,500-key synthetic bundle manifest is complete.
- M09 is the only mechanism that expands one semantic source field into multiple encoded columns: eight complete one-hot columns. Other mechanisms have identity semantic-to-encoded group mappings under the current frozen representation; M10's two injected columns have distinct roles and remain distinct groups.
- The primary natural and semantic sensitivity budget is 20%, with 20 governance seeds and LR as the directly comparable learner.
