# T0-B Amendment 007 — R1 Runner Run-ID Prefix Fix

R1 runner seal (6391036) had inconsistent run_id prefixes between make_gov_row and the resume loop, causing resume to double rows. Fixed to use consistent `t0b1r|` prefix everywhere.
