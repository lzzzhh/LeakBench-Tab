# Generated evidence macros

`result_macros.tex` is generated only from the complete confirmatory release:

```bash
python source_data/generate_result_macros.py
```

Do not hand-edit or seed this directory with pilot values.  Without a valid
generated file, both PDFs compile with an explicit **RESULTS BLOCKED** banner
and all empirical values remain `PENDING`.

The release validator always rebuilds this file, checks its embedded
`paper_claims.json` SHA-256, and records the macro-file SHA-256.  The artifact
builder rejects a stale claims file, stale validation report, or stale macros.
