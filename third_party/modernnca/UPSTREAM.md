# ModernNCA — Vendored Official Source

- Repository: https://github.com/qile2000/LAMDA-TALENT
- Commit: 08301d670a7c854bcf3a73298763484ba58eecdb
- Retrieved: 2026-07-14
- License: MIT (see LICENSE)
- Modified: NO (byte-identical)

## Vendored files
| local | upstream | byte-identical |
|-------|----------|----------------|
| modernNCA.py | TALENT/model/models/modernNCA.py | YES |
| tabr_utils.py | TALENT/model/lib/tabr/utils.py | YES |
| TALENT/model/lib/tabr/utils.py | (same) | YES (import-path shim) |

The `TALENT/` subtree is a shim so the unmodified `modernNCA.py`'s
`from TALENT.model.lib.tabr.utils import make_module` resolves to the
byte-identical vendored utils. No core nn.Module code was altered.
