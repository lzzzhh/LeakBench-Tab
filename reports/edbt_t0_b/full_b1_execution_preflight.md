# Full T0-B1 Execution Preflight

5,500 keys, 803,000 downstream rows, 792,000 selection rows, zero planned
failure rows, and 64 shards (85-86 keys each).

- Mode: `production`
- Scientific freeze: `ff347b0657e8faf5d0ec1a4ca283185ffe2f5845`
- Execution contract: `v1`
- Execution-tool seal: `156b6e887bc97d0099ec0a700e748ae5cf561f6d`
- Plan-manifest SHA-256: `b36551fe7cce614204a8417f11ae55079339b4638c42dadd6ba7b81ac938f2f1`
- Run-plan SHA-256: `4fdfe21c3861ef45fcab693d8af6e64de27bd92e972af077bee00d5b84da3aef`
- Strict plan validation: `PASS`
- Shared preflight validator: `PASS` (`Errors: 0`)
- Production runner validate-only, shard 0: `PASS`

The execution-tool seal points to the code-only contract-closure commit. The
plan and receipt were regenerated afterwards and bind that seal; no Full-B1
outcome was generated or observed during this correction.
