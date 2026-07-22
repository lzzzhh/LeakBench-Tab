# Full T0-B1 Execution Preflight

5,500 keys, 803,000 downstream rows, 792,000 selection rows, zero planned
failure rows, and 64 shards (85-86 keys each).

- Mode: `production`
- Scientific freeze: `ff347b0657e8faf5d0ec1a4ca283185ffe2f5845`
- Execution contract: `v1`
- Execution-tool seal: `f33ca0ab194c335eff226002dbffc416d203f254`
- Plan-manifest SHA-256: `b5581c6b26238ff792ed746d04d78f464b55a2a8a2007ab6a9acdf6813945812`
- Run-plan SHA-256: `4fdfe21c3861ef45fcab693d8af6e64de27bd92e972af077bee00d5b84da3aef`
- Strict plan validation: `PASS`
- Shared preflight validator: `PASS` (`Errors: 0`)
- Production runner validate-only, shard 0: `PASS`
- One-key production canary: `PASS` (146/146 planned runs)
- Canary shard admission: `PASS`
- Canary strict global merge: `PASS`
- Canary R10d full-result validation: `PASS`

The execution-tool seal points to the code-only declared-input and runtime-
binding closure commit. The plan and receipt were regenerated afterwards and
bind that seal; no Full-B1 outcome was generated or observed during this
correction.
