# Source mapping and migration references

External projects and historical experiments are references, not runtime
 dependencies. Ideas enter IPA only through local contracts, adapters, tests and
evidence.

| Source concept | IPA boundary | Status |
|---|---|---|
| Canonical document | `contracts/`, `ipa.storage`, ingestion | current |
| Hybrid retrieval | `ipa.indexes`, future `ipa.agentic` | current/experimental |
| QueryIR/EvidenceSet/ContextPackage | `ipa.agentic` facade | experimental |
| Reporter | `ipa.reporter` | experimental capability |
| Tutor contracts | `contracts/`, `ipa.tutor` facade | contracts current, runtime planned |
| EKS | `knowledge/`, `tools/eks_mcp_server.py` | dev-time current |

No external code path is authoritative for IPA. External concepts must be
reimplemented behind the local contracts when adopted.
