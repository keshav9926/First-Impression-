# app/agent/ — the Phase 3 analysis agent.
# A hand-rolled ReAct loop (react.py) drives three tools (tools.py) over the
# ingested content, then a schema-constrained call (report.py) turns what it
# gathered into the structured First Impression report.
