# release_v1

Audited self-play curricula: admitted problems per round and lineage with Opus verdicts and solver agreement, fully
audited fixed pools, planted premise-deletion twins (as patches), the exact judge prompts and the judge-free UMWP probe.
Read DATASHEET.md first. MANIFEST.tsv lists every file with its SHA-256, size and row count; build_release.py rebuilds
everything from the project tree.

Quick start:
    import json
    rows = [json.loads(l) for l in open("audited_pools/qwen3_4b_Base_own_a3_s0_r3.jsonl")]
    share = sum(r["opus_verdict"] != "SOLVABLE" for r in rows if r["in_audit_sample200"]) / 200
    # ids are unique only within a run: join rows across runs on problem_sha256
    # judge-free reads J-P2 / J-P3 from the released dumps: cd probe && python3 reproduce_j1.py
    # twins: cd twins && python3 rebuild_twins.py [--gsm_symbolic_dir DIR]   (needs GSM-Symbolic; see the script header);
    #        the output is a drop-in --twins_file for probe/code/c0_probe.py
