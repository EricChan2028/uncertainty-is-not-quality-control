"""Label-free comparison of the A4i loop-round guards with own round 4 (same proposals): admitted counts, share of admitted
proposals whose majority (counting unknown) is unknown, and the planted-tracer ratio. python3 guard_cmp.py"""
import json
B = "/home/tione/notebook/rsi_rl/dirB/runs"
for d in ("qwen3_4b_Base_loop_uv_a4_s0_r4", "qwen3_4b_Base_own_a3_s0_r4", "qwen3_4b_Base_loop_uv_a4_s1_r4", "qwen3_4b_Base_own_a3_s1_r4", "qwen3_4b_Base_loop_uv_a4_s2_r4", "qwen3_4b_Base_own_a3_s2_r4"):
    st = json.load(open(f"{B}/{d}/guard_stats.json"))
    recs = [json.loads(l) for l in open(f"{B}/{d}/guard_records.jsonl")]
    prop = [r for r in recs if r["admit"] and r["kind"] == "proposed"]
    unk = sum(r.get("abstain", 0) > r["p_mode"] for r in prop)
    ti, td = st.get("tracer_intact_admit_rate"), st.get("tracer_deleted_admit_rate")
    ratio = td / ti if ti else float("nan")
    print(f"{d[14:]:24s} admitted {st.get('n_admitted')} | proposals admitted {len(prop)} of {sum(r['kind'] == 'proposed' for r in recs)}"
          f" | unknown-majority {unk} ({100 * unk / max(1, len(prop)):.1f}%) | tracer ratio {ratio:.2f} (deleted {td:.2f}, intact {ti:.2f})")
