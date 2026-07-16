# Stage C — TP-shard-aware weight sync (implementation spec)

**Status:** trainer seam landed (config knob + guard); Pie-side changes specified below,
**to be implemented and validated on NVLink hardware** (TP>1 over no-P2P is pathological —
this box reports `CNS` for every GPU pair). Nothing here is validated yet.

## TL;DR — this is smaller than it looks

The hard part (per-rank weight sharding) **already exists**:

- The vllm driver **already spawns `tp_degree` worker processes per DP replica** (leader +
  followers) via `calculate_topology` + mp-spawn — `driver/vllm/.../_bridge/_launcher.py:196-234`,
  `_bridge/worker.py:52-210`.
- vLLM's **own `weight_loader` already narrows a full tensor to each rank's shard** once the
  TP process group is initialized — `vllm/model_executor/layers/linear.py` (`ColumnParallelLinear.weight_loader:537`,
  `QKVParallelLinear.weight_loader:696`, using `get_tensor_model_parallel_rank/world_size`).
  `Qwen2ForCausalLM.load_weights` (`vllm/model_executor/models/qwen2.py:428`) maps HF names →
  fused params and calls those loaders. **`model.load_weights` does NOT assume a full model** —
  each rank auto-narrows.

**The one gap:** the weight-sync path stops at the group **leader**. `engine.update_weights`
is called only on `tp_rank==0` and is never broadcast to followers, so ranks `1..tp-1` keep
**stale shards**. The forward path and `load_adapter` already solve this exact problem — we
mirror their broadcast.

## Current weight-sync trace (branch `features/RL`) — verified

```
trainer: reduce_tensor(full bf16 param) → pickled CUDA-IPC handle
  client.update_weights(handles, device_idx)            client/python/.../client.py:316
  → ClientMessage::UpdateWeights{corr_id,device_idx,handles}  client/rust/src/message.rs:37
  → server dispatch                                     runtime/src/server.rs:690
  → handle_update_weights → driver::update_weights(device_idx, handles)  handler.rs:327-340
  → get_channel(device_idx) → shmem "/pie_shmem_g{idx}" runtime/src/driver/ops.rs:222, channel.rs:20
  → LEADER _shmem_loop: _METHOD_UPDATE_WEIGHTS          driver/vllm/.../_bridge/worker.py:628
        engine.update_weights(handles)   ← STOPS HERE, no broadcast
  → engine.update_weights: args[6]=dev_index (IPC remap) → model.load_weights → sync
                                                          driver/vllm/.../engine.py:711-735
```

Invariants that make this **TP-agnostic on the Rust side** (no Rust changes needed):
- `device_idx` (client) == `DriverId` (Rust) == shmem group id `/pie_shmem_g{idx}` == **one TP
  group (one DP replica)**. The Rust runtime has no notion of `tp_rank`/`tensor_parallel_size`
  (grep-confirmed: those live only in the Python drivers). `DriverSpec` (`runtime/src/driver.rs:41`)
  has no device/TP fields.
- The wire schema (`WeightHandle{name,blob}`, `UpdateWeightsRequest{handles}`,
  `driver/bridge/src/schema.rs:346-352`) is TP-agnostic.

## The changes

### 1. Pie bridge — broadcast handles to followers (primary change)

**File:** `driver/vllm/src/pie_driver_vllm/_bridge/worker.py`

- **Leader (`~line 628`, `_METHOD_UPDATE_WEIGHTS`):** when `config.world_size > 1`, broadcast the
  handles to the TP group *before* applying locally — **mirror `_handle_load_adapter_v2` (~line 583)**,
  which already does `runtime_ops.broadcast_struct({"type":"LOAD_ADAPTER", ...}, src=0)`. Use the
  same `broadcast_struct` mechanism the STEP_KWARGS / LOAD_ADAPTER paths use.
- **Follower (`_follower_loop`, lines 743-781):** add an `elif msg_type == "UPDATE_WEIGHTS":`
  branch that reconstructs `handles` from the broadcast and calls `engine.update_weights(handles)`.
  (Today the follower loop handles only STEP/STEP_KWARGS/INIT_ADAPTER/UPDATE_ADAPTER/LOAD_ADAPTER/
  SAVE_ADAPTER — there is no UPDATE_WEIGHTS case.)

After this, each rank's `engine.update_weights` → `model.load_weights` → vLLM `weight_loader`
narrows to that rank's shard automatically. **`engine.py:734` needs no change.**

### 2. The IPC-handle semantics — the one real design decision

`engine.update_weights` (`engine.py:726-731`) rebuilds the CUDA-IPC handle and sets
`args[6] = dev_index` (this rank's own GPU). Today trainer and driver **share one physical GPU**,
so the handle is same-GPU zero-copy. Under TP each follower is on a **different** GPU than the
trainer tensor's source GPU. Two ways to resolve it:

- **Option A — broadcast one full-tensor handle, followers open cross-GPU (needs NVLink/P2P).**
  Broadcast the *same* handle to every rank; each follower opens it against the **source** device
  (not `dev_index`) and reads cross-GPU over NVLink, then `load_weights` narrows. Requires
  `args[6]` to become the *source* device for followers, and P2P/NVLink to map the handle. Minimal
  trainer change (still one push per replica). **Fails on no-P2P** — which is why Stage C is
  NVLink-gated.
- **Option B — trainer publishes one full-tensor handle per GPU (keeps same-GPU IPC).** Each
  trainer rank colocated with a Pie TP worker publishes a handle for the full model on *its* GPU;
  each follower opens its **own-GPU** handle (`args[6]=dev_index`, unchanged). No cross-GPU IPC, so
  it would even work without NVLink — but it needs the full model resident on every GPU (FSDP
  `summon_full_params` already materializes it transiently) and a trainer-side fan-out that pushes
  per-GPU. More trainer work; fewer hardware constraints.

**Recommendation: start with Option A** (matches verl/vLLM's "send full tensor, shard on receive"
idiom and the broadcast pattern already in the bridge; the only new semantics is the source-device
open). Revisit Option B only if you want TP without NVLink.

### 3. Trainer side — already seamed; finish per chosen option

Already landed (commit adding `topology.resolve_tp_size` + the `train_grpo` guard):
- `pie.tensor_parallel_size` config knob (`config.yaml`), `resolve_tp_size` (`topology.py`, unit-tested).
- `train_grpo.train()` raises `NotImplementedError` for `tp_size != 1` pointing here.

To finish (once Pie-side lands):
- **Option A:** replace the guard with a normal push — no trainer fan-out needed (one push per
  replica; the bridge broadcasts inside the group). `extract_hf_state` unchanged (full tensors).
- **Option B:** add a fan-out in `sync_weights` that pushes the full tensor to each TP device index
  of the replica. Reuse Stage B's `device_idx_for_rank` to enumerate the group's device indices
  (verl's `vllm_rollout/utils.py:46-69` has the index formula:
  `tp_rank = local_rank % tp_size; dp_local_rank*tp_size + tp_rank`).

### 4. Rust — no changes for Option A

`DriverId`/`device_idx` already routes to the TP group leader; the schema is TP-agnostic; the
broadcast happens inside the Python group. **Only** if you choose to send *pre-sharded* per-rank
tensors from the client (rejected — duplicates vLLM's sharding convention) would the wire enum need
a `tp_rank` field.

## Hazards

- **Wrong slice loads silently-wrong weights.** A mis-narrowed shard leaves the engine
  coherent-but-wrong (same failure class as a name mismatch). Must validate against a full-model
  reference (below). Mitigated by *reusing vLLM's own `weight_loader`* rather than hand-rolling
  the row/column/fused-QKV/gate-up rules.
- **`tensor_parallel_size` must divide head/hidden dims** — inherit vLLM's constraint; assert early.
- **Per-rank CUDA-IPC device identity** — Option A's source-device open must target the *actual*
  device of the trainer tensor; Stage B's mapping must yield real device indices.
- **Do not enable on this box.** `tp_size=1` stays the default; the `train_grpo` guard blocks TP>1.

## Verification (on NVLink hardware)

1. **Shard-reassembly equivalence:** with `tp_size=2`, after a sync, gather each rank's shard back
   to a full tensor and assert equality with `extract_hf_state`'s full tensor for all ~338 params
   (bf16 tolerance). Directly tests the narrowing.
2. **Behavioral parity:** TP=2 engine vs TP=1 engine loaded from the same checkpoint → identical
   greedy outputs on a fixed prompt set.
3. **End-to-end:** a few GRPO steps at TP=2 tracking the TP=1 held-out heval curve
   (0.515→~0.66 band), no NaN/divergence.

## Effort estimate

- Pie bridge broadcast + follower case (worker.py): small, mirrors `_handle_load_adapter_v2`.
- IPC source-device open (engine.py Option A): small but subtle (the one thing to get exactly right).
- Trainer: Option A ≈ delete the guard; Option B ≈ a short fan-out loop.
- Rust: none (Option A).
- The bulk of the risk is **verification** (#1 shard-reassembly), not new code — because the
  sharding logic is vLLM's, already battle-tested.
