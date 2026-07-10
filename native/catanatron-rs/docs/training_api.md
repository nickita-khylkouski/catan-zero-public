# Neural-Net Training API

The Rust engine owns Catan rules, legal action generation, and deterministic
state transitions. Neural-network inference and training should live outside the
rules engine in Python, Torch, JAX, ONNX Runtime, or another trainer.

Recommended loop:

```text
Rust BatchEnv -> observations + legal masks -> GPU model -> masked action ids -> Rust BatchEnv.step
```

Do not put CUDA, optimizers, replay buffers, or model checkpoint loading in the
core rules engine. Catan rule stepping mutates irregular game state, dynamically
generates legal actions, and branches through prompts, trades, robber moves, and
discard/dev-card flows; it is not a fixed-shape tensor kernel. The GPU boundary
is batched model inference/training.

## BatchEnv

`catanatron_rs.BatchEnv` is the dependency-free Python binding for batched
external-policy stepping.

Constructor:

```python
env = catanatron_rs.BatchEnv(
    num_envs,
    colors=["RED", "BLUE", "WHITE", "ORANGE"],
    seed=1,
    player_kind="simple",
    player_kinds=None,
    discard_limit=7,
    friendly_robber=False,
    vps_to_win=10,
    map_kind="BASE",
    number_placement="official_spiral",
    channels_first=False,
    turn_limit=1000,
)
```

Observation tuple returned by `observe()`, `reset()`, and `step(action_indices)`:

```text
(
  observations,       # flat f32 list
  observation_shape,  # (batch, 21, 11, channels) or (batch, channels, 21, 11)
  legal_masks,        # flat u8 list, shape (batch, action_space_size)
  mask_shape,         # (batch, action_space_size)
  rewards,            # f32 list, one per env
  dones,              # bool list
  winners,            # optional color strings
  current_colors,     # next actor color strings
)
```

For lower Python overhead, use `observe_bytes()`, `reset_bytes()`, and
`step_bytes(action_indices)`. They return the same metadata, rewards, done flags,
winner values, and current colors, but observations are little-endian `float32`
bytes and legal masks are `uint8` bytes. Torch can consume them with
`torch.frombuffer`.

For repeated training loops, allocate mutable byte buffers once and ask Rust to
fill them in place:

```python
layout = env.byte_buffer_layout()
obs_buf = bytearray(layout["observations_nbytes"])
mask_buf = bytearray(layout["legal_masks_nbytes"])
feature_buf = bytearray(layout["features_nbytes"])

obs_shape, mask_shape, rewards, dones, winners, colors = env.reset_bytes_into(
    obs_buf,
    mask_buf,
)
feature_shape = env.feature_vectors_bytes_into(feature_buf)
```

`torch.frombuffer(bytearray_obj, dtype=...)` creates a CPU tensor view over the
same mutable memory. Keep the bytearray alive and copy tensors to the GPU, or
clone CPU tensors, before reusing the buffer for the next environment step.
Buffers must have the exact byte lengths reported by `byte_buffer_layout()`;
Rust never resizes them.

The `*_bytes_into` APIs accept `bytearray` objects. The matching
`*_into_buffer` APIs accept any writable C-contiguous Python buffer with the
exact byte length reported by `byte_buffer_layout()`, including `memoryview`
objects over NumPy arrays or pinned Torch CPU tensors viewed as `uint8`.
Rust writes the same little-endian bytes either way; trainers should reinterpret
the host bytes as `float32` observations/features and `uint8` masks, then copy to
the accelerator.

Scalar feature batches are exposed separately so existing observation callers do
not need to change:

```python
feature_names = env.feature_ordering()
feature_schema_hash = env.feature_schema_hash()
action_space_hash = env.action_space_hash()
features, feature_shape = env.feature_vectors()
feature_bytes, feature_shape = env.feature_vectors_bytes()
feature_shape = env.feature_vectors_bytes_into(feature_buf)
feature_shape = env.feature_vectors_into_buffer(feature_buffer)
```

Feature vectors are row-major `float32` with shape `(batch, len(feature_names))`.
The bytes variant returns little-endian `float32` bytes. Rows use the current
actor perspective for each environment. Missing fields for the cached schema
are zero-filled, so width stays stable within one `BatchEnv` map/player schema
across game phases. Store `feature_schema_hash` and `action_space_hash` beside
model checkpoints and replay buffers.

Reward is from the pre-step actor perspective: `+1.0` if that actor wins on the
step, `-1.0` if another player wins, otherwise `0.0`. Timeout at `turn_limit`
marks an environment done with no winner.

## Action Masks

The action space is fixed per `(player_colors, map_kind)` and exposed by:

```python
env.action_space_len()
env.action_space_json()
```

Legal masks are dense fixed-width `uint8` rows. `1` means the action id is legal
for the current state; `0` means illegal. Done environments return all-zero mask
rows. Masked logits should set illegal action ids to `-inf` before sampling or
argmax.

The mask is a policy-selection surface built from enumerated
`playable_actions`, not a replacement for execute-time validation. Domestic
trade offers are currently schema-visible as `OFFER_TRADE`, but arbitrary
concrete offer payloads are validated through the lower-level execute path
rather than enumerated into the policy mask. Accept/reject/confirm trade
responses are maskable after a trade has entered the decision flow.

The native CSV exporter writes both:

- `legal_action_indices.csv`: sparse semicolon-separated legal ids
- `legal_action_masks.csv`: dense `A_0..A_N` mask columns for NN losses

## Hot-Path Benchmark

After installing the `catanatron_rs` Python extension, measure the PyO3
byte-buffer training path with:

```bash
python examples/training_hotpaths.py --batches 64 256 1024 4096
```

This benchmark covers `observe_bytes_into`, `feature_vectors_bytes_into`,
`step_bytes_into`, and the generic `*_into_buffer` memoryview variants. It
intentionally does not require Torch or CUDA; GPU-transfer benchmarking belongs
in the model training loop.

## Board Tensor Shape

Board tensors use these channel counts:

```text
channels = 2 * num_players + 12
2 players: 16 channels
3 players: 18 channels
4 players: 20 channels
```

Default `channels_first=False` shape is `(batch, 21, 11, channels)`.
`channels_first=True` shape is `(batch, channels, 21, 11)`.

## Stability

Action ids are stable within a versioned action-space schema. The action-space
builder sorts robber coordinates so dense mask columns do not depend on hash map
iteration order.
