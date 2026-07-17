# A1 H100 pilot and scale topology

The checked-in A1 pre-wave template is intentionally a one-node pilot. It
selects the first complete host from
`configs/gpu_fleet_h100_8x6.json`, uses GPUs 0-7, and renders three ordered
jobs on every GPU:

| Category | Games/GPU | Fleet games |
|---|---:|---:|
| current producer | 8 | 64 |
| recent history | 2 | 16 |
| hard negative | 1 | 8 |

The pilot therefore seals 8 lanes, 24 jobs, and 88 selected games. It uses the
same global coherent-public n128 science contract as a full wave. Only topology
and quotas are smaller. The bounded attempts per GPU are 13, 4, and 2, so the
32-seed lane block in the template is sufficient.

The production executor remains manifest-driven: it derives lane and job counts
from the sealed lock and does not assume 64 GPUs. A dry-run must report exactly
8 lanes and 24 jobs before the operator may use `--go`.

After the pilot completes generation, harvest, training, and evaluation without
contract drift, scale by changing the sealed draft quota policy to
`balanced_prefix_v1` and the seed block size to at least 1,000. That selects all
six identical 8-GPU hosts from the same manifest (48 lanes and 144 jobs) while
preserving the 9,600/1,800/600 source totals and the global n128 recipe.

Do not edit host addresses, reorder hosts, or select an arbitrary subset after
sealing. Any such change creates a different manifest hash and requires a new
contract.
