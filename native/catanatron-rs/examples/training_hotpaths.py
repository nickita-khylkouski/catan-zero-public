import argparse
import time

import catanatron_rs


def bench(name, fn, batch_size, nbytes, iters, warmup):
    for _ in range(warmup):
        fn()

    start_ns = time.perf_counter_ns()
    for _ in range(iters):
        result = fn()
    elapsed_ns = time.perf_counter_ns() - start_ns

    seconds = elapsed_ns / 1_000_000_000
    us_iter = elapsed_ns / iters / 1_000
    ns_env = elapsed_ns / iters / batch_size if batch_size else 0.0
    gb_s = (nbytes * iters) / seconds / 1_000_000_000 if seconds else 0.0
    print(
        f"{name} batch={batch_size} iters={iters} "
        f"us_iter={us_iter:.3f} ns_env={ns_env:.1f} gb_s={gb_s:.3f}"
    )
    return result


def first_legal_actions(mask_buf, batch_size, action_space_len):
    actions = []
    for row in range(batch_size):
        start = row * action_space_len
        end = start + action_space_len
        row_mask = mask_buf[start:end]
        try:
            actions.append(row_mask.index(1))
        except ValueError:
            actions.append(0)
    return actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", type=int, nargs="+", default=[64, 256, 1024, 4096])
    parser.add_argument("--target-envs", type=int, default=262_144)
    parser.add_argument("--min-iters", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--channels-first", action="store_true")
    args = parser.parse_args()

    for batch_size in args.batches:
        env = catanatron_rs.BatchEnv(
            batch_size,
            colors=["RED", "BLUE", "WHITE", "ORANGE"],
            seed=1,
            player_kind="simple",
            channels_first=args.channels_first,
        )
        layout = env.byte_buffer_layout()
        obs_buf = bytearray(layout["observations_nbytes"])
        mask_buf = bytearray(layout["legal_masks_nbytes"])
        feature_buf = bytearray(layout["features_nbytes"])
        obs_view = memoryview(obs_buf)
        mask_view = memoryview(mask_buf)
        feature_view = memoryview(feature_buf)
        action_space_len = env.action_space_len()
        iters = max(args.min_iters, args.target_envs // max(batch_size, 1))

        env.reset_bytes_into(obs_buf, mask_buf)

        def step_first_legal():
            actions = first_legal_actions(mask_buf, batch_size, action_space_len)
            return env.step_bytes_into(actions, obs_buf, mask_buf)

        def observe_bytes_into_new_buffers():
            new_obs = bytearray(layout["observations_nbytes"])
            new_mask = bytearray(layout["legal_masks_nbytes"])
            return env.observe_bytes_into(new_obs, new_mask)

        observation_nbytes = layout["observations_nbytes"] + layout["legal_masks_nbytes"]
        bench(
            "observe_list_allocating",
            env.observe,
            batch_size,
            observation_nbytes,
            iters,
            args.warmup,
        )
        bench(
            "observe_bytes_allocating",
            env.observe_bytes,
            batch_size,
            observation_nbytes,
            iters,
            args.warmup,
        )
        bench(
            "observe_bytes_into_reused",
            lambda: env.observe_bytes_into(obs_buf, mask_buf),
            batch_size,
            observation_nbytes,
            iters,
            args.warmup,
        )
        bench(
            "observe_into_buffer_memoryview",
            lambda: env.observe_into_buffer(obs_view, mask_view),
            batch_size,
            observation_nbytes,
            iters,
            args.warmup,
        )
        bench(
            "observe_bytes_into_new_bytearrays",
            observe_bytes_into_new_buffers,
            batch_size,
            observation_nbytes,
            iters,
            args.warmup,
        )
        bench(
            "feature_vectors_bytes_into",
            lambda: env.feature_vectors_bytes_into(feature_buf),
            batch_size,
            layout["features_nbytes"],
            iters,
            args.warmup,
        )
        bench(
            "feature_vectors_into_buffer_memoryview",
            lambda: env.feature_vectors_into_buffer(feature_view),
            batch_size,
            layout["features_nbytes"],
            iters,
            args.warmup,
        )
        bench(
            "step_bytes_into_first_legal",
            step_first_legal,
            batch_size,
            layout["observations_nbytes"] + layout["legal_masks_nbytes"],
            iters,
            args.warmup,
        )
        bench(
            "step_into_buffer_first_legal_memoryview",
            lambda: env.step_into_buffer(
                first_legal_actions(mask_buf, batch_size, action_space_len),
                obs_view,
                mask_view,
            ),
            batch_size,
            layout["observations_nbytes"] + layout["legal_masks_nbytes"],
            iters,
            args.warmup,
        )


if __name__ == "__main__":
    main()
