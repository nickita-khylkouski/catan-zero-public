import torch

import catanatron_rs


def masked_sample(logits, masks):
    masked = logits.masked_fill(masks == 0, float("-inf"))
    return torch.distributions.Categorical(logits=masked).sample()


def main():
    env = catanatron_rs.BatchEnv(
        64,
        colors=["RED", "BLUE", "WHITE", "ORANGE"],
        seed=1,
        player_kind="simple",
        channels_first=True,
    )

    feature_names = env.feature_ordering()
    layout = env.byte_buffer_layout()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_pinned = device == "cuda"
    obs_host = torch.empty(
        layout["observations_nbytes"], dtype=torch.uint8, pin_memory=use_pinned
    )
    mask_host = torch.empty(
        layout["legal_masks_nbytes"], dtype=torch.uint8, pin_memory=use_pinned
    )
    feature_host = torch.empty(
        layout["features_nbytes"], dtype=torch.uint8, pin_memory=use_pinned
    )
    obs_buf = memoryview(obs_host.numpy())
    mask_buf = memoryview(mask_host.numpy())
    feature_buf = memoryview(feature_host.numpy())
    obs_shape, mask_shape, rewards, dones, winners, colors = env.reset_into_buffer(
        obs_buf, mask_buf
    )

    for _ in range(128):
        observations = obs_host.view(torch.float32).reshape(obs_shape).to(
            device, non_blocking=use_pinned
        )
        legal_masks = mask_host.reshape(mask_shape).to(
            device, non_blocking=use_pinned
        ).bool()
        feature_shape = env.feature_vectors_into_buffer(feature_buf)
        features = feature_host.view(torch.float32).reshape(feature_shape).to(
            device, non_blocking=use_pinned
        )

        # Replace this with a real policy network. This random policy demonstrates
        # the Rust-env -> GPU-model -> action-id -> Rust-env boundary.
        assert features.shape[1] == len(feature_names)
        logits = torch.randn((mask_shape[0], mask_shape[1]), device=device)
        actions = masked_sample(logits, legal_masks).cpu().tolist()

        obs_shape, mask_shape, rewards, dones, winners, colors = env.step_into_buffer(
            actions,
            obs_buf,
            mask_buf,
        )
        if all(dones):
            obs_shape, mask_shape, rewards, dones, winners, colors = env.reset_into_buffer(
                obs_buf,
                mask_buf,
            )

    print("batch", obs_shape, "action_space", env.action_space_len())


if __name__ == "__main__":
    main()
