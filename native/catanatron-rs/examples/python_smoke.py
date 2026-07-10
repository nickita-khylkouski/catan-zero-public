import json

import catanatron_rs


def main():
    game = catanatron_rs.Game.simple(["RED", "BLUE"], seed=1)
    playable = json.loads(game.playable_actions_json())
    assert playable[0][1] == "BUILD_SETTLEMENT"

    record = json.loads(game.play_tick())
    assert record["action"][1] == "BUILD_SETTLEMENT"
    assert game.state_index() == 1

    snapshot = json.loads(game.json_snapshot())
    assert "current_playable_actions" in snapshot
    assert len(game.sample_vector(game.current_color())) > 0

    normalized = catanatron_rs.action_from_json('["BLUE","BUILD_ROAD",[1,0]]')
    assert normalized == '["BLUE","BUILD_ROAD",[0,1]]'

    env = catanatron_rs.BatchEnv(2, colors=["RED", "BLUE"], seed=1)
    obs, obs_shape, masks, mask_shape, rewards, dones, winners, colors = env.reset()
    assert obs_shape == (2, 21, 11, 16)
    assert len(obs) == 2 * 21 * 11 * 16
    assert mask_shape == (2, env.action_space_len())
    feature_ordering = env.feature_ordering()
    features, feature_shape = env.feature_vectors()
    assert feature_shape == (2, len(feature_ordering))
    assert len(features) == 2 * len(feature_ordering)
    feature_bytes, byte_feature_shape = env.feature_vectors_bytes()
    assert byte_feature_shape == feature_shape
    assert len(feature_bytes) == len(features) * 4
    layout = env.byte_buffer_layout()
    obs_buf = bytearray(layout["observations_nbytes"])
    mask_buf = bytearray(layout["legal_masks_nbytes"])
    feature_buf = bytearray(layout["features_nbytes"])
    into_obs_shape, into_mask_shape, *_ = env.observe_bytes_into(obs_buf, mask_buf)
    assert into_obs_shape == obs_shape
    assert into_mask_shape == mask_shape
    assert len(obs_buf) == len(obs) * 4
    assert len(mask_buf) == len(masks)
    assert env.feature_vectors_bytes_into(feature_buf) == feature_shape
    assert len(feature_buf) == len(features) * 4
    actions = []
    for offset in range(0, len(masks), env.action_space_len()):
        row = masks[offset : offset + env.action_space_len()]
        actions.append(row.index(1))
    into_obs_shape, into_mask_shape, into_rewards, into_dones, _, _ = env.step_bytes_into(
        actions,
        obs_buf,
        mask_buf,
    )
    assert into_obs_shape == obs_shape
    assert into_mask_shape == mask_shape
    assert len(into_rewards) == 2
    assert len(into_dones) == 2

    list_env = catanatron_rs.BatchEnv(2, colors=["RED", "BLUE"], seed=1)
    list_env.reset()
    _, _, next_masks, next_mask_shape, next_rewards, next_dones, _, _ = list_env.step(actions)
    assert next_mask_shape == mask_shape
    assert len(next_masks) == len(masks)
    assert len(next_rewards) == 2
    assert len(next_dones) == 2


if __name__ == "__main__":
    main()
