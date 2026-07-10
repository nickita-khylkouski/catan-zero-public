use catanatron_rs::{
    BuildingType, Color, Game, MapKind, NumberPlacement, Player, Resource,
    player_num_resource_cards,
};

fn assert_game_invariants(game: &Game) {
    let mut resource_totals = game.state.resource_freqdeck;
    for color in game.state.colors.iter().copied() {
        let ps = game.state.player_state(color);
        for resource in Resource::ALL {
            resource_totals[resource.idx()] += ps.resources[resource.idx()];
        }
    }
    assert_eq!(
        resource_totals, [19; 5],
        "resource cards should be conserved"
    );

    let mut dev_total = game.state.development_listdeck.len();
    for color in game.state.colors.iter().copied() {
        let ps = game.state.player_state(color);
        dev_total += ps
            .dev_cards
            .iter()
            .map(|count| usize::from(*count))
            .sum::<usize>();
        dev_total += ps
            .played_dev_cards
            .iter()
            .map(|count| usize::from(*count))
            .sum::<usize>();
    }
    assert_eq!(dev_total, 25, "development cards should be conserved");

    for color in game.state.colors.iter().copied() {
        let ps = game.state.player_state(color);
        let roads = game
            .state
            .board
            .roads
            .values()
            .filter(|owner| **owner == color)
            .count();
        let settlements = game
            .state
            .board
            .buildings
            .values()
            .filter(|(owner, building)| *owner == color && *building == BuildingType::Settlement)
            .count();
        let cities = game
            .state
            .board
            .buildings
            .values()
            .filter(|(owner, building)| *owner == color && *building == BuildingType::City)
            .count();

        assert_eq!(usize::from(ps.roads_available) + roads, 15);
        assert_eq!(usize::from(ps.settlements_available) + settlements, 5);
        assert_eq!(usize::from(ps.cities_available) + cities, 4);
        assert_eq!(
            player_num_resource_cards(&game.state, color, None),
            ps.resources.iter().sum::<u8>()
        );
        assert!(ps.actual_victory_points >= ps.victory_points - 2);
    }
}

fn bot_roster(seed: usize) -> Vec<Player> {
    match seed % 4 {
        0 => vec![
            Player::random(Color::Red),
            Player::weighted_random(Color::Blue),
            Player::victory_point(Color::White),
            Player::value_function(Color::Orange),
        ],
        1 => vec![
            Player::simple(Color::Red),
            Player::random(Color::Blue),
            Player::playout(Color::White, 1),
            Player::weighted_random(Color::Orange),
        ],
        2 => vec![
            Player::alpha_beta(Color::Red, 1),
            Player::random(Color::Blue),
            Player::victory_point(Color::White),
            Player::weighted_random(Color::Orange),
        ],
        _ => vec![
            Player::value_function(Color::Red),
            Player::playout(Color::Blue, 1),
            Player::random(Color::White),
            Player::weighted_random(Color::Orange),
        ],
    }
}

#[test]
fn mixed_bots_survive_maps_rules_and_seeds() {
    let scenarios = [
        (MapKind::Base, NumberPlacement::OfficialSpiral, false),
        (MapKind::Base, NumberPlacement::Random, true),
        (MapKind::Mini, NumberPlacement::Random, false),
        (MapKind::Tournament, NumberPlacement::Random, true),
    ];

    for (scenario_index, (map_kind, number_placement, friendly_robber)) in
        scenarios.into_iter().enumerate()
    {
        for seed in 0..4 {
            let mut game = Game::with_options_and_map_options(
                bot_roster(seed),
                Some((scenario_index * 100 + seed) as u64),
                7,
                friendly_robber,
                6,
                map_kind,
                number_placement,
            );
            let _ = game.play();
            assert!(game.state.num_turns <= 1000);
            assert_game_invariants(&game);
        }
    }
}

#[test]
fn larger_random_batch_keeps_core_invariants() {
    for seed in 100..140 {
        let mut game = Game::with_options(
            vec![
                Player::random(Color::Red),
                Player::random(Color::Blue),
                Player::random(Color::White),
                Player::random(Color::Orange),
            ],
            Some(seed),
            7,
            seed % 2 == 0,
            5,
        );
        let _ = game.play();
        assert_game_invariants(&game);
    }
}
