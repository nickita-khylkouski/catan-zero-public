//! Deterministic native-tree throughput microbenchmark.
//!
//! The evaluator is intentionally cheap so this isolates traversal, game
//! cloning, arena allocation, and tree bookkeeping rather than neural time.
//! Use a release build; the final checksum makes semantic drift visible.

use std::collections::HashMap;
use std::time::Instant;

use catanatron_rs::{Color, Game, Player};
use gumbel_mcts::{Evaluation, Evaluator, GumbelMctsEngine, SearchConfig};

struct UniformEvaluator;

impl Evaluator for UniformEvaluator {
    fn evaluate(
        &mut self,
        _game: &Game,
        legal_action_indices: &[usize],
        _root_color: Color,
    ) -> Result<Evaluation, String> {
        let probability = 1.0 / legal_action_indices.len().max(1) as f64;
        Ok((
            legal_action_indices
                .iter()
                .map(|&action| (action, probability))
                .collect::<HashMap<_, _>>(),
            0.125,
            0.0,
        ))
    }
}

fn main() {
    let iterations = std::env::args()
        .nth(1)
        .map(|value| value.parse::<usize>().expect("iterations must be usize"))
        .unwrap_or(250);
    let started = Instant::now();
    let mut checksum = 0_u64;
    let mut simulations = 0_i64;
    for iteration in 0..iterations {
        let game = Game::new(
            vec![Player::simple(Color::Red), Player::simple(Color::Blue)],
            Some(10_000 + iteration as u64),
        );
        let mut engine = GumbelMctsEngine::new(SearchConfig {
            seed: 20_000 + iteration as u64,
            n_full: 128,
            n_fast: 128,
            p_full: 1.0,
            lazy_interior_chance: true,
            ..SearchConfig::default()
        });
        let result = engine
            .search(&game, &mut UniformEvaluator, Some(true))
            .expect("native search failed");
        checksum = checksum.wrapping_add(result.selected_action as u64);
        simulations += i64::from(result.simulations_used);
    }
    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "iterations={iterations} simulations={simulations} elapsed_sec={elapsed:.6} searches_per_sec={:.3} checksum={checksum}",
        iterations as f64 / elapsed,
    );
}
