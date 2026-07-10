use catanatron_rs::{Color, Game, Player};
use std::time::Instant;

fn main() {
    let games: usize = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(1_000);

    let start = Instant::now();
    let mut completed_turns = 0usize;
    let mut wins = 0usize;

    for i in 0..games {
        let players = vec![
            Player::random(Color::Red),
            Player::random(Color::Blue),
            Player::random(Color::White),
            Player::random(Color::Orange),
        ];
        let mut game = Game::with_options(players, Some(i as u64), 7, false, 10);
        if game.play().is_some() {
            wins += 1;
        }
        completed_turns += game.state.num_turns;
    }

    let elapsed = start.elapsed();
    let games_per_second = games as f64 / elapsed.as_secs_f64();
    let turns_per_second = completed_turns as f64 / elapsed.as_secs_f64();

    println!("games={games}");
    println!("wins={wins}");
    println!("turns={completed_turns}");
    println!("elapsed_ms={:.3}", elapsed.as_secs_f64() * 1_000.0);
    println!("games_per_second={games_per_second:.2}");
    println!("turns_per_second={turns_per_second:.2}");
}
