//! Gumbel-AlphaZero MCTS — pure Rust library.
//!
//! No pyo3 dependency. The evaluator is a trait callback.
//! The Python binding lives in catanatron-rs-python.

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;
use std::f64;

use catanatron_rs::{
    generate_playable_actions, execute_spectrum, game_to_json_value, action_to_json_value,
    Action, ActionValue, ActionType, Color,
    Game, State,
};

// ---------------------------------------------------------------------------
// Evaluator trait — implemented by the Python binding
// ---------------------------------------------------------------------------

pub trait Evaluator {
    /// Evaluate a single game position.
    /// Returns (priors: action_index -> probability, value, uncertainty).
    /// `root_color` is the color of the player at the root of the search —
    /// the evaluator must flip the value sign when `acting_color != root_color`.
    fn evaluate(&mut self, game: &Game, legal_action_indices: &[usize], root_color: Color)
        -> Result<(HashMap<usize, f64>, f64, f64), String>;

    /// Evaluate multiple game positions in a batch.
    /// Default implementation calls evaluate() one at a time.
    fn evaluate_many(&mut self, requests: &[(Game, Vec<usize>, Color)])
        -> Result<Vec<(HashMap<usize, f64>, f64, f64)>, String>
    {
        let mut out = Vec::with_capacity(requests.len());
        for (game, legal, rc) in requests {
            out.push(self.evaluate(game, legal, *rc)?);
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct SearchConfig {
    pub colors: Vec<Color>,
    pub max_depth: i32,
    pub seed: u64,
    pub max_root_candidates: usize,
    pub max_root_candidates_wide: usize,
    pub wide_candidates_threshold: usize,
    pub exact_budget_sh: bool,
    pub exact_budget_sh_min_n: i32,
    pub c_visit: f64,
    pub c_scale: f64,
    pub temperature: f64,
    pub play_sh_winner: bool,
    pub prior_temperature: f64,
    pub n_full: i32,
    pub n_fast: i32,
    pub p_full: f64,
    pub n_full_wide: Option<i32>,
    pub raw_policy_above_width: Option<usize>,
    pub lazy_interior_chance: bool,
    pub root_candidate_cap: Option<usize>,
    pub policy_target_min_visits: i32,
    pub rescale_noise_floor_c: f64,
    pub sigma_eval: f64,
    pub variance_aware_q: bool,
    pub variance_aware_k: f64,
    pub variance_aware_closed_form_js: bool,
    pub uncertainty_backup_weighting: bool,
    pub uncertainty_backup_a: f64,
    pub uncertainty_backup_exp: f64,
    pub uncertainty_backup_cap: f64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            colors: vec![Color::Red, Color::Blue],
            max_depth: 80, seed: 0,
            max_root_candidates: 16, max_root_candidates_wide: 54, wide_candidates_threshold: 24,
            exact_budget_sh: false, exact_budget_sh_min_n: 0,
            c_visit: 50.0, c_scale: 0.1, temperature: 0.0, play_sh_winner: false,
            prior_temperature: 1.0, n_full: 64, n_fast: 16, p_full: 0.25,
            n_full_wide: None, raw_policy_above_width: None, lazy_interior_chance: false,
            root_candidate_cap: None, policy_target_min_visits: 0,
            rescale_noise_floor_c: 0.0, sigma_eval: 0.79,
            variance_aware_q: false, variance_aware_k: 1.0, variance_aware_closed_form_js: false,
            uncertainty_backup_weighting: false, uncertainty_backup_a: 0.25,
            uncertainty_backup_exp: 1.0, uncertainty_backup_cap: 1.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Tree nodes
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct ActionStats {
    prior: f64, visits: i32, value_sum: f64, value_sq_sum: f64,
    weighted_value_sum: f64, weight_sum: f64,
    children: HashMap<usize, usize>,
    probabilities: HashMap<usize, f64>,
    afterstate_value: Option<f64>,
}

impl ActionStats {
    fn new(prior: f64) -> Self {
        Self { prior, visits: 0, value_sum: 0.0, value_sq_sum: 0.0,
            weighted_value_sum: 0.0, weight_sum: 0.0,
            children: HashMap::new(), probabilities: HashMap::new(), afterstate_value: None }
    }
    #[inline] fn q(&self) -> f64 { if self.visits <= 0 { 0.0 } else { self.value_sum / self.visits as f64 } }
    #[inline] fn weighted_q(&self) -> f64 { if self.weight_sum <= 0.0 { self.q() } else { self.weighted_value_sum / self.weight_sum } }
    #[inline] fn q_variance(&self) -> f64 {
        if self.visits < 2 { return 0.0; }
        let mean = self.value_sum / self.visits as f64;
        let mean_sq = self.value_sq_sum / self.visits as f64;
        (mean_sq - mean * mean).max(0.0)
    }
}

#[derive(Clone, Debug)]
struct Node {
    game: Game, root_color: Color, prior_value: f64, prior_uncertainty: f64,
    visits: i32, value_sum: f64,
    actions: HashMap<usize, ActionStats>,
    action_logits: HashMap<usize, f64>,
    playable_actions: Vec<Action>,
    expanded: bool,
}

impl Node {
    fn new(game: Game, root_color: Color) -> Self {
        Self { game, root_color, prior_value: 0.0, prior_uncertainty: 0.0,
            visits: 0, value_sum: 0.0, actions: HashMap::new(),
            action_logits: HashMap::new(), playable_actions: Vec::new(), expanded: false }
    }
    #[inline] fn value(&self) -> f64 { if self.visits <= 0 { self.prior_value } else { self.value_sum / self.visits as f64 } }
}

struct Arena { nodes: Vec<Node> }
impl Arena {
    fn new() -> Self { Self { nodes: Vec::with_capacity(512) } }
    #[inline] fn alloc(&mut self, node: Node) -> usize { let i = self.nodes.len(); self.nodes.push(node); i }
    #[inline] fn get(&self, i: usize) -> &Node { &self.nodes[i] }
    #[inline] fn get_mut(&mut self, i: usize) -> &mut Node { &mut self.nodes[i] }
}

// ---------------------------------------------------------------------------
// Search result
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub selected_action: usize,
    pub improved_policy: Vec<(usize, f64)>,
    pub visit_counts: Vec<(usize, i32)>,
    pub q_values: Vec<(usize, f64)>,
    pub priors: Vec<(usize, f64)>,
    pub root_value: f64,
    pub used_full_search: bool,
    pub simulations_used: i32,
    pub afterstate_values: Vec<(usize, f64)>,
}

// ---------------------------------------------------------------------------
// Engine
// ---------------------------------------------------------------------------

pub struct GumbelMctsEngine {
    config: SearchConfig,
    rng: ChaCha8Rng,
    /// Pending leaf evaluations for batched GPU calls.
    /// Each entry: (node_idx, path_from_root, game, legal_indices, root_color)
    pending_leaves: Vec<(usize, Vec<(usize, Option<usize>)>, Game, Vec<usize>, Color)>,
    /// Batch size for deferred evaluation (0 = disabled, evaluate immediately)
    batch_size: usize,
}

impl GumbelMctsEngine {
    pub fn new(config: SearchConfig) -> Self {
        Self {
            rng: ChaCha8Rng::seed_from_u64(config.seed),
            config,
            pending_leaves: Vec::new(),
            batch_size: 0, // 0 = no deferred batching (immediate eval)
        }
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    // -----------------------------------------------------------------------
    // Public search
    // -----------------------------------------------------------------------
    pub fn search<E: Evaluator>(
        &mut self, game: &Game, evaluator: &mut E, force_full: Option<bool>,
    ) -> Result<SearchResult, String> {
        let root_color = game.state.current_color();
        let legal_actions = generate_playable_actions(&game.state);
        if legal_actions.is_empty() { return Err("no legal actions".into()); }
        if legal_actions.len() == 1 {
            return self.forced_single_action(game, &legal_actions, root_color, evaluator);
        }
        if let Some(width) = self.config.raw_policy_above_width {
            if legal_actions.len() > width {
                return self.raw_policy_root(game, root_color, &legal_actions, evaluator);
            }
        }
        let use_full = match force_full { Some(f) => f, None => self.rng.gen::<f64>() < self.config.p_full };
        let n_full_eff = if let Some(nw) = self.config.n_full_wide {
            if legal_actions.len() > self.config.wide_candidates_threshold { nw } else { self.config.n_full }
        } else { self.config.n_full };
        let n_simulations = if use_full { n_full_eff } else { self.config.n_fast }.max(1);

        let mut root_node = Node::new(game.clone(), root_color);
        root_node.playable_actions = legal_actions.clone();
        let mut arena = Arena::new();
        let root_idx = arena.alloc(root_node);
        self.expand_node(&mut arena, root_idx, evaluator)?;

        // GPU optimization: pre-expand ALL root children in one batch.
        // Only beneficial when batched evaluation is enabled (batch_size > 0).
        // Without batching, this just adds overhead (evaluates children that
        // might never be visited).
        if self.batch_size > 0 {
            self.pre_expand_root_children(&mut arena, root_idx, evaluator)?;
        }

        let (sh_winner, used) = self.run_root_search(&mut arena, root_idx, n_simulations, evaluator)?;
        let completed_q = self.completed_q(&arena, root_idx, root_color);
        let improved_policy = self.improved_policy(&arena, root_idx, &completed_q);
        let root = arena.get(root_idx);
        let visit_counts: Vec<_> = root.actions.iter().map(|(&i, s)| (i, s.visits)).collect();
        let q_values: Vec<_> = root.actions.iter().filter(|(_, s)| s.visits > 0).map(|(&i, s)| (i, s.q())).collect();
        let priors: Vec<_> = root.actions.iter().map(|(&i, s)| (i, s.prior)).collect();
        let afterstate_values: Vec<_> = root.actions.iter().filter_map(|(&i, s)| s.afterstate_value.map(|v| (i, v))).collect();

        let selected = if self.config.play_sh_winner { sh_winner }
        else if self.config.temperature > 0.0 { self.sample_categorical(&improved_policy) }
        else { improved_policy.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|(i, _)| *i).unwrap_or(sh_winner) };

        let training_policy = if self.config.policy_target_min_visits > 0 {
            self.prune_policy_target(&improved_policy, &visit_counts)
        } else { improved_policy.clone() };

        Ok(SearchResult {
            selected_action: selected, improved_policy: training_policy,
            visit_counts, q_values, priors, root_value: root.value(),
            used_full_search: use_full, simulations_used: used, afterstate_values,
        })
    }

    // -----------------------------------------------------------------------
    // Node expansion — calls evaluator
    // -----------------------------------------------------------------------
    fn expand_node<E: Evaluator>(&mut self, arena: &mut Arena, node_idx: usize, evaluator: &mut E) -> Result<f64, String> {
        let game = arena.get(node_idx).game.clone();
        let root_color = arena.get(node_idx).root_color;
        let legal_actions = generate_playable_actions(&game.state);
        let legal_indices: Vec<usize> = (0..legal_actions.len()).collect();
        let (priors, value, uncertainty) = evaluator.evaluate(&game, &legal_indices, root_color)?;
        self.finish_expand(arena, node_idx, legal_actions, priors, value, uncertainty)
    }

    /// GPU optimization: pre-expand ALL root children in ONE batched call.
    ///
    /// For each root action, computes the child state(s) and evaluates them
    /// in a single `evaluate_many` call. This turns N separate evaluator
    /// calls into 1 batched call — critical for GPU efficiency.
    ///
    /// For Roll actions (chance nodes), enumerates ALL outcomes and evaluates
    /// each outcome state. For other actions, evaluates the single resulting state.
    fn pre_expand_root_children<E: Evaluator>(&mut self, arena: &mut Arena, root_idx: usize, evaluator: &mut E) -> Result<(), String> {
        let root_color = arena.get(root_idx).root_color;
        let actions: Vec<Action> = arena.get(root_idx).playable_actions.clone();
        if actions.is_empty() { return Ok(()); }

        // Collect ALL child states that need evaluation
        // For each action, compute the child state(s)
        let mut batch_requests: Vec<(Game, Vec<usize>, Color)> = Vec::new();
        let mut action_children: Vec<(usize, Vec<(Game, f64)>)> = Vec::new(); // (action_idx, [(child_game, prob)])

        for (action_idx, action) in actions.iter().enumerate() {
            let parent_game = &arena.get(root_idx).game;
            let outcomes = execute_spectrum(parent_game, action);
            let total_prob: f64 = outcomes.iter().map(|(_, p)| *p).sum();
            if total_prob <= 0.0 || outcomes.is_empty() { continue; }

            let mut children_for_action = Vec::with_capacity(outcomes.len());
            for (child_game, prob) in &outcomes {
                let legal = generate_playable_actions(&child_game.state);
                let legal_indices: Vec<usize> = (0..legal.len()).collect();
                batch_requests.push((child_game.clone(), legal_indices, root_color));
                children_for_action.push((child_game.clone(), *prob / total_prob));
            }
            action_children.push((action_idx, children_for_action));
        }

        if batch_requests.is_empty() { return Ok(()); }

        // ONE batched evaluation call for ALL root children
        let results = evaluator.evaluate_many(&batch_requests)?;

        // Create child nodes and store evaluation results
        let mut result_idx = 0;
        for (action_idx, children) in action_children {
            let mut child_indices = Vec::with_capacity(children.len());
            let mut afterstate_value = 0.0;
            for (child_game, prob) in children {
                let (priors, value, uncertainty) = &results[result_idx];
                result_idx += 1;
                let legal = generate_playable_actions(&child_game.state);
                let child_idx = arena.alloc(Node::new(child_game, root_color));
                self.finish_expand(arena, child_idx, legal, priors.clone(), *value, *uncertainty)?;
                child_indices.push((child_idx, prob));
                afterstate_value += prob * value.clamp(-1.0, 1.0);
            }

            // Link children to root's action stats
            let node = arena.get_mut(root_idx);
            if let Some(stats) = node.actions.get_mut(&action_idx) {
                for (i, (child_idx, prob)) in child_indices.into_iter().enumerate() {
                    stats.children.insert(i, child_idx);
                    stats.probabilities.insert(i, prob);
                }
                stats.afterstate_value = Some(afterstate_value);
            }
        }

        Ok(())
    }

    fn finish_expand(&self, arena: &mut Arena, node_idx: usize, legal_actions: Vec<Action>, mut priors: HashMap<usize, f64>, value: f64, uncertainty: f64) -> Result<f64, String> {
        let node = arena.get_mut(node_idx);
        if !legal_actions.is_empty() {
            let floor = priors.values().filter(|&&p| p > 0.0).fold(f64::INFINITY, |a, &b| a.min(b));
            let floor = if floor == f64::INFINITY { 1.0 } else { floor * 0.01 };
            for i in 0..legal_actions.len() { if !priors.contains_key(&i) { priors.insert(i, floor); } }
            let total: f64 = (0..legal_actions.len()).map(|i| priors.get(&i).copied().unwrap_or(0.0)).sum();
            let prior_temp = self.config.prior_temperature.max(1e-6);
            node.playable_actions = legal_actions;
            node.actions.clear(); node.action_logits.clear();
            for i in 0..node.playable_actions.len() {
                let p = priors.get(&i).copied().unwrap_or(0.0) / total;
                node.actions.insert(i, ActionStats::new(p));
                node.action_logits.insert(i, (p.max(1e-8)).ln() / prior_temp);
            }
        }
        node.prior_value = value.clamp(-1.0, 1.0);
        node.prior_uncertainty = uncertainty.max(0.0);
        node.expanded = true;
        Ok(node.prior_value)
    }

    // -----------------------------------------------------------------------
    // Root search — with batched leaf evaluation for GPU
    // -----------------------------------------------------------------------
    fn run_root_search<E: Evaluator>(&mut self, arena: &mut Arena, root_idx: usize, n_simulations: i32, evaluator: &mut E) -> Result<(usize, i32), String> {
        let legal: Vec<usize> = arena.get(root_idx).actions.keys().copied().collect();
        let num_legal = legal.len();
        let m = self.root_candidate_count(num_legal);
        let mut gumbel: HashMap<usize, f64> = HashMap::new();
        for &aid in &legal { gumbel.insert(aid, self.sample_gumbel()); }
        let logits: HashMap<usize, f64> = arena.get(root_idx).action_logits.clone();
        let mut top_k: Vec<usize> = legal.clone();
        top_k.sort_by(|&a, &b| {
            let sa = gumbel[&a] + logits.get(&a).copied().unwrap_or(0.0);
            let sb = gumbel[&b] + logits.get(&b).copied().unwrap_or(0.0);
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });
        top_k.truncate(m);
        let mut remaining = top_k.clone();

        let use_batching = self.batch_size > 0;

        if self.config.exact_budget_sh && n_simulations >= self.config.exact_budget_sh_min_n {
            let phases = exact_budget_sh_phases(m as i32, n_simulations);
            let mut used = 0;
            for &(count, budget) in &phases {
                let visit: Vec<usize> = remaining.iter().take(count as usize).copied().collect();
                for &aid in &visit {
                    for _ in 0..budget {
                        if use_batching {
                            self.simulate_deferred(arena, root_idx, 0, Some(aid), Vec::new(), evaluator)?;
                            self.flush_pending_if_full(arena, evaluator)?;
                        } else {
                            self.simulate(arena, root_idx, 0, Some(aid), evaluator)?;
                        }
                        used += 1;
                    }
                }
                // Flush any remaining pending leaves before re-ranking
                if use_batching { self.flush_pending(arena, evaluator)?; }
                let cq = self.completed_q(arena, root_idx, arena.get(root_idx).root_color);
                let rq = self.rescaled_completed_q_with_noise(arena, root_idx, &cq);
                let scale = self.sigma_scale(arena, root_idx);
                remaining.sort_by(|&a, &b| {
                    let sa = gumbel[&a] + logits.get(&a).copied().unwrap_or(0.0) + scale * rq.get(&a).copied().unwrap_or(0.0);
                    let sb = gumbel[&b] + logits.get(&b).copied().unwrap_or(0.0) + scale * rq.get(&b).copied().unwrap_or(0.0);
                    sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            return Ok((*remaining.first().unwrap_or(&top_k[0]), used));
        }

        let schedule = sequential_halving_schedule(m as i32, n_simulations);
        let mut used = 0;
        for &(count, budget) in &schedule {
            for &aid in &remaining {
                for _ in 0..budget {
                    if use_batching {
                        self.simulate_deferred(arena, root_idx, 0, Some(aid), Vec::new(), evaluator)?;
                        self.flush_pending_if_full(arena, evaluator)?;
                    } else {
                        self.simulate(arena, root_idx, 0, Some(aid), evaluator)?;
                    }
                    used += 1;
                }
            }
            // Flush any remaining pending leaves before re-ranking
            if use_batching { self.flush_pending(arena, evaluator)?; }
            let cq = self.completed_q(arena, root_idx, arena.get(root_idx).root_color);
            let rq = self.rescaled_completed_q_with_noise(arena, root_idx, &cq);
            let scale = self.sigma_scale(arena, root_idx);
            remaining.sort_by(|&a, &b| {
                let sa = gumbel[&a] + logits.get(&a).copied().unwrap_or(0.0) + scale * rq.get(&a).copied().unwrap_or(0.0);
                let sb = gumbel[&b] + logits.get(&b).copied().unwrap_or(0.0) + scale * rq.get(&b).copied().unwrap_or(0.0);
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });
            remaining.truncate((count / 2).max(1) as usize);
        }
        Ok((*remaining.first().unwrap_or(&top_k[0]), used))
    }

    // -----------------------------------------------------------------------
    // Deferred simulation — records path, defers leaf evaluation
    // -----------------------------------------------------------------------
    fn simulate_deferred<E: Evaluator>(
        &mut self, arena: &mut Arena, node_idx: usize, depth: i32,
        forced_action: Option<usize>, mut path: Vec<(usize, Option<usize>)>,
        evaluator: &mut E,
    ) -> Result<f64, String> {
        let winner = arena.get(node_idx).game.winning_color();
        if let Some(w) = winner {
            let root_color = arena.get(node_idx).root_color;
            let value = if w == root_color { 1.0 } else { -1.0 };
            // Backpropagate terminal value immediately
            path.push((node_idx, None));
            self.backpropagate(arena, &path, value);
            return Ok(value);
        }
        if depth >= self.config.max_depth {
            if !arena.get(node_idx).expanded {
                // Defer this expansion
                let game = arena.get(node_idx).game.clone();
                let root_color = arena.get(node_idx).root_color;
                let legal = generate_playable_actions(&game.state);
                let legal_indices: Vec<usize> = (0..legal.len()).collect();
                path.push((node_idx, None));
                self.pending_leaves.push((node_idx, path.clone(), game, legal_indices, root_color));
                // Backpropagate placeholder (0.0) — will be corrected after eval
                self.backpropagate(arena, &path, 0.0);
                return Ok(0.0);
            }
            let value = arena.get(node_idx).prior_value;
            path.push((node_idx, None));
            self.backpropagate(arena, &path, value);
            return Ok(value);
        }
        if !arena.get(node_idx).expanded {
            let game = arena.get(node_idx).game.clone();
            let legal = generate_playable_actions(&game.state);
            if legal.len() == 1 {
                let root_color = arena.get(node_idx).root_color;
                let mut node = arena.get_mut(node_idx);
                node.playable_actions = legal;
                node.actions.insert(0, ActionStats::new(1.0));
                node.action_logits.insert(0, 0.0);
                let winner = node.game.winning_color();
                node.prior_value = match winner { Some(w) => if w == root_color { 1.0 } else { -1.0 }, None => 0.0 };
                node.expanded = true;
                path.push((node_idx, Some(0)));
                return self.simulate_deferred(arena, node_idx, depth, forced_action, path, evaluator);
            }
            // Defer this expansion
            let root_color = arena.get(node_idx).root_color;
            let legal_indices: Vec<usize> = (0..legal.len()).collect();
            path.push((node_idx, None));
            self.pending_leaves.push((node_idx, path.clone(), game, legal_indices, root_color));
            self.backpropagate(arena, &path, 0.0);
            return Ok(0.0);
        }
        if arena.get(node_idx).actions.is_empty() {
            let value = arena.get(node_idx).prior_value;
            path.push((node_idx, None));
            self.backpropagate(arena, &path, value);
            return Ok(value);
        }
        let action_idx = if let Some(fa) = forced_action { fa } else { self.select_nonroot_action(arena, node_idx)? };
        let action_type = arena.get(node_idx).playable_actions[action_idx].action_type;
        path.push((node_idx, Some(action_idx)));

        if action_type == ActionType::Roll && !(self.config.lazy_interior_chance && depth > 0) {
            self.traverse_roll_deferred(arena, node_idx, action_idx, depth, path, evaluator)
        } else {
            self.traverse_single_sample_deferred(arena, node_idx, action_idx, depth, path, evaluator)
        }
    }

    fn traverse_roll_deferred<E: Evaluator>(
        &mut self, arena: &mut Arena, node_idx: usize, action_idx: usize, depth: i32,
        path: Vec<(usize, Option<usize>)>, evaluator: &mut E,
    ) -> Result<f64, String> {
        let needs_enum = arena.get(node_idx).actions.get(&action_idx).map_or(true, |s| s.children.is_empty());
        if needs_enum { self.enumerate_outcomes(arena, node_idx, action_idx, evaluator)?; }
        let outcome_index = {
            let stats = arena.get(node_idx).actions.get(&action_idx).unwrap();
            let probs: Vec<(usize, f64)> = stats.probabilities.iter().map(|(&k, &v)| (k, v)).collect();
            self.sample_outcome(&probs)
        };
        let child_idx = arena.get(node_idx).actions.get(&action_idx).unwrap().children[&outcome_index];
        // Path already has (node_idx, Some(action_idx)) from simulate_deferred
        // backpropagate will handle visits and value_sum updates
        let value = self.simulate_deferred(arena, child_idx, depth + 1, None, path, evaluator)?;
        Ok(value)
    }

    fn traverse_single_sample_deferred<E: Evaluator>(
        &mut self, arena: &mut Arena, node_idx: usize, action_idx: usize, depth: i32,
        path: Vec<(usize, Option<usize>)>, evaluator: &mut E,
    ) -> Result<f64, String> {
        let child_exists = arena.get(node_idx).actions.get(&action_idx).map_or(false, |s| !s.children.is_empty());
        if !child_exists {
            let action = arena.get(node_idx).playable_actions[action_idx].clone();
            let game = arena.get(node_idx).game.clone();
            let root_color = arena.get(node_idx).root_color;
            let outcomes = execute_spectrum(&game, &action);
            let total_prob: f64 = outcomes.iter().map(|(_, p)| *p).sum();
            let mut new_child_indices: Vec<usize> = Vec::with_capacity(outcomes.len());
            for (child_game, _) in &outcomes {
                let child_node = Node::new(child_game.clone(), root_color);
                new_child_indices.push(arena.alloc(child_node));
            }
            let node = arena.get_mut(node_idx);
            let stats = node.actions.get_mut(&action_idx).unwrap();
            for (i, child_idx) in new_child_indices.into_iter().enumerate() {
                stats.children.insert(i, child_idx);
                stats.probabilities.insert(i, outcomes[i].1 / total_prob);
            }
        }
        let outcome_index = {
            let stats = arena.get(node_idx).actions.get(&action_idx).unwrap();
            let probs: Vec<(usize, f64)> = stats.probabilities.iter().map(|(&k, &v)| (k, v)).collect();
            self.sample_outcome(&probs)
        };
        let child_idx = arena.get(node_idx).actions.get(&action_idx).unwrap().children[&outcome_index];
        // Path already has (node_idx, Some(action_idx)) from simulate_deferred
        let value = self.simulate_deferred(arena, child_idx, depth + 1, None, path, evaluator)?;
        Ok(value)
    }

    /// Backpropagate a value along a path.
    /// path: Vec of (node_idx, action_idx) from root to leaf.
    /// For each node, increment visits and value_sum.
    /// For each action (if Some), increment action stats.
    fn backpropagate(&self, arena: &mut Arena, path: &[(usize, Option<usize>)], value: f64) {
        for &(node_idx, action_idx) in path {
            let node = arena.get_mut(node_idx);
            node.visits += 1;
            node.value_sum += value;
            if let Some(ai) = action_idx {
                if let Some(stats) = node.actions.get_mut(&ai) {
                    stats.visits += 1;
                    stats.value_sum += value;
                    stats.value_sq_sum += value * value;
                }
            }
        }
    }

    /// Add a value to value_sum along a path WITHOUT incrementing visits.
    /// Used by flush_pending to correct the placeholder (0.0) with the real value.
    fn backpropagate_value_only(&self, arena: &mut Arena, path: &[(usize, Option<usize>)], value: f64) {
        for &(node_idx, action_idx) in path {
            let node = arena.get_mut(node_idx);
            node.value_sum += value;
            if let Some(ai) = action_idx {
                if let Some(stats) = node.actions.get_mut(&ai) {
                    stats.value_sum += value;
                    stats.value_sq_sum += value * value;
                }
            }
        }
    }

    /// Flush pending leaves if batch is full
    fn flush_pending_if_full<E: Evaluator>(&mut self, arena: &mut Arena, evaluator: &mut E) -> Result<(), String> {
        if self.pending_leaves.len() >= self.batch_size {
            self.flush_pending(arena, evaluator)?;
        }
        Ok(())
    }

    /// Batch-evaluate all pending leaves and fix backpropagation
    fn flush_pending<E: Evaluator>(&mut self, arena: &mut Arena, evaluator: &mut E) -> Result<(), String> {
        if self.pending_leaves.is_empty() { return Ok(()); }

        // Extract batch requests
        let requests: Vec<(Game, Vec<usize>, Color)> = self.pending_leaves.iter()
            .map(|(_, _, game, legal, rc)| (game.clone(), legal.clone(), *rc))
            .collect();

        // ONE batched evaluation call
        let results = evaluator.evaluate_many(&requests)?;

        // Drain pending leaves into a separate vec to avoid borrow conflicts
        let pending = std::mem::take(&mut self.pending_leaves);

        // Update each pending leaf with its evaluation result
        // and fix the backpropagation (replace placeholder 0.0 with real value)
        for ((node_idx, path, _, _, _), (priors, value, uncertainty)) in
            pending.into_iter().zip(results.into_iter())
        {
            // Update the node's prior value
            let legal = generate_playable_actions(&arena.get(node_idx).game.state);
            self.finish_expand(arena, node_idx, legal, priors, value, uncertainty)?;

            // Fix backpropagation: add real value to value_sum
            // (visits were already incremented during deferred simulation)
            let clamped_value = value.clamp(-1.0, 1.0);
            self.backpropagate_value_only(arena, &path, clamped_value);
        }

        Ok(())
    }

    // -----------------------------------------------------------------------
    // Simulation — ALL in Rust
    // -----------------------------------------------------------------------
    fn simulate<E: Evaluator>(&mut self, arena: &mut Arena, node_idx: usize, depth: i32, forced_action: Option<usize>, evaluator: &mut E) -> Result<f64, String> {
        let winner = arena.get(node_idx).game.winning_color();
        if let Some(w) = winner {
            let root_color = arena.get(node_idx).root_color;
            return Ok(if w == root_color { 1.0 } else { -1.0 });
        }
        if depth >= self.config.max_depth {
            if !arena.get(node_idx).expanded { self.expand_node(arena, node_idx, evaluator)?; }
            return Ok(arena.get(node_idx).prior_value);
        }
        if !arena.get(node_idx).expanded {
            let game = arena.get(node_idx).game.clone();
            let legal = generate_playable_actions(&game.state);
            if legal.len() == 1 {
                let root_color = arena.get(node_idx).root_color;
                let mut node = arena.get_mut(node_idx);
                node.playable_actions = legal;
                node.actions.insert(0, ActionStats::new(1.0));
                node.action_logits.insert(0, 0.0);
                let winner = node.game.winning_color();
                node.prior_value = match winner { Some(w) => if w == root_color { 1.0 } else { -1.0 }, None => 0.0 };
                node.expanded = true;
                return self.simulate(arena, node_idx, depth, forced_action, evaluator);
            }
            let value = self.expand_node(arena, node_idx, evaluator)?;
            let node = arena.get_mut(node_idx);
            node.visits += 1; node.value_sum += value;
            return Ok(value);
        }
        if arena.get(node_idx).actions.is_empty() {
            let value = arena.get(node_idx).prior_value;
            let node = arena.get_mut(node_idx);
            node.visits += 1; node.value_sum += value;
            return Ok(value);
        }
        let action_idx = if let Some(fa) = forced_action { fa } else { self.select_nonroot_action(arena, node_idx)? };
        let action_type = arena.get(node_idx).playable_actions[action_idx].action_type;

        let value = if action_type == ActionType::Roll && !(self.config.lazy_interior_chance && depth > 0) {
            self.traverse_roll(arena, node_idx, action_idx, depth, evaluator)?
        } else {
            self.traverse_single_sample(arena, node_idx, action_idx, depth, evaluator)?
        };
        let node = arena.get_mut(node_idx);
        node.visits += 1; node.value_sum += value;
        Ok(value)
    }

    // -----------------------------------------------------------------------
    // Chance traversal
    // -----------------------------------------------------------------------
    fn traverse_roll<E: Evaluator>(&mut self, arena: &mut Arena, node_idx: usize, action_idx: usize, depth: i32, evaluator: &mut E) -> Result<f64, String> {
        let needs_enum = arena.get(node_idx).actions.get(&action_idx).map_or(true, |s| s.children.is_empty());
        if needs_enum { self.enumerate_outcomes(arena, node_idx, action_idx, evaluator)?; }
        let outcome_index = {
            let stats = arena.get(node_idx).actions.get(&action_idx).unwrap();
            let probs: Vec<(usize, f64)> = stats.probabilities.iter().map(|(&k, &v)| (k, v)).collect();
            self.sample_outcome(&probs)
        };
        let child_idx = arena.get(node_idx).actions.get(&action_idx).unwrap().children[&outcome_index];
        self.simulate(arena, child_idx, depth + 1, None, evaluator)?;
        let value = {
            let stats = arena.get(node_idx).actions.get(&action_idx).unwrap();
            let mut v = 0.0;
            for (&idx, &child_idx) in &stats.children {
                let prob = stats.probabilities.get(&idx).copied().unwrap_or(0.0);
                v += prob * arena.get(child_idx).value();
            }
            v
        };
        let node = arena.get_mut(node_idx);
        let stats = node.actions.get_mut(&action_idx).unwrap();
        stats.visits += 1; stats.value_sum += value; stats.value_sq_sum += value * value;
        if self.config.uncertainty_backup_weighting {
            let w = self.backup_weight(node.prior_uncertainty);
            stats.weight_sum += w; stats.weighted_value_sum += w * value;
        }
        Ok(value)
    }

    fn traverse_single_sample<E: Evaluator>(&mut self, arena: &mut Arena, node_idx: usize, action_idx: usize, depth: i32, evaluator: &mut E) -> Result<f64, String> {
        let child_exists = arena.get(node_idx).actions.get(&action_idx).map_or(false, |s| !s.children.is_empty());
        if !child_exists {
            let action = arena.get(node_idx).playable_actions[action_idx].clone();
            let game = arena.get(node_idx).game.clone();
            let root_color = arena.get(node_idx).root_color;
            let outcomes = execute_spectrum(&game, &action);
            let total_prob: f64 = outcomes.iter().map(|(_, p)| *p).sum();
            let child_games: Vec<Game> = outcomes.iter().map(|(g, _)| g.clone()).collect();
            let probs: Vec<f64> = outcomes.iter().map(|(_, p)| *p / total_prob).collect();
            let mut new_child_indices: Vec<usize> = Vec::with_capacity(child_games.len());
            for child_game in child_games.into_iter() {
                let child_node = Node::new(child_game, root_color);
                new_child_indices.push(arena.alloc(child_node));
            }
            let node = arena.get_mut(node_idx);
            let stats = node.actions.get_mut(&action_idx).unwrap();
            for (i, child_idx) in new_child_indices.into_iter().enumerate() {
                stats.children.insert(i, child_idx);
                stats.probabilities.insert(i, probs[i]);
            }
        }
        let outcome_index = {
            let stats = arena.get(node_idx).actions.get(&action_idx).unwrap();
            let probs: Vec<(usize, f64)> = stats.probabilities.iter().map(|(&k, &v)| (k, v)).collect();
            self.sample_outcome(&probs)
        };
        let child_idx = arena.get(node_idx).actions.get(&action_idx).unwrap().children[&outcome_index];
        let value = self.simulate(arena, child_idx, depth + 1, None, evaluator)?;
        let child_unc = if self.config.uncertainty_backup_weighting { arena.get(child_idx).prior_uncertainty } else { 0.0 };
        let node = arena.get_mut(node_idx);
        let stats = node.actions.get_mut(&action_idx).unwrap();
        stats.visits += 1; stats.value_sum += value; stats.value_sq_sum += value * value;
        if self.config.uncertainty_backup_weighting {
            let w = self.backup_weight(child_unc);
            stats.weight_sum += w; stats.weighted_value_sum += w * value;
        }
        Ok(value)
    }

    fn enumerate_outcomes<E: Evaluator>(&mut self, arena: &mut Arena, node_idx: usize, action_idx: usize, evaluator: &mut E) -> Result<(), String> {
        let action = arena.get(node_idx).playable_actions[action_idx].clone();
        let game = arena.get(node_idx).game.clone();
        let root_color = arena.get(node_idx).root_color;
        let outcomes = execute_spectrum(&game, &action);
        let total_prob: f64 = outcomes.iter().map(|(_, p)| *p).sum();
        if total_prob <= 0.0 || outcomes.is_empty() { return Ok(()); }
        let child_games: Vec<Game> = outcomes.iter().map(|(g, _)| g.clone()).collect();
        let mut new_child_indices: Vec<usize> = Vec::with_capacity(child_games.len());
        for child_game in &child_games {
            let child_node = Node::new(child_game.clone(), root_color);
            new_child_indices.push(arena.alloc(child_node));
        }
        // Evaluate all children — batch if possible
        let mut requests: Vec<(Game, Vec<usize>, Color)> = Vec::new();
        for &child_idx in &new_child_indices {
            let child_game = arena.get(child_idx).game.clone();
            let legal = generate_playable_actions(&child_game.state);
            let legal_indices: Vec<usize> = (0..legal.len()).collect();
            requests.push((child_game, legal_indices, root_color));
        }
        let results = evaluator.evaluate_many(&requests)?;
        for (i, &child_idx) in new_child_indices.iter().enumerate() {
            let (priors, value, unc) = &results[i];
            let legal = generate_playable_actions(&arena.get(child_idx).game.state);
            self.finish_expand(arena, child_idx, legal, priors.clone(), *value, *unc)?;
        }
        let afterstate_value: f64 = {
            let mut weighted = 0.0;
            for (i, (_, prob)) in outcomes.iter().enumerate() {
                weighted += prob * arena.get(new_child_indices[i]).prior_value;
            }
            weighted / total_prob
        };
        let node = arena.get_mut(node_idx);
        let stats = node.actions.get_mut(&action_idx).unwrap();
        stats.afterstate_value = Some(afterstate_value);
        for (i, (_, prob)) in outcomes.into_iter().enumerate() {
            stats.probabilities.insert(i, prob / total_prob);
            stats.children.insert(i, new_child_indices[i]);
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Completed-Q / improved policy
    // -----------------------------------------------------------------------
    fn sigma_scale(&self, arena: &Arena, node_idx: usize) -> f64 {
        let max_visits = arena.get(node_idx).actions.values().map(|s| s.visits).max().unwrap_or(0);
        (self.config.c_visit + max_visits as f64) * self.config.c_scale
    }

    fn completed_q(&self, arena: &Arena, node_idx: usize, root_color: Color) -> HashMap<usize, f64> {
        let node = arena.get(node_idx);
        let root_to_act = node.game.state.current_color() == root_color;
        let sign = if root_to_act { 1.0 } else { -1.0 };
        let use_weighted = self.config.uncertainty_backup_weighting;
        let total_child_visits: i32 = node.actions.values().map(|s| s.visits).sum();
        let mut visited_prior_sum = 0.0; let mut visited_q_sum = 0.0;
        for stats in node.actions.values() {
            if stats.visits > 0 {
                visited_prior_sum += stats.prior;
                let q = if use_weighted { stats.weighted_q() } else { stats.q() };
                visited_q_sum += stats.prior * (sign * q);
            }
        }
        let weighted_q = if visited_prior_sum > 0.0 { visited_q_sum / visited_prior_sum } else { 0.0 };
        let node_value = sign * node.prior_value;
        let v_mix = (node_value + total_child_visits as f64 * weighted_q) / (1.0 + total_child_visits as f64);
        let mut completed: HashMap<usize, f64> = HashMap::new();
        for (&aid, stats) in &node.actions {
            if stats.visits > 0 {
                let q = if use_weighted { stats.weighted_q() } else { stats.q() };
                completed.insert(aid, sign * q);
            } else { completed.insert(aid, v_mix); }
        }
        if self.config.variance_aware_q {
            self.shrink_completed_q_by_variance(arena, node_idx, &mut completed, v_mix);
        }
        completed
    }

    fn shrink_completed_q_by_variance(&self, arena: &Arena, node_idx: usize, completed: &mut HashMap<usize, f64>, v_mix: f64) {
        let node = arena.get(node_idx);
        let visited: Vec<(usize, &ActionStats)> = node.actions.iter().filter(|(_, s)| s.visits > 0).map(|(&k, v)| (k, v)).collect();
        if visited.len() < 2 { return; }
        let visited_qs: Vec<f64> = visited.iter().map(|(id, _)| completed[id]).collect();
        let mean_q: f64 = visited_qs.iter().sum::<f64>() / visited_qs.len() as f64;
        let signal_var: f64 = visited_qs.iter().map(|q| (q - mean_q).powi(2)).sum::<f64>() / visited_qs.len() as f64;
        if signal_var <= 0.0 { return; }
        if self.config.variance_aware_closed_form_js {
            let se_sqs: Vec<f64> = visited.iter().map(|(_, s)| s.q_variance() / s.visits as f64).collect();
            let mean_se_sq: f64 = se_sqs.iter().sum::<f64>() / se_sqs.len() as f64;
            let lam = signal_var / (signal_var + mean_se_sq);
            for (id, _) in &visited {
                let q = completed[id];
                completed.insert(*id, v_mix + lam * (q - v_mix));
            }
        } else {
            let k = self.config.variance_aware_k;
            for (id, stats) in &visited {
                let se_sq = stats.q_variance() / stats.visits as f64;
                let shrink = signal_var / (signal_var + k * se_sq);
                let q = completed[id];
                completed.insert(*id, v_mix + shrink * (q - v_mix));
            }
        }
    }

    fn rescale_completed_q(&self, cq: &HashMap<usize, f64>) -> HashMap<usize, f64> {
        if cq.is_empty() { return HashMap::new(); }
        let min_q = cq.values().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_q = cq.values().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let denom = (max_q - min_q) + 1e-8;
        cq.iter().map(|(&k, &v)| (k, (v - min_q) / denom)).collect()
    }

    fn rescaled_completed_q_with_noise(&self, arena: &Arena, node_idx: usize, cq: &HashMap<usize, f64>) -> HashMap<usize, f64> {
        let rescaled = self.rescale_completed_q(cq);
        self.apply_noise_floor(arena, node_idx, cq, &rescaled)
    }

    fn apply_noise_floor(&self, arena: &Arena, node_idx: usize, cq: &HashMap<usize, f64>, rescaled: &HashMap<usize, f64>) -> HashMap<usize, f64> {
        let c = self.config.rescale_noise_floor_c;
        if c <= 0.0 || rescaled.is_empty() { return rescaled.clone(); }
        let values: Vec<f64> = cq.values().copied().collect();
        let raw_spread = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)) - values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let visits: Vec<i32> = arena.get(node_idx).actions.values().map(|s| s.visits).collect();
        let mean_visits = if visits.is_empty() { 0.0 } else { visits.iter().map(|&v| v as f64).sum::<f64>() / visits.len() as f64 };
        let noise_floor = if mean_visits <= 0.0 { f64::INFINITY } else { c * self.config.sigma_eval / mean_visits.sqrt() };
        let denom = raw_spread + noise_floor;
        let alpha = if denom <= 0.0 || noise_floor.is_infinite() { 0.0 } else { raw_spread / denom };
        rescaled.iter().map(|(&k, &v)| (k, 0.5 + alpha * (v - 0.5))).collect()
    }

    fn improved_policy(&self, arena: &Arena, node_idx: usize, cq: &HashMap<usize, f64>) -> Vec<(usize, f64)> {
        let scale = self.sigma_scale(arena, node_idx);
        let rq = self.rescaled_completed_q_with_noise(arena, node_idx, cq);
        let node = arena.get(node_idx);
        let mut scores: Vec<(usize, f64)> = node.actions.keys().map(|&aid| {
            let logit = node.action_logits.get(&aid).copied().unwrap_or(0.0);
            let rqv = rq.get(&aid).copied().unwrap_or(0.0);
            (aid, logit + scale * rqv)
        }).collect();
        let max_score = scores.iter().fold(f64::NEG_INFINITY, |a, &(_, b)| a.max(b));
        for (_, s) in scores.iter_mut() { *s = ((*s - max_score).clamp(-40.0, 40.0)).exp(); }
        let total: f64 = scores.iter().map(|(_, s)| *s).sum();
        if total > 0.0 { for (_, s) in scores.iter_mut() { *s /= total; } }
        scores
    }

    fn select_nonroot_action(&self, arena: &Arena, node_idx: usize) -> Result<usize, String> {
        let root_color = arena.get(node_idx).root_color;
        let cq = self.completed_q(arena, node_idx, root_color);
        let improved = self.improved_policy(arena, node_idx, &cq);
        let total_visits: i32 = arena.get(node_idx).actions.values().map(|s| s.visits).sum();
        let node = arena.get(node_idx);
        let mut best: Option<usize> = None;
        let mut best_score = f64::NEG_INFINITY;
        for (aid, prob) in &improved {
            let visits = node.actions.get(aid).map(|s| s.visits).unwrap_or(0);
            let score = prob - visits as f64 / (1.0 + total_visits as f64);
            if score > best_score { best_score = score; best = Some(*aid); }
        }
        best.ok_or_else(|| "cannot select from empty node".into())
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------
    #[inline] fn sample_gumbel(&mut self) -> f64 {
        let u = self.rng.gen::<f64>().clamp(1e-12, 1.0 - 1e-12);
        -(-u.ln()).ln()
    }
    #[inline] fn sample_outcome(&mut self, outcomes: &[(usize, f64)]) -> usize {
        if outcomes.len() == 1 { return outcomes[0].0; }
        let draw = self.rng.gen::<f64>();
        let mut cum = 0.0;
        for &(idx, prob) in outcomes { cum += prob; if draw <= cum { return idx; } }
        outcomes.last().map(|&(idx, _)| idx).unwrap_or(0)
    }
    #[inline] fn sample_categorical(&mut self, policy: &[(usize, f64)]) -> usize {
        if policy.is_empty() { return 0; }
        let draw = self.rng.gen::<f64>();
        let mut cum = 0.0;
        for &(aid, prob) in policy { cum += prob; if draw <= cum { return aid; } }
        policy.last().map(|&(aid, _)| aid).unwrap_or(0)
    }
    #[inline] fn backup_weight(&self, uncertainty: f64) -> f64 {
        let err = uncertainty.max(0.0);
        self.config.uncertainty_backup_cap.min(self.config.uncertainty_backup_a * err.powf(self.config.uncertainty_backup_exp))
    }
    fn root_candidate_count(&self, num_legal: usize) -> usize {
        if let Some(cap) = self.config.root_candidate_cap { num_legal.min(cap).max(1) }
        else if num_legal > self.config.wide_candidates_threshold { num_legal.min(self.config.max_root_candidates_wide).max(1) }
        else { num_legal.min(self.config.max_root_candidates).max(1) }
    }
    fn prune_policy_target(&self, policy: &[(usize, f64)], visits: &[(usize, i32)]) -> Vec<(usize, f64)> {
        if self.config.policy_target_min_visits <= 0 || policy.is_empty() { return policy.to_vec(); }
        let visit_map: HashMap<usize, i32> = visits.iter().copied().collect();
        let kept_mass: f64 = policy.iter().filter(|(aid, _)| visit_map.get(aid).copied().unwrap_or(0) >= self.config.policy_target_min_visits).map(|(_, p)| *p).sum();
        if kept_mass <= 0.0 { return policy.to_vec(); }
        policy.iter().map(|(aid, prob)| {
            if visit_map.get(aid).copied().unwrap_or(0) >= self.config.policy_target_min_visits { (*aid, *prob / kept_mass) } else { (*aid, 0.0) }
        }).collect()
    }

    // -----------------------------------------------------------------------
    // Special paths
    // -----------------------------------------------------------------------
    fn forced_single_action<E: Evaluator>(&mut self, game: &Game, legal: &[Action], root_color: Color, evaluator: &mut E) -> Result<SearchResult, String> {
        let action = &legal[0];
        if action.action_type != ActionType::Roll {
            let (_, value, _) = evaluator.evaluate(game, &[0], root_color)?;
            return Ok(SearchResult {
                selected_action: 0, improved_policy: vec![(0, 1.0)], visit_counts: vec![(0, 1)],
                q_values: vec![], priors: vec![(0, 1.0)], root_value: value.clamp(-1.0, 1.0),
                used_full_search: true, simulations_used: 0, afterstate_values: vec![],
            });
        }
        let outcomes = execute_spectrum(game, action);
        let total_prob: f64 = outcomes.iter().map(|(_, p)| *p).sum();
        let mut weighted = 0.0;
        for (child_game, prob) in &outcomes {
            let (_, value, _) = evaluator.evaluate(child_game, &[0], root_color)?;
            weighted += prob * value;
        }
        let root_value = if total_prob > 0.0 { weighted / total_prob } else { 0.0 };
        Ok(SearchResult {
            selected_action: 0, improved_policy: vec![(0, 1.0)], visit_counts: vec![(0, 1)],
            q_values: vec![], priors: vec![(0, 1.0)], root_value,
            used_full_search: true, simulations_used: 0, afterstate_values: vec![(0, root_value)],
        })
    }

    fn raw_policy_root<E: Evaluator>(&mut self, game: &Game, root_color: Color, legal: &[Action], evaluator: &mut E) -> Result<SearchResult, String> {
        let legal_indices: Vec<usize> = (0..legal.len()).collect();
        let (priors, value, _) = evaluator.evaluate(game, &legal_indices, root_color)?;
        let mut node = Node::new(game.clone(), root_color);
        node.playable_actions = legal.to_vec();
        let mut arena = Arena::new();
        let root_idx = arena.alloc(node);
        self.finish_expand(&mut arena, root_idx, legal.to_vec(), priors, value, 0.0)?;
        let priors_vec: Vec<(usize, f64)> = arena.get(root_idx).actions.iter().map(|(&k, v)| (k, v.prior)).collect();
        if priors_vec.is_empty() { return Err("no legal actions at raw-policy root".into()); }
        let selected = priors_vec.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)).map(|(i, _)| *i).unwrap_or(0);
        Ok(SearchResult {
            selected_action: selected, improved_policy: priors_vec.clone(), visit_counts: vec![],
            q_values: vec![], priors: priors_vec, root_value: arena.get(root_idx).value(),
            used_full_search: false, simulations_used: 0, afterstate_values: vec![],
        })
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

pub fn sequential_halving_schedule(m: i32, n_simulations: i32) -> Vec<(i32, i32)> {
    let m = m.max(1);
    let num_rounds = if m > 1 { (m as f64).log2().ceil() as i32 } else { 1 };
    let mut schedule = Vec::new();
    let mut count = m;
    for _ in 0..num_rounds {
        let budget = (n_simulations / (num_rounds * count)).max(1);
        schedule.push((count, budget));
        count = (count / 2).max(1);
    }
    schedule
}

pub fn exact_budget_sh_phases(m: i32, n_simulations: i32) -> Vec<(i32, i32)> {
    let m = m.max(1);
    let n = n_simulations.max(1);
    if m == 1 { return vec![(1, n)]; }
    let log2max = (m as f64).log2().ceil() as i32;
    let mut phases = Vec::new();
    let mut total = 0;
    let mut considered = m;
    while total < n {
        let extra = (n as f64 / (log2max as f64 * considered as f64)).floor() as i32;
        let extra = extra.max(1);
        let full_passes = extra.min((n - total) / considered);
        if full_passes > 0 {
            phases.push((considered, full_passes));
            total += considered * full_passes;
        }
        if full_passes < extra {
            let leftover = n - total;
            if leftover > 0 { phases.push((leftover, 1)); total = n; }
            break;
        }
        considered = (considered / 2).max(2);
    }
    phases
}
