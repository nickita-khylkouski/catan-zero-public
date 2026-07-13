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
    execute_spectrum, generate_playable_actions, Action, ActionSpace, ActionType, ActionValue,
    Color, Game, MapKind,
};

pub type Evaluation = (HashMap<usize, f64>, f64, f64);
pub type EvaluationRequest<'a> = (&'a Game, Vec<usize>, Color);

// ---------------------------------------------------------------------------
// Evaluator trait — implemented by the Python binding
// ---------------------------------------------------------------------------

pub trait Evaluator {
    /// Evaluate a single game position.
    /// Returns (priors: action_index -> probability, value, uncertainty).
    /// `root_color` is the color of the player at the root of the search —
    /// the evaluator must flip the value sign when `acting_color != root_color`.
    fn evaluate(
        &mut self,
        game: &Game,
        legal_action_indices: &[usize],
        root_color: Color,
    ) -> Result<Evaluation, String>;

    /// Evaluate a search root.  The default is deliberately identical to a
    /// normal leaf; the Python bridge overrides this only when D6 root
    /// averaging is explicitly enabled.  Keeping the distinction native
    /// prevents an interior node in the same Catan turn from accidentally
    /// receiving the expensive/root-only D6 treatment.
    fn evaluate_root(
        &mut self,
        game: &Game,
        legal_action_indices: &[usize],
        root_color: Color,
    ) -> Result<Evaluation, String> {
        self.evaluate(game, legal_action_indices, root_color)
    }

    /// Evaluate multiple game positions in a batch.
    /// Default implementation calls evaluate() one at a time.
    fn evaluate_many(
        &mut self,
        requests: &[EvaluationRequest<'_>],
    ) -> Result<Vec<Evaluation>, String> {
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
    pub map_kind: MapKind,
    pub max_depth: i32,
    pub seed: u64,
    pub max_root_candidates: usize,
    pub max_root_candidates_wide: usize,
    pub wide_candidates_threshold: usize,
    pub exact_budget_sh: bool,
    pub exact_budget_sh_min_n: i32,
    pub c_visit: f64,
    pub c_scale: f64,
    /// Fixed max-child-visit reference for budget-invariant sigma calibration.
    /// None preserves the historical realized-max-visits transform.
    pub sigma_reference_visits: Option<i32>,
    pub temperature: f64,
    pub play_sh_winner: bool,
    pub prior_temperature: f64,
    pub n_full: i32,
    pub n_fast: i32,
    pub p_full: f64,
    pub n_full_wide: Option<i32>,
    pub n_full_wide_threshold: Option<usize>,
    pub wide_roots_always_full: bool,
    pub raw_policy_above_width: Option<usize>,
    pub lazy_interior_chance: bool,
    pub root_candidate_cap: Option<usize>,
    pub policy_target_min_visits: i32,
    pub rescale_noise_floor_c: f64,
    /// Apply D1 only at the attested BUILD_INITIAL_ROAD decision root.
    /// Interior nodes and every other public root keep historical min-max Q.
    pub rescale_noise_floor_initial_road_only: bool,
    /// Public root prompt attested by the authoritative Python orchestration
    /// layer. Information-set particles must never supply this from their
    /// sampled hidden-world state.
    pub attested_root_phase: Option<String>,
    pub sigma_eval: f64,
    pub variance_aware_q: bool,
    pub variance_aware_k: f64,
    pub variance_aware_closed_form_js: bool,
    pub uncertainty_backup_weighting: bool,
    pub uncertainty_backup_a: f64,
    pub uncertainty_backup_exp: f64,
    pub uncertainty_backup_cap: f64,
    /// Stop a determinized tree as soon as play leaves the root actor's turn.
    /// This is the actor-turn PIMC horizon used by information-set search.
    pub stop_at_root_turn_boundary: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            colors: vec![Color::Red, Color::Blue],
            map_kind: MapKind::Base,
            max_depth: 80,
            seed: 0,
            max_root_candidates: 16,
            max_root_candidates_wide: 54,
            wide_candidates_threshold: 24,
            exact_budget_sh: false,
            exact_budget_sh_min_n: 0,
            c_visit: 50.0,
            c_scale: 0.1,
            sigma_reference_visits: None,
            temperature: 0.0,
            play_sh_winner: false,
            prior_temperature: 1.0,
            n_full: 64,
            n_fast: 16,
            p_full: 0.25,
            n_full_wide: None,
            n_full_wide_threshold: None,
            wide_roots_always_full: false,
            raw_policy_above_width: None,
            lazy_interior_chance: false,
            root_candidate_cap: None,
            policy_target_min_visits: 0,
            rescale_noise_floor_c: 0.0,
            rescale_noise_floor_initial_road_only: false,
            attested_root_phase: None,
            sigma_eval: 0.79,
            variance_aware_q: false,
            variance_aware_k: 1.0,
            variance_aware_closed_form_js: false,
            uncertainty_backup_weighting: false,
            uncertainty_backup_a: 0.25,
            uncertainty_backup_exp: 1.0,
            uncertainty_backup_cap: 1.0,
            stop_at_root_turn_boundary: false,
        }
    }
}

#[inline]
fn calibrated_sigma_scale(
    c_visit: f64,
    c_scale: f64,
    realized_max_visits: i32,
    reference_visits: Option<i32>,
) -> f64 {
    let visits = reference_visits.unwrap_or(realized_max_visits);
    (c_visit + visits as f64) * c_scale
}

fn temperature_scale_policy(
    policy: &[(usize, f64)],
    temperature: f64,
) -> Result<Vec<(usize, f64)>, String> {
    if !temperature.is_finite() || temperature <= 0.0 {
        return Err("policy sampling temperature must be finite and > 0".into());
    }
    if temperature == 1.0 || policy.is_empty() {
        return Ok(policy.to_vec());
    }
    let mut log_weights = Vec::with_capacity(policy.len());
    let mut max_log_weight = f64::NEG_INFINITY;
    for &(action, probability) in policy {
        let log_weight = if probability.is_finite() && probability > 0.0 {
            probability.ln() / temperature
        } else {
            f64::NEG_INFINITY
        };
        max_log_weight = max_log_weight.max(log_weight);
        log_weights.push((action, log_weight));
    }
    if !max_log_weight.is_finite() {
        return Err("policy sampling distribution has no positive finite mass".into());
    }
    let mut scaled = Vec::with_capacity(policy.len());
    let mut total = 0.0;
    for (action, log_weight) in log_weights {
        let weight = if log_weight.is_finite() {
            (log_weight - max_log_weight).exp()
        } else {
            0.0
        };
        total += weight;
        scaled.push((action, weight));
    }
    if !total.is_finite() || total <= 0.0 {
        return Err("policy sampling temperature normalization failed".into());
    }
    for (_, probability) in &mut scaled {
        *probability /= total;
    }
    Ok(scaled)
}

// ---------------------------------------------------------------------------
// Tree nodes
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct ActionStats {
    prior: f64,
    visits: i32,
    value_sum: f64,
    value_sq_sum: f64,
    weighted_value_sum: f64,
    weight_sum: f64,
    children: HashMap<usize, usize>,
    probabilities: HashMap<usize, f64>,
    afterstate_value: Option<f64>,
}

impl ActionStats {
    fn new(prior: f64) -> Self {
        Self {
            prior,
            visits: 0,
            value_sum: 0.0,
            value_sq_sum: 0.0,
            weighted_value_sum: 0.0,
            weight_sum: 0.0,
            children: HashMap::new(),
            probabilities: HashMap::new(),
            afterstate_value: None,
        }
    }
    #[inline]
    fn q(&self) -> f64 {
        if self.visits <= 0 {
            0.0
        } else {
            self.value_sum / self.visits as f64
        }
    }
    #[inline]
    fn weighted_q(&self) -> f64 {
        if self.weight_sum <= 0.0 {
            self.q()
        } else {
            self.weighted_value_sum / self.weight_sum
        }
    }
    #[inline]
    fn q_variance(&self) -> f64 {
        if self.visits < 2 {
            return 0.0;
        }
        let mean = self.value_sum / self.visits as f64;
        let mean_sq = self.value_sq_sum / self.visits as f64;
        (mean_sq - mean * mean).max(0.0)
    }
}

#[derive(Clone, Debug)]
struct Node {
    game: Game,
    root_color: Color,
    prior_value: f64,
    prior_uncertainty: f64,
    visits: i32,
    value_sum: f64,
    actions: HashMap<usize, ActionStats>,
    action_logits: HashMap<usize, f64>,
    playable_actions: Vec<Action>,
    expanded: bool,
}

impl Node {
    fn new(game: Game, root_color: Color) -> Self {
        Self {
            game,
            root_color,
            prior_value: 0.0,
            prior_uncertainty: 0.0,
            visits: 0,
            value_sum: 0.0,
            actions: HashMap::new(),
            action_logits: HashMap::new(),
            playable_actions: Vec::new(),
            expanded: false,
        }
    }
    #[inline]
    fn value(&self) -> f64 {
        if self.visits <= 0 {
            self.prior_value
        } else {
            self.value_sum / self.visits as f64
        }
    }
}

struct Arena {
    nodes: Vec<Node>,
}
impl Arena {
    fn new() -> Self {
        Self {
            nodes: Vec::with_capacity(512),
        }
    }
    #[inline]
    fn alloc(&mut self, node: Node) -> usize {
        let i = self.nodes.len();
        self.nodes.push(node);
        i
    }
    #[inline]
    fn get(&self, i: usize) -> &Node {
        &self.nodes[i]
    }
    #[inline]
    fn get_mut(&mut self, i: usize) -> &mut Node {
        &mut self.nodes[i]
    }
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
    /// Root-actor completed-Q for every legal root action.  Unvisited actions
    /// carry the ordinary per-world mctx mixed-value completion.
    pub completed_q_values: Vec<(usize, f64)>,
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
    root_turn: usize,
    /// Immutable for one search recipe; building this per action was
    /// O(legal-width x full-action-space) and erased the native-loop win.
    action_space: ActionSpace,
}

impl GumbelMctsEngine {
    pub fn new(config: SearchConfig) -> Self {
        assert!(config.temperature.is_finite(), "temperature must be finite");
        assert!(
            config
                .sigma_reference_visits
                .map_or(true, |visits| visits >= 0),
            "sigma_reference_visits must be non-negative"
        );
        let action_space = ActionSpace::new(&config.colors, config.map_kind);
        Self {
            rng: ChaCha8Rng::seed_from_u64(config.seed),
            config,
            root_turn: 0,
            action_space,
        }
    }

    fn action_ids(&self, actions: &[Action]) -> Result<Vec<usize>, String> {
        actions
            .iter()
            .enumerate()
            .map(|(local, action)| {
                self.action_space.index(action).ok_or_else(|| {
                    format!("playable action {local} is absent from configured action space")
                })
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Public search
    // -----------------------------------------------------------------------
    pub fn search<E: Evaluator>(
        &mut self,
        game: &Game,
        evaluator: &mut E,
        force_full: Option<bool>,
    ) -> Result<SearchResult, String> {
        if self.config.rescale_noise_floor_initial_road_only
            && self
                .config
                .attested_root_phase
                .as_deref()
                .map_or(true, str::is_empty)
        {
            return Err(
                "initial-road-only D1 requires a non-empty authoritative root-phase attestation"
                    .into(),
            );
        }
        let root_color = game.state.current_color();
        self.root_turn = game.state.num_turns;
        let legal_actions = generate_playable_actions(&game.state);
        if legal_actions.is_empty() {
            return Err("no legal actions".into());
        }
        if legal_actions.len() == 1 {
            return self.forced_single_action(game, &legal_actions, root_color, evaluator);
        }
        if let Some(width) = self.config.raw_policy_above_width {
            if legal_actions.len() > width {
                return self.raw_policy_root(game, root_color, &legal_actions, evaluator);
            }
        }
        let wide_budget_root = self.config.n_full_wide.is_some()
            && match self.config.n_full_wide_threshold {
                Some(threshold) => legal_actions.len() >= threshold,
                None => legal_actions.len() > self.config.wide_candidates_threshold,
            };
        let use_full = match force_full {
            Some(f) => f,
            None if wide_budget_root && self.config.wide_roots_always_full => true,
            None => self.rng.gen::<f64>() < self.config.p_full,
        };
        let n_full_eff = if let Some(nw) = self.config.n_full_wide {
            if wide_budget_root {
                nw
            } else {
                self.config.n_full
            }
        } else {
            self.config.n_full
        };
        let n_simulations = if use_full {
            n_full_eff
        } else {
            self.config.n_fast
        }
        .max(1);

        let mut root_node = Node::new(game.clone(), root_color);
        root_node.playable_actions = legal_actions.clone();
        let mut arena = Arena::new();
        let root_idx = arena.alloc(root_node);
        self.expand_root_node(&mut arena, root_idx, evaluator)?;

        let (sh_winner, used) =
            self.run_root_search(&mut arena, root_idx, n_simulations, evaluator)?;
        let completed_q = self.completed_q(&arena, root_idx, root_color);
        let improved_policy = self.improved_policy(&arena, root_idx, &completed_q);
        let root = arena.get(root_idx);
        let result_action_space = self.action_space.clone();
        let to_global = |local: usize| -> Result<usize, String> {
            result_action_space
                .index(&root.playable_actions[local])
                .ok_or_else(|| {
                    format!("playable action {local} is absent from configured action space")
                })
        };
        let local_visit_counts: Vec<_> = root
            .actions
            .iter()
            .map(|(&local, stats)| (local, stats.visits))
            .collect();
        let mut visit_counts: Vec<_> = local_visit_counts
            .iter()
            .map(|(local, visits)| Ok((to_global(*local)?, *visits)))
            .collect::<Result<_, String>>()?;
        let mut q_values: Vec<_> = root
            .actions
            .iter()
            .filter(|(_, s)| s.visits > 0)
            .map(|(&i, s)| Ok((to_global(i)?, s.q())))
            .collect::<Result<_, String>>()?;
        let mut priors: Vec<_> = root
            .actions
            .iter()
            .map(|(&i, s)| Ok((to_global(i)?, s.prior)))
            .collect::<Result<_, String>>()?;
        let mut completed_q_values: Vec<_> = completed_q
            .iter()
            .map(|(&local, value)| Ok((to_global(local)?, *value)))
            .collect::<Result<_, String>>()?;
        let mut afterstate_values: Vec<_> = root
            .actions
            .iter()
            .filter_map(|(&i, s)| s.afterstate_value.map(|v| Ok((to_global(i)?, v))))
            .collect::<Result<_, String>>()?;

        let selected = if self.config.play_sh_winner {
            sh_winner
        } else if self.config.temperature > 0.0 {
            let sampling_policy =
                temperature_scale_policy(&improved_policy, self.config.temperature)?;
            self.sample_categorical(&sampling_policy)
        } else {
            improved_policy
                .iter()
                .max_by(|a, b| {
                    a.1.partial_cmp(&b.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| root.actions[&a.0].visits.cmp(&root.actions[&b.0].visits))
                        .then_with(|| {
                            root.actions[&a.0]
                                .prior
                                .partial_cmp(&root.actions[&b.0].prior)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .then_with(|| {
                            // Python max() preserves the first legal-action
                            // insertion on an exact policy/visit/prior tie.
                            // Local indices follow that playable-action order;
                            // global-ID order can differ on real states.
                            b.0.cmp(&a.0)
                        })
                })
                .map(|(i, _)| *i)
                .unwrap_or(sh_winner)
        };

        let training_policy = if self.config.policy_target_min_visits > 0 {
            self.prune_policy_target(&improved_policy, &local_visit_counts)
        } else {
            improved_policy.clone()
        };

        let selected_global = to_global(selected)?;
        let mut training_policy = training_policy
            .into_iter()
            .map(|(local, probability)| Ok((to_global(local)?, probability)))
            .collect::<Result<Vec<_>, String>>()?;
        visit_counts.sort_unstable_by_key(|(action, _)| *action);
        q_values.sort_unstable_by_key(|(action, _)| *action);
        priors.sort_unstable_by_key(|(action, _)| *action);
        completed_q_values.sort_unstable_by_key(|(action, _)| *action);
        afterstate_values.sort_unstable_by_key(|(action, _)| *action);
        training_policy.sort_unstable_by_key(|(action, _)| *action);
        Ok(SearchResult {
            selected_action: selected_global,
            improved_policy: training_policy,
            visit_counts,
            q_values,
            priors,
            root_value: root.value(),
            completed_q_values,
            used_full_search: use_full,
            simulations_used: used,
            afterstate_values,
        })
    }

    // -----------------------------------------------------------------------
    // Node expansion — calls evaluator
    // -----------------------------------------------------------------------
    fn expand_node<E: Evaluator>(
        &mut self,
        arena: &mut Arena,
        node_idx: usize,
        evaluator: &mut E,
    ) -> Result<f64, String> {
        let node = arena.get(node_idx);
        let root_color = node.root_color;
        let legal_actions = generate_playable_actions(&node.game.state);
        let legal_indices = self.action_ids(&legal_actions)?;
        let (priors, value, uncertainty) =
            evaluator.evaluate(&node.game, &legal_indices, root_color)?;
        self.finish_expand(arena, node_idx, legal_actions, priors, value, uncertainty)
    }

    fn expand_root_node<E: Evaluator>(
        &mut self,
        arena: &mut Arena,
        node_idx: usize,
        evaluator: &mut E,
    ) -> Result<f64, String> {
        let node = arena.get(node_idx);
        let root_color = node.root_color;
        let legal_actions = generate_playable_actions(&node.game.state);
        let legal_indices = self.action_ids(&legal_actions)?;
        let (priors, value, uncertainty) =
            evaluator.evaluate_root(&node.game, &legal_indices, root_color)?;
        self.finish_expand(arena, node_idx, legal_actions, priors, value, uncertainty)
    }

    fn finish_expand(
        &self,
        arena: &mut Arena,
        node_idx: usize,
        legal_actions: Vec<Action>,
        priors: HashMap<usize, f64>,
        value: f64,
        uncertainty: f64,
    ) -> Result<f64, String> {
        let node = arena.get_mut(node_idx);
        if !legal_actions.is_empty() {
            let mut local_priors = HashMap::with_capacity(legal_actions.len());
            for (local, action) in legal_actions.iter().enumerate() {
                let global = self.action_space.index(action).ok_or_else(|| {
                    format!("playable action {local} is absent from configured action space")
                })?;
                if let Some(probability) = priors.get(&global) {
                    local_priors.insert(local, *probability);
                }
            }
            let floor = local_priors
                .values()
                .filter(|&&p| p > 0.0)
                .fold(f64::INFINITY, |a, &b| a.min(b));
            let floor = if floor == f64::INFINITY {
                1.0
            } else {
                floor * 0.01
            };
            for i in 0..legal_actions.len() {
                local_priors.entry(i).or_insert(floor);
            }
            let total: f64 = (0..legal_actions.len())
                .map(|i| local_priors.get(&i).copied().unwrap_or(0.0))
                .sum();
            let prior_temp = self.config.prior_temperature.max(1e-6);
            node.playable_actions = legal_actions;
            node.actions.clear();
            node.action_logits.clear();
            for i in 0..node.playable_actions.len() {
                let p = local_priors.get(&i).copied().unwrap_or(0.0) / total;
                node.actions.insert(i, ActionStats::new(p));
                node.action_logits
                    .insert(i, (p.max(1e-8)).ln() / prior_temp);
            }
        }
        node.prior_value = value.clamp(-1.0, 1.0);
        node.prior_uncertainty = uncertainty.max(0.0);
        node.expanded = true;
        Ok(node.prior_value)
    }

    // -----------------------------------------------------------------------
    // Root search
    // -----------------------------------------------------------------------
    fn run_root_search<E: Evaluator>(
        &mut self,
        arena: &mut Arena,
        root_idx: usize,
        n_simulations: i32,
        evaluator: &mut E,
    ) -> Result<(usize, i32), String> {
        let mut legal: Vec<usize> = arena.get(root_idx).actions.keys().copied().collect();
        legal.sort_unstable();
        let num_legal = legal.len();
        let m = self.root_candidate_count(num_legal);
        let mut gumbel: HashMap<usize, f64> = HashMap::new();
        for &aid in &legal {
            gumbel.insert(aid, self.sample_gumbel());
        }
        let logits: HashMap<usize, f64> = arena.get(root_idx).action_logits.clone();
        let mut top_k: Vec<usize> = legal.clone();
        top_k.sort_by(|&a, &b| {
            let sa = gumbel[&a] + logits.get(&a).copied().unwrap_or(0.0);
            let sb = gumbel[&b] + logits.get(&b).copied().unwrap_or(0.0);
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });
        top_k.truncate(m);
        let mut remaining = top_k.clone();

        if self.config.exact_budget_sh && n_simulations >= self.config.exact_budget_sh_min_n {
            let phases = exact_budget_sh_phases(m as i32, n_simulations);
            let mut used = 0;
            for &(count, budget) in &phases {
                let mut visit: Vec<usize> =
                    remaining.iter().take(count as usize).copied().collect();
                for &aid in &visit {
                    for _ in 0..budget {
                        self.simulate(arena, root_idx, 0, Some(aid), evaluator)?;
                        used += 1;
                    }
                }
                let cq = self.completed_q(arena, root_idx, arena.get(root_idx).root_color);
                let rq = self.rescaled_completed_q_with_noise(arena, root_idx, &cq);
                let scale = self.sigma_scale(arena, root_idx);
                visit = rerank_candidates(visit, |action| {
                    gumbel[&action]
                        + logits.get(&action).copied().unwrap_or(0.0)
                        + scale * rq.get(&action).copied().unwrap_or(0.0)
                });
                // Exact-budget phases eliminate candidates just like the
                // reference SH operator. Sorting the old full `remaining`
                // vector let previously eliminated actions re-enter/win.
                remaining = visit;
            }
            return Ok((*remaining.first().unwrap_or(&top_k[0]), used));
        }

        let schedule = sequential_halving_schedule(m as i32, n_simulations);
        let mut used = 0;
        for &(count, budget) in &schedule {
            for &aid in &remaining {
                for _ in 0..budget {
                    self.simulate(arena, root_idx, 0, Some(aid), evaluator)?;
                    used += 1;
                }
            }
            let cq = self.completed_q(arena, root_idx, arena.get(root_idx).root_color);
            let rq = self.rescaled_completed_q_with_noise(arena, root_idx, &cq);
            let scale = self.sigma_scale(arena, root_idx);
            remaining.sort_by(|&a, &b| {
                let sa = gumbel[&a]
                    + logits.get(&a).copied().unwrap_or(0.0)
                    + scale * rq.get(&a).copied().unwrap_or(0.0);
                let sb = gumbel[&b]
                    + logits.get(&b).copied().unwrap_or(0.0)
                    + scale * rq.get(&b).copied().unwrap_or(0.0);
                sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
            });
            remaining.truncate((count / 2).max(1) as usize);
        }
        Ok((*remaining.first().unwrap_or(&top_k[0]), used))
    }

    // -----------------------------------------------------------------------
    // Simulation — ALL in Rust
    // -----------------------------------------------------------------------
    fn simulate<E: Evaluator>(
        &mut self,
        arena: &mut Arena,
        node_idx: usize,
        depth: i32,
        forced_action: Option<usize>,
        evaluator: &mut E,
    ) -> Result<f64, String> {
        let winner = arena.get(node_idx).game.winning_color();
        if let Some(w) = winner {
            let root_color = arena.get(node_idx).root_color;
            return Ok(if w == root_color { 1.0 } else { -1.0 });
        }
        if self.is_root_turn_boundary(arena, node_idx, depth) {
            if !arena.get(node_idx).expanded {
                self.expand_node(arena, node_idx, evaluator)?;
            }
            let value = arena.get(node_idx).prior_value;
            let node = arena.get_mut(node_idx);
            node.visits += 1;
            node.value_sum += value;
            return Ok(value);
        }
        if depth >= self.config.max_depth {
            if !arena.get(node_idx).expanded {
                self.expand_node(arena, node_idx, evaluator)?;
            }
            return Ok(arena.get(node_idx).prior_value);
        }
        if !arena.get(node_idx).expanded {
            let legal = generate_playable_actions(&arena.get(node_idx).game.state);
            if legal.len() == 1 {
                let root_color = arena.get(node_idx).root_color;
                let node = arena.get_mut(node_idx);
                node.playable_actions = legal;
                node.actions.insert(0, ActionStats::new(1.0));
                node.action_logits.insert(0, 0.0);
                let winner = node.game.winning_color();
                node.prior_value = match winner {
                    Some(w) => {
                        if w == root_color {
                            1.0
                        } else {
                            -1.0
                        }
                    }
                    None => 0.0,
                };
                node.expanded = true;
                return self.simulate(arena, node_idx, depth, forced_action, evaluator);
            }
            let value = self.expand_node(arena, node_idx, evaluator)?;
            let node = arena.get_mut(node_idx);
            node.visits += 1;
            node.value_sum += value;
            return Ok(value);
        }
        if arena.get(node_idx).actions.is_empty() {
            let value = arena.get(node_idx).prior_value;
            let node = arena.get_mut(node_idx);
            node.visits += 1;
            node.value_sum += value;
            return Ok(value);
        }
        let action_idx = if let Some(fa) = forced_action {
            fa
        } else {
            self.select_nonroot_action(arena, node_idx)?
        };
        let expectation_backup =
            self.expectation_backup(&arena.get(node_idx).playable_actions[action_idx], depth);
        let value = if expectation_backup {
            self.traverse_roll(arena, node_idx, action_idx, depth, evaluator)?
        } else {
            self.traverse_single_sample(arena, node_idx, action_idx, depth, evaluator)?
        };
        let node = arena.get_mut(node_idx);
        node.visits += 1;
        node.value_sum += value;
        Ok(value)
    }

    #[inline]
    fn is_root_turn_boundary(&self, arena: &Arena, node_idx: usize, depth: i32) -> bool {
        if !self.config.stop_at_root_turn_boundary || depth <= 0 {
            return false;
        }
        let node = arena.get(node_idx);
        node.game.state.current_color() != node.root_color
            || node.game.state.num_turns != self.root_turn
    }

    #[inline]
    fn expectation_backup(&self, action: &Action, depth: i32) -> bool {
        match action.action_type {
            ActionType::Roll => !(self.config.lazy_interior_chance && depth > 0),
            // F7: these small hidden-outcome spaces are exact expectation
            // backups in the reference search, never single samples.
            ActionType::MoveRobber => {
                matches!(action.value, ActionValue::Robber(_, Some(_)))
            }
            ActionType::BuyDevelopmentCard => true,
            _ => false,
        }
    }

    // -----------------------------------------------------------------------
    // Chance traversal
    // -----------------------------------------------------------------------
    fn traverse_roll<E: Evaluator>(
        &mut self,
        arena: &mut Arena,
        node_idx: usize,
        action_idx: usize,
        depth: i32,
        evaluator: &mut E,
    ) -> Result<f64, String> {
        let needs_enum = arena
            .get(node_idx)
            .actions
            .get(&action_idx)
            .is_none_or(|s| s.children.is_empty());
        if needs_enum {
            self.enumerate_outcomes(arena, node_idx, action_idx, evaluator)?;
        }
        let outcome_index = {
            let stats = arena.get(node_idx).actions.get(&action_idx).unwrap();
            let mut probs: Vec<(usize, f64)> =
                stats.probabilities.iter().map(|(&k, &v)| (k, v)).collect();
            probs.sort_unstable_by_key(|(index, _)| *index);
            self.sample_outcome(&probs)
        };
        let child_idx = arena
            .get(node_idx)
            .actions
            .get(&action_idx)
            .unwrap()
            .children[&outcome_index];
        self.simulate(arena, child_idx, depth + 1, None, evaluator)?;
        let value = {
            let stats = arena.get(node_idx).actions.get(&action_idx).unwrap();
            let mut v = 0.0;
            let mut child_indices = stats.children.keys().copied().collect::<Vec<_>>();
            child_indices.sort_unstable();
            for idx in child_indices {
                let child_idx = stats.children[&idx];
                let prob = stats.probabilities.get(&idx).copied().unwrap_or(0.0);
                v += prob * arena.get(child_idx).value();
            }
            v
        };
        let node = arena.get_mut(node_idx);
        let stats = node.actions.get_mut(&action_idx).unwrap();
        stats.visits += 1;
        stats.value_sum += value;
        stats.value_sq_sum += value * value;
        if self.config.uncertainty_backup_weighting {
            let w = self.backup_weight(node.prior_uncertainty);
            stats.weight_sum += w;
            stats.weighted_value_sum += w * value;
        }
        Ok(value)
    }

    fn traverse_single_sample<E: Evaluator>(
        &mut self,
        arena: &mut Arena,
        node_idx: usize,
        action_idx: usize,
        depth: i32,
        evaluator: &mut E,
    ) -> Result<f64, String> {
        let child_exists = arena
            .get(node_idx)
            .actions
            .get(&action_idx)
            .is_some_and(|s| !s.children.is_empty());
        if !child_exists {
            let root_color = arena.get(node_idx).root_color;
            let outcomes = {
                let node = arena.get(node_idx);
                execute_spectrum(&node.game, &node.playable_actions[action_idx])
            };
            let total_prob: f64 = outcomes.iter().map(|(_, p)| *p).sum();
            let probs: Vec<f64> = outcomes.iter().map(|(_, p)| *p / total_prob).collect();
            let mut new_child_indices: Vec<usize> = Vec::with_capacity(outcomes.len());
            for (child_game, _) in outcomes {
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
            let mut probs: Vec<(usize, f64)> =
                stats.probabilities.iter().map(|(&k, &v)| (k, v)).collect();
            probs.sort_unstable_by_key(|(index, _)| *index);
            self.sample_outcome(&probs)
        };
        let child_idx = arena
            .get(node_idx)
            .actions
            .get(&action_idx)
            .unwrap()
            .children[&outcome_index];
        let value = self.simulate(arena, child_idx, depth + 1, None, evaluator)?;
        let child_unc = if self.config.uncertainty_backup_weighting {
            arena.get(child_idx).prior_uncertainty
        } else {
            0.0
        };
        let node = arena.get_mut(node_idx);
        let stats = node.actions.get_mut(&action_idx).unwrap();
        stats.visits += 1;
        stats.value_sum += value;
        stats.value_sq_sum += value * value;
        if self.config.uncertainty_backup_weighting {
            let w = self.backup_weight(child_unc);
            stats.weight_sum += w;
            stats.weighted_value_sum += w * value;
        }
        Ok(value)
    }

    fn enumerate_outcomes<E: Evaluator>(
        &mut self,
        arena: &mut Arena,
        node_idx: usize,
        action_idx: usize,
        evaluator: &mut E,
    ) -> Result<(), String> {
        let root_color = arena.get(node_idx).root_color;
        let outcomes = {
            let node = arena.get(node_idx);
            execute_spectrum(&node.game, &node.playable_actions[action_idx])
        };
        let total_prob: f64 = outcomes.iter().map(|(_, p)| *p).sum();
        if total_prob <= 0.0 || outcomes.is_empty() {
            return Ok(());
        }
        let mut outcome_probabilities = Vec::with_capacity(outcomes.len());
        let mut new_child_indices: Vec<usize> = Vec::with_capacity(outcomes.len());
        for (child_game, probability) in outcomes {
            outcome_probabilities.push(probability);
            let child_node = Node::new(child_game, root_color);
            new_child_indices.push(arena.alloc(child_node));
        }
        // Evaluate all children — batch if possible
        let mut requests = Vec::with_capacity(new_child_indices.len());
        for &child_idx in &new_child_indices {
            let child_game = &arena.get(child_idx).game;
            let legal = generate_playable_actions(&child_game.state);
            let legal_indices = self.action_ids(&legal)?;
            requests.push((child_game, legal_indices, root_color));
        }
        let results = evaluator.evaluate_many(&requests)?;
        if results.len() != requests.len() {
            return Err(format!(
                "evaluator batch length mismatch: requested {}, received {}",
                requests.len(),
                results.len()
            ));
        }
        for (i, &child_idx) in new_child_indices.iter().enumerate() {
            let (priors, value, unc) = &results[i];
            let legal = generate_playable_actions(&arena.get(child_idx).game.state);
            self.finish_expand(arena, child_idx, legal, priors.clone(), *value, *unc)?;
        }
        let afterstate_value: f64 = {
            let mut weighted = 0.0;
            for (i, prob) in outcome_probabilities.iter().enumerate() {
                weighted += prob * arena.get(new_child_indices[i]).prior_value;
            }
            weighted / total_prob
        };
        let node = arena.get_mut(node_idx);
        let stats = node.actions.get_mut(&action_idx).unwrap();
        stats.afterstate_value = Some(afterstate_value);
        for (i, prob) in outcome_probabilities.into_iter().enumerate() {
            stats.probabilities.insert(i, prob / total_prob);
            stats.children.insert(i, new_child_indices[i]);
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Completed-Q / improved policy
    // -----------------------------------------------------------------------
    fn sigma_scale(&self, arena: &Arena, node_idx: usize) -> f64 {
        let max_visits = self.config.sigma_reference_visits.unwrap_or_else(|| {
            arena
                .get(node_idx)
                .actions
                .values()
                .map(|s| s.visits)
                .max()
                .unwrap_or(0)
        });
        calibrated_sigma_scale(
            self.config.c_visit,
            self.config.c_scale,
            max_visits,
            self.config.sigma_reference_visits,
        )
    }

    fn completed_q(
        &self,
        arena: &Arena,
        node_idx: usize,
        root_color: Color,
    ) -> HashMap<usize, f64> {
        let node = arena.get(node_idx);
        let root_to_act = node.game.state.current_color() == root_color;
        let sign = if root_to_act { 1.0 } else { -1.0 };
        let use_weighted = self.config.uncertainty_backup_weighting;
        let total_child_visits: i32 = node.actions.values().map(|s| s.visits).sum();
        let mut visited_prior_sum = 0.0;
        let mut visited_q_sum = 0.0;
        let mut action_ids = node.actions.keys().copied().collect::<Vec<_>>();
        action_ids.sort_unstable();
        for action_id in action_ids {
            let stats = &node.actions[&action_id];
            if stats.visits > 0 {
                visited_prior_sum += stats.prior;
                let q = if use_weighted {
                    stats.weighted_q()
                } else {
                    stats.q()
                };
                visited_q_sum += stats.prior * (sign * q);
            }
        }
        let weighted_q = if visited_prior_sum > 0.0 {
            visited_q_sum / visited_prior_sum
        } else {
            0.0
        };
        let node_value = sign * node.prior_value;
        let v_mix = (node_value + total_child_visits as f64 * weighted_q)
            / (1.0 + total_child_visits as f64);
        let mut completed: HashMap<usize, f64> = HashMap::new();
        for (&aid, stats) in &node.actions {
            if stats.visits > 0 {
                let q = if use_weighted {
                    stats.weighted_q()
                } else {
                    stats.q()
                };
                completed.insert(aid, sign * q);
            } else {
                completed.insert(aid, v_mix);
            }
        }
        if self.config.variance_aware_q {
            self.shrink_completed_q_by_variance(arena, node_idx, &mut completed, v_mix);
        }
        completed
    }

    fn shrink_completed_q_by_variance(
        &self,
        arena: &Arena,
        node_idx: usize,
        completed: &mut HashMap<usize, f64>,
        v_mix: f64,
    ) {
        let node = arena.get(node_idx);
        let mut visited: Vec<(usize, &ActionStats)> = node
            .actions
            .iter()
            .filter(|(_, s)| s.visits > 0)
            .map(|(&k, v)| (k, v))
            .collect();
        visited.sort_unstable_by_key(|(action_id, _)| *action_id);
        if visited.len() < 2 {
            return;
        }
        let visited_qs: Vec<f64> = visited.iter().map(|(id, _)| completed[id]).collect();
        let mean_q: f64 = visited_qs.iter().sum::<f64>() / visited_qs.len() as f64;
        let signal_var: f64 =
            visited_qs.iter().map(|q| (q - mean_q).powi(2)).sum::<f64>() / visited_qs.len() as f64;
        if signal_var <= 0.0 {
            return;
        }
        if self.config.variance_aware_closed_form_js {
            let se_sqs: Vec<f64> = visited
                .iter()
                .map(|(_, s)| s.q_variance() / s.visits as f64)
                .collect();
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
        if cq.is_empty() {
            return HashMap::new();
        }
        let min_q = cq.values().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_q = cq.values().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let denom = (max_q - min_q) + 1e-8;
        cq.iter().map(|(&k, &v)| (k, (v - min_q) / denom)).collect()
    }

    fn rescaled_completed_q_with_noise(
        &self,
        arena: &Arena,
        node_idx: usize,
        cq: &HashMap<usize, f64>,
    ) -> HashMap<usize, f64> {
        let rescaled = self.rescale_completed_q(cq);
        self.apply_noise_floor(arena, node_idx, cq, &rescaled)
    }

    fn apply_noise_floor(
        &self,
        arena: &Arena,
        node_idx: usize,
        cq: &HashMap<usize, f64>,
        rescaled: &HashMap<usize, f64>,
    ) -> HashMap<usize, f64> {
        let c = self.config.rescale_noise_floor_c;
        if c <= 0.0 || rescaled.is_empty() {
            return rescaled.clone();
        }
        if self.config.rescale_noise_floor_initial_road_only
            && (node_idx != 0
                || self.config.attested_root_phase.as_deref() != Some("BUILD_INITIAL_ROAD"))
        {
            return rescaled.clone();
        }
        let values: Vec<f64> = cq.values().copied().collect();
        let raw_spread = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b))
            - values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let visits: Vec<i32> = arena
            .get(node_idx)
            .actions
            .values()
            .map(|s| s.visits)
            .collect();
        let mean_visits = if visits.is_empty() {
            0.0
        } else {
            visits.iter().map(|&v| v as f64).sum::<f64>() / visits.len() as f64
        };
        let noise_floor = if mean_visits <= 0.0 {
            f64::INFINITY
        } else {
            c * self.config.sigma_eval / mean_visits.sqrt()
        };
        let denom = raw_spread + noise_floor;
        let alpha = if denom <= 0.0 || noise_floor.is_infinite() {
            0.0
        } else {
            raw_spread / denom
        };
        rescaled
            .iter()
            .map(|(&k, &v)| (k, 0.5 + alpha * (v - 0.5)))
            .collect()
    }

    fn improved_policy(
        &self,
        arena: &Arena,
        node_idx: usize,
        cq: &HashMap<usize, f64>,
    ) -> Vec<(usize, f64)> {
        let scale = self.sigma_scale(arena, node_idx);
        let rq = self.rescaled_completed_q_with_noise(arena, node_idx, cq);
        let node = arena.get(node_idx);
        let mut scores: Vec<(usize, f64)> = node
            .actions
            .keys()
            .map(|&aid| {
                let logit = node.action_logits.get(&aid).copied().unwrap_or(0.0);
                let rqv = rq.get(&aid).copied().unwrap_or(0.0);
                (aid, logit + scale * rqv)
            })
            .collect();
        scores.sort_unstable_by_key(|(action_id, _)| *action_id);
        let max_score = scores.iter().fold(f64::NEG_INFINITY, |a, &(_, b)| a.max(b));
        for (_, s) in scores.iter_mut() {
            *s = ((*s - max_score).clamp(-40.0, 40.0)).exp();
        }
        let total: f64 = scores.iter().map(|(_, s)| *s).sum();
        if total > 0.0 {
            for (_, s) in scores.iter_mut() {
                *s /= total;
            }
        }
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
            if score > best_score {
                best_score = score;
                best = Some(*aid);
            }
        }
        best.ok_or_else(|| "cannot select from empty node".into())
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------
    #[inline]
    fn sample_gumbel(&mut self) -> f64 {
        let u = self.rng.gen::<f64>().clamp(1e-12, 1.0 - 1e-12);
        -(-u.ln()).ln()
    }
    #[inline]
    fn sample_outcome(&mut self, outcomes: &[(usize, f64)]) -> usize {
        if outcomes.len() == 1 {
            return outcomes[0].0;
        }
        let draw = self.rng.gen::<f64>();
        let mut cum = 0.0;
        for &(idx, prob) in outcomes {
            cum += prob;
            if draw <= cum {
                return idx;
            }
        }
        outcomes.last().map(|&(idx, _)| idx).unwrap_or(0)
    }
    #[inline]
    fn sample_categorical(&mut self, policy: &[(usize, f64)]) -> usize {
        if policy.is_empty() {
            return 0;
        }
        let draw = self.rng.gen::<f64>();
        let mut cum = 0.0;
        for &(aid, prob) in policy {
            cum += prob;
            if draw <= cum {
                return aid;
            }
        }
        policy.last().map(|&(aid, _)| aid).unwrap_or(0)
    }
    #[inline]
    fn backup_weight(&self, uncertainty: f64) -> f64 {
        let err = uncertainty.max(0.0);
        self.config
            .uncertainty_backup_cap
            .min(self.config.uncertainty_backup_a * err.powf(self.config.uncertainty_backup_exp))
    }
    fn root_candidate_count(&self, num_legal: usize) -> usize {
        if let Some(cap) = self.config.root_candidate_cap {
            num_legal.min(cap).max(1)
        } else if num_legal > self.config.wide_candidates_threshold {
            num_legal.min(self.config.max_root_candidates_wide).max(1)
        } else {
            num_legal.min(self.config.max_root_candidates).max(1)
        }
    }
    fn prune_policy_target(
        &self,
        policy: &[(usize, f64)],
        visits: &[(usize, i32)],
    ) -> Vec<(usize, f64)> {
        if self.config.policy_target_min_visits <= 0 || policy.is_empty() {
            return policy.to_vec();
        }
        let visit_map: HashMap<usize, i32> = visits.iter().copied().collect();
        let kept_mass: f64 = policy
            .iter()
            .filter(|(aid, _)| {
                visit_map.get(aid).copied().unwrap_or(0) >= self.config.policy_target_min_visits
            })
            .map(|(_, p)| *p)
            .sum();
        if kept_mass <= 0.0 {
            return policy.to_vec();
        }
        policy
            .iter()
            .map(|(aid, prob)| {
                if visit_map.get(aid).copied().unwrap_or(0) >= self.config.policy_target_min_visits
                {
                    (*aid, *prob / kept_mass)
                } else {
                    (*aid, 0.0)
                }
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // Special paths
    // -----------------------------------------------------------------------
    fn forced_single_action<E: Evaluator>(
        &mut self,
        game: &Game,
        legal: &[Action],
        root_color: Color,
        evaluator: &mut E,
    ) -> Result<SearchResult, String> {
        let action = &legal[0];
        let action_id = self.action_ids(legal)?[0];
        if action.action_type != ActionType::Roll {
            let legal_ids = self.action_ids(legal)?;
            let (_, value, _) = evaluator.evaluate_root(game, &legal_ids, root_color)?;
            return Ok(SearchResult {
                selected_action: action_id,
                improved_policy: vec![(action_id, 1.0)],
                visit_counts: vec![(action_id, 1)],
                q_values: vec![],
                priors: vec![(action_id, 1.0)],
                root_value: value.clamp(-1.0, 1.0),
                completed_q_values: vec![(action_id, value.clamp(-1.0, 1.0))],
                used_full_search: true,
                simulations_used: 0,
                afterstate_values: vec![],
            });
        }
        let outcomes = execute_spectrum(game, action);
        let total_prob: f64 = outcomes.iter().map(|(_, p)| *p).sum();
        let mut requests = Vec::with_capacity(outcomes.len());
        for (child_game, _) in &outcomes {
            let child_legal = generate_playable_actions(&child_game.state);
            let child_ids = self.action_ids(&child_legal)?;
            requests.push((child_game, child_ids, root_color));
        }
        let evaluations = evaluator.evaluate_many(&requests)?;
        if evaluations.len() != outcomes.len() {
            return Err(format!(
                "evaluator batch length mismatch: requested {}, received {}",
                outcomes.len(),
                evaluations.len()
            ));
        }
        let weighted = outcomes
            .iter()
            .zip(evaluations)
            .map(|((_, probability), (_, value, _))| probability * value)
            .sum::<f64>();
        let root_value = if total_prob > 0.0 {
            weighted / total_prob
        } else {
            0.0
        };
        Ok(SearchResult {
            selected_action: action_id,
            improved_policy: vec![(action_id, 1.0)],
            visit_counts: vec![(action_id, 1)],
            q_values: vec![],
            priors: vec![(action_id, 1.0)],
            root_value,
            completed_q_values: vec![(action_id, root_value)],
            used_full_search: true,
            simulations_used: 0,
            afterstate_values: vec![(action_id, root_value)],
        })
    }

    fn raw_policy_root<E: Evaluator>(
        &mut self,
        game: &Game,
        root_color: Color,
        legal: &[Action],
        evaluator: &mut E,
    ) -> Result<SearchResult, String> {
        let legal_indices = self.action_ids(legal)?;
        let (priors, value, _) = evaluator.evaluate_root(game, &legal_indices, root_color)?;
        let mut node = Node::new(game.clone(), root_color);
        node.playable_actions = legal.to_vec();
        let mut arena = Arena::new();
        let root_idx = arena.alloc(node);
        self.finish_expand(&mut arena, root_idx, legal.to_vec(), priors, value, 0.0)?;
        let mut priors_vec: Vec<(usize, f64)> = arena
            .get(root_idx)
            .actions
            .iter()
            .map(|(&local, stats)| Ok((self.action_ids(&legal[local..=local])?[0], stats.prior)))
            .collect::<Result<_, String>>()?;
        priors_vec.sort_unstable_by_key(|(action, _)| *action);
        if priors_vec.is_empty() {
            return Err("no legal actions at raw-policy root".into());
        }
        let selected = priors_vec
            .iter()
            .max_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.0.cmp(&a.0))
            })
            .map(|(i, _)| *i)
            .unwrap_or(0);
        let completed_q_values = priors_vec
            .iter()
            .map(|(action, _)| (*action, arena.get(root_idx).prior_value))
            .collect();
        Ok(SearchResult {
            selected_action: selected,
            improved_policy: priors_vec.clone(),
            visit_counts: vec![],
            q_values: vec![],
            priors: priors_vec,
            root_value: arena.get(root_idx).value(),
            completed_q_values,
            used_full_search: false,
            simulations_used: 0,
            afterstate_values: vec![],
        })
    }
}

// ---------------------------------------------------------------------------
// Free functions
// ---------------------------------------------------------------------------

pub fn sequential_halving_schedule(m: i32, n_simulations: i32) -> Vec<(i32, i32)> {
    let m = m.max(1);
    let num_rounds = if m > 1 {
        (m as f64).log2().ceil() as i32
    } else {
        1
    };
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
    if m == 1 {
        return vec![(1, n)];
    }
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
            if leftover > 0 {
                phases.push((leftover, 1));
            }
            break;
        }
        considered = (considered / 2).max(2);
    }
    phases
}

fn rerank_candidates<F>(mut survivors: Vec<usize>, mut score: F) -> Vec<usize>
where
    F: FnMut(usize) -> f64,
{
    survivors.sort_by(|a, b| {
        score(*b)
            .partial_cmp(&score(*a))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.cmp(b))
    });
    survivors
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigma_reference_visits_is_budget_invariant_and_default_is_legacy() {
        let close = |actual: f64, expected: f64| {
            assert!(
                (actual - expected).abs() < 1.0e-12,
                "{actual} != {expected}"
            );
        };
        close(calibrated_sigma_scale(50.0, 0.1, 8, None), 5.8);
        close(calibrated_sigma_scale(50.0, 0.1, 32, None), 8.2);
        close(calibrated_sigma_scale(50.0, 0.1, 8, Some(12)), 6.2);
        close(calibrated_sigma_scale(50.0, 0.1, 32, Some(12)), 6.2);
    }

    #[test]
    fn gameplay_temperature_scales_policy_without_changing_t1() {
        let close = |actual: f64, expected: f64| {
            assert!(
                (actual - expected).abs() < 1.0e-12,
                "{actual} != {expected}"
            );
        };
        let policy = vec![(3, 0.8), (7, 0.2), (11, 0.0)];
        assert_eq!(temperature_scale_policy(&policy, 1.0).unwrap(), policy);

        let sharp = temperature_scale_policy(&policy, 0.5).unwrap();
        close(sharp[0].1, 16.0 / 17.0);
        close(sharp[1].1, 1.0 / 17.0);
        close(sharp[2].1, 0.0);

        let soft = temperature_scale_policy(&policy, 2.0).unwrap();
        close(soft[0].1, 2.0 / 3.0);
        close(soft[1].1, 1.0 / 3.0);
        close(soft[2].1, 0.0);

        assert!(temperature_scale_policy(&policy, 0.0).is_err());
        assert!(temperature_scale_policy(&policy, f64::NAN).is_err());
    }

    #[test]
    fn zero_gameplay_temperature_keeps_deterministic_argmax_selection() {
        let game = opening(3);
        let config = SearchConfig {
            temperature: 0.0,
            n_full: 8,
            n_fast: 8,
            p_full: 1.0,
            max_depth: 3,
            ..Default::default()
        };
        let first = GumbelMctsEngine::new(config.clone())
            .search(&game, &mut CountingEvaluator::default(), Some(true))
            .unwrap();
        let second = GumbelMctsEngine::new(config)
            .search(&game, &mut CountingEvaluator::default(), Some(true))
            .unwrap();
        assert_eq!(first.selected_action, second.selected_action);

        let visits = first
            .visit_counts
            .iter()
            .copied()
            .collect::<HashMap<_, _>>();
        let priors = first.priors.iter().copied().collect::<HashMap<_, _>>();
        let selected_policy = first
            .improved_policy
            .iter()
            .find(|(action, _)| *action == first.selected_action)
            .unwrap()
            .1;
        let selected_visits = visits[&first.selected_action];
        let selected_prior = priors[&first.selected_action];
        assert!(first.improved_policy.iter().all(|(action, probability)| {
            selected_policy > *probability
                || (selected_policy == *probability
                    && (selected_visits > visits[action]
                        || (selected_visits == visits[action] && selected_prior >= priors[action])))
        }));
    }

    #[test]
    fn initial_road_d1_is_root_only_and_exact_off_phase() {
        let game = opening(3);
        let mut node = Node::new(game, Color::Red);
        let mut first = ActionStats::new(0.5);
        first.visits = 4;
        let mut second = ActionStats::new(0.5);
        second.visits = 4;
        node.actions.insert(0, first);
        node.actions.insert(1, second);
        let mut arena = Arena::new();
        let root_idx = arena.alloc(node.clone());
        let interior_idx = arena.alloc(node);
        assert_eq!(root_idx, 0);
        assert_eq!(interior_idx, 1);
        let completed_q = HashMap::from([(0, 0.400004), (1, 0.399996)]);

        let plain =
            GumbelMctsEngine::new(SearchConfig::default()).rescale_completed_q(&completed_q);
        let road = GumbelMctsEngine::new(SearchConfig {
            rescale_noise_floor_c: 8.0,
            sigma_eval: 0.98,
            rescale_noise_floor_initial_road_only: true,
            attested_root_phase: Some("BUILD_INITIAL_ROAD".to_string()),
            ..Default::default()
        });
        let play = GumbelMctsEngine::new(SearchConfig {
            rescale_noise_floor_c: 8.0,
            sigma_eval: 0.98,
            rescale_noise_floor_initial_road_only: true,
            attested_root_phase: Some("PLAY_TURN".to_string()),
            ..Default::default()
        });

        assert_ne!(
            road.rescaled_completed_q_with_noise(&arena, root_idx, &completed_q),
            plain
        );
        assert_eq!(
            road.rescaled_completed_q_with_noise(&arena, interior_idx, &completed_q),
            plain
        );
        assert_eq!(
            play.rescaled_completed_q_with_noise(&arena, root_idx, &completed_q),
            plain
        );
    }

    #[test]
    fn initial_road_d1_fails_closed_without_authoritative_attestation() {
        let config = SearchConfig {
            rescale_noise_floor_c: 8.0,
            rescale_noise_floor_initial_road_only: true,
            ..Default::default()
        };
        let error = GumbelMctsEngine::new(config)
            .search(&opening(5), &mut CountingEvaluator::default(), Some(true))
            .unwrap_err();
        assert!(error.contains("authoritative root-phase attestation"));
    }
    use catanatron_rs::{Coordinate, Player};

    #[derive(Default)]
    struct CountingEvaluator {
        leaf_calls: usize,
        root_calls: usize,
    }

    impl Evaluator for CountingEvaluator {
        fn evaluate(
            &mut self,
            _game: &Game,
            legal: &[usize],
            _root_color: Color,
        ) -> Result<(HashMap<usize, f64>, f64, f64), String> {
            self.leaf_calls += 1;
            let probability = 1.0 / legal.len().max(1) as f64;
            Ok((
                legal.iter().map(|id| (*id, probability)).collect(),
                0.125,
                0.0,
            ))
        }

        fn evaluate_root(
            &mut self,
            game: &Game,
            legal: &[usize],
            root_color: Color,
        ) -> Result<(HashMap<usize, f64>, f64, f64), String> {
            self.root_calls += 1;
            self.evaluate(game, legal, root_color)
        }
    }

    fn opening(seed: u64) -> Game {
        Game::new(
            vec![Player::simple(Color::Red), Player::simple(Color::Blue)],
            Some(seed),
        )
    }

    #[test]
    fn root_callback_is_distinct_and_results_use_global_action_ids() {
        let game = opening(7);
        let expected_ids = generate_playable_actions(&game.state)
            .iter()
            .map(|action| {
                ActionSpace::new(&[Color::Red, Color::Blue], MapKind::Base)
                    .index(action)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        assert!(
            expected_ids.iter().any(|id| *id >= 54),
            "opening IDs must be global, not local offsets"
        );
        let mut evaluator = CountingEvaluator::default();
        let config = SearchConfig {
            n_full: 8,
            n_fast: 8,
            p_full: 1.0,
            max_depth: 3,
            ..Default::default()
        };
        let result = GumbelMctsEngine::new(config)
            .search(&game, &mut evaluator, Some(true))
            .unwrap();
        assert_eq!(evaluator.root_calls, 1);
        assert!(expected_ids.contains(&result.selected_action));
        assert!(result
            .priors
            .iter()
            .all(|(id, _)| expected_ids.contains(id)));
        assert!(result
            .improved_policy
            .iter()
            .all(|(id, _)| expected_ids.contains(id)));
    }

    #[test]
    fn actor_turn_horizon_never_increases_leaf_work() {
        let game = opening(11);
        let mut unrestricted_eval = CountingEvaluator::default();
        let mut bounded_eval = CountingEvaluator::default();
        let mut config = SearchConfig {
            n_full: 128,
            n_fast: 128,
            p_full: 1.0,
            max_depth: 12,
            ..Default::default()
        };
        GumbelMctsEngine::new(config.clone())
            .search(&game, &mut unrestricted_eval, Some(true))
            .unwrap();
        config.stop_at_root_turn_boundary = true;
        let bounded = GumbelMctsEngine::new(config)
            .search(&game, &mut bounded_eval, Some(true))
            .unwrap();
        assert!(bounded_eval.leaf_calls <= unrestricted_eval.leaf_calls);
        assert!(bounded.simulations_used >= 32);
    }

    #[test]
    fn explicit_wide_budget_threshold_is_inclusive_and_can_force_full() {
        let game = opening(13);
        let width = generate_playable_actions(&game.state).len();
        let mut evaluator = CountingEvaluator::default();
        let config = SearchConfig {
            n_full: 4,
            n_fast: 2,
            p_full: 0.0,
            n_full_wide: Some(8),
            n_full_wide_threshold: Some(width),
            wide_roots_always_full: true,
            exact_budget_sh: true,
            ..Default::default()
        };
        let result = GumbelMctsEngine::new(config)
            .search(&game, &mut evaluator, None)
            .unwrap();
        assert!(result.used_full_search);
        assert_eq!(result.simulations_used, 8);
    }

    #[test]
    fn exact_budget_phases_spend_exactly_and_only_shrink_survivors() {
        let phases = exact_budget_sh_phases(54, 128);
        assert_eq!(
            phases
                .iter()
                .map(|(count, budget)| count * budget)
                .sum::<i32>(),
            128
        );
        assert!(phases.windows(2).all(|pair| pair[1].0 <= pair[0].0));
        let ranked = rerank_candidates(vec![2, 4], |action| {
            if action == 99 {
                1_000.0
            } else {
                action as f64
            }
        });
        assert_eq!(ranked, vec![4, 2]);
        assert!(
            !ranked.contains(&99),
            "an eliminated candidate cannot re-enter"
        );
    }

    #[test]
    fn seeded_search_is_deterministic_across_fresh_hashmaps() {
        let game = opening(17);
        for closed_form in [false, true] {
            let mut signatures = Vec::new();
            for _ in 0..8 {
                let config = SearchConfig {
                    seed: 91,
                    n_full: 32,
                    n_fast: 32,
                    p_full: 1.0,
                    exact_budget_sh: true,
                    max_depth: 5,
                    variance_aware_q: true,
                    variance_aware_closed_form_js: closed_form,
                    ..Default::default()
                };
                let result = GumbelMctsEngine::new(config)
                    .search(&game, &mut CountingEvaluator::default(), Some(true))
                    .unwrap();
                let mut visits = result.visit_counts;
                visits.sort_unstable_by_key(|(action, _)| *action);
                let mut policy = result.improved_policy;
                policy.sort_unstable_by_key(|(action, _)| *action);
                signatures.push((result.selected_action, visits, policy));
            }
            assert!(signatures.windows(2).all(|pair| pair[0] == pair[1]));
        }
    }

    #[test]
    fn robber_without_victim_is_not_an_expectation_backup() {
        let engine = GumbelMctsEngine::new(SearchConfig::default());
        let no_victim = Action::new(
            Color::Red,
            ActionType::MoveRobber,
            ActionValue::Robber(Coordinate(0, 0, 0), None),
        );
        let with_victim = Action::new(
            Color::Red,
            ActionType::MoveRobber,
            ActionValue::Robber(Coordinate(0, 0, 0), Some(Color::Blue)),
        );
        assert!(!engine.expectation_backup(&no_victim, 1));
        assert!(engine.expectation_backup(&with_victim, 1));
    }

    #[test]
    fn policy_pruning_uses_local_visits_before_global_id_export() {
        let game = opening(29);
        let config = SearchConfig {
            seed: 5,
            n_full: 128,
            n_fast: 128,
            p_full: 1.0,
            exact_budget_sh: true,
            policy_target_min_visits: 2,
            ..Default::default()
        };
        let result = GumbelMctsEngine::new(config)
            .search(&game, &mut CountingEvaluator::default(), Some(true))
            .unwrap();
        let visits = result
            .visit_counts
            .iter()
            .copied()
            .collect::<HashMap<_, _>>();
        assert_eq!(result.improved_policy.len(), visits.len());
        for (action, probability) in result.improved_policy {
            assert_eq!(
                probability == 0.0,
                visits[&action] < 2,
                "policy zero-mask must use the matching global action's local visit count",
            );
        }
    }

    #[test]
    fn exact_sh_winner_survives_every_phase_and_has_max_visits() {
        let game = opening(31);
        for seed in 0..24 {
            let config = SearchConfig {
                seed,
                n_full: 128,
                n_fast: 128,
                p_full: 1.0,
                exact_budget_sh: true,
                play_sh_winner: true,
                max_depth: 5,
                ..Default::default()
            };
            let result = GumbelMctsEngine::new(config)
                .search(&game, &mut CountingEvaluator::default(), Some(true))
                .unwrap();
            let max_visits = result
                .visit_counts
                .iter()
                .map(|(_, visits)| *visits)
                .max()
                .unwrap();
            let winner_visits = result
                .visit_counts
                .iter()
                .find(|(action, _)| *action == result.selected_action)
                .unwrap()
                .1;
            assert_eq!(winner_visits, max_visits);
        }
    }
}
