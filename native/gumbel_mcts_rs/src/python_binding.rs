use ::catanatron_rs::init_python_module;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::exceptions::PyRuntimeError;

use gumbel_mcts::{
    GumbelMctsEngine, SearchConfig, Evaluator,
};

// Use :: to disambiguate from the pymodule function name
use ::catanatron_rs as ctrs;
use ctrs::{Game, Color, generate_playable_actions, game_to_json_value, action_to_json_value};
use ctrs::python_bindings::PyGame;

// ---------------------------------------------------------------------------
// PyO3 Evaluator — wraps a Python callable, passes PyGame for rust_featurize
// ---------------------------------------------------------------------------

struct PyEvaluator {
    eval_fn: PyObject,
    eval_many_fn: Option<PyObject>,
}

impl Evaluator for PyEvaluator {
    fn evaluate(&mut self, game: &Game, legal_action_indices: &[usize], root_color: Color)
        -> Result<(std::collections::HashMap<usize, f64>, f64, f64), String>
    {
        Python::with_gil(|py| {
            // Create a PyGame from the native Game — this lets the evaluator
            // use the rust_featurize path directly (no JSON round-trip)
            let py_game = PyGame { game: game.clone() };
            let py_obj = Py::new(py, py_game).map_err(|e| e.to_string())?;
            let legal: Vec<usize> = legal_action_indices.to_vec();
            let rc = color_to_string(root_color);

            // Call eval_fn(game, legal, root_color=rc)
            let result = self.eval_fn.bind(py).call1((py_obj, legal, rc))
                .map_err(|e| e.to_string())?;
            parse_eval_result(&result)
        })
    }

    fn evaluate_many(&mut self, requests: &[(Game, Vec<usize>, Color)])
        -> Result<Vec<(std::collections::HashMap<usize, f64>, f64, f64)>, String>
    {
        if requests.is_empty() { return Ok(Vec::new()); }

        let eval_many = match &self.eval_many_fn {
            Some(f) => f,
            None => {
                let mut out = Vec::with_capacity(requests.len());
                for (game, legal, rc) in requests {
                    out.push(self.evaluate(game, legal, *rc)?);
                }
                return Ok(out);
            }
        };

        Python::with_gil(|py| {
            let py_list = PyList::empty(py);
            for (game, legal, rc) in requests {
                let py_game = PyGame { game: game.clone() };
                let py_obj = Py::new(py, py_game).map_err(|e| e.to_string())?;
                let rc_str = color_to_string(*rc);
                py_list.append((py_obj, legal.clone(), rc_str)).map_err(|e| e.to_string())?;
            }

            let result = eval_many.bind(py).call1((py_list,)).map_err(|e| e.to_string())?;
            let result_list = result.downcast::<PyList>().map_err(|e| e.to_string())?;
            let mut out = Vec::with_capacity(result_list.len());
            for item in result_list.iter() {
                out.push(parse_eval_result(&item)?);
            }
            Ok(out)
        })
    }
}

/// Convert Color to the string representation expected by the Python evaluator
fn color_to_string(c: Color) -> String {
    match c {
        Color::Red => "RED".to_string(),
        Color::Blue => "BLUE".to_string(),
        Color::Orange => "ORANGE".to_string(),
        Color::White => "WHITE".to_string(),
    }
}

fn parse_eval_result(result: &Bound<'_, PyAny>) -> Result<(std::collections::HashMap<usize, f64>, f64, f64), String> {
    let tuple = result.downcast::<PyTuple>().map_err(|e| e.to_string())?;
    let item0 = tuple.get_item(0).map_err(|e| e.to_string())?;
    let priors_dict = item0.downcast::<PyDict>().map_err(|e| e.to_string())?;
    let value: f64 = tuple.get_item(1).map_err(|e| e.to_string())?.extract::<f64>().map_err(|e| e.to_string())?;
    let uncertainty: f64 = if tuple.len() > 2 {
        tuple.get_item(2).map_err(|e| e.to_string())?.extract::<f64>().map_err(|e| e.to_string())?
    } else { 0.0 };

    let mut priors = std::collections::HashMap::with_capacity(priors_dict.len());
    for (key, val) in priors_dict.iter() {
        let aid: usize = key.extract::<usize>().map_err(|e| e.to_string())?;
        let p: f64 = val.extract::<f64>().map_err(|e| e.to_string())?;
        priors.insert(aid, p);
    }
    Ok((priors, value, uncertainty))
}

// ---------------------------------------------------------------------------
// GameWrapper — backward compat pyclass
// ---------------------------------------------------------------------------

#[pyclass(name = "GameWrapper")]
struct GameWrapper {
    game: Game,
}

impl GameWrapper {
    fn new(game: Game) -> Self { Self { game } }
}

#[pymethods]
impl GameWrapper {
    fn json_snapshot(&self) -> PyResult<String> {
        serde_json::to_string(&game_to_json_value(&self.game))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("snapshot: {}", e)))
    }

    fn current_color(&self) -> PyResult<String> {
        Ok(color_to_string(self.game.state.current_color()))
    }

    fn playable_action_indices(&self, _colors: Vec<String>, _map_kind: Option<String>) -> PyResult<Vec<usize>> {
        let legal = generate_playable_actions(&self.game.state);
        Ok((0..legal.len()).collect())
    }

    fn playable_actions_json(&self) -> PyResult<String> {
        let legal = generate_playable_actions(&self.game.state);
        let actions: Vec<serde_json::Value> = legal.iter().map(action_to_json_value).collect();
        Ok(serde_json::to_string(&actions).unwrap_or_default())
    }

    fn winning_color(&self) -> PyResult<Option<String>> {
        Ok(self.game.winning_color().map(color_to_string))
    }

    fn copy(&self) -> PyResult<GameWrapper> {
        Ok(GameWrapper { game: self.game.clone() })
    }
    fn __copy__(&self) -> PyResult<GameWrapper> { self.copy() }
    fn __deepcopy__(&self, _memo: Option<PyObject>) -> PyResult<GameWrapper> { self.copy() }

    fn num_turns(&self) -> PyResult<usize> { Ok(self.game.state.num_turns) }
    fn is_initial_build_phase(&self) -> PyResult<bool> { Ok(self.game.state.is_initial_build_phase) }
    fn vps_to_win(&self) -> PyResult<i16> { Ok(self.game.vps_to_win) }
    fn seed(&self) -> PyResult<Option<u64>> { Ok(self.game.seed) }

    /// Convert to a catanatron_rs.Game (PyGame) so the evaluator can use
    /// the rust_featurize path directly — no JSON round-trip needed.
    fn to_game(&self) -> PyResult<PyGame> {
        Ok(PyGame { game: self.game.clone() })
    }
}

fn color_to_string(c: Color) -> String {
    match c {
        Color::Red => "RED", Color::Blue => "BLUE",
        Color::Orange => "ORANGE", Color::White => "WHITE",
    }.into()
}

fn string_to_color(s: &str) -> Option<Color> {
    match s.to_uppercase().as_str() {
        "RED" => Some(Color::Red), "BLUE" => Some(Color::Blue),
        "ORANGE" => Some(Color::Orange), "WHITE" => Some(Color::White),
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// search pyfunction
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(signature = (game, evaluator, config_dict, evaluator_many=None, force_full=None))]
fn gumbel_search(
    py: Python,
    game: &Bound<'_, PyAny>,
    evaluator: PyObject,
    config_dict: &Bound<'_, PyDict>,
    evaluator_many: Option<PyObject>,
    force_full: Option<bool>,
) -> PyResult<Py<PyDict>> {
    let py_game = game.cast::<PyGame>()?;
    let native_game = py_game.borrow().game.clone();

    let mut config = SearchConfig::default();
    if let Some(v) = config_dict.get_item("max_depth")? { config.max_depth = v.extract()?; }
    if let Some(v) = config_dict.get_item("seed")? { config.seed = v.extract()?; }
    if let Some(v) = config_dict.get_item("c_visit")? { config.c_visit = v.extract()?; }
    if let Some(v) = config_dict.get_item("c_scale")? { config.c_scale = v.extract()?; }
    if let Some(v) = config_dict.get_item("temperature")? { config.temperature = v.extract()?; }
    if let Some(v) = config_dict.get_item("play_sh_winner")? { config.play_sh_winner = v.extract()?; }
    if let Some(v) = config_dict.get_item("prior_temperature")? { config.prior_temperature = v.extract()?; }
    if let Some(v) = config_dict.get_item("n_full")? { config.n_full = v.extract()?; }
    if let Some(v) = config_dict.get_item("n_fast")? { config.n_fast = v.extract()?; }
    if let Some(v) = config_dict.get_item("p_full")? { config.p_full = v.extract()?; }
    if let Some(v) = config_dict.get_item("n_full_wide")? { config.n_full_wide = Some(v.extract()?); }
    if let Some(v) = config_dict.get_item("raw_policy_above_width")? { config.raw_policy_above_width = Some(v.extract()?); }
    if let Some(v) = config_dict.get_item("lazy_interior_chance")? { config.lazy_interior_chance = v.extract()?; }
    if let Some(v) = config_dict.get_item("root_candidate_cap")? { config.root_candidate_cap = Some(v.extract()?); }
    if let Some(v) = config_dict.get_item("policy_target_min_visits")? { config.policy_target_min_visits = v.extract()?; }
    if let Some(v) = config_dict.get_item("max_root_candidates")? { config.max_root_candidates = v.extract()?; }
    if let Some(v) = config_dict.get_item("max_root_candidates_wide")? { config.max_root_candidates_wide = v.extract()?; }
    if let Some(v) = config_dict.get_item("wide_candidates_threshold")? { config.wide_candidates_threshold = v.extract()?; }
    if let Some(v) = config_dict.get_item("exact_budget_sh")? { config.exact_budget_sh = v.extract()?; }
    if let Some(v) = config_dict.get_item("exact_budget_sh_min_n")? { config.exact_budget_sh_min_n = v.extract()?; }
    if let Some(v) = config_dict.get_item("rescale_noise_floor_c")? { config.rescale_noise_floor_c = v.extract()?; }
    if let Some(v) = config_dict.get_item("sigma_eval")? { config.sigma_eval = v.extract()?; }
    if let Some(v) = config_dict.get_item("variance_aware_q")? { config.variance_aware_q = v.extract()?; }
    if let Some(v) = config_dict.get_item("variance_aware_k")? { config.variance_aware_k = v.extract()?; }
    if let Some(v) = config_dict.get_item("variance_aware_closed_form_js")? { config.variance_aware_closed_form_js = v.extract()?; }
    if let Some(v) = config_dict.get_item("uncertainty_backup_weighting")? { config.uncertainty_backup_weighting = v.extract()?; }
    if let Some(v) = config_dict.get_item("uncertainty_backup_a")? { config.uncertainty_backup_a = v.extract()?; }
    if let Some(v) = config_dict.get_item("uncertainty_backup_exp")? { config.uncertainty_backup_exp = v.extract()?; }
    if let Some(v) = config_dict.get_item("uncertainty_backup_cap")? { config.uncertainty_backup_cap = v.extract()?; }
    if let Some(v) = config_dict.get_item("colors")? {
        let colors: Vec<String> = v.extract()?;
        config.colors = colors.iter().filter_map(|s| string_to_color(s)).collect();
    }

    let mut py_evaluator = PyEvaluator { eval_fn: evaluator, eval_many_fn: evaluator_many };
    let mut engine = GumbelMctsEngine::new(config);
    // Enable batched leaf evaluation for GPU efficiency
    if let Some(v) = config_dict.get_item("batch_size")? {
        let batch_size: usize = v.extract()?;
        if batch_size > 0 {
            engine = engine.with_batch_size(batch_size);
        }
    }
    let result = engine.search(&native_game, &mut py_evaluator, force_full)
        .map_err(|e| PyRuntimeError::new_err(e))?;

    let out = PyDict::new(py);
    out.set_item("selected_action", result.selected_action)?;
    let policy_dict = PyDict::new(py);
    for (aid, prob) in &result.improved_policy { policy_dict.set_item(aid, *prob)?; }
    out.set_item("improved_policy", policy_dict)?;
    let visits_dict = PyDict::new(py);
    for (aid, vis) in &result.visit_counts { visits_dict.set_item(aid, *vis)?; }
    out.set_item("visit_counts", visits_dict)?;
    let q_dict = PyDict::new(py);
    for (aid, q) in &result.q_values { q_dict.set_item(aid, *q)?; }
    out.set_item("q_values", q_dict)?;
    let priors_dict = PyDict::new(py);
    for (aid, p) in &result.priors { priors_dict.set_item(aid, *p)?; }
    out.set_item("priors", priors_dict)?;
    out.set_item("root_value", result.root_value)?;
    out.set_item("used_full_search", result.used_full_search)?;
    out.set_item("simulations_used", result.simulations_used)?;
    let asv_dict = PyDict::new(py);
    for (aid, v) in &result.afterstate_values { asv_dict.set_item(aid, *v)?; }
    out.set_item("afterstate_values", asv_dict)?;
    Ok(out.into())
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn catanatron_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    init_python_module(module)?;
    module.add_function(wrap_pyfunction!(gumbel_search, module)?)?;
    module.add_class::<GameWrapper>()?;
    Ok(())
}
