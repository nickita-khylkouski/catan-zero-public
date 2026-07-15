use ::catanatron_rs::init_python_module;
use pyo3::prelude::*;

// Use :: to disambiguate from the pymodule function name
use ::catanatron_rs as ctrs;
use ctrs::python_bindings::PyGame;
use ctrs::{
    Color, Game, action_to_json_value, game_to_json_value, generate_playable_actions,
    public_card_deductions_to_json_value,
};

#[path = "../../../gumbel_mcts_rs/src/python_binding.rs"]
mod gumbel_binding;

// ---------------------------------------------------------------------------
// GameWrapper — backward compat pyclass
// ---------------------------------------------------------------------------

#[pyclass(name = "GameWrapper")]
struct GameWrapper {
    game: Game,
}

#[pymethods]
impl GameWrapper {
    fn json_snapshot(&self) -> PyResult<String> {
        serde_json::to_string(&game_to_json_value(&self.game))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("snapshot: {}", e)))
    }

    fn public_card_deductions_json(&self, observer: &str) -> PyResult<String> {
        let observer = parse_public_color(observer)?;
        let value = public_card_deductions_to_json_value(&self.game, observer)
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        serde_json::to_string(&value).map_err(|error| {
            pyo3::exceptions::PyValueError::new_err(format!(
                "public card deductions: {error}"
            ))
        })
    }

    fn current_color(&self) -> PyResult<String> {
        Ok(color_to_string(self.game.state.current_color()))
    }

    fn playable_action_indices(
        &self,
        _colors: Vec<String>,
        _map_kind: Option<String>,
    ) -> PyResult<Vec<usize>> {
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
        Ok(GameWrapper {
            game: self.game.clone(),
        })
    }
    fn __copy__(&self) -> PyResult<GameWrapper> {
        self.copy()
    }
    fn __deepcopy__(&self, _memo: Option<Py<PyAny>>) -> PyResult<GameWrapper> {
        self.copy()
    }

    fn num_turns(&self) -> PyResult<usize> {
        Ok(self.game.state.num_turns)
    }
    fn is_initial_build_phase(&self) -> PyResult<bool> {
        Ok(self.game.state.is_initial_build_phase)
    }
    fn vps_to_win(&self) -> PyResult<i16> {
        Ok(self.game.vps_to_win)
    }
    fn seed(&self) -> PyResult<Option<u64>> {
        Ok(self.game.seed)
    }

    /// Convert to a catanatron_rs.Game (PyGame) so the evaluator can use
    /// the rust_featurize path directly — no JSON round-trip needed.
    fn to_game(&self) -> PyResult<PyGame> {
        Ok(PyGame {
            game: self.game.clone(),
        })
    }
}

fn color_to_string(c: Color) -> String {
    match c {
        Color::Red => "RED",
        Color::Blue => "BLUE",
        Color::Orange => "ORANGE",
        Color::White => "WHITE",
    }
    .into()
}

fn parse_public_color(value: &str) -> PyResult<Color> {
    match value.to_ascii_uppercase().as_str() {
        "RED" => Ok(Color::Red),
        "BLUE" => Ok(Color::Blue),
        "ORANGE" => Ok(Color::Orange),
        "WHITE" => Ok(Color::White),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown color: {value}"
        ))),
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

#[pymodule]
fn catanatron_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
    init_python_module(module)?;
    module.add_class::<GameWrapper>()?;
    gumbel_binding::register(module)?;
    Ok(())
}
