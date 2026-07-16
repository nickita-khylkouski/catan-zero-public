//! A Rust implementation of Catanatron's core game engine.
//!
//! The Python project stores most state in dictionaries and relies on
//! `networkx` for static graph operations. This crate keeps the same game
//! concepts but uses compact enums, fixed resource decks, and adjacency lists.

use rand::prelude::*;
use rayon::prelude::*;
use serde_json::{Value, json};
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::sync::OnceLock;

pub type NodeId = usize;
pub type Edge = (NodeId, NodeId);
pub type FreqDeck = [u8; 5];
type RoadAwardUpdate = (Option<Color>, Option<Color>, HashMap<Color, usize>);

pub const ROAD_COST: FreqDeck = [1, 1, 0, 0, 0];
pub const SETTLEMENT_COST: FreqDeck = [1, 1, 1, 1, 0];
pub const CITY_COST: FreqDeck = [0, 0, 0, 2, 3];
pub const DEVELOPMENT_CARD_COST: FreqDeck = [0, 0, 1, 1, 1];

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Resource {
    Wood,
    Brick,
    Sheep,
    Wheat,
    Ore,
}

impl Resource {
    pub const ALL: [Resource; 5] = [
        Resource::Wood,
        Resource::Brick,
        Resource::Sheep,
        Resource::Wheat,
        Resource::Ore,
    ];

    pub const fn idx(self) -> usize {
        match self {
            Resource::Wood => 0,
            Resource::Brick => 1,
            Resource::Sheep => 2,
            Resource::Wheat => 3,
            Resource::Ore => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum DevCard {
    Knight,
    YearOfPlenty,
    Monopoly,
    RoadBuilding,
    VictoryPoint,
}

impl DevCard {
    pub const ALL: [DevCard; 5] = [
        DevCard::Knight,
        DevCard::YearOfPlenty,
        DevCard::Monopoly,
        DevCard::RoadBuilding,
        DevCard::VictoryPoint,
    ];

    pub const fn idx(self) -> usize {
        match self {
            DevCard::Knight => 0,
            DevCard::YearOfPlenty => 1,
            DevCard::Monopoly => 2,
            DevCard::RoadBuilding => 3,
            DevCard::VictoryPoint => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum BuildingType {
    Settlement,
    City,
    Road,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum Color {
    Red,
    Blue,
    Orange,
    White,
}

impl Color {
    pub const ALL: [Color; 4] = [Color::Red, Color::Blue, Color::Orange, Color::White];
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Coordinate(pub i8, pub i8, pub i8);

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum Direction {
    East,
    Southeast,
    Southwest,
    West,
    Northwest,
    Northeast,
}

impl Direction {
    const ALL: [Direction; 6] = [
        Direction::East,
        Direction::Southeast,
        Direction::Southwest,
        Direction::West,
        Direction::Northwest,
        Direction::Northeast,
    ];

    const REVERSED: [Direction; 6] = [
        Direction::Northeast,
        Direction::Northwest,
        Direction::West,
        Direction::Southwest,
        Direction::Southeast,
        Direction::East,
    ];

    const fn unit(self) -> Coordinate {
        match self {
            Direction::Northeast => Coordinate(1, 0, -1),
            Direction::Southwest => Coordinate(-1, 0, 1),
            Direction::Northwest => Coordinate(0, 1, -1),
            Direction::Southeast => Coordinate(0, -1, 1),
            Direction::East => Coordinate(1, -1, 0),
            Direction::West => Coordinate(-1, 1, 0),
        }
    }
}

fn add_coord(a: Coordinate, b: Coordinate) -> Coordinate {
    Coordinate(a.0 + b.0, a.1 + b.1, a.2 + b.2)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum NodeRef {
    North,
    Northeast,
    Southeast,
    South,
    Southwest,
    Northwest,
}

impl NodeRef {
    const ALL: [NodeRef; 6] = [
        NodeRef::North,
        NodeRef::Northeast,
        NodeRef::Southeast,
        NodeRef::South,
        NodeRef::Southwest,
        NodeRef::Northwest,
    ];

    const fn idx(self) -> usize {
        match self {
            NodeRef::North => 0,
            NodeRef::Northeast => 1,
            NodeRef::Southeast => 2,
            NodeRef::South => 3,
            NodeRef::Southwest => 4,
            NodeRef::Northwest => 5,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
enum EdgeRef {
    East,
    Southeast,
    Southwest,
    West,
    Northwest,
    Northeast,
}

impl EdgeRef {
    const ALL: [EdgeRef; 6] = [
        EdgeRef::East,
        EdgeRef::Southeast,
        EdgeRef::Southwest,
        EdgeRef::West,
        EdgeRef::Northwest,
        EdgeRef::Northeast,
    ];

    const fn idx(self) -> usize {
        match self {
            EdgeRef::East => 0,
            EdgeRef::Southeast => 1,
            EdgeRef::Southwest => 2,
            EdgeRef::West => 3,
            EdgeRef::Northwest => 4,
            EdgeRef::Northeast => 5,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LandTile {
    pub id: usize,
    pub resource: Option<Resource>,
    pub number: Option<u8>,
    pub coordinate: Coordinate,
    nodes: [NodeId; 6],
    edges: [Edge; 6],
}

impl LandTile {
    pub fn nodes(&self) -> &[NodeId; 6] {
        &self.nodes
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Port {
    pub id: usize,
    pub resource: Option<Resource>,
    direction: Direction,
    nodes: [NodeId; 6],
    edges: [Edge; 6],
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum Tile {
    Land(LandTile),
    Port(Port),
    Water {
        nodes: [NodeId; 6],
        edges: [Edge; 6],
    },
}

impl Tile {
    fn nodes(&self) -> &[NodeId; 6] {
        match self {
            Tile::Land(tile) => &tile.nodes,
            Tile::Port(tile) => &tile.nodes,
            Tile::Water { nodes, .. } => nodes,
        }
    }

    fn edges(&self) -> &[Edge; 6] {
        match self {
            Tile::Land(tile) => &tile.edges,
            Tile::Port(tile) => &tile.edges,
            Tile::Water { edges, .. } => edges,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MapKind {
    Base,
    Mini,
    Tournament,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NumberPlacement {
    OfficialSpiral,
    Random,
}

#[derive(Clone, Debug)]
pub struct CatanMap {
    tiles: HashMap<Coordinate, Tile>,
    pub land_tiles: HashMap<Coordinate, LandTile>,
    pub port_nodes: HashMap<Option<Resource>, HashSet<NodeId>>,
    pub land_nodes: HashSet<NodeId>,
    land_edges: Vec<Edge>,
    pub adjacent_tiles: HashMap<NodeId, Vec<LandTile>>,
    pub node_production: HashMap<NodeId, [f64; 5]>,
    pub tiles_by_id: HashMap<usize, LandTile>,
    pub ports_by_id: HashMap<usize, Port>,
}

impl CatanMap {
    pub fn base() -> Self {
        Self::from_template(MapKind::Base, NumberPlacement::OfficialSpiral)
    }

    pub fn mini() -> Self {
        Self::from_template(MapKind::Mini, NumberPlacement::OfficialSpiral)
    }

    pub fn from_template(kind: MapKind, number_placement: NumberPlacement) -> Self {
        let mut rng = thread_rng();
        Self::from_template_with_rng(kind, number_placement, &mut rng)
    }

    pub fn from_template_with_rng<R: Rng + ?Sized>(
        kind: MapKind,
        number_placement: NumberPlacement,
        rng: &mut R,
    ) -> Self {
        let tiles = initialize_tiles(kind, number_placement, rng);
        Self::from_tiles(tiles)
    }

    fn from_tiles(tiles: HashMap<Coordinate, Tile>) -> Self {
        let land_tiles: HashMap<_, _> = tiles
            .iter()
            .filter_map(|(coord, tile)| match tile {
                Tile::Land(land) => Some((*coord, land.clone())),
                _ => None,
            })
            .collect();

        let mut port_nodes: HashMap<Option<Resource>, HashSet<NodeId>> = HashMap::new();
        for tile in tiles.values() {
            let Tile::Port(port) = tile else { continue };
            let (a, b) = port_direction_node_refs(port.direction);
            port_nodes
                .entry(port.resource)
                .or_default()
                .insert(port.nodes[a.idx()]);
            port_nodes
                .entry(port.resource)
                .or_default()
                .insert(port.nodes[b.idx()]);
        }

        let mut land_nodes = HashSet::new();
        for tile in land_tiles.values() {
            land_nodes.extend(tile.nodes);
        }

        let mut land_edge_set = HashSet::new();
        for tile in land_tiles.values() {
            for edge in tile.edges {
                land_edge_set.insert(canonical_edge(edge));
            }
        }
        let mut land_edges: Vec<_> = land_edge_set.into_iter().collect();
        land_edges.sort_unstable();

        let mut adjacent_tiles: HashMap<NodeId, Vec<LandTile>> = HashMap::new();
        for tile in land_tiles.values() {
            for node in tile.nodes {
                adjacent_tiles.entry(node).or_default().push(tile.clone());
            }
        }

        let mut node_production = HashMap::new();
        for (&node, tiles) in &adjacent_tiles {
            let mut production = [0.0; 5];
            for tile in tiles {
                if let (Some(resource), Some(number)) = (tile.resource, tile.number) {
                    production[resource.idx()] += number_probability(number);
                }
            }
            node_production.insert(node, production);
        }

        let tiles_by_id = land_tiles.values().map(|t| (t.id, t.clone())).collect();
        let ports_by_id = tiles
            .values()
            .filter_map(|tile| match tile {
                Tile::Port(port) => Some((port.id, port.clone())),
                _ => None,
            })
            .collect();

        CatanMap {
            tiles,
            land_tiles,
            port_nodes,
            land_nodes,
            land_edges,
            adjacent_tiles,
            node_production,
            tiles_by_id,
            ports_by_id,
        }
    }

    fn land_edges(&self) -> Vec<Edge> {
        self.land_edges.clone()
    }

    pub fn tile_count(&self) -> usize {
        self.tiles.len()
    }
}

#[derive(Clone, Copy)]
enum Topology {
    Land,
    Water,
    Port(Direction),
}

fn base_numbers() -> Vec<u8> {
    vec![2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12]
}

fn official_spiral_numbers() -> [u8; 18] {
    [5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11]
}

fn base_tile_resources() -> Vec<Option<Resource>> {
    let mut resources = Vec::new();
    resources.extend([Some(Resource::Wood); 4]);
    resources.extend([Some(Resource::Brick); 3]);
    resources.extend([Some(Resource::Sheep); 4]);
    resources.extend([Some(Resource::Wheat); 4]);
    resources.extend([Some(Resource::Ore); 3]);
    resources.push(None);
    resources
}

fn base_port_resources() -> Vec<Option<Resource>> {
    vec![
        Some(Resource::Wood),
        Some(Resource::Brick),
        Some(Resource::Sheep),
        Some(Resource::Wheat),
        Some(Resource::Ore),
        None,
        None,
        None,
        None,
    ]
}

fn tournament_numbers() -> Vec<u8> {
    vec![10, 8, 3, 6, 2, 5, 10, 8, 4, 11, 12, 9, 5, 4, 9, 11, 3, 6]
}

fn tournament_port_resources() -> Vec<Option<Resource>> {
    vec![
        None,
        Some(Resource::Sheep),
        None,
        Some(Resource::Ore),
        Some(Resource::Wheat),
        None,
        Some(Resource::Wood),
        Some(Resource::Brick),
        None,
    ]
}

fn tournament_tile_resources() -> Vec<Option<Resource>> {
    vec![
        None,
        Some(Resource::Wood),
        Some(Resource::Sheep),
        Some(Resource::Sheep),
        Some(Resource::Wood),
        Some(Resource::Wheat),
        Some(Resource::Wood),
        Some(Resource::Wheat),
        Some(Resource::Brick),
        Some(Resource::Sheep),
        Some(Resource::Brick),
        Some(Resource::Sheep),
        Some(Resource::Wheat),
        Some(Resource::Wheat),
        Some(Resource::Ore),
        Some(Resource::Brick),
        Some(Resource::Ore),
        Some(Resource::Wood),
        Some(Resource::Ore),
        None,
    ]
}

fn mini_numbers() -> Vec<u8> {
    vec![3, 4, 5, 6, 8, 9, 10]
}

fn mini_tile_resources() -> Vec<Option<Resource>> {
    vec![
        Some(Resource::Wood),
        None,
        Some(Resource::Brick),
        Some(Resource::Sheep),
        Some(Resource::Wheat),
        Some(Resource::Wheat),
        Some(Resource::Ore),
    ]
}

fn topology(kind: MapKind) -> Vec<(Coordinate, Topology)> {
    let base_land = vec![
        Coordinate(0, 0, 0),
        Coordinate(1, -1, 0),
        Coordinate(0, -1, 1),
        Coordinate(-1, 0, 1),
        Coordinate(-1, 1, 0),
        Coordinate(0, 1, -1),
        Coordinate(1, 0, -1),
    ];
    if kind == MapKind::Mini {
        let mut items: Vec<_> = base_land.into_iter().map(|c| (c, Topology::Land)).collect();
        items.extend(
            [
                Coordinate(2, -2, 0),
                Coordinate(1, -2, 1),
                Coordinate(0, -2, 2),
                Coordinate(-1, -1, 2),
                Coordinate(-2, 0, 2),
                Coordinate(-2, 1, 1),
                Coordinate(-2, 2, 0),
                Coordinate(-1, 2, -1),
                Coordinate(0, 2, -2),
                Coordinate(1, 1, -2),
                Coordinate(2, 0, -2),
                Coordinate(2, -1, -1),
            ]
            .into_iter()
            .map(|c| (c, Topology::Water)),
        );
        return items;
    }

    let mut items: Vec<_> = base_land.into_iter().map(|c| (c, Topology::Land)).collect();
    items.extend(
        [
            Coordinate(2, -2, 0),
            Coordinate(1, -2, 1),
            Coordinate(0, -2, 2),
            Coordinate(-1, -1, 2),
            Coordinate(-2, 0, 2),
            Coordinate(-2, 1, 1),
            Coordinate(-2, 2, 0),
            Coordinate(-1, 2, -1),
            Coordinate(0, 2, -2),
            Coordinate(1, 1, -2),
            Coordinate(2, 0, -2),
            Coordinate(2, -1, -1),
        ]
        .into_iter()
        .map(|c| (c, Topology::Land)),
    );
    items.extend([
        (Coordinate(3, -3, 0), Topology::Port(Direction::West)),
        (Coordinate(2, -3, 1), Topology::Water),
        (Coordinate(1, -3, 2), Topology::Port(Direction::Northwest)),
        (Coordinate(0, -3, 3), Topology::Water),
        (Coordinate(-1, -2, 3), Topology::Port(Direction::Northwest)),
        (Coordinate(-2, -1, 3), Topology::Water),
        (Coordinate(-3, 0, 3), Topology::Port(Direction::Northeast)),
        (Coordinate(-3, 1, 2), Topology::Water),
        (Coordinate(-3, 2, 1), Topology::Port(Direction::East)),
        (Coordinate(-3, 3, 0), Topology::Water),
        (Coordinate(-2, 3, -1), Topology::Port(Direction::East)),
        (Coordinate(-1, 3, -2), Topology::Water),
        (Coordinate(0, 3, -3), Topology::Port(Direction::Southeast)),
        (Coordinate(1, 2, -3), Topology::Water),
        (Coordinate(2, 1, -3), Topology::Port(Direction::Southwest)),
        (Coordinate(3, 0, -3), Topology::Water),
        (Coordinate(3, -1, -2), Topology::Port(Direction::Southwest)),
        (Coordinate(3, -2, -1), Topology::Water),
    ]);
    items
}

fn initialize_tiles<R: Rng + ?Sized>(
    kind: MapKind,
    number_placement: NumberPlacement,
    rng: &mut R,
) -> HashMap<Coordinate, Tile> {
    let (mut numbers, mut tile_resources, mut port_resources, should_shuffle) = match kind {
        MapKind::Base => (
            base_numbers(),
            base_tile_resources(),
            base_port_resources(),
            true,
        ),
        MapKind::Mini => (mini_numbers(), mini_tile_resources(), Vec::new(), true),
        MapKind::Tournament => (
            tournament_numbers(),
            tournament_tile_resources(),
            tournament_port_resources(),
            false,
        ),
    };
    if should_shuffle {
        numbers.shuffle(rng);
        tile_resources.shuffle(rng);
        port_resources.shuffle(rng);
    }

    let mut all_tiles = HashMap::new();
    let mut node_autoinc = 0;
    let mut tile_autoinc = 0;
    let mut port_autoinc = 0;

    for (coordinate, tile_type) in topology(kind) {
        let (nodes, edges, next_node) = get_nodes_and_edges(&all_tiles, coordinate, node_autoinc);
        node_autoinc = next_node;
        let tile = match tile_type {
            Topology::Port(direction) => {
                let resource = port_resources.pop().expect("port resource");
                let port = Port {
                    id: port_autoinc,
                    resource,
                    direction,
                    nodes,
                    edges,
                };
                port_autoinc += 1;
                Tile::Port(port)
            }
            Topology::Land => {
                let resource = tile_resources.pop().expect("tile resource");
                let number = resource.map(|_| numbers.pop().expect("number"));
                let tile = LandTile {
                    id: tile_autoinc,
                    resource,
                    number,
                    coordinate,
                    nodes,
                    edges,
                };
                tile_autoinc += 1;
                Tile::Land(tile)
            }
            Topology::Water => Tile::Water { nodes, edges },
        };
        all_tiles.insert(coordinate, tile);
    }

    if kind != MapKind::Tournament && number_placement == NumberPlacement::OfficialSpiral {
        let start = if kind == MapKind::Base {
            Coordinate(2, -2, 0)
        } else {
            Coordinate(1, -1, 0)
        };
        let mut i = 0;
        for coordinate in spiral_land_coordinates(&all_tiles, start) {
            if let Some(Tile::Land(tile)) = all_tiles.get_mut(&coordinate)
                && tile.resource.is_some()
            {
                tile.number = Some(official_spiral_numbers()[i]);
                i += 1;
            }
        }
    }

    all_tiles
}

fn get_nodes_and_edges(
    tiles: &HashMap<Coordinate, Tile>,
    coordinate: Coordinate,
    mut node_autoinc: usize,
) -> ([NodeId; 6], [Edge; 6], usize) {
    let mut nodes: [Option<NodeId>; 6] = [None; 6];
    let mut edges: [Option<Edge>; 6] = [None; 6];

    for direction in Direction::ALL {
        let coord = add_coord(coordinate, direction.unit());
        let Some(neighbor) = tiles.get(&coord) else {
            continue;
        };
        let nn = neighbor.nodes();
        let ne = neighbor.edges();
        match direction {
            Direction::East => {
                nodes[NodeRef::Northeast.idx()] = Some(nn[NodeRef::Northwest.idx()]);
                nodes[NodeRef::Southeast.idx()] = Some(nn[NodeRef::Southwest.idx()]);
                edges[EdgeRef::East.idx()] = Some(ne[EdgeRef::West.idx()]);
            }
            Direction::Southeast => {
                nodes[NodeRef::South.idx()] = Some(nn[NodeRef::Northwest.idx()]);
                nodes[NodeRef::Southeast.idx()] = Some(nn[NodeRef::North.idx()]);
                edges[EdgeRef::Southeast.idx()] = Some(ne[EdgeRef::Northwest.idx()]);
            }
            Direction::Southwest => {
                nodes[NodeRef::South.idx()] = Some(nn[NodeRef::Northeast.idx()]);
                nodes[NodeRef::Southwest.idx()] = Some(nn[NodeRef::North.idx()]);
                edges[EdgeRef::Southwest.idx()] = Some(ne[EdgeRef::Northeast.idx()]);
            }
            Direction::West => {
                nodes[NodeRef::Northwest.idx()] = Some(nn[NodeRef::Northeast.idx()]);
                nodes[NodeRef::Southwest.idx()] = Some(nn[NodeRef::Southeast.idx()]);
                edges[EdgeRef::West.idx()] = Some(ne[EdgeRef::East.idx()]);
            }
            Direction::Northwest => {
                nodes[NodeRef::North.idx()] = Some(nn[NodeRef::Southeast.idx()]);
                nodes[NodeRef::Northwest.idx()] = Some(nn[NodeRef::South.idx()]);
                edges[EdgeRef::Northwest.idx()] = Some(ne[EdgeRef::Southeast.idx()]);
            }
            Direction::Northeast => {
                nodes[NodeRef::North.idx()] = Some(nn[NodeRef::Southwest.idx()]);
                nodes[NodeRef::Northeast.idx()] = Some(nn[NodeRef::South.idx()]);
                edges[EdgeRef::Northeast.idx()] = Some(ne[EdgeRef::Southwest.idx()]);
            }
        }
    }

    let mut final_nodes = [0; 6];
    for node_ref in NodeRef::ALL {
        final_nodes[node_ref.idx()] = match nodes[node_ref.idx()] {
            Some(node) => node,
            None => {
                let node = node_autoinc;
                node_autoinc += 1;
                node
            }
        };
    }

    let mut final_edges = [(0, 0); 6];
    for edge_ref in EdgeRef::ALL {
        final_edges[edge_ref.idx()] = match edges[edge_ref.idx()] {
            Some(edge) => canonical_edge(edge),
            None => {
                let (a, b) = edge_nodes(edge_ref);
                canonical_edge((final_nodes[a.idx()], final_nodes[b.idx()]))
            }
        };
    }

    (final_nodes, final_edges, node_autoinc)
}

fn edge_nodes(edge_ref: EdgeRef) -> (NodeRef, NodeRef) {
    match edge_ref {
        EdgeRef::East => (NodeRef::Northeast, NodeRef::Southeast),
        EdgeRef::Southeast => (NodeRef::Southeast, NodeRef::South),
        EdgeRef::Southwest => (NodeRef::South, NodeRef::Southwest),
        EdgeRef::West => (NodeRef::Southwest, NodeRef::Northwest),
        EdgeRef::Northwest => (NodeRef::Northwest, NodeRef::North),
        EdgeRef::Northeast => (NodeRef::North, NodeRef::Northeast),
    }
}

fn port_direction_node_refs(direction: Direction) -> (NodeRef, NodeRef) {
    match direction {
        Direction::West => (NodeRef::Northwest, NodeRef::Southwest),
        Direction::Northwest => (NodeRef::North, NodeRef::Northwest),
        Direction::Northeast => (NodeRef::Northeast, NodeRef::North),
        Direction::East => (NodeRef::Southeast, NodeRef::Northeast),
        Direction::Southeast => (NodeRef::South, NodeRef::Southeast),
        Direction::Southwest => (NodeRef::Southwest, NodeRef::South),
    }
}

fn spiral_land_coordinates(
    tiles: &HashMap<Coordinate, Tile>,
    start: Coordinate,
) -> Vec<Coordinate> {
    let is_land = |coord: Coordinate| matches!(tiles.get(&coord), Some(Tile::Land(_)));
    assert!(is_land(start), "start must be land");
    let total = tiles
        .values()
        .filter(|t| matches!(t, Tile::Land(_)))
        .count();
    let mut direction = None;
    for (i, candidate) in Direction::REVERSED.iter().enumerate() {
        let previous =
            Direction::REVERSED[(i + Direction::REVERSED.len() - 1) % Direction::REVERSED.len()];
        if is_land(add_coord(start, candidate.unit()))
            && !is_land(add_coord(start, previous.unit()))
        {
            direction = Some(*candidate);
            break;
        }
    }
    let mut direction = direction.expect("initial spiral direction");
    let mut visited = HashSet::new();
    let mut coord = start;
    let mut out = Vec::with_capacity(total);
    while visited.len() < total {
        if visited.insert(coord) {
            out.push(coord);
        }
        let next = add_coord(coord, direction.unit());
        if is_land(next) && !visited.contains(&next) {
            coord = next;
            continue;
        }
        let index = Direction::REVERSED
            .iter()
            .position(|d| *d == direction)
            .unwrap();
        direction = Direction::REVERSED[(index + 1) % Direction::REVERSED.len()];
    }
    out
}

fn number_probability(number: u8) -> f64 {
    match number {
        2 | 12 => 1.0 / 36.0,
        3 | 11 => 2.0 / 36.0,
        4 | 10 => 3.0 / 36.0,
        5 | 9 => 4.0 / 36.0,
        6 | 8 => 5.0 / 36.0,
        7 => 6.0 / 36.0,
        _ => 0.0,
    }
}

fn cached_tournament_map() -> CatanMap {
    static MAP: OnceLock<CatanMap> = OnceLock::new();
    MAP.get_or_init(|| {
        let mut rng = SmallRng::seed_from_u64(0);
        CatanMap::from_template_with_rng(
            MapKind::Tournament,
            NumberPlacement::OfficialSpiral,
            &mut rng,
        )
    })
    .clone()
}

fn canonical_edge(edge: Edge) -> Edge {
    if edge.0 <= edge.1 {
        edge
    } else {
        (edge.1, edge.0)
    }
}

#[derive(Clone, Debug)]
pub struct Board {
    pub map: CatanMap,
    pub buildings: HashMap<NodeId, (Color, BuildingType)>,
    pub roads: HashMap<Edge, Color>,
    connected_components: HashMap<Color, Vec<HashSet<NodeId>>>,
    board_buildable_ids: HashSet<NodeId>,
    road_lengths: HashMap<Color, usize>,
    pub road_color: Option<Color>,
    pub road_length: usize,
    pub robber_coordinate: Coordinate,
    adjacency: HashMap<NodeId, Vec<NodeId>>,
    land_edges: HashSet<Edge>,
}

impl Default for Board {
    fn default() -> Self {
        Self::new(CatanMap::base())
    }
}

impl Board {
    pub fn new(map: CatanMap) -> Self {
        let land_edges: HashSet<_> = map.land_edges().into_iter().collect();
        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for &(a, b) in &land_edges {
            adjacency.entry(a).or_default().push(b);
            adjacency.entry(b).or_default().push(a);
        }
        for neighbors in adjacency.values_mut() {
            neighbors.sort_unstable();
        }
        let robber_coordinate = map
            .land_tiles
            .iter()
            .find_map(|(coord, tile)| tile.resource.is_none().then_some(*coord))
            .expect("at least one desert");
        let board_buildable_ids = map.land_nodes.clone();
        Board {
            map,
            buildings: HashMap::new(),
            roads: HashMap::new(),
            connected_components: HashMap::new(),
            board_buildable_ids,
            road_lengths: HashMap::new(),
            road_color: None,
            road_length: 0,
            robber_coordinate,
            adjacency,
            land_edges,
        }
    }

    pub fn build_settlement(
        &mut self,
        color: Color,
        node_id: NodeId,
        initial_build_phase: bool,
    ) -> Result<RoadAwardUpdate, String> {
        if !self
            .buildable_node_ids(color, initial_build_phase)
            .contains(&node_id)
        {
            return Err("invalid settlement placement".into());
        }
        if self.buildings.contains_key(&node_id) {
            return Err("a building already exists there".into());
        }
        Ok(self.build_settlement_known_valid(color, node_id, initial_build_phase))
    }

    fn build_settlement_known_valid(
        &mut self,
        color: Color,
        node_id: NodeId,
        initial_build_phase: bool,
    ) -> RoadAwardUpdate {
        let previous_road_color = self.road_color;
        self.buildings
            .insert(node_id, (color, BuildingType::Settlement));
        if initial_build_phase {
            self.connected_components
                .entry(color)
                .or_default()
                .push(HashSet::from([node_id]));
        } else {
            self.recompute_components_and_roads();
        }
        self.board_buildable_ids.remove(&node_id);
        if let Some(neighbors) = self.adjacency.get(&node_id) {
            for neighbor in neighbors {
                self.board_buildable_ids.remove(neighbor);
            }
        }
        (
            previous_road_color,
            self.road_color,
            self.road_lengths.clone(),
        )
    }

    pub fn build_road(&mut self, color: Color, edge: Edge) -> Result<RoadAwardUpdate, String> {
        let edge = canonical_edge(edge);
        if !self.buildable_edges(color).contains(&edge) {
            return Err("invalid road placement".into());
        }
        Ok(self.build_road_known_valid(color, edge))
    }

    fn build_road_known_valid(&mut self, color: Color, edge: Edge) -> RoadAwardUpdate {
        let edge = canonical_edge(edge);
        let previous_road_color = self.road_color;
        self.roads.insert(edge, color);
        self.recompute_components_and_roads_for_color(color);
        self.recompute_road_award();
        (
            previous_road_color,
            self.road_color,
            self.road_lengths.clone(),
        )
    }

    pub fn build_city(&mut self, color: Color, node_id: NodeId) -> Result<(), String> {
        match self.buildings.get_mut(&node_id) {
            Some((owner, building)) if *owner == color && *building == BuildingType::Settlement => {
                *building = BuildingType::City;
                Ok(())
            }
            _ => Err("invalid city placement".into()),
        }
    }

    pub fn buildable_node_ids(&self, color: Color, initial_build_phase: bool) -> Vec<NodeId> {
        if initial_build_phase {
            let mut out: Vec<_> = self.board_buildable_ids.iter().copied().collect();
            out.sort_unstable();
            return out;
        }
        let mut out = Vec::new();
        for component in self.connected_components.get(&color).into_iter().flatten() {
            out.extend(
                component
                    .iter()
                    .copied()
                    .filter(|node| self.board_buildable_ids.contains(node)),
            );
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    pub fn buildable_edges(&self, color: Color) -> Vec<Edge> {
        let mut out = Vec::new();
        for component in self.connected_components.get(&color).into_iter().flatten() {
            for &node in component {
                if self.is_enemy_node(node, color) {
                    continue;
                }
                for neighbor in self.adjacency.get(&node).into_iter().flatten() {
                    let edge = canonical_edge((node, *neighbor));
                    if self.land_edges.contains(&edge) && !self.roads.contains_key(&edge) {
                        out.push(edge);
                    }
                }
            }
        }
        out.sort_unstable();
        out.dedup();
        out
    }

    pub fn get_player_port_resources(&self, color: Color) -> HashSet<Option<Resource>> {
        let mut resources = HashSet::new();
        for (&resource, nodes) in &self.map.port_nodes {
            if nodes.iter().any(|node| self.is_friendly_node(*node, color)) {
                resources.insert(resource);
            }
        }
        resources
    }

    pub fn find_connected_components(&self, color: Color) -> Vec<HashSet<NodeId>> {
        self.connected_components
            .get(&color)
            .cloned()
            .unwrap_or_default()
    }

    pub fn continuous_roads_by_player(&self, color: Color) -> Vec<Vec<Edge>> {
        self.find_connected_components(color)
            .iter()
            .map(|component| self.longest_acyclic_path(component, color))
            .collect()
    }

    pub fn get_node_color(&self, node_id: NodeId) -> Option<Color> {
        self.buildings.get(&node_id).map(|(color, _)| *color)
    }

    pub fn get_edge_color(&self, edge: Edge) -> Option<Color> {
        self.roads.get(&canonical_edge(edge)).copied()
    }

    pub fn is_enemy_node(&self, node_id: NodeId, color: Color) -> bool {
        self.get_node_color(node_id).is_some_and(|c| c != color)
    }

    pub fn is_friendly_node(&self, node_id: NodeId, color: Color) -> bool {
        self.get_node_color(node_id) == Some(color)
    }

    pub fn is_friendly_road(&self, edge: Edge, color: Color) -> bool {
        self.get_edge_color(edge) == Some(color)
    }

    fn dfs_walk(&self, start: NodeId, color: Color) -> HashSet<NodeId> {
        let mut agenda = vec![start];
        let mut visited = HashSet::new();
        while let Some(node) = agenda.pop() {
            if !visited.insert(node) {
                continue;
            }
            if node != start && self.is_enemy_node(node, color) {
                continue;
            }
            for neighbor in self.adjacency.get(&node).into_iter().flatten() {
                if !visited.contains(neighbor) && self.is_friendly_road((node, *neighbor), color) {
                    agenda.push(*neighbor);
                }
            }
        }
        visited
    }

    fn recompute_components_and_roads(&mut self) {
        self.connected_components.clear();
        self.road_lengths.clear();

        for color in Color::ALL {
            self.recompute_components_and_roads_for_color(color);
        }
        self.recompute_road_award();
    }

    fn recompute_components_and_roads_for_color(&mut self, color: Color) {
        let mut starts: HashSet<NodeId> = HashSet::new();
        for (&node, &(owner, _)) in &self.buildings {
            if owner == color {
                starts.insert(node);
            }
        }
        for (&(a, b), &owner) in &self.roads {
            if owner == color {
                starts.insert(a);
                starts.insert(b);
            }
        }

        let mut seen = HashSet::new();
        let mut components = Vec::new();
        for start in starts {
            if seen.contains(&start) {
                continue;
            }
            let component = self.dfs_walk(start, color);
            seen.extend(component.iter().copied());
            components.push(component);
        }

        let longest = components
            .iter()
            .map(|component| self.longest_acyclic_path(component, color).len())
            .max()
            .unwrap_or(0);
        if longest > 0 {
            self.road_lengths.insert(color, longest);
        } else {
            self.road_lengths.remove(&color);
        }
        if components.is_empty() {
            self.connected_components.remove(&color);
        } else {
            self.connected_components.insert(color, components);
        }
    }

    fn recompute_road_award(&mut self) {
        let mut qualifying = [0usize; Color::ALL.len()];
        for (index, color) in Color::ALL.iter().enumerate() {
            let length = self.road_lengths.get(color).copied().unwrap_or(0);
            if length >= 5 {
                qualifying[index] = length;
            }
        }
        let max_length = qualifying.iter().copied().max().unwrap_or(0);
        if max_length == 0 {
            self.road_color = None;
            self.road_length = 0;
            return;
        }
        // Official rules: the incumbent keeps the card on ties; a challenger
        // must have a strictly longer road to take it.
        if let Some(incumbent) = self.road_color {
            let incumbent_length = qualifying[color_order(incumbent) as usize];
            if incumbent_length == max_length {
                self.road_length = incumbent_length;
                return;
            }
        }
        let mut leaders = Color::ALL
            .iter()
            .copied()
            .filter(|color| qualifying[color_order(*color) as usize] == max_length);
        let leader = leaders.next().unwrap();
        if leaders.next().is_none() {
            self.road_color = Some(leader);
            self.road_length = max_length;
        } else {
            // No qualified incumbent and several players tie for the longest
            // road: the card is set aside until one is strictly longest.
            self.road_color = None;
            self.road_length = 0;
        }
    }

    fn longest_acyclic_path(&self, node_set: &HashSet<NodeId>, color: Color) -> Vec<Edge> {
        let mut best = Vec::new();
        for &start in node_set {
            let mut agenda = vec![(start, Vec::<Edge>::new())];
            while let Some((node, path)) = agenda.pop() {
                let mut expanded = false;
                for neighbor in self.adjacency.get(&node).into_iter().flatten() {
                    let edge = canonical_edge((node, *neighbor));
                    if !self.is_friendly_road(edge, color) || path.contains(&edge) {
                        continue;
                    }
                    let mut next_path = path.clone();
                    next_path.push(edge);
                    if self.is_enemy_node(*neighbor, color) {
                        // The edge into an enemy building counts toward road
                        // length; the path just cannot continue through it.
                        if next_path.len() > best.len() {
                            best = next_path;
                        }
                        expanded = true;
                        continue;
                    }
                    agenda.push((*neighbor, next_path));
                    expanded = true;
                }
                if !expanded && path.len() > best.len() {
                    best = path;
                }
            }
        }
        best
    }
}

pub fn node_distances() -> HashMap<NodeId, HashMap<NodeId, usize>> {
    let board = Board::default();
    let mut distances = HashMap::new();
    for &start in &board.map.land_nodes {
        let mut seen = HashMap::from([(start, 0usize)]);
        let mut queue = VecDeque::from([start]);
        while let Some(node) = queue.pop_front() {
            let dist = seen[&node];
            for neighbor in board.adjacency.get(&node).into_iter().flatten() {
                if !seen.contains_key(neighbor) {
                    seen.insert(*neighbor, dist + 1);
                    queue.push_back(*neighbor);
                }
            }
        }
        distances.insert(start, seen);
    }
    distances
}

pub const BOARD_TENSOR_WIDTH: usize = 21;
pub const BOARD_TENSOR_HEIGHT: usize = 11;
const BOARD_TENSOR_PATH_PAIRS: [(NodeId, NodeId); 6] =
    [(82, 93), (79, 94), (42, 25), (41, 26), (73, 59), (72, 60)];
const PARALLEL_BOARD_TENSOR_BATCH_MIN_ROWS: usize = 8;
const PARALLEL_SAMPLE_VECTOR_BATCH_MIN_ROWS: usize = 8;
type TensorPoint = (usize, usize);
type TensorNodeMap = HashMap<NodeId, TensorPoint>;
type TensorEdgeMap = HashMap<Edge, TensorPoint>;

fn shortest_node_path(
    adjacency: &HashMap<NodeId, Vec<NodeId>>,
    start: NodeId,
    end: NodeId,
) -> Option<Vec<NodeId>> {
    let mut parent: HashMap<NodeId, NodeId> = HashMap::new();
    let mut queue = VecDeque::from([start]);
    parent.insert(start, start);

    while let Some(node) = queue.pop_front() {
        if node == end {
            let mut path = vec![end];
            let mut current = end;
            while current != start {
                current = parent[&current];
                path.push(current);
            }
            path.reverse();
            return Some(path);
        }
        for neighbor in adjacency.get(&node).into_iter().flatten() {
            if !parent.contains_key(neighbor) {
                parent.insert(*neighbor, node);
                queue.push_back(*neighbor);
            }
        }
    }
    None
}

#[cfg(test)]
fn board_tensor_node_edge_maps() -> (TensorNodeMap, TensorEdgeMap) {
    let (node_map, edge_map) = board_tensor_node_edge_maps_static();
    (node_map.clone(), edge_map.clone())
}

fn board_tensor_node_edge_maps_static() -> &'static (TensorNodeMap, TensorEdgeMap) {
    static MAPS: OnceLock<(TensorNodeMap, TensorEdgeMap)> = OnceLock::new();
    MAPS.get_or_init(build_board_tensor_node_edge_maps)
}

fn build_board_tensor_node_edge_maps() -> (TensorNodeMap, TensorEdgeMap) {
    let board = Board::default();
    let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
    for tile in board.map.tiles.values() {
        for edge in tile.edges() {
            let (a, b) = canonical_edge(*edge);
            adjacency.entry(a).or_default().push(b);
            adjacency.entry(b).or_default().push(a);
        }
    }
    for neighbors in adjacency.values_mut() {
        neighbors.sort_unstable();
        neighbors.dedup();
    }
    let paths: Vec<Vec<NodeId>> = BOARD_TENSOR_PATH_PAIRS
        .iter()
        .map(|&(start, end)| {
            shortest_node_path(&adjacency, start, end).expect("static board tensor path exists")
        })
        .collect();

    let mut node_map = HashMap::new();
    let mut edge_map = HashMap::new();
    for (i, path) in paths.iter().enumerate() {
        for (j, &node) in path.iter().enumerate() {
            node_map.insert(node, (2 * j, 2 * i));

            let node_has_down_edge = (i + j) % 2 == 0;
            if node_has_down_edge && i + 1 < paths.len() {
                let next_node = paths[i + 1][j];
                edge_map.insert(canonical_edge((node, next_node)), (2 * j, 2 * i + 1));
            }
            if j + 1 < path.len() {
                edge_map.insert(canonical_edge((node, path[j + 1])), (2 * j + 1, 2 * i));
            }
        }
    }
    (node_map, edge_map)
}

fn offset_to_cube(offset_x: i8, offset_y: i8) -> Coordinate {
    let parity = offset_y & 1;
    let x = offset_x - ((offset_y - parity) / 2);
    let z = offset_y;
    let y = -x - z;
    Coordinate(x, y, z)
}

#[cfg(test)]
fn board_tensor_tile_coordinate_map() -> HashMap<Coordinate, (usize, usize)> {
    board_tensor_tile_coordinate_map_static().clone()
}

fn board_tensor_tile_coordinate_map_static() -> &'static HashMap<Coordinate, (usize, usize)> {
    static TILE_MAP: OnceLock<HashMap<Coordinate, (usize, usize)>> = OnceLock::new();
    TILE_MAP.get_or_init(build_board_tensor_tile_coordinate_map)
}

fn build_board_tensor_tile_coordinate_map() -> HashMap<Coordinate, (usize, usize)> {
    let mut tile_map = HashMap::new();
    let width_step = 4;
    let height_step = 2;
    for i in 0..(BOARD_TENSOR_HEIGHT / height_step) {
        for j in 0..(BOARD_TENSOR_WIDTH / width_step) {
            let offset_x = -2 + j as i8;
            let offset_y = -2 + i as i8;
            let maybe_odd_offset = (i % 2) * 2;
            tile_map.insert(
                offset_to_cube(offset_x, offset_y),
                (height_step * i, width_step * j + maybe_odd_offset),
            );
        }
    }
    tile_map
}

fn board_tensor_plane_index(channel: usize, x: usize, y: usize) -> usize {
    (channel * BOARD_TENSOR_WIDTH + x) * BOARD_TENSOR_HEIGHT + y
}

fn board_tensor_index(
    channels_first: bool,
    channels: usize,
    channel: usize,
    x: usize,
    y: usize,
) -> usize {
    if channels_first {
        board_tensor_plane_index(channel, x, y)
    } else {
        (x * BOARD_TENSOR_HEIGHT + y) * channels + channel
    }
}

pub const fn board_tensor_channels(num_players: usize) -> usize {
    2 * num_players + 5 + 1 + 6
}

pub const fn board_tensor_flat_len(num_players: usize) -> usize {
    board_tensor_channels(num_players) * BOARD_TENSOR_WIDTH * BOARD_TENSOR_HEIGHT
}

pub const fn board_tensor_shape(num_players: usize, channels_first: bool) -> (usize, usize, usize) {
    let channels = board_tensor_channels(num_players);
    if channels_first {
        (channels, BOARD_TENSOR_WIDTH, BOARD_TENSOR_HEIGHT)
    } else {
        (BOARD_TENSOR_WIDTH, BOARD_TENSOR_HEIGHT, channels)
    }
}

trait BoardTensorScalar: Copy {
    fn zero() -> Self;
    fn from_f64(value: f64) -> Self;
    fn add_assign(&mut self, value: f64);
}

impl BoardTensorScalar for f64 {
    fn zero() -> Self {
        0.0
    }

    fn from_f64(value: f64) -> Self {
        value
    }

    fn add_assign(&mut self, value: f64) {
        *self += value;
    }
}

impl BoardTensorScalar for f32 {
    fn zero() -> Self {
        0.0
    }

    fn from_f64(value: f64) -> Self {
        value as f32
    }

    fn add_assign(&mut self, value: f64) {
        *self += value as f32;
    }
}

fn mark_tile_nodes<T: BoardTensorScalar>(
    out: &mut [T],
    channels_first: bool,
    channels: usize,
    channel: usize,
    x: usize,
    y: usize,
    value: f64,
) {
    for (dx, dy) in [(0, 0), (2, 0), (4, 0), (0, 2), (2, 2), (4, 2)] {
        out[board_tensor_index(channels_first, channels, channel, x + dx, y + dy)]
            .add_assign(value);
    }
}

fn write_f32_le_at(out: &mut [u8], index: usize, value: f64) {
    let start = index * 4;
    out[start..start + 4].copy_from_slice(&(value as f32).to_le_bytes());
}

fn add_f32_le_at(out: &mut [u8], index: usize, value: f64) {
    let start = index * 4;
    let mut bytes = [0_u8; 4];
    bytes.copy_from_slice(&out[start..start + 4]);
    let updated = f32::from_le_bytes(bytes) + value as f32;
    out[start..start + 4].copy_from_slice(&updated.to_le_bytes());
}

fn mark_tile_nodes_f32_le_bytes(
    out: &mut [u8],
    channels_first: bool,
    channels: usize,
    channel: usize,
    x: usize,
    y: usize,
    value: f64,
) {
    for (dx, dy) in [(0, 0), (2, 0), (4, 0), (0, 2), (2, 2), (4, 2)] {
        add_f32_le_at(
            out,
            board_tensor_index(channels_first, channels, channel, x + dx, y + dy),
            value,
        );
    }
}

fn fill_board_tensor_flat_into<T: BoardTensorScalar>(
    game: &Game,
    p0_color: Color,
    channels_first: bool,
    out: &mut [T],
) -> Result<(usize, usize, usize), String> {
    let n = game.state.colors.len();
    let channels = board_tensor_channels(n);
    let expected = board_tensor_flat_len(n);
    if out.len() != expected {
        return Err(format!(
            "board tensor output length mismatch: got {}, expected {expected}",
            out.len()
        ));
    }
    out.fill(T::zero());
    let (node_map, edge_map) = board_tensor_node_edge_maps_static();

    for (i, color) in player_colors_from_perspective(&game.state.colors, p0_color) {
        for (&node_id, &(owner, building)) in &game.state.board.buildings {
            if owner != color {
                continue;
            }
            if let Some(&(x, y)) = node_map.get(&node_id) {
                let value = match building {
                    BuildingType::Settlement => 1.0,
                    BuildingType::City => 2.0,
                    BuildingType::Road => 0.0,
                };
                if value > 0.0 {
                    out[board_tensor_index(channels_first, channels, 2 * i, x, y)] =
                        T::from_f64(value);
                }
            }
        }

        for (&edge, &owner) in &game.state.board.roads {
            if owner != color {
                continue;
            }
            if let Some(&(x, y)) = edge_map.get(&canonical_edge(edge)) {
                out[board_tensor_index(channels_first, channels, 2 * i + 1, x, y)] =
                    T::from_f64(1.0);
            }
        }
    }

    let tile_map = board_tensor_tile_coordinate_map_static();
    for (&coordinate, tile) in &game.state.board.map.land_tiles {
        let Some(resource) = tile.resource else {
            continue;
        };
        let Some(&(y, x)) = tile_map.get(&coordinate) else {
            continue;
        };
        let probability = tile.number.map(number_probability).unwrap_or(0.0);
        mark_tile_nodes(
            out,
            channels_first,
            channels,
            2 * n + resource.idx(),
            x,
            y,
            probability,
        );
    }

    if let Some(&(y, x)) = tile_map.get(&game.state.board.robber_coordinate) {
        mark_tile_nodes(out, channels_first, channels, 2 * n + 5, x, y, 1.0);
    }

    for (&resource, node_ids) in &game.state.board.map.port_nodes {
        let channel_delta = resource.map_or(5, Resource::idx);
        let channel = 2 * n + 5 + 1 + channel_delta;
        for node_id in node_ids {
            if let Some(&(x, y)) = node_map.get(node_id) {
                out[board_tensor_index(channels_first, channels, channel, x, y)] = T::from_f64(1.0);
            }
        }
    }

    Ok(board_tensor_shape(n, channels_first))
}

pub fn fill_board_tensor_flat_f32_le_bytes(
    game: &Game,
    p0_color: Color,
    channels_first: bool,
    out: &mut [u8],
) -> Result<(usize, usize, usize), String> {
    let n = game.state.colors.len();
    let channels = board_tensor_channels(n);
    let values_len = board_tensor_flat_len(n);
    let expected = values_len.checked_mul(4).ok_or_else(|| {
        format!("board tensor byte length overflow for {values_len} float32 values")
    })?;
    if out.len() != expected {
        return Err(format!(
            "board tensor byte output length mismatch: got {}, expected {expected}",
            out.len()
        ));
    }
    out.fill(0);
    let (node_map, edge_map) = board_tensor_node_edge_maps_static();

    for (i, color) in player_colors_from_perspective(&game.state.colors, p0_color) {
        for (&node_id, &(owner, building)) in &game.state.board.buildings {
            if owner != color {
                continue;
            }
            if let Some(&(x, y)) = node_map.get(&node_id) {
                let value = match building {
                    BuildingType::Settlement => 1.0,
                    BuildingType::City => 2.0,
                    BuildingType::Road => 0.0,
                };
                if value > 0.0 {
                    write_f32_le_at(
                        out,
                        board_tensor_index(channels_first, channels, 2 * i, x, y),
                        value,
                    );
                }
            }
        }

        for (&edge, &owner) in &game.state.board.roads {
            if owner != color {
                continue;
            }
            if let Some(&(x, y)) = edge_map.get(&canonical_edge(edge)) {
                write_f32_le_at(
                    out,
                    board_tensor_index(channels_first, channels, 2 * i + 1, x, y),
                    1.0,
                );
            }
        }
    }

    let tile_map = board_tensor_tile_coordinate_map_static();
    for (&coordinate, tile) in &game.state.board.map.land_tiles {
        let Some(resource) = tile.resource else {
            continue;
        };
        let Some(&(y, x)) = tile_map.get(&coordinate) else {
            continue;
        };
        let probability = tile.number.map(number_probability).unwrap_or(0.0);
        mark_tile_nodes_f32_le_bytes(
            out,
            channels_first,
            channels,
            2 * n + resource.idx(),
            x,
            y,
            probability,
        );
    }

    if let Some(&(y, x)) = tile_map.get(&game.state.board.robber_coordinate) {
        mark_tile_nodes_f32_le_bytes(out, channels_first, channels, 2 * n + 5, x, y, 1.0);
    }

    for (&resource, node_ids) in &game.state.board.map.port_nodes {
        let channel_delta = resource.map_or(5, Resource::idx);
        let channel = 2 * n + 5 + 1 + channel_delta;
        for node_id in node_ids {
            if let Some(&(x, y)) = node_map.get(node_id) {
                write_f32_le_at(
                    out,
                    board_tensor_index(channels_first, channels, channel, x, y),
                    1.0,
                );
            }
        }
    }

    Ok(board_tensor_shape(n, channels_first))
}

pub fn fill_board_tensor_flat(
    game: &Game,
    p0_color: Color,
    channels_first: bool,
    out: &mut [f64],
) -> Result<(usize, usize, usize), String> {
    fill_board_tensor_flat_into(game, p0_color, channels_first, out)
}

pub fn fill_board_tensor_flat_f32(
    game: &Game,
    p0_color: Color,
    channels_first: bool,
    out: &mut [f32],
) -> Result<(usize, usize, usize), String> {
    fill_board_tensor_flat_into(game, p0_color, channels_first, out)
}

pub fn create_board_tensor_flat(
    game: &Game,
    p0_color: Color,
    channels_first: bool,
) -> (Vec<f64>, (usize, usize, usize)) {
    let mut out = vec![0.0; board_tensor_flat_len(game.state.colors.len())];
    let shape = fill_board_tensor_flat(game, p0_color, channels_first, &mut out)
        .expect("allocated board tensor output has correct length");
    (out, shape)
}

pub fn fill_board_tensor_batch_flat(
    samples: &[(&Game, Color)],
    channels_first: bool,
    out: &mut [f64],
) -> Result<(usize, usize, usize, usize), String> {
    if samples.is_empty() {
        return Ok(if channels_first {
            (0, 0, BOARD_TENSOR_WIDTH, BOARD_TENSOR_HEIGHT)
        } else {
            (0, BOARD_TENSOR_WIDTH, BOARD_TENSOR_HEIGHT, 0)
        });
    }

    let num_players = samples[0].0.state.colors.len();
    let row_len = board_tensor_flat_len(num_players);
    let expected = row_len * samples.len();
    if out.len() != expected {
        return Err(format!(
            "board tensor batch output length mismatch: got {}, expected {expected}",
            out.len()
        ));
    }
    for (game, _) in samples {
        if game.state.colors.len() != num_players {
            return Err("all games in a board tensor batch must have the same player count".into());
        }
    }

    for (index, (game, color)) in samples.iter().enumerate() {
        let start = index * row_len;
        let end = start + row_len;
        fill_board_tensor_flat(game, *color, channels_first, &mut out[start..end])?;
    }

    let (a, b, c) = board_tensor_shape(num_players, channels_first);
    Ok((samples.len(), a, b, c))
}

pub fn fill_board_tensor_batch_flat_f32(
    samples: &[(&Game, Color)],
    channels_first: bool,
    out: &mut [f32],
) -> Result<(usize, usize, usize, usize), String> {
    if samples.is_empty() {
        return Ok(if channels_first {
            (0, 0, BOARD_TENSOR_WIDTH, BOARD_TENSOR_HEIGHT)
        } else {
            (0, BOARD_TENSOR_WIDTH, BOARD_TENSOR_HEIGHT, 0)
        });
    }

    let num_players = samples[0].0.state.colors.len();
    let row_len = board_tensor_flat_len(num_players);
    let expected = row_len * samples.len();
    if out.len() != expected {
        return Err(format!(
            "board tensor batch output length mismatch: got {}, expected {expected}",
            out.len()
        ));
    }
    for (game, _) in samples {
        if game.state.colors.len() != num_players {
            return Err("all games in a board tensor batch must have the same player count".into());
        }
    }

    if samples.len() >= PARALLEL_BOARD_TENSOR_BATCH_MIN_ROWS {
        samples
            .par_iter()
            .zip(out.par_chunks_mut(row_len))
            .try_for_each(|((game, color), row)| {
                fill_board_tensor_flat_f32(game, *color, channels_first, row).map(|_| ())
            })?;
    } else {
        for (index, (game, color)) in samples.iter().enumerate() {
            let start = index * row_len;
            let end = start + row_len;
            fill_board_tensor_flat_f32(game, *color, channels_first, &mut out[start..end])?;
        }
    }

    let (a, b, c) = board_tensor_shape(num_players, channels_first);
    Ok((samples.len(), a, b, c))
}

pub fn fill_board_tensor_batch_flat_f32_le_bytes(
    samples: &[(&Game, Color)],
    channels_first: bool,
    out: &mut [u8],
) -> Result<(usize, usize, usize, usize), String> {
    if samples.is_empty() {
        if !out.is_empty() {
            return Err(format!(
                "board tensor byte batch output length mismatch: got {}, expected 0",
                out.len()
            ));
        }
        return Ok(if channels_first {
            (0, 0, BOARD_TENSOR_WIDTH, BOARD_TENSOR_HEIGHT)
        } else {
            (0, BOARD_TENSOR_WIDTH, BOARD_TENSOR_HEIGHT, 0)
        });
    }

    let num_players = samples[0].0.state.colors.len();
    let row_len = board_tensor_flat_len(num_players);
    let row_bytes = row_len.checked_mul(4).ok_or_else(|| {
        format!("board tensor byte row length overflow for {row_len} float32 values")
    })?;
    let expected = row_bytes.checked_mul(samples.len()).ok_or_else(|| {
        format!(
            "board tensor byte batch length overflow: {} rows * {row_bytes} bytes",
            samples.len()
        )
    })?;
    if out.len() != expected {
        return Err(format!(
            "board tensor byte batch output length mismatch: got {}, expected {expected}",
            out.len()
        ));
    }
    for (game, _) in samples {
        if game.state.colors.len() != num_players {
            return Err("all games in a board tensor batch must have the same player count".into());
        }
    }

    if samples.len() >= PARALLEL_BOARD_TENSOR_BATCH_MIN_ROWS {
        samples
            .par_iter()
            .zip(out.par_chunks_mut(row_bytes))
            .try_for_each(|((game, color), row)| {
                fill_board_tensor_flat_f32_le_bytes(game, *color, channels_first, row).map(|_| ())
            })?;
    } else {
        for ((game, color), row) in samples.iter().zip(out.chunks_mut(row_bytes)) {
            fill_board_tensor_flat_f32_le_bytes(game, *color, channels_first, row)?;
        }
    }

    let (a, b, c) = board_tensor_shape(num_players, channels_first);
    Ok((samples.len(), a, b, c))
}

pub fn create_board_tensor_batch_flat_f32(
    samples: &[(&Game, Color)],
    channels_first: bool,
) -> (Vec<f32>, (usize, usize, usize, usize)) {
    if samples.is_empty() {
        let shape = if channels_first {
            (0, 0, BOARD_TENSOR_WIDTH, BOARD_TENSOR_HEIGHT)
        } else {
            (0, BOARD_TENSOR_WIDTH, BOARD_TENSOR_HEIGHT, 0)
        };
        return (Vec::new(), shape);
    }
    let row_len = board_tensor_flat_len(samples[0].0.state.colors.len());
    let mut out = vec![0.0; row_len * samples.len()];
    let shape = fill_board_tensor_batch_flat_f32(samples, channels_first, &mut out)
        .expect("allocated board tensor batch output has correct length");
    (out, shape)
}

pub fn create_board_tensor_batch_flat(
    samples: &[(&Game, Color)],
    channels_first: bool,
) -> (Vec<f64>, (usize, usize, usize, usize)) {
    if samples.is_empty() {
        let shape = if channels_first {
            (0, 0, BOARD_TENSOR_WIDTH, BOARD_TENSOR_HEIGHT)
        } else {
            (0, BOARD_TENSOR_WIDTH, BOARD_TENSOR_HEIGHT, 0)
        };
        return (Vec::new(), shape);
    }
    let row_len = board_tensor_flat_len(samples[0].0.state.colors.len());
    let mut out = vec![0.0; row_len * samples.len()];
    let shape = fill_board_tensor_batch_flat(samples, channels_first, &mut out)
        .expect("allocated board tensor batch output has correct length");
    (out, shape)
}

pub fn create_current_board_tensor_batch_flat(
    games: &[Game],
    channels_first: bool,
) -> (Vec<f64>, (usize, usize, usize, usize)) {
    let samples = games
        .iter()
        .map(|game| (game, game.state.current_color()))
        .collect::<Vec<_>>();
    create_board_tensor_batch_flat(&samples, channels_first)
}

pub fn create_current_board_tensor_batch_flat_f32(
    games: &[Game],
    channels_first: bool,
) -> (Vec<f32>, (usize, usize, usize, usize)) {
    let samples = games
        .iter()
        .map(|game| (game, game.state.current_color()))
        .collect::<Vec<_>>();
    if samples.is_empty() {
        let shape = if channels_first {
            (0, 0, BOARD_TENSOR_WIDTH, BOARD_TENSOR_HEIGHT)
        } else {
            (0, BOARD_TENSOR_WIDTH, BOARD_TENSOR_HEIGHT, 0)
        };
        return (Vec::new(), shape);
    }
    create_board_tensor_batch_flat_f32(&samples, channels_first)
}

fn action_space_entry(action_type: ActionType, value: Value) -> Value {
    json!([action_type_name(action_type), value])
}

pub fn action_space_json_value(player_colors: &[Color], map_kind: MapKind) -> Value {
    let catan_map = match map_kind {
        MapKind::Base => CatanMap::base(),
        MapKind::Mini => CatanMap::mini(),
        MapKind::Tournament => {
            CatanMap::from_template(MapKind::Tournament, NumberPlacement::OfficialSpiral)
        }
    };
    let num_nodes = catan_map.land_nodes.len();
    let mut actions = Vec::new();

    actions.push(action_space_entry(ActionType::Roll, Value::Null));
    actions.extend(Resource::ALL.map(|resource| {
        action_space_entry(ActionType::DiscardResource, json!(resource_name(resource)))
    }));
    actions.extend(
        catan_map
            .land_edges()
            .into_iter()
            .map(|edge| action_space_entry(ActionType::BuildRoad, edge_json(edge))),
    );
    actions.extend(
        (0..num_nodes).map(|node| action_space_entry(ActionType::BuildSettlement, json!(node))),
    );
    actions
        .extend((0..num_nodes).map(|node| action_space_entry(ActionType::BuildCity, json!(node))));
    actions.push(action_space_entry(
        ActionType::BuyDevelopmentCard,
        Value::Null,
    ));
    actions.push(action_space_entry(ActionType::PlayKnightCard, Value::Null));
    for i in 0..Resource::ALL.len() {
        for j in i..Resource::ALL.len() {
            actions.push(action_space_entry(
                ActionType::PlayYearOfPlenty,
                json!([
                    resource_name(Resource::ALL[i]),
                    resource_name(Resource::ALL[j])
                ]),
            ));
        }
    }
    actions.extend(Resource::ALL.map(|resource| {
        action_space_entry(
            ActionType::PlayYearOfPlenty,
            json!([resource_name(resource)]),
        )
    }));
    actions.push(action_space_entry(
        ActionType::PlayRoadBuilding,
        Value::Null,
    ));
    actions.extend(Resource::ALL.map(|resource| {
        action_space_entry(ActionType::PlayMonopoly, json!(resource_name(resource)))
    }));

    let mut robber_coordinates = catan_map.land_tiles.keys().copied().collect::<Vec<_>>();
    robber_coordinates.sort_unstable();
    for coordinate in robber_coordinates {
        actions.push(action_space_entry(
            ActionType::MoveRobber,
            json!([coordinate_json(coordinate), Value::Null]),
        ));
        for color in player_colors {
            actions.push(action_space_entry(
                ActionType::MoveRobber,
                json!([coordinate_json(coordinate), color_name(*color)]),
            ));
        }
    }

    for offering in Resource::ALL {
        for asking in Resource::ALL {
            if offering == asking {
                continue;
            }
            actions.push(action_space_entry(
                ActionType::MaritimeTrade,
                json!([
                    resource_name(offering),
                    resource_name(offering),
                    resource_name(offering),
                    resource_name(offering),
                    resource_name(asking)
                ]),
            ));
            actions.push(action_space_entry(
                ActionType::MaritimeTrade,
                json!([
                    resource_name(offering),
                    resource_name(offering),
                    resource_name(offering),
                    Value::Null,
                    resource_name(asking)
                ]),
            ));
            actions.push(action_space_entry(
                ActionType::MaritimeTrade,
                json!([
                    resource_name(offering),
                    resource_name(offering),
                    Value::Null,
                    Value::Null,
                    resource_name(asking)
                ]),
            ));
        }
    }
    actions.push(action_space_entry(ActionType::OfferTrade, Value::Null));
    actions.push(action_space_entry(ActionType::AcceptTrade, Value::Null));
    actions.push(action_space_entry(ActionType::RejectTrade, Value::Null));
    for color in player_colors {
        actions.push(action_space_entry(
            ActionType::ConfirmTrade,
            json!(color_name(*color)),
        ));
    }
    actions.push(action_space_entry(ActionType::CancelTrade, Value::Null));
    actions.push(action_space_entry(ActionType::EndTurn, Value::Null));

    Value::Array(actions)
}

#[derive(Clone, Debug)]
pub struct ActionSpace {
    entries: Vec<Value>,
    index_by_key: HashMap<ActionSpaceKey, usize>,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct ActionSpaceKey {
    action_type: ActionType,
    value: ActionSpaceValueKey,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
enum ActionSpaceValueKey {
    None,
    Node(NodeId),
    Edge(Edge),
    Resource(Resource),
    Resources(Vec<Resource>),
    Robber(Coordinate, Option<Color>),
    MaritimeTrade([Option<Resource>; 4], Resource),
    ConfirmTrade(Color),
}

impl ActionSpace {
    pub fn new(player_colors: &[Color], map_kind: MapKind) -> Self {
        let entries = action_space_json_value(player_colors, map_kind)
            .as_array()
            .cloned()
            .unwrap_or_default();
        let index_by_key = entries
            .iter()
            .enumerate()
            .filter_map(|(index, value)| {
                action_space_key_from_value(value)
                    .ok()
                    .map(|key| (key, index))
            })
            .collect();
        Self {
            entries,
            index_by_key,
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn json_value(&self) -> Value {
        Value::Array(self.entries.clone())
    }

    pub fn index(&self, action: &Action) -> Option<usize> {
        normalized_action_space_key(action).and_then(|key| self.index_by_key.get(&key).copied())
    }
}

fn action_space_key_from_value(value: &Value) -> Result<ActionSpaceKey, String> {
    let values = value
        .as_array()
        .ok_or_else(|| "action-space entry must be a two-element array".to_string())?;
    if values.len() != 2 {
        return Err("action-space entry must be a two-element array".to_string());
    }
    let action_type = parse_action_type(json_str(&values[0], "action-space action type")?)?;
    let raw_value = &values[1];
    let value = match action_type {
        ActionType::Roll
        | ActionType::BuyDevelopmentCard
        | ActionType::PlayKnightCard
        | ActionType::PlayRoadBuilding
        | ActionType::OfferTrade
        | ActionType::AcceptTrade
        | ActionType::RejectTrade
        | ActionType::CancelTrade
        | ActionType::EndTurn => ActionSpaceValueKey::None,
        ActionType::BuildRoad => ActionSpaceValueKey::Edge(parse_edge_value(raw_value)?),
        ActionType::BuildSettlement | ActionType::BuildCity => {
            ActionSpaceValueKey::Node(json_u64(raw_value, "action-space node id")? as usize)
        }
        ActionType::DiscardResource | ActionType::PlayMonopoly => ActionSpaceValueKey::Resource(
            parse_resource(json_str(raw_value, "action-space resource")?)?,
        ),
        ActionType::PlayYearOfPlenty => {
            let resources = raw_value
                .as_array()
                .ok_or_else(|| "action-space Year of Plenty value must be an array".to_string())?
                .iter()
                .map(|value| parse_resource(json_str(value, "action-space resource")?))
                .collect::<Result<Vec<_>, _>>()?;
            ActionSpaceValueKey::Resources(resources)
        }
        ActionType::MoveRobber => {
            let values = raw_value.as_array().ok_or_else(|| {
                "action-space robber value must be [coordinate, victim]".to_string()
            })?;
            if values.len() != 2 {
                return Err("action-space robber value must be [coordinate, victim]".to_string());
            }
            let coordinate = parse_coordinate_value(&values[0])?;
            let victim = if values[1].is_null() {
                None
            } else {
                Some(parse_color(json_str(
                    &values[1],
                    "action-space robber victim",
                )?)?)
            };
            ActionSpaceValueKey::Robber(coordinate, victim)
        }
        ActionType::MaritimeTrade => {
            let values = raw_value.as_array().ok_or_else(|| {
                "action-space maritime trade value must be a five-element array".to_string()
            })?;
            if values.len() != 5 {
                return Err(
                    "action-space maritime trade value must be a five-element array".to_string(),
                );
            }
            let mut offering = [None; 4];
            for (index, value) in values.iter().take(4).enumerate() {
                offering[index] = if value.is_null() {
                    None
                } else {
                    Some(parse_resource(json_str(
                        value,
                        "action-space maritime offered resource",
                    )?)?)
                };
            }
            let asking = parse_resource(json_str(
                &values[4],
                "action-space maritime requested resource",
            )?)?;
            ActionSpaceValueKey::MaritimeTrade(offering, asking)
        }
        ActionType::ConfirmTrade => ActionSpaceValueKey::ConfirmTrade(parse_color(json_str(
            raw_value,
            "action-space confirm trade color",
        )?)?),
    };
    Ok(ActionSpaceKey { action_type, value })
}

fn normalized_action_space_key(action: &Action) -> Option<ActionSpaceKey> {
    let value = match action.action_type {
        ActionType::Roll
        | ActionType::BuyDevelopmentCard
        | ActionType::PlayKnightCard
        | ActionType::PlayRoadBuilding
        | ActionType::OfferTrade
        | ActionType::AcceptTrade
        | ActionType::RejectTrade
        | ActionType::CancelTrade
        | ActionType::EndTurn => ActionSpaceValueKey::None,
        ActionType::BuildRoad => match action.value {
            ActionValue::Edge(edge) => ActionSpaceValueKey::Edge(canonical_edge(edge)),
            _ => return None,
        },
        ActionType::BuildSettlement | ActionType::BuildCity => match action.value {
            ActionValue::Node(node) => ActionSpaceValueKey::Node(node),
            _ => return None,
        },
        ActionType::DiscardResource | ActionType::PlayMonopoly => match action.value {
            ActionValue::Resource(resource) => ActionSpaceValueKey::Resource(resource),
            _ => return None,
        },
        ActionType::PlayYearOfPlenty => match &action.value {
            ActionValue::Resources(resources) => ActionSpaceValueKey::Resources(resources.clone()),
            _ => return None,
        },
        ActionType::MoveRobber => match action.value {
            ActionValue::Robber(coordinate, victim) => {
                ActionSpaceValueKey::Robber(coordinate, victim)
            }
            _ => return None,
        },
        ActionType::MaritimeTrade => match action.value {
            ActionValue::MaritimeTrade(offering, asking) => {
                ActionSpaceValueKey::MaritimeTrade(offering, asking)
            }
            _ => return None,
        },
        ActionType::ConfirmTrade => match action.value {
            ActionValue::ConfirmTrade(_, color) => ActionSpaceValueKey::ConfirmTrade(color),
            _ => return None,
        },
    };
    Some(ActionSpaceKey {
        action_type: action.action_type,
        value,
    })
}

#[cfg(test)]
fn action_space_legacy_string_key(action: &Action) -> Result<String, serde_json::Error> {
    let value = match action.action_type {
        ActionType::Roll
        | ActionType::BuyDevelopmentCard
        | ActionType::PlayKnightCard
        | ActionType::PlayRoadBuilding
        | ActionType::OfferTrade
        | ActionType::AcceptTrade
        | ActionType::RejectTrade
        | ActionType::CancelTrade
        | ActionType::EndTurn => Value::Null,
        ActionType::ConfirmTrade => match action.value {
            ActionValue::ConfirmTrade(_, color) => json!(color_name(color)),
            _ => Value::Null,
        },
        _ => action_value_to_json(&action.value),
    };
    serde_json::to_string(&action_space_entry(action.action_type, value))
}

pub fn action_space_index(
    action: &Action,
    player_colors: &[Color],
    map_kind: MapKind,
) -> Option<usize> {
    ActionSpace::new(player_colors, map_kind).index(action)
}

pub fn legal_action_indices_with_space(
    game: &Game,
    action_space: &ActionSpace,
) -> Result<Vec<usize>, String> {
    game.playable_actions
        .iter()
        .map(|action| {
            action_space
                .index(action)
                .ok_or_else(|| format!("playable action missing from action space: {action:?}"))
        })
        .collect()
}

pub fn fill_legal_action_mask_with_space(
    game: &Game,
    action_space: &ActionSpace,
    out: &mut [u8],
) -> Result<(), String> {
    if out.len() != action_space.len() {
        return Err(format!(
            "legal action mask output length mismatch: got {}, expected {}",
            out.len(),
            action_space.len()
        ));
    }
    out.fill(0);
    fill_legal_action_mask_assume_zeroed_with_space(game, action_space, out)
}

pub fn validate_legal_action_mask_with_space(
    game: &Game,
    action_space: &ActionSpace,
) -> Result<(), String> {
    for action in &game.playable_actions {
        action_space
            .index(action)
            .ok_or_else(|| format!("playable action missing from action space: {action:?}"))?;
    }
    Ok(())
}

pub fn fill_legal_action_mask_assume_zeroed_with_space(
    game: &Game,
    action_space: &ActionSpace,
    out: &mut [u8],
) -> Result<(), String> {
    if out.len() != action_space.len() {
        return Err(format!(
            "legal action mask output length mismatch: got {}, expected {}",
            out.len(),
            action_space.len()
        ));
    }
    for action in &game.playable_actions {
        let index = action_space
            .index(action)
            .ok_or_else(|| format!("playable action missing from action space: {action:?}"))?;
        out[index] = 1;
    }
    Ok(())
}

pub fn legal_action_mask_from_indices(indices: &[usize], action_space_size: usize) -> Vec<u8> {
    let mut mask = vec![0; action_space_size];
    for &index in indices {
        if index < action_space_size {
            mask[index] = 1;
        }
    }
    mask
}

pub fn legal_action_mask_with_space(
    game: &Game,
    action_space: &ActionSpace,
) -> Result<Vec<u8>, String> {
    let mut mask = vec![0; action_space.len()];
    fill_legal_action_mask_with_space(game, action_space, &mut mask)?;
    Ok(mask)
}

pub fn legal_action_mask(
    game: &Game,
    player_colors: &[Color],
    map_kind: MapKind,
) -> Result<Vec<u8>, String> {
    let action_space = ActionSpace::new(player_colors, map_kind);
    legal_action_mask_with_space(game, &action_space)
}

pub fn legal_action_mask_f32_with_space(
    game: &Game,
    action_space: &ActionSpace,
) -> Result<Vec<f32>, String> {
    Ok(legal_action_mask_with_space(game, action_space)?
        .into_iter()
        .map(f32::from)
        .collect())
}

pub fn action_type_index(action_type: ActionType) -> usize {
    match action_type {
        ActionType::Roll => 0,
        ActionType::MoveRobber => 1,
        ActionType::DiscardResource => 2,
        ActionType::BuildRoad => 3,
        ActionType::BuildSettlement => 4,
        ActionType::BuildCity => 5,
        ActionType::BuyDevelopmentCard => 6,
        ActionType::PlayKnightCard => 7,
        ActionType::PlayYearOfPlenty => 8,
        ActionType::PlayMonopoly => 9,
        ActionType::PlayRoadBuilding => 10,
        ActionType::MaritimeTrade => 11,
        ActionType::OfferTrade => 12,
        ActionType::AcceptTrade => 13,
        ActionType::RejectTrade => 14,
        ActionType::ConfirmTrade => 15,
        ActionType::CancelTrade => 16,
        ActionType::EndTurn => 17,
    }
}

fn player_colors_from_perspective(colors: &[Color], p0_color: Color) -> Vec<(usize, Color)> {
    let start = colors
        .iter()
        .position(|color| *color == p0_color)
        .unwrap_or(0);
    (0..colors.len())
        .map(|i| (i, colors[(start + i) % colors.len()]))
        .collect()
}

pub fn starting_resource_bank() -> FreqDeck {
    [19, 19, 19, 19, 19]
}

pub fn starting_devcard_bank() -> Vec<DevCard> {
    let mut deck = Vec::with_capacity(25);
    deck.extend([DevCard::Knight; 14]);
    deck.extend([DevCard::YearOfPlenty; 2]);
    deck.extend([DevCard::RoadBuilding; 2]);
    deck.extend([DevCard::Monopoly; 2]);
    deck.extend([DevCard::VictoryPoint; 5]);
    deck
}

pub fn freqdeck_from_resources(resources: impl IntoIterator<Item = Resource>) -> FreqDeck {
    let mut deck = [0; 5];
    for resource in resources {
        deck[resource.idx()] += 1;
    }
    deck
}

pub fn freqdeck_contains(deck: FreqDeck, wanted: FreqDeck) -> bool {
    deck.iter().zip(wanted).all(|(a, b)| *a >= b)
}

pub fn freqdeck_add(a: FreqDeck, b: FreqDeck) -> FreqDeck {
    [
        a[0] + b[0],
        a[1] + b[1],
        a[2] + b[2],
        a[3] + b[3],
        a[4] + b[4],
    ]
}

pub fn freqdeck_subtract(a: FreqDeck, b: FreqDeck) -> FreqDeck {
    [
        a[0] - b[0],
        a[1] - b[1],
        a[2] - b[2],
        a[3] - b[3],
        a[4] - b[4],
    ]
}

#[derive(Clone, Debug)]
pub struct Player {
    pub color: Color,
    pub kind: PlayerKind,
}

impl Player {
    pub fn simple(color: Color) -> Self {
        Self {
            color,
            kind: PlayerKind::Simple,
        }
    }

    pub fn random(color: Color) -> Self {
        Self {
            color,
            kind: PlayerKind::Random,
        }
    }

    pub fn weighted_random(color: Color) -> Self {
        Self {
            color,
            kind: PlayerKind::WeightedRandom,
        }
    }

    pub fn victory_point(color: Color) -> Self {
        Self {
            color,
            kind: PlayerKind::VictoryPoint,
        }
    }

    pub fn value_function(color: Color) -> Self {
        Self {
            color,
            kind: PlayerKind::ValueFunction { epsilon: None },
        }
    }

    pub fn strategic_value(color: Color) -> Self {
        Self {
            color,
            kind: PlayerKind::StrategicValue { epsilon: None },
        }
    }

    pub fn alpha_beta(color: Color, depth: u8) -> Self {
        Self {
            color,
            kind: PlayerKind::AlphaBeta {
                depth,
                same_turn: false,
                pruning: false,
                epsilon: None,
            },
        }
    }

    pub fn strategic_alpha_beta(color: Color, depth: u8) -> Self {
        Self {
            color,
            kind: PlayerKind::StrategicAlphaBeta {
                depth,
                same_turn: false,
                pruning: true,
                epsilon: None,
            },
        }
    }

    pub fn same_turn_alpha_beta(color: Color, depth: u8) -> Self {
        Self {
            color,
            kind: PlayerKind::AlphaBeta {
                depth,
                same_turn: true,
                pruning: false,
                epsilon: None,
            },
        }
    }

    pub fn playout(color: Color, playouts_per_action: u16) -> Self {
        Self {
            color,
            kind: PlayerKind::Playout {
                playouts_per_action,
            },
        }
    }

    pub fn strategic_playout(color: Color, playouts_per_action: u16) -> Self {
        Self {
            color,
            kind: PlayerKind::StrategicPlayout {
                playouts_per_action,
            },
        }
    }

    pub fn ensemble(color: Color) -> Self {
        Self {
            color,
            kind: PlayerKind::Champion,
        }
    }

    pub fn champion(color: Color) -> Self {
        Self {
            color,
            kind: PlayerKind::Champion,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlayerKind {
    Simple,
    Random,
    WeightedRandom,
    VictoryPoint,
    ValueFunction {
        epsilon: Option<u8>,
    },
    StrategicValue {
        epsilon: Option<u8>,
    },
    AlphaBeta {
        depth: u8,
        same_turn: bool,
        pruning: bool,
        epsilon: Option<u8>,
    },
    StrategicAlphaBeta {
        depth: u8,
        same_turn: bool,
        pruning: bool,
        epsilon: Option<u8>,
    },
    Playout {
        playouts_per_action: u16,
    },
    StrategicPlayout {
        playouts_per_action: u16,
    },
    Champion,
}

#[derive(Clone, Debug)]
pub struct PlayerState {
    pub victory_points: i16,
    pub actual_victory_points: i16,
    pub roads_available: u8,
    pub settlements_available: u8,
    pub cities_available: u8,
    pub has_road: bool,
    pub has_army: bool,
    pub has_rolled: bool,
    pub has_played_development_card_in_turn: bool,
    pub resources: FreqDeck,
    pub dev_cards: [u8; 5],
    pub played_dev_cards: [u8; 5],
    pub owned_at_start: [bool; 5],
    pub longest_road_length: usize,
}

impl Default for PlayerState {
    fn default() -> Self {
        Self {
            victory_points: 0,
            actual_victory_points: 0,
            roads_available: 15,
            settlements_available: 5,
            cities_available: 4,
            has_road: false,
            has_army: false,
            has_rolled: false,
            has_played_development_card_in_turn: false,
            resources: [0; 5],
            dev_cards: [0; 5],
            played_dev_cards: [0; 5],
            owned_at_start: [false; 5],
            longest_road_length: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ActionPrompt {
    BuildInitialSettlement,
    BuildInitialRoad,
    PlayTurn,
    Discard,
    MoveRobber,
    DecideTrade,
    DecideAcceptees,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub enum ActionType {
    Roll,
    MoveRobber,
    DiscardResource,
    BuildRoad,
    BuildSettlement,
    BuildCity,
    BuyDevelopmentCard,
    PlayKnightCard,
    PlayYearOfPlenty,
    PlayMonopoly,
    PlayRoadBuilding,
    MaritimeTrade,
    OfferTrade,
    AcceptTrade,
    RejectTrade,
    ConfirmTrade,
    CancelTrade,
    EndTurn,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum ActionValue {
    None,
    Node(NodeId),
    Edge(Edge),
    Resource(Resource),
    Resources(Vec<Resource>),
    Robber(Coordinate, Option<Color>),
    MaritimeTrade([Option<Resource>; 4], Resource),
    Trade([u8; 10]),
    ConfirmTrade([u8; 10], Color),
    Dice(u8, u8),
    DevCard(DevCard),
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Action {
    pub color: Color,
    pub action_type: ActionType,
    pub value: ActionValue,
}

impl Action {
    pub fn new(color: Color, action_type: ActionType, value: ActionValue) -> Self {
        Self {
            color,
            action_type,
            value,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionRecord {
    pub action: Action,
    pub result: ActionValue,
}

#[derive(Clone, Debug)]
pub struct State {
    pub players: Vec<Player>,
    pub colors: Vec<Color>,
    pub board: Board,
    pub player_state: Vec<PlayerState>,
    pub color_to_index: HashMap<Color, usize>,
    pub resource_freqdeck: FreqDeck,
    pub development_listdeck: Vec<DevCard>,
    pub buildings_by_color: HashMap<Color, HashMap<BuildingType, Vec<usize>>>,
    pub action_records: Vec<ActionRecord>,
    /// Legal-action width aligned with `action_records`, but only when that
    /// width is derivable from public state. Zero means "unknown/private" and
    /// must never be used to suppress an event. In particular, regular
    /// PLAY_TURN and DISCARD widths depend on hidden cards and stay zero.
    pub action_public_legal_counts: Vec<u16>,
    pub num_turns: usize,
    pub current_player_index: usize,
    pub current_turn_index: usize,
    pub current_prompt: ActionPrompt,
    pub is_initial_build_phase: bool,
    pub is_discarding: bool,
    pub discard_counts: Vec<u8>,
    pub is_moving_knight: bool,
    pub is_road_building: bool,
    pub free_roads_available: u8,
    pub is_resolving_trade: bool,
    pub current_trade: [u8; 11],
    pub acceptees: Vec<bool>,
    pub discard_limit: u8,
    pub friendly_robber: bool,
}

impl State {
    pub fn new(
        players: Vec<Player>,
        seed: Option<u64>,
        discard_limit: u8,
        friendly_robber: bool,
        board: Board,
    ) -> Self {
        let mut rng: SmallRng = match seed {
            Some(seed) => SeedableRng::seed_from_u64(seed),
            None => SeedableRng::from_entropy(),
        };
        let mut players = players;
        players.shuffle(&mut rng);
        let colors: Vec<_> = players.iter().map(|p| p.color).collect();
        let color_to_index = colors.iter().enumerate().map(|(i, c)| (*c, i)).collect();
        let mut development_listdeck = starting_devcard_bank();
        development_listdeck.shuffle(&mut rng);
        let buildings_by_color = colors
            .iter()
            .map(|&color| {
                let mut by_type = HashMap::new();
                by_type.insert(BuildingType::Settlement, Vec::new());
                by_type.insert(BuildingType::City, Vec::new());
                by_type.insert(BuildingType::Road, Vec::new());
                (color, by_type)
            })
            .collect();
        let num_players = players.len();
        Self {
            players,
            colors,
            board,
            player_state: vec![PlayerState::default(); num_players],
            color_to_index,
            resource_freqdeck: starting_resource_bank(),
            development_listdeck,
            buildings_by_color,
            action_records: Vec::new(),
            action_public_legal_counts: Vec::new(),
            num_turns: 0,
            current_player_index: 0,
            current_turn_index: 0,
            current_prompt: ActionPrompt::BuildInitialSettlement,
            is_initial_build_phase: true,
            is_discarding: false,
            discard_counts: vec![0; num_players],
            is_moving_knight: false,
            is_road_building: false,
            free_roads_available: 0,
            is_resolving_trade: false,
            current_trade: [0; 11],
            acceptees: vec![false; num_players],
            discard_limit,
            friendly_robber,
        }
    }

    pub fn current_color(&self) -> Color {
        self.colors[self.current_player_index]
    }

    pub fn player_index(&self, color: Color) -> usize {
        self.color_to_index[&color]
    }

    pub fn player_state(&self, color: Color) -> &PlayerState {
        &self.player_state[self.player_index(color)]
    }

    pub fn player_state_mut(&mut self, color: Color) -> &mut PlayerState {
        let index = self.player_index(color);
        &mut self.player_state[index]
    }
}

pub fn generate_playable_actions(state: &State) -> Vec<Action> {
    let color = state.current_color();
    match state.current_prompt {
        ActionPrompt::BuildInitialSettlement => settlement_possibilities(state, color, true),
        ActionPrompt::BuildInitialRoad => initial_road_possibilities(state, color),
        ActionPrompt::MoveRobber => {
            let mut actions = robber_possibilities(state, color);
            sort_actions(&mut actions);
            actions
        }
        ActionPrompt::Discard => discard_possibilities(state, color),
        ActionPrompt::PlayTurn => {
            if state.is_road_building {
                road_building_possibilities(state, color, false)
            } else {
                let mut actions = Vec::new();
                if !state.player_state(color).has_rolled {
                    actions.push(Action::new(color, ActionType::Roll, ActionValue::None));
                } else {
                    actions.extend(road_building_possibilities(state, color, true));
                    actions.extend(settlement_possibilities(state, color, false));
                    actions.extend(city_possibilities(state, color));
                    if player_can_afford_dev_card(state, color)
                        && !state.development_listdeck.is_empty()
                    {
                        actions.push(Action::new(
                            color,
                            ActionType::BuyDevelopmentCard,
                            ActionValue::None,
                        ));
                    }
                }
                if player_can_play_dev(state, color, DevCard::Knight) {
                    actions.push(Action::new(
                        color,
                        ActionType::PlayKnightCard,
                        ActionValue::None,
                    ));
                }
                if player_can_play_dev(state, color, DevCard::YearOfPlenty) {
                    actions.extend(year_of_plenty_possibilities(color, state.resource_freqdeck));
                }
                if player_can_play_dev(state, color, DevCard::Monopoly) {
                    actions.extend(Resource::ALL.map(|r| {
                        Action::new(color, ActionType::PlayMonopoly, ActionValue::Resource(r))
                    }));
                }
                if can_play_road_building_card(state, color) {
                    actions.push(Action::new(
                        color,
                        ActionType::PlayRoadBuilding,
                        ActionValue::None,
                    ));
                }
                if state.player_state(color).has_rolled {
                    actions.extend(maritime_trade_possibilities(state, color));
                    actions.push(Action::new(color, ActionType::EndTurn, ActionValue::None));
                }
                actions
            }
        }
        ActionPrompt::DecideTrade => {
            let mut actions = Vec::new();
            let asked = freqdeck_from_trade_slice(&state.current_trade[5..10]);
            if freqdeck_contains(state.player_state(color).resources, asked) {
                actions.push(Action::new(
                    color,
                    ActionType::AcceptTrade,
                    ActionValue::Trade(current_trade_10(state)),
                ));
            }
            actions.push(Action::new(
                color,
                ActionType::RejectTrade,
                ActionValue::Trade(current_trade_10(state)),
            ));
            actions
        }
        ActionPrompt::DecideAcceptees => {
            let mut actions = Vec::new();
            for confirm_color in Color::ALL {
                if state
                    .colors
                    .iter()
                    .zip(&state.acceptees)
                    .any(|(&other, &accepted)| other == confirm_color && accepted)
                {
                    actions.push(Action::new(
                        color,
                        ActionType::ConfirmTrade,
                        ActionValue::ConfirmTrade(current_trade_10(state), confirm_color),
                    ));
                }
            }
            actions.push(Action::new(
                color,
                ActionType::CancelTrade,
                ActionValue::None,
            ));
            actions
        }
    }
}

fn random_sorted_playable_action<R: Rng + ?Sized>(state: &State, rng: &mut R) -> Option<Action> {
    let total = count_sorted_playable_actions(state);
    if total == 0 {
        return None;
    }
    let mut index = rng.gen_range(0..total);
    nth_sorted_playable_action(state, &mut index)
}

fn count_sorted_playable_actions(state: &State) -> usize {
    let color = state.current_color();
    match state.current_prompt {
        ActionPrompt::BuildInitialSettlement => state.board.buildable_node_ids(color, true).len(),
        ActionPrompt::BuildInitialRoad => initial_road_possibilities(state, color).len(),
        ActionPrompt::MoveRobber => robber_possibilities(state, color).len(),
        ActionPrompt::Discard => Resource::ALL
            .into_iter()
            .filter(|resource| state.player_state(color).resources[resource.idx()] > 0)
            .count(),
        ActionPrompt::PlayTurn => count_sorted_play_turn_actions(state, color),
        ActionPrompt::DecideTrade => {
            let asked = freqdeck_from_trade_slice(&state.current_trade[5..10]);
            1 + usize::from(freqdeck_contains(
                state.player_state(color).resources,
                asked,
            ))
        }
        ActionPrompt::DecideAcceptees => {
            1 + state.acceptees.iter().filter(|accepted| **accepted).count()
        }
    }
}

fn count_sorted_play_turn_actions(state: &State, color: Color) -> usize {
    if state.is_road_building {
        return road_building_possibilities(state, color, false).len();
    }

    let mut count = 0;
    if !state.player_state(color).has_rolled {
        count += 1;
    } else {
        count += road_building_possibilities(state, color, true).len();
        count += settlement_possibilities(state, color, false).len();
        count += city_possibilities(state, color).len();
        count += usize::from(
            player_can_afford_dev_card(state, color) && !state.development_listdeck.is_empty(),
        );
    }

    count += usize::from(player_can_play_dev(state, color, DevCard::Knight));
    if player_can_play_dev(state, color, DevCard::YearOfPlenty) {
        count += count_year_of_plenty_possibilities(state.resource_freqdeck);
    }
    if player_can_play_dev(state, color, DevCard::Monopoly) {
        count += Resource::ALL.len();
    }
    count += usize::from(can_play_road_building_card(state, color));

    if state.player_state(color).has_rolled {
        count += count_maritime_trade_possibilities(
            state.player_state(color).resources,
            state.resource_freqdeck,
            &state.board.get_player_port_resources(color),
        );
        count += 1;
    }

    count
}

fn nth_sorted_playable_action(state: &State, index: &mut usize) -> Option<Action> {
    let color = state.current_color();
    match state.current_prompt {
        ActionPrompt::BuildInitialSettlement => {
            nth_from_actions(index, settlement_possibilities(state, color, true))
        }
        ActionPrompt::BuildInitialRoad => {
            nth_from_actions(index, initial_road_possibilities(state, color))
        }
        ActionPrompt::MoveRobber => {
            let mut actions = robber_possibilities(state, color);
            sort_actions(&mut actions);
            nth_from_actions(index, actions)
        }
        ActionPrompt::Discard => {
            for resource in Resource::ALL {
                if state.player_state(color).resources[resource.idx()] > 0 && take_index(index) {
                    return Some(Action::new(
                        color,
                        ActionType::DiscardResource,
                        ActionValue::Resource(resource),
                    ));
                }
            }
            None
        }
        ActionPrompt::PlayTurn => nth_sorted_play_turn_action(state, color, index),
        ActionPrompt::DecideTrade => {
            let asked = freqdeck_from_trade_slice(&state.current_trade[5..10]);
            if freqdeck_contains(state.player_state(color).resources, asked) && take_index(index) {
                return Some(Action::new(
                    color,
                    ActionType::AcceptTrade,
                    ActionValue::Trade(current_trade_10(state)),
                ));
            }
            take_index(index).then(|| {
                Action::new(
                    color,
                    ActionType::RejectTrade,
                    ActionValue::Trade(current_trade_10(state)),
                )
            })
        }
        ActionPrompt::DecideAcceptees => {
            for confirm_color in Color::ALL {
                if state
                    .colors
                    .iter()
                    .zip(&state.acceptees)
                    .any(|(&other, &accepted)| other == confirm_color && accepted)
                    && take_index(index)
                {
                    return Some(Action::new(
                        color,
                        ActionType::ConfirmTrade,
                        ActionValue::ConfirmTrade(current_trade_10(state), confirm_color),
                    ));
                }
            }
            take_index(index)
                .then(|| Action::new(color, ActionType::CancelTrade, ActionValue::None))
        }
    }
}

fn nth_sorted_play_turn_action(state: &State, color: Color, index: &mut usize) -> Option<Action> {
    if state.is_road_building {
        return nth_from_actions(index, road_building_possibilities(state, color, false));
    }

    if !state.player_state(color).has_rolled {
        if take_index(index) {
            return Some(Action::new(color, ActionType::Roll, ActionValue::None));
        }
    } else {
        if let Some(action) =
            nth_from_actions(index, road_building_possibilities(state, color, true))
        {
            return Some(action);
        }
        if let Some(action) = nth_from_actions(index, settlement_possibilities(state, color, false))
        {
            return Some(action);
        }
        if let Some(action) = nth_from_actions(index, city_possibilities(state, color)) {
            return Some(action);
        }
        if player_can_afford_dev_card(state, color)
            && !state.development_listdeck.is_empty()
            && take_index(index)
        {
            return Some(Action::new(
                color,
                ActionType::BuyDevelopmentCard,
                ActionValue::None,
            ));
        }
    }

    if player_can_play_dev(state, color, DevCard::Knight) && take_index(index) {
        return Some(Action::new(
            color,
            ActionType::PlayKnightCard,
            ActionValue::None,
        ));
    }
    if player_can_play_dev(state, color, DevCard::YearOfPlenty)
        && let Some(action) = nth_from_actions(
            index,
            year_of_plenty_possibilities(color, state.resource_freqdeck),
        )
    {
        return Some(action);
    }
    if player_can_play_dev(state, color, DevCard::Monopoly) {
        for resource in Resource::ALL {
            if take_index(index) {
                return Some(Action::new(
                    color,
                    ActionType::PlayMonopoly,
                    ActionValue::Resource(resource),
                ));
            }
        }
    }
    if can_play_road_building_card(state, color) && take_index(index) {
        return Some(Action::new(
            color,
            ActionType::PlayRoadBuilding,
            ActionValue::None,
        ));
    }
    if state.player_state(color).has_rolled {
        if let Some(action) = nth_from_actions(index, maritime_trade_possibilities(state, color)) {
            return Some(action);
        }
        if take_index(index) {
            return Some(Action::new(color, ActionType::EndTurn, ActionValue::None));
        }
    }
    None
}

fn nth_from_actions(index: &mut usize, actions: Vec<Action>) -> Option<Action> {
    if *index < actions.len() {
        Some(actions[*index].clone())
    } else {
        *index -= actions.len();
        None
    }
}

fn take_index(index: &mut usize) -> bool {
    if *index == 0 {
        true
    } else {
        *index -= 1;
        false
    }
}

fn sort_actions(actions: &mut [Action]) {
    actions.sort_by(action_cmp);
}

fn action_cmp(a: &Action, b: &Action) -> std::cmp::Ordering {
    color_order(a.color)
        .cmp(&color_order(b.color))
        .then_with(|| action_type_order(a.action_type).cmp(&action_type_order(b.action_type)))
        .then_with(|| action_value_cmp(&a.value, &b.value))
}

fn color_order(color: Color) -> u8 {
    match color {
        Color::Red => 0,
        Color::Blue => 1,
        Color::Orange => 2,
        Color::White => 3,
    }
}

fn action_type_order(action_type: ActionType) -> u8 {
    match action_type {
        ActionType::Roll => 0,
        ActionType::MoveRobber => 1,
        ActionType::DiscardResource => 2,
        ActionType::BuildRoad => 3,
        ActionType::BuildSettlement => 4,
        ActionType::BuildCity => 5,
        ActionType::BuyDevelopmentCard => 6,
        ActionType::PlayKnightCard => 7,
        ActionType::PlayYearOfPlenty => 8,
        ActionType::PlayMonopoly => 9,
        ActionType::PlayRoadBuilding => 10,
        ActionType::MaritimeTrade => 11,
        ActionType::OfferTrade => 12,
        ActionType::AcceptTrade => 13,
        ActionType::RejectTrade => 14,
        ActionType::ConfirmTrade => 15,
        ActionType::CancelTrade => 16,
        ActionType::EndTurn => 17,
    }
}

fn resource_order(resource: Resource) -> u8 {
    resource.idx() as u8
}

fn dev_card_order(card: DevCard) -> u8 {
    card.idx() as u8
}

fn coordinate_order(coordinate: Coordinate) -> (i8, i8, i8) {
    (coordinate.0, coordinate.1, coordinate.2)
}

fn action_value_rank(value: &ActionValue) -> u8 {
    match value {
        ActionValue::None => 0,
        ActionValue::Node(_) => 1,
        ActionValue::Edge(_) => 2,
        ActionValue::Resource(_) => 3,
        ActionValue::Resources(_) => 4,
        ActionValue::Robber(_, _) => 5,
        ActionValue::MaritimeTrade(_, _) => 6,
        ActionValue::Trade(_) => 7,
        ActionValue::ConfirmTrade(_, _) => 8,
        ActionValue::Dice(_, _) => 9,
        ActionValue::DevCard(_) => 10,
    }
}

fn option_resource_order(resource: Option<Resource>) -> i16 {
    resource.map(resource_order).map_or(-1, i16::from)
}

fn option_color_order(color: Option<Color>) -> i16 {
    color.map(color_order).map_or(-1, i16::from)
}

fn action_value_cmp(a: &ActionValue, b: &ActionValue) -> Ordering {
    let rank_cmp = action_value_rank(a).cmp(&action_value_rank(b));
    if rank_cmp != Ordering::Equal {
        return rank_cmp;
    }
    match (a, b) {
        (ActionValue::None, ActionValue::None) => Ordering::Equal,
        (ActionValue::Node(a), ActionValue::Node(b)) => a.cmp(b),
        (ActionValue::Edge(a), ActionValue::Edge(b)) => a.cmp(b),
        (ActionValue::Resource(a), ActionValue::Resource(b)) => {
            resource_order(*a).cmp(&resource_order(*b))
        }
        (ActionValue::Resources(a), ActionValue::Resources(b)) => a
            .iter()
            .map(|resource| resource_order(*resource))
            .cmp(b.iter().map(|resource| resource_order(*resource))),
        (
            ActionValue::Robber(a_coordinate, a_victim),
            ActionValue::Robber(b_coordinate, b_victim),
        ) => coordinate_order(*a_coordinate)
            .cmp(&coordinate_order(*b_coordinate))
            .then_with(|| option_color_order(*a_victim).cmp(&option_color_order(*b_victim))),
        (
            ActionValue::MaritimeTrade(a_offering, a_asking),
            ActionValue::MaritimeTrade(b_offering, b_asking),
        ) => a_offering
            .iter()
            .copied()
            .map(option_resource_order)
            .cmp(b_offering.iter().copied().map(option_resource_order))
            .then_with(|| resource_order(*a_asking).cmp(&resource_order(*b_asking))),
        (ActionValue::Trade(a), ActionValue::Trade(b)) => a.cmp(b),
        (
            ActionValue::ConfirmTrade(a_trade, a_color),
            ActionValue::ConfirmTrade(b_trade, b_color),
        ) => a_trade
            .cmp(b_trade)
            .then_with(|| color_order(*a_color).cmp(&color_order(*b_color))),
        (ActionValue::Dice(a1, a2), ActionValue::Dice(b1, b2)) => (a1, a2).cmp(&(b1, b2)),
        (ActionValue::DevCard(a), ActionValue::DevCard(b)) => {
            dev_card_order(*a).cmp(&dev_card_order(*b))
        }
        _ => Ordering::Equal,
    }
}

fn current_trade_10(state: &State) -> [u8; 10] {
    let mut out = [0; 10];
    out.copy_from_slice(&state.current_trade[..10]);
    out
}

pub fn road_building_possibilities(state: &State, color: Color, check_money: bool) -> Vec<Action> {
    if state.player_state(color).roads_available == 0 {
        return Vec::new();
    }
    if check_money && !freqdeck_contains(state.player_state(color).resources, ROAD_COST) {
        return Vec::new();
    }
    state
        .board
        .buildable_edges(color)
        .into_iter()
        .map(|edge| Action::new(color, ActionType::BuildRoad, ActionValue::Edge(edge)))
        .collect()
}

fn can_play_road_building_card(state: &State, color: Color) -> bool {
    player_can_play_dev(state, color, DevCard::RoadBuilding)
        && state.player_state(color).roads_available > 0
        && !state.board.buildable_edges(color).is_empty()
}

pub fn settlement_possibilities(state: &State, color: Color, initial: bool) -> Vec<Action> {
    if !initial {
        let ps = state.player_state(color);
        if !freqdeck_contains(ps.resources, SETTLEMENT_COST) || ps.settlements_available == 0 {
            return Vec::new();
        }
    }
    state
        .board
        .buildable_node_ids(color, initial)
        .into_iter()
        .map(|node| Action::new(color, ActionType::BuildSettlement, ActionValue::Node(node)))
        .collect()
}

pub fn initial_road_possibilities(state: &State, color: Color) -> Vec<Action> {
    let last_settlement = state.buildings_by_color[&color][&BuildingType::Settlement]
        .last()
        .copied();
    let Some(last_settlement) = last_settlement else {
        return Vec::new();
    };
    state
        .board
        .buildable_edges(color)
        .into_iter()
        .filter(|edge| edge.0 == last_settlement || edge.1 == last_settlement)
        .map(|edge| Action::new(color, ActionType::BuildRoad, ActionValue::Edge(edge)))
        .collect()
}

pub fn city_possibilities(state: &State, color: Color) -> Vec<Action> {
    let ps = state.player_state(color);
    if !freqdeck_contains(ps.resources, CITY_COST) || ps.cities_available == 0 {
        return Vec::new();
    }
    let mut nodes = state.buildings_by_color[&color][&BuildingType::Settlement].clone();
    nodes.sort_unstable();
    nodes
        .into_iter()
        .map(|node| Action::new(color, ActionType::BuildCity, ActionValue::Node(node)))
        .collect()
}

pub fn discard_possibilities(state: &State, color: Color) -> Vec<Action> {
    if state.discard_counts[state.player_index(color)] == 0 {
        return Vec::new();
    }
    Resource::ALL
        .into_iter()
        .filter(|r| state.player_state(color).resources[r.idx()] > 0)
        .map(|r| Action::new(color, ActionType::DiscardResource, ActionValue::Resource(r)))
        .collect()
}

pub fn robber_possibilities(state: &State, color: Color) -> Vec<Action> {
    let mut actions = Vec::new();
    let mut resource_counts = [0u8; 4];
    for &player_color in &state.colors {
        resource_counts[color_order(player_color) as usize] =
            player_num_resource_cards(state, player_color, None);
    }
    for (&coordinate, tile) in &state.board.map.land_tiles {
        if coordinate == state.board.robber_coordinate {
            continue;
        }
        let mut seen = [false; 4];
        let mut to_steal = Vec::new();
        for &node in &tile.nodes {
            if let Some((candidate, _)) = state.board.buildings.get(&node)
                && *candidate != color
            {
                let index = color_order(*candidate) as usize;
                if resource_counts[index] > 0 && !seen[index] {
                    seen[index] = true;
                    to_steal.push(*candidate);
                }
            }
        }
        if to_steal.is_empty() {
            actions.push(Action::new(
                color,
                ActionType::MoveRobber,
                ActionValue::Robber(coordinate, None),
            ));
        } else {
            for enemy in to_steal {
                actions.push(Action::new(
                    color,
                    ActionType::MoveRobber,
                    ActionValue::Robber(coordinate, Some(enemy)),
                ));
            }
        }
    }
    if !state.friendly_robber {
        return actions;
    }
    let filtered: Vec<_> = actions
        .iter()
        .filter(|action| !robber_action_blocks_low_vp_enemy(state, color, action))
        .cloned()
        .collect();
    if filtered.is_empty() {
        actions
    } else {
        filtered
    }
}

fn robber_action_blocks_low_vp_enemy(state: &State, color: Color, action: &Action) -> bool {
    let ActionValue::Robber(coordinate, _) = action.value else {
        return false;
    };
    let Some(tile) = state.board.map.land_tiles.get(&coordinate) else {
        return false;
    };
    for &node in &tile.nodes {
        let Some((candidate, _)) = state.board.buildings.get(&node) else {
            continue;
        };
        if *candidate != color && state.player_state(*candidate).actual_victory_points < 3 {
            return true;
        }
    }
    false
}

pub fn year_of_plenty_possibilities(color: Color, bank: FreqDeck) -> Vec<Action> {
    let mut actions = Vec::new();
    let mut single_seen = [false; 5];
    for i in 0..Resource::ALL.len() {
        for j in i..Resource::ALL.len() {
            let first = Resource::ALL[i];
            let second = Resource::ALL[j];
            let wanted = freqdeck_from_resources([first, second]);
            if freqdeck_contains(bank, wanted) {
                actions.push(Action::new(
                    color,
                    ActionType::PlayYearOfPlenty,
                    ActionValue::Resources(vec![first, second]),
                ));
            } else {
                if bank[first.idx()] >= 1 {
                    push_year_of_plenty_single(&mut actions, &mut single_seen, color, first);
                }
                if bank[second.idx()] >= 1 {
                    push_year_of_plenty_single(&mut actions, &mut single_seen, color, second);
                }
            }
        }
    }
    sort_actions(&mut actions);
    actions
}

fn count_year_of_plenty_possibilities(bank: FreqDeck) -> usize {
    let mut count = 0;
    let mut single_seen = [false; 5];
    for i in 0..Resource::ALL.len() {
        for j in i..Resource::ALL.len() {
            let first = Resource::ALL[i];
            let second = Resource::ALL[j];
            let wanted = freqdeck_from_resources([first, second]);
            if freqdeck_contains(bank, wanted) {
                count += 1;
            } else {
                if bank[first.idx()] >= 1 && !single_seen[first.idx()] {
                    single_seen[first.idx()] = true;
                    count += 1;
                }
                if bank[second.idx()] >= 1 && !single_seen[second.idx()] {
                    single_seen[second.idx()] = true;
                    count += 1;
                }
            }
        }
    }
    count
}

fn push_year_of_plenty_single(
    actions: &mut Vec<Action>,
    single_seen: &mut [bool; 5],
    color: Color,
    resource: Resource,
) {
    let index = resource.idx();
    if single_seen[index] {
        return;
    }
    single_seen[index] = true;
    actions.push(Action::new(
        color,
        ActionType::PlayYearOfPlenty,
        ActionValue::Resources(vec![resource]),
    ));
}

pub fn maritime_trade_possibilities(state: &State, color: Color) -> Vec<Action> {
    inner_maritime_trade_possibilities(
        state.player_state(color).resources,
        state.resource_freqdeck,
        &state.board.get_player_port_resources(color),
    )
    .into_iter()
    .map(|(offer, ask)| {
        Action::new(
            color,
            ActionType::MaritimeTrade,
            ActionValue::MaritimeTrade(offer, ask),
        )
    })
    .collect()
}

pub fn inner_maritime_trade_possibilities(
    hand: FreqDeck,
    bank: FreqDeck,
    port_resources: &HashSet<Option<Resource>>,
) -> Vec<([Option<Resource>; 4], Resource)> {
    let mut rates = [4u8; 5];
    if port_resources.contains(&None) {
        rates = [3; 5];
    }
    for resource in port_resources.iter().flatten() {
        rates[resource.idx()] = 2;
    }
    let mut offers = Vec::new();
    for resource in Resource::ALL {
        let rate = rates[resource.idx()];
        if hand[resource.idx()] < rate {
            continue;
        }
        let mut offer = [None; 4];
        for slot in offer.iter_mut().take(rate as usize) {
            *slot = Some(resource);
        }
        for ask in Resource::ALL {
            if ask != resource && bank[ask.idx()] > 0 {
                offers.push((offer, ask));
            }
        }
    }
    offers.sort_by_key(|(offer, ask)| (*offer, *ask));
    offers
}

fn count_maritime_trade_possibilities(
    hand: FreqDeck,
    bank: FreqDeck,
    port_resources: &HashSet<Option<Resource>>,
) -> usize {
    let mut rates = [4u8; 5];
    if port_resources.contains(&None) {
        rates = [3; 5];
    }
    for resource in port_resources.iter().flatten() {
        rates[resource.idx()] = 2;
    }

    let mut count = 0;
    for resource in Resource::ALL {
        if hand[resource.idx()] < rates[resource.idx()] {
            continue;
        }
        count += Resource::ALL
            .into_iter()
            .filter(|ask| *ask != resource && bank[ask.idx()] > 0)
            .count();
    }
    count
}

pub fn is_valid_trade(value: [u8; 10]) -> bool {
    let offering = &value[..5];
    let asking = &value[5..];
    if offering.iter().sum::<u8>() == 0 || asking.iter().sum::<u8>() == 0 {
        return false;
    }
    offering.iter().zip(asking).all(|(a, b)| *a == 0 || *b == 0)
}

pub fn apply_action(
    state: &mut State,
    action: Action,
    replay_result: Option<ActionValue>,
    rng: &mut impl Rng,
    record_action: bool,
) -> Result<ActionRecord, String> {
    apply_action_inner(state, action, replay_result, rng, record_action, false)
}

fn apply_action_known_valid(
    state: &mut State,
    action: Action,
    replay_result: Option<ActionValue>,
    rng: &mut impl Rng,
    record_action: bool,
) -> Result<ActionRecord, String> {
    apply_action_inner(state, action, replay_result, rng, record_action, true)
}

fn apply_action_inner(
    state: &mut State,
    action: Action,
    replay_result: Option<ActionValue>,
    rng: &mut impl Rng,
    record_action: bool,
    known_valid: bool,
) -> Result<ActionRecord, String> {
    let record = match action.action_type {
        ActionType::EndTurn => apply_end_turn(state, action),
        ActionType::BuildSettlement => apply_build_settlement(state, action, known_valid),
        ActionType::BuildRoad => apply_build_road(state, action, known_valid),
        ActionType::BuildCity => apply_build_city(state, action),
        ActionType::BuyDevelopmentCard => apply_buy_development_card(state, action, replay_result),
        ActionType::Roll => apply_roll(state, action, replay_result, rng),
        ActionType::DiscardResource => apply_discard(state, action),
        ActionType::MoveRobber => apply_move_robber(state, action, replay_result, rng),
        ActionType::PlayKnightCard => apply_play_knight_card(state, action),
        ActionType::PlayYearOfPlenty => apply_play_year_of_plenty(state, action),
        ActionType::PlayMonopoly => apply_play_monopoly(state, action),
        ActionType::PlayRoadBuilding => apply_play_road_building(state, action),
        ActionType::MaritimeTrade => apply_maritime_trade(state, action),
        ActionType::OfferTrade => apply_offer_trade(state, action),
        ActionType::AcceptTrade => apply_accept_trade(state, action),
        ActionType::RejectTrade => apply_reject_trade(state, action),
        ActionType::ConfirmTrade => apply_confirm_trade(state, action),
        ActionType::CancelTrade => apply_cancel_trade(state, action),
    }?;
    if record_action {
        // Keep the optional public-width sidecar exactly aligned even when a
        // caller previously injected records without width metadata. The Game
        // execution boundary fills the newest zero only when the pre-action
        // width is public; direct low-level callers remain safely unknown.
        state
            .action_public_legal_counts
            .truncate(state.action_records.len());
        state
            .action_public_legal_counts
            .resize(state.action_records.len(), 0);
        state.action_records.push(record.clone());
        state.action_public_legal_counts.push(0);
    }
    Ok(record)
}

fn apply_end_turn(state: &mut State, action: Action) -> Result<ActionRecord, String> {
    player_clean_turn(state, action.color);
    advance_turn(state, 1);
    state.current_prompt = ActionPrompt::PlayTurn;
    Ok(ActionRecord {
        action,
        result: ActionValue::None,
    })
}

fn apply_build_settlement(
    state: &mut State,
    action: Action,
    known_valid: bool,
) -> Result<ActionRecord, String> {
    let ActionValue::Node(node) = action.value else {
        return Err("settlement action requires node".into());
    };
    if state.is_initial_build_phase {
        if known_valid {
            state
                .board
                .build_settlement_known_valid(action.color, node, true);
        } else {
            state.board.build_settlement(action.color, node, true)?;
        }
        build_settlement_state(state, action.color, node, true);
        let buildings = &state.buildings_by_color[&action.color][&BuildingType::Settlement];
        if buildings.len() == 2 {
            let adjacent = state
                .board
                .map
                .adjacent_tiles
                .get(&node)
                .cloned()
                .unwrap_or_default();
            for tile in adjacent {
                if let Some(resource) = tile.resource {
                    state.resource_freqdeck[resource.idx()] -= 1;
                    state.player_state_mut(action.color).resources[resource.idx()] += 1;
                }
            }
        }
        state.current_prompt = ActionPrompt::BuildInitialRoad;
    } else {
        let (previous, road_color, lengths) = if known_valid {
            state
                .board
                .build_settlement_known_valid(action.color, node, false)
        } else {
            state.board.build_settlement(action.color, node, false)?
        };
        build_settlement_state(state, action.color, node, false);
        state.resource_freqdeck = freqdeck_add(state.resource_freqdeck, SETTLEMENT_COST);
        maintain_longest_road(state, previous, road_color, &lengths);
    }
    Ok(ActionRecord {
        action,
        result: ActionValue::None,
    })
}

fn apply_build_road(
    state: &mut State,
    action: Action,
    known_valid: bool,
) -> Result<ActionRecord, String> {
    let ActionValue::Edge(edge) = action.value else {
        return Err("road action requires edge".into());
    };
    if state.is_initial_build_phase {
        if known_valid {
            state.board.build_road_known_valid(action.color, edge);
        } else {
            state.board.build_road(action.color, edge)?;
        }
        build_road_state(state, action.color, canonical_edge(edge), true);
        let num_buildings: usize = state
            .colors
            .iter()
            .map(|c| state.buildings_by_color[c][&BuildingType::Settlement].len())
            .sum();
        let num_players = state.colors.len();
        if num_buildings < num_players {
            advance_turn(state, 1);
            state.current_prompt = ActionPrompt::BuildInitialSettlement;
        } else if num_buildings == num_players {
            state.current_prompt = ActionPrompt::BuildInitialSettlement;
        } else if num_buildings == 2 * num_players {
            state.is_initial_build_phase = false;
            state.current_prompt = ActionPrompt::PlayTurn;
        } else {
            advance_turn(state, -1);
            state.current_prompt = ActionPrompt::BuildInitialSettlement;
        }
    } else if state.is_road_building && state.free_roads_available > 0 {
        let (previous, road_color, lengths) = if known_valid {
            state.board.build_road_known_valid(action.color, edge)
        } else {
            state.board.build_road(action.color, edge)?
        };
        build_road_state(state, action.color, canonical_edge(edge), true);
        maintain_longest_road(state, previous, road_color, &lengths);
        state.free_roads_available -= 1;
        if state.free_roads_available == 0
            || road_building_possibilities(state, action.color, false).is_empty()
        {
            state.is_road_building = false;
            state.free_roads_available = 0;
        }
    } else {
        let (previous, road_color, lengths) = if known_valid {
            state.board.build_road_known_valid(action.color, edge)
        } else {
            state.board.build_road(action.color, edge)?
        };
        build_road_state(state, action.color, canonical_edge(edge), false);
        maintain_longest_road(state, previous, road_color, &lengths);
    }
    Ok(ActionRecord {
        action,
        result: ActionValue::None,
    })
}

fn apply_build_city(state: &mut State, action: Action) -> Result<ActionRecord, String> {
    let ActionValue::Node(node) = action.value else {
        return Err("city action requires node".into());
    };
    state.board.build_city(action.color, node)?;
    build_city_state(state, action.color, node);
    state.resource_freqdeck = freqdeck_add(state.resource_freqdeck, CITY_COST);
    Ok(ActionRecord {
        action,
        result: ActionValue::None,
    })
}

fn apply_buy_development_card(
    state: &mut State,
    action: Action,
    replay_result: Option<ActionValue>,
) -> Result<ActionRecord, String> {
    if state.development_listdeck.is_empty() {
        return Err("no more development cards".into());
    }
    if !player_can_afford_dev_card(state, action.color) {
        return Err("not enough resources for development card".into());
    }
    let card = match replay_result {
        Some(ActionValue::DevCard(card)) => {
            let pos = state
                .development_listdeck
                .iter()
                .position(|c| *c == card)
                .ok_or("replay dev card not in deck")?;
            state.development_listdeck.remove(pos)
        }
        _ => state.development_listdeck.pop().unwrap(),
    };
    buy_dev_card(state, action.color, card)?;
    state.resource_freqdeck = freqdeck_add(state.resource_freqdeck, DEVELOPMENT_CARD_COST);
    let action = Action::new(action.color, action.action_type, ActionValue::DevCard(card));
    Ok(ActionRecord {
        action,
        result: ActionValue::DevCard(card),
    })
}

fn apply_roll(
    state: &mut State,
    action: Action,
    replay_result: Option<ActionValue>,
    rng: &mut impl Rng,
) -> Result<ActionRecord, String> {
    state.player_state_mut(action.color).has_rolled = true;
    let (d1, d2) = match replay_result {
        Some(ActionValue::Dice(a, b)) => (a, b),
        _ => (rng.gen_range(1..=6), rng.gen_range(1..=6)),
    };
    let number = d1 + d2;
    let action = Action::new(action.color, action.action_type, ActionValue::Dice(d1, d2));
    if number == 7 {
        let mut first = None;
        for (i, color) in state.colors.iter().copied().enumerate() {
            let cards = player_num_resource_cards(state, color, None);
            let discard = if cards > state.discard_limit {
                cards / 2
            } else {
                0
            };
            state.discard_counts[i] = discard;
            if discard > 0 && first.is_none() {
                first = Some(i);
            }
        }
        if let Some(index) = first {
            state.current_player_index = index;
            state.current_prompt = ActionPrompt::Discard;
            state.is_discarding = true;
        } else {
            state.discard_counts.fill(0);
            state.current_prompt = ActionPrompt::MoveRobber;
            state.is_moving_knight = true;
        }
    } else {
        let (payout, _) = yield_resources(&state.board, state.resource_freqdeck, number);
        for (color, deck) in payout {
            player_freqdeck_add(state, color, deck);
            state.resource_freqdeck = freqdeck_subtract(state.resource_freqdeck, deck);
        }
        state.current_prompt = ActionPrompt::PlayTurn;
    }
    Ok(ActionRecord {
        action,
        result: ActionValue::Dice(d1, d2),
    })
}

fn apply_discard(state: &mut State, action: Action) -> Result<ActionRecord, String> {
    let ActionValue::Resource(resource) = action.value else {
        return Err("discard requires resource".into());
    };
    let index = state.player_index(action.color);
    if state.discard_counts[index] == 0 {
        return Err("player is not required to discard".into());
    }
    player_deck_draw(state, action.color, resource, 1)?;
    state.resource_freqdeck[resource.idx()] += 1;
    state.discard_counts[index] -= 1;
    if state.discard_counts[index] == 0 {
        if let Some(next) = ((state.current_player_index + 1)..state.colors.len())
            .find(|&i| state.discard_counts[i] > 0)
        {
            state.current_player_index = next;
        } else {
            state.current_player_index = state.current_turn_index;
            state.current_prompt = ActionPrompt::MoveRobber;
            state.is_discarding = false;
            state.is_moving_knight = true;
            state.discard_counts.fill(0);
        }
    }
    Ok(ActionRecord {
        action,
        result: ActionValue::Resource(resource),
    })
}

fn apply_move_robber(
    state: &mut State,
    action: Action,
    replay_result: Option<ActionValue>,
    rng: &mut impl Rng,
) -> Result<ActionRecord, String> {
    let ActionValue::Robber(coordinate, robbed_color) = action.value else {
        return Err("move robber requires robber value".into());
    };
    let mut robbed_resource = None;
    if let Some(enemy) = robbed_color {
        let resource = match replay_result {
            Some(ActionValue::Resource(resource)) => resource,
            _ => {
                player_deck_random_select(state, enemy, rng).ok_or("enemy has no cards to steal")?
            }
        };
        player_deck_draw(state, enemy, resource, 1)?;
        player_deck_replenish(state, action.color, resource, 1);
        robbed_resource = Some(resource);
    }
    state.board.robber_coordinate = coordinate;
    state.current_prompt = ActionPrompt::PlayTurn;
    state.is_moving_knight = false;
    Ok(ActionRecord {
        action,
        result: robbed_resource
            .map(ActionValue::Resource)
            .unwrap_or(ActionValue::None),
    })
}

fn apply_play_knight_card(state: &mut State, action: Action) -> Result<ActionRecord, String> {
    if !player_can_play_dev(state, action.color, DevCard::Knight) {
        return Err("player cannot play knight".into());
    }
    play_dev_card(state, action.color, DevCard::Knight)?;
    state.current_prompt = ActionPrompt::MoveRobber;
    state.is_moving_knight = true;
    Ok(ActionRecord {
        action,
        result: ActionValue::None,
    })
}

fn apply_play_year_of_plenty(state: &mut State, action: Action) -> Result<ActionRecord, String> {
    let ActionValue::Resources(cards) = action.value.clone() else {
        return Err("year of plenty requires resources".into());
    };
    if !player_can_play_dev(state, action.color, DevCard::YearOfPlenty) {
        return Err("player cannot play year of plenty".into());
    }
    let selected = freqdeck_from_resources(cards);
    if !freqdeck_contains(state.resource_freqdeck, selected) {
        return Err("not enough bank resources".into());
    }
    player_freqdeck_add(state, action.color, selected);
    state.resource_freqdeck = freqdeck_subtract(state.resource_freqdeck, selected);
    play_dev_card(state, action.color, DevCard::YearOfPlenty)?;
    state.current_prompt = ActionPrompt::PlayTurn;
    Ok(ActionRecord {
        action,
        result: ActionValue::None,
    })
}

fn apply_play_monopoly(state: &mut State, action: Action) -> Result<ActionRecord, String> {
    let ActionValue::Resource(resource) = action.value else {
        return Err("monopoly requires resource".into());
    };
    if !player_can_play_dev(state, action.color, DevCard::Monopoly) {
        return Err("player cannot play monopoly".into());
    }
    let mut stolen = [0; 5];
    for color in state.colors.clone() {
        if color != action.color {
            let amount = state.player_state(color).resources[resource.idx()];
            if amount > 0 {
                stolen[resource.idx()] += amount;
                player_deck_draw(state, color, resource, amount)?;
            }
        }
    }
    player_freqdeck_add(state, action.color, stolen);
    play_dev_card(state, action.color, DevCard::Monopoly)?;
    state.current_prompt = ActionPrompt::PlayTurn;
    Ok(ActionRecord {
        action,
        result: ActionValue::None,
    })
}

fn apply_play_road_building(state: &mut State, action: Action) -> Result<ActionRecord, String> {
    if !player_can_play_dev(state, action.color, DevCard::RoadBuilding) {
        return Err("player cannot play road building".into());
    }
    play_dev_card(state, action.color, DevCard::RoadBuilding)?;
    state.is_road_building = true;
    state.free_roads_available = 2;
    state.current_prompt = ActionPrompt::PlayTurn;
    Ok(ActionRecord {
        action,
        result: ActionValue::None,
    })
}

fn apply_maritime_trade(state: &mut State, action: Action) -> Result<ActionRecord, String> {
    let ActionValue::MaritimeTrade(offering, asking) = action.value else {
        return Err("maritime trade requires value".into());
    };
    let offered_resources = offering.into_iter().flatten();
    let offer_deck = freqdeck_from_resources(offered_resources);
    let ask_deck = freqdeck_from_resources([asking]);
    if !freqdeck_contains(state.player_state(action.color).resources, offer_deck) {
        return Err("not enough resources for trade".into());
    }
    if !freqdeck_contains(state.resource_freqdeck, ask_deck) {
        return Err("bank lacks requested resource".into());
    }
    player_freqdeck_subtract(state, action.color, offer_deck)?;
    state.resource_freqdeck = freqdeck_add(state.resource_freqdeck, offer_deck);
    player_freqdeck_add(state, action.color, ask_deck);
    state.resource_freqdeck = freqdeck_subtract(state.resource_freqdeck, ask_deck);
    Ok(ActionRecord {
        action,
        result: ActionValue::None,
    })
}

fn apply_offer_trade(state: &mut State, action: Action) -> Result<ActionRecord, String> {
    let ActionValue::Trade(trade) = action.value else {
        return Err("offer trade requires trade".into());
    };
    if !is_valid_trade(trade) {
        return Err("invalid trade".into());
    }
    let offering = freqdeck_from_trade_slice(&trade[..5]);
    if !freqdeck_contains(state.player_state(action.color).resources, offering) {
        return Err("not enough resources for trade offer".into());
    }
    state.is_resolving_trade = true;
    state.current_trade[..10].copy_from_slice(&trade);
    state.current_trade[10] = state.current_turn_index as u8;
    state.current_player_index = state
        .colors
        .iter()
        .position(|c| *c != action.color)
        .ok_or("no other player")?;
    state.current_prompt = ActionPrompt::DecideTrade;
    Ok(ActionRecord {
        action,
        result: ActionValue::None,
    })
}

fn apply_accept_trade(state: &mut State, action: Action) -> Result<ActionRecord, String> {
    let index = state.player_index(action.color);
    state.acceptees[index] = true;
    if let Some(next) = ((state.current_player_index + 1)..state.colors.len())
        .find(|&i| state.colors[i] != action.color)
    {
        state.current_player_index = next;
    } else {
        state.current_player_index = state.current_turn_index;
        state.current_prompt = ActionPrompt::DecideAcceptees;
    }
    Ok(ActionRecord {
        action,
        result: ActionValue::None,
    })
}

fn apply_reject_trade(state: &mut State, action: Action) -> Result<ActionRecord, String> {
    if let Some(next) = ((state.current_player_index + 1)..state.colors.len())
        .find(|&i| state.colors[i] != action.color)
    {
        state.current_player_index = next;
    } else if state.acceptees.iter().any(|x| *x) {
        state.current_player_index = state.current_turn_index;
        state.current_prompt = ActionPrompt::DecideAcceptees;
    } else {
        reset_trading_state(state);
        state.current_player_index = state.current_turn_index;
        state.current_prompt = ActionPrompt::PlayTurn;
    }
    Ok(ActionRecord {
        action,
        result: ActionValue::None,
    })
}

fn apply_confirm_trade(state: &mut State, action: Action) -> Result<ActionRecord, String> {
    let ActionValue::ConfirmTrade(trade, enemy) = action.value else {
        return Err("confirm trade requires value".into());
    };
    if trade != current_trade_10(state) {
        return Err("confirm trade does not match current trade".into());
    }
    if !state.acceptees[state.player_index(enemy)] {
        return Err("confirm trade target has not accepted".into());
    }
    let offering = freqdeck_from_trade_slice(&trade[..5]);
    let asking = freqdeck_from_trade_slice(&trade[5..]);
    player_freqdeck_subtract(state, action.color, offering)?;
    player_freqdeck_add(state, action.color, asking);
    player_freqdeck_subtract(state, enemy, asking)?;
    player_freqdeck_add(state, enemy, offering);
    reset_trading_state(state);
    state.current_player_index = state.current_turn_index;
    state.current_prompt = ActionPrompt::PlayTurn;
    Ok(ActionRecord {
        action,
        result: ActionValue::None,
    })
}

fn apply_cancel_trade(state: &mut State, action: Action) -> Result<ActionRecord, String> {
    reset_trading_state(state);
    state.current_player_index = state.current_turn_index;
    state.current_prompt = ActionPrompt::PlayTurn;
    Ok(ActionRecord {
        action,
        result: ActionValue::None,
    })
}

pub fn yield_resources(
    board: &Board,
    bank: FreqDeck,
    number: u8,
) -> (HashMap<Color, FreqDeck>, Vec<Resource>) {
    let mut intended: HashMap<Color, FreqDeck> = HashMap::new();
    let mut totals = [0; 5];
    for (&coordinate, tile) in &board.map.land_tiles {
        if tile.number != Some(number) || board.robber_coordinate == coordinate {
            continue;
        }
        let Some(resource) = tile.resource else {
            continue;
        };
        for node in tile.nodes {
            match board.buildings.get(&node) {
                Some(&(color, BuildingType::Settlement)) => {
                    intended.entry(color).or_insert([0; 5])[resource.idx()] += 1;
                    totals[resource.idx()] += 1;
                }
                Some(&(color, BuildingType::City)) => {
                    intended.entry(color).or_insert([0; 5])[resource.idx()] += 2;
                    totals[resource.idx()] += 2;
                }
                _ => {}
            }
        }
    }
    let depleted: Vec<_> = Resource::ALL
        .into_iter()
        .filter(|r| bank[r.idx()] < totals[r.idx()])
        .collect();
    let mut payout = HashMap::new();
    for (color, deck) in intended {
        let mut final_deck = [0; 5];
        for resource in Resource::ALL {
            if !depleted.contains(&resource) {
                final_deck[resource.idx()] = deck[resource.idx()];
            }
        }
        payout.insert(color, final_deck);
    }
    (payout, depleted)
}

fn maintain_longest_road(
    state: &mut State,
    previous: Option<Color>,
    road_color: Option<Color>,
    lengths: &HashMap<Color, usize>,
) {
    for (&color, &length) in lengths {
        state.player_state_mut(color).longest_road_length = length;
    }
    if previous == road_color {
        return;
    }
    if let Some(winner) = road_color {
        let winner_state = state.player_state_mut(winner);
        winner_state.has_road = true;
        winner_state.victory_points += 2;
        winner_state.actual_victory_points += 2;
    }
    if let Some(loser) = previous {
        let loser_state = state.player_state_mut(loser);
        loser_state.has_road = false;
        loser_state.victory_points -= 2;
        loser_state.actual_victory_points -= 2;
    }
}

fn maintain_largest_army(
    state: &mut State,
    color: Color,
    previous_color: Option<Color>,
    previous_size: u8,
) {
    let candidate = state.player_state(color).played_dev_cards[DevCard::Knight.idx()];
    if candidate < 3 {
        return;
    }
    if let Some(previous_color) = previous_color {
        if previous_size < candidate && previous_color != color {
            let ps = state.player_state_mut(color);
            ps.has_army = true;
            ps.victory_points += 2;
            ps.actual_victory_points += 2;
            let ls = state.player_state_mut(previous_color);
            ls.has_army = false;
            ls.victory_points -= 2;
            ls.actual_victory_points -= 2;
        }
    } else {
        let ps = state.player_state_mut(color);
        ps.has_army = true;
        ps.victory_points += 2;
        ps.actual_victory_points += 2;
    }
}

fn largest_army(state: &State) -> (Option<Color>, u8) {
    for color in state.colors.iter().copied() {
        let ps = state.player_state(color);
        if ps.has_army {
            return (Some(color), ps.played_dev_cards[DevCard::Knight.idx()]);
        }
    }
    (None, 0)
}

fn build_settlement_state(state: &mut State, color: Color, node: NodeId, is_free: bool) {
    state
        .buildings_by_color
        .get_mut(&color)
        .unwrap()
        .get_mut(&BuildingType::Settlement)
        .unwrap()
        .push(node);
    let ps = state.player_state_mut(color);
    ps.settlements_available -= 1;
    ps.victory_points += 1;
    ps.actual_victory_points += 1;
    if !is_free {
        ps.resources[Resource::Wood.idx()] -= 1;
        ps.resources[Resource::Brick.idx()] -= 1;
        ps.resources[Resource::Sheep.idx()] -= 1;
        ps.resources[Resource::Wheat.idx()] -= 1;
    }
}

fn build_road_state(state: &mut State, color: Color, edge: Edge, is_free: bool) {
    state
        .buildings_by_color
        .get_mut(&color)
        .unwrap()
        .get_mut(&BuildingType::Road)
        .unwrap()
        .push(edge.0);
    let ps = state.player_state_mut(color);
    ps.roads_available -= 1;
    if !is_free {
        ps.resources[Resource::Wood.idx()] -= 1;
        ps.resources[Resource::Brick.idx()] -= 1;
        state.resource_freqdeck = freqdeck_add(state.resource_freqdeck, ROAD_COST);
    }
}

fn build_city_state(state: &mut State, color: Color, node: NodeId) {
    let by_color = state.buildings_by_color.get_mut(&color).unwrap();
    let settlements = by_color.get_mut(&BuildingType::Settlement).unwrap();
    if let Some(pos) = settlements.iter().position(|n| *n == node) {
        settlements.remove(pos);
    }
    by_color.get_mut(&BuildingType::City).unwrap().push(node);
    let ps = state.player_state_mut(color);
    ps.settlements_available += 1;
    ps.cities_available -= 1;
    ps.victory_points += 1;
    ps.actual_victory_points += 1;
    ps.resources[Resource::Wheat.idx()] -= 2;
    ps.resources[Resource::Ore.idx()] -= 3;
}

pub fn player_can_afford_dev_card(state: &State, color: Color) -> bool {
    freqdeck_contains(state.player_state(color).resources, DEVELOPMENT_CARD_COST)
}

pub fn player_can_play_dev(state: &State, color: Color, card: DevCard) -> bool {
    let ps = state.player_state(color);
    !ps.has_played_development_card_in_turn
        && ps.dev_cards[card.idx()] >= 1
        && ps.owned_at_start[card.idx()]
}

pub fn player_freqdeck_add(state: &mut State, color: Color, deck: FreqDeck) {
    let ps = state.player_state_mut(color);
    for (i, value) in deck.into_iter().enumerate() {
        ps.resources[i] += value;
    }
}

pub fn player_freqdeck_subtract(
    state: &mut State,
    color: Color,
    deck: FreqDeck,
) -> Result<(), String> {
    if !freqdeck_contains(state.player_state(color).resources, deck) {
        return Err("player resource deck underflow".into());
    }
    let ps = state.player_state_mut(color);
    for (i, value) in deck.into_iter().enumerate() {
        ps.resources[i] -= value;
    }
    Ok(())
}

pub fn buy_dev_card(state: &mut State, color: Color, card: DevCard) -> Result<(), String> {
    if !player_can_afford_dev_card(state, color) {
        return Err("not enough resources".into());
    }
    let ps = state.player_state_mut(color);
    ps.dev_cards[card.idx()] += 1;
    if card == DevCard::VictoryPoint {
        ps.actual_victory_points += 1;
    }
    ps.resources[Resource::Sheep.idx()] -= 1;
    ps.resources[Resource::Wheat.idx()] -= 1;
    ps.resources[Resource::Ore.idx()] -= 1;
    Ok(())
}

pub fn player_num_resource_cards(state: &State, color: Color, card: Option<Resource>) -> u8 {
    let resources = state.player_state(color).resources;
    match card {
        Some(resource) => resources[resource.idx()],
        None => resources.iter().sum(),
    }
}

pub fn player_deck_draw(
    state: &mut State,
    color: Color,
    resource: Resource,
    amount: u8,
) -> Result<(), String> {
    let ps = state.player_state_mut(color);
    if ps.resources[resource.idx()] < amount {
        return Err("not enough cards to draw".into());
    }
    ps.resources[resource.idx()] -= amount;
    Ok(())
}

pub fn player_deck_replenish(state: &mut State, color: Color, resource: Resource, amount: u8) {
    state.player_state_mut(color).resources[resource.idx()] += amount;
}

fn player_deck_random_select(state: &State, color: Color, rng: &mut impl Rng) -> Option<Resource> {
    let resources = state.player_state(color).resources;
    let total: u8 = resources.iter().sum();
    if total == 0 {
        return None;
    }
    let mut ticket = rng.gen_range(0..total);
    for resource in Resource::ALL {
        let count = resources[resource.idx()];
        if ticket < count {
            return Some(resource);
        }
        ticket -= count;
    }
    None
}

pub fn play_dev_card(state: &mut State, color: Color, card: DevCard) -> Result<(), String> {
    let (previous_army_color, previous_army_size) = if card == DevCard::Knight {
        largest_army(state)
    } else {
        (None, 0)
    };
    let ps = state.player_state_mut(color);
    if ps.dev_cards[card.idx()] == 0 {
        return Err("no dev card in hand".into());
    }
    ps.dev_cards[card.idx()] -= 1;
    ps.has_played_development_card_in_turn = true;
    ps.played_dev_cards[card.idx()] += 1;
    if card == DevCard::Knight {
        maintain_largest_army(state, color, previous_army_color, previous_army_size);
    }
    Ok(())
}

pub fn player_clean_turn(state: &mut State, color: Color) {
    let ps = state.player_state_mut(color);
    ps.has_played_development_card_in_turn = false;
    ps.has_rolled = false;
    for card in DevCard::ALL {
        if card != DevCard::VictoryPoint {
            ps.owned_at_start[card.idx()] = ps.dev_cards[card.idx()] > 0;
        }
    }
}

fn advance_turn(state: &mut State, direction: isize) {
    let len = state.colors.len() as isize;
    let next = (state.current_player_index as isize + direction).rem_euclid(len) as usize;
    state.current_player_index = next;
    state.current_turn_index = next;
    state.num_turns += 1;
}

fn reset_trading_state(state: &mut State) {
    state.is_resolving_trade = false;
    state.current_trade = [0; 11];
    state.acceptees.fill(false);
}

fn freqdeck_from_trade_slice(slice: &[u8]) -> FreqDeck {
    [slice[0], slice[1], slice[2], slice[3], slice[4]]
}

const WEIGHTED_RANDOM_CITY_WEIGHT: usize = 10_000;
const WEIGHTED_RANDOM_SETTLEMENT_WEIGHT: usize = 1_000;
const WEIGHTED_RANDOM_DEV_CARD_WEIGHT: usize = 100;
const TRANSLATE_VARIETY: f64 = 4.0;

#[derive(Clone, Copy, Debug)]
pub struct ValueWeights {
    pub public_vps: f64,
    pub production: f64,
    pub enemy_production: f64,
    pub num_tiles: f64,
    pub buildable_nodes: f64,
    pub longest_road: f64,
    pub hand_synergy: f64,
    pub hand_resources: f64,
    pub discard_penalty: f64,
    pub hand_devs: f64,
    pub army_size: f64,
}

impl Default for ValueWeights {
    fn default() -> Self {
        Self {
            public_vps: 3e14,
            production: 1e8,
            enemy_production: -1e8,
            num_tiles: 1.0,
            buildable_nodes: 1e3,
            longest_road: 10.0,
            hand_synergy: 1e2,
            hand_resources: 1.0,
            discard_penalty: -5.0,
            hand_devs: 10.0,
            army_size: 10.1,
        }
    }
}

pub fn weighted_random_action<R: Rng + ?Sized>(actions: &[Action], rng: &mut R) -> Option<Action> {
    let total_weight: usize = actions
        .iter()
        .map(|action| match action.action_type {
            ActionType::BuildCity => WEIGHTED_RANDOM_CITY_WEIGHT,
            ActionType::BuildSettlement => WEIGHTED_RANDOM_SETTLEMENT_WEIGHT,
            ActionType::BuyDevelopmentCard => WEIGHTED_RANDOM_DEV_CARD_WEIGHT,
            _ => 1,
        })
        .sum();
    if total_weight == 0 {
        return None;
    }
    let mut ticket = rng.gen_range(0..total_weight);
    for action in actions {
        let weight = match action.action_type {
            ActionType::BuildCity => WEIGHTED_RANDOM_CITY_WEIGHT,
            ActionType::BuildSettlement => WEIGHTED_RANDOM_SETTLEMENT_WEIGHT,
            ActionType::BuyDevelopmentCard => WEIGHTED_RANDOM_DEV_CARD_WEIGHT,
            _ => 1,
        };
        if ticket < weight {
            return Some(action.clone());
        }
        ticket -= weight;
    }
    actions.last().cloned()
}

pub fn victory_point_action<R: Rng + ?Sized>(
    game: &Game,
    color: Color,
    actions: &[Action],
    rng: &mut R,
) -> Option<Action> {
    let mut best_value = i16::MIN;
    let mut best_actions = Vec::new();
    for action in actions {
        let mut game_copy = game.clone();
        if game_copy.execute(action.clone(), false, None).is_err() {
            continue;
        }
        let value = game_copy.state.player_state(color).actual_victory_points;
        match value.cmp(&best_value) {
            std::cmp::Ordering::Greater => {
                best_value = value;
                best_actions.clear();
                best_actions.push(action.clone());
            }
            std::cmp::Ordering::Equal => best_actions.push(action.clone()),
            std::cmp::Ordering::Less => {}
        }
    }
    best_actions.choose(rng).cloned()
}

pub fn value_production_from_features(
    sample: &FeatureMap,
    player_name: &str,
    include_variety: bool,
) -> f64 {
    let proba_point = 2.778 / 100.0;
    let keys = [
        format!("EFFECTIVE_{player_name}_WHEAT_PRODUCTION"),
        format!("EFFECTIVE_{player_name}_ORE_PRODUCTION"),
        format!("EFFECTIVE_{player_name}_SHEEP_PRODUCTION"),
        format!("EFFECTIVE_{player_name}_WOOD_PRODUCTION"),
        format!("EFFECTIVE_{player_name}_BRICK_PRODUCTION"),
    ];
    let prod_sum: f64 = keys
        .iter()
        .map(|key| sample.get(key).copied().unwrap_or(0.0))
        .sum();
    let prod_variety = keys
        .iter()
        .filter(|key| sample.get(*key).copied().unwrap_or(0.0) != 0.0)
        .count() as f64
        * TRANSLATE_VARIETY
        * proba_point;
    prod_sum + if include_variety { prod_variety } else { 0.0 }
}

pub fn base_value(game: &Game, p0_color: Color, weights: ValueWeights) -> f64 {
    let production_sample = production_features(game, p0_color, true);
    let production = value_production_from_features(&production_sample, "P0", true);
    let enemy_production = value_production_from_features(&production_sample, "P1", false);
    let ps = game.state.player_state(p0_color);
    let hand = ps.resources;
    let distance_to_city = (2u8.saturating_sub(hand[Resource::Wheat.idx()]) as f64
        + 3u8.saturating_sub(hand[Resource::Ore.idx()]) as f64)
        / 5.0;
    let distance_to_settlement = (1u8.saturating_sub(hand[Resource::Wheat.idx()]) as f64
        + 1u8.saturating_sub(hand[Resource::Sheep.idx()]) as f64
        + 1u8.saturating_sub(hand[Resource::Brick.idx()]) as f64
        + 1u8.saturating_sub(hand[Resource::Wood.idx()]) as f64)
        / 4.0;
    let hand_synergy = (2.0 - distance_to_city - distance_to_settlement) / 2.0;
    let num_in_hand = player_num_resource_cards(&game.state, p0_color, None) as f64;
    let discard_penalty = if num_in_hand > 7.0 {
        weights.discard_penalty
    } else {
        0.0
    };

    let owned_nodes: Vec<_> = game
        .state
        .board
        .buildings
        .iter()
        .filter_map(|(&node, &(owner, building))| {
            (owner == p0_color && matches!(building, BuildingType::Settlement | BuildingType::City))
                .then_some(node)
        })
        .collect();
    let mut owned_tiles = HashSet::new();
    for node in owned_nodes {
        if let Some(tiles) = game.state.board.map.adjacent_tiles.get(&node) {
            owned_tiles.extend(tiles.iter().map(|tile| tile.id));
        }
    }

    let num_buildable_nodes = game.state.board.buildable_node_ids(p0_color, false).len() as f64;
    let longest_road_factor = if num_buildable_nodes == 0.0 {
        weights.longest_road
    } else {
        0.1
    };
    let hand_devs: u8 = ps.dev_cards.iter().sum();

    ps.victory_points as f64 * weights.public_vps
        + production * weights.production
        + enemy_production * weights.enemy_production
        + hand_synergy * weights.hand_synergy
        + num_buildable_nodes * weights.buildable_nodes
        + owned_tiles.len() as f64 * weights.num_tiles
        + num_in_hand * weights.hand_resources
        + discard_penalty
        + ps.longest_road_length as f64 * longest_road_factor
        + hand_devs as f64 * weights.hand_devs
        + ps.played_dev_cards[DevCard::Knight.idx()] as f64 * weights.army_size
}

fn production_profile(state: &State, color: Color, robber_aware: bool) -> [f64; 5] {
    let mut out = [0.0; 5];
    for (&coordinate, tile) in &state.board.map.land_tiles {
        if robber_aware && coordinate == state.board.robber_coordinate {
            continue;
        }
        let (Some(resource), Some(number)) = (tile.resource, tile.number) else {
            continue;
        };
        let probability = number_probability(number);
        for node in tile.nodes {
            let Some(&(owner, building)) = state.board.buildings.get(&node) else {
                continue;
            };
            if owner != color {
                continue;
            }
            let multiplier = match building {
                BuildingType::Settlement => 1.0,
                BuildingType::City => 2.0,
                BuildingType::Road => 0.0,
            };
            out[resource.idx()] += probability * multiplier;
        }
    }
    out
}

fn production_sum(profile: [f64; 5]) -> f64 {
    profile.into_iter().sum()
}

fn production_variety(profile: [f64; 5]) -> f64 {
    profile
        .into_iter()
        .filter(|production| *production > 0.0)
        .count() as f64
}

fn weighted_resource_production(profile: [f64; 5]) -> f64 {
    profile[Resource::Wood.idx()] * 0.95
        + profile[Resource::Brick.idx()] * 0.95
        + profile[Resource::Sheep.idx()] * 0.9
        + profile[Resource::Wheat.idx()] * 1.45
        + profile[Resource::Ore.idx()] * 1.55
}

fn cost_readiness(hand: FreqDeck, cost: FreqDeck) -> f64 {
    let needed: u8 = cost.iter().sum();
    if needed == 0 {
        return 1.0;
    }
    let missing: u8 = Resource::ALL
        .into_iter()
        .map(|resource| cost[resource.idx()].saturating_sub(hand[resource.idx()]))
        .sum();
    1.0 - f64::from(missing) / f64::from(needed)
}

fn port_synergy(state: &State, color: Color, profile: [f64; 5]) -> f64 {
    let ports = state.board.get_player_port_resources(color);
    let mut value = 0.0;
    if ports.contains(&None) {
        value += production_sum(profile) * 0.8;
    }
    for resource in ports.iter().flatten().copied() {
        value += profile[resource.idx()] * 2.2;
    }
    value
}

fn max_opponent_metric(game: &Game, color: Color, metric: impl Fn(Color) -> f64) -> f64 {
    game.state
        .colors
        .iter()
        .copied()
        .filter(|candidate| *candidate != color)
        .map(metric)
        .fold(0.0, f64::max)
}

pub fn strategic_value(game: &Game, color: Color) -> f64 {
    if game.winning_color() == Some(color) {
        return 1.0e18;
    }
    if game.winning_color().is_some() {
        return -1.0e18;
    }

    let ps = game.state.player_state(color);
    let profile = production_profile(&game.state, color, true);
    let raw_profile = production_profile(&game.state, color, false);
    let production = production_sum(profile);
    let weighted_production = weighted_resource_production(profile);
    let variety = production_variety(raw_profile);
    let hand_cards = player_num_resource_cards(&game.state, color, None) as f64;
    let hand = ps.resources;
    let city_ready = cost_readiness(hand, CITY_COST);
    let settlement_ready = cost_readiness(hand, SETTLEMENT_COST);
    let dev_ready = cost_readiness(hand, DEVELOPMENT_CARD_COST);
    let city_engine = profile[Resource::Ore.idx()] * 2.6
        + profile[Resource::Wheat.idx()] * 2.2
        + profile[Resource::Sheep.idx()] * 0.8;
    let expansion_engine = profile[Resource::Wood.idx()] * 1.3
        + profile[Resource::Brick.idx()] * 1.3
        + profile[Resource::Wheat.idx()] * 0.7
        + profile[Resource::Sheep.idx()] * 0.7;
    let hand_devs: u8 = ps.dev_cards.iter().sum();
    let playable_knights = ps.dev_cards[DevCard::Knight.idx()];
    let hidden_vps = ps.dev_cards[DevCard::VictoryPoint.idx()];
    let army_pressure = f64::from(ps.played_dev_cards[DevCard::Knight.idx()])
        + f64::from(playable_knights) * 0.65
        + if ps.has_army { 1.5 } else { 0.0 };
    let road_pressure = ps.longest_road_length as f64 + if ps.has_road { 1.5 } else { 0.0 };
    let buildable_nodes = game.state.board.buildable_node_ids(color, false).len() as f64;
    let port_value = port_synergy(&game.state, color, raw_profile);
    let discard_risk = if hand_cards > f64::from(game.state.discard_limit) {
        (hand_cards - f64::from(game.state.discard_limit)) * 4.5
    } else {
        0.0
    };

    let opponent_score = max_opponent_metric(game, color, |opponent| {
        let ops = game.state.player_state(opponent);
        f64::from(ops.actual_victory_points) * 1000.0
            + weighted_resource_production(production_profile(&game.state, opponent, true)) * 160.0
            + f64::from(ops.dev_cards[DevCard::VictoryPoint.idx()]) * 650.0
            + f64::from(ops.played_dev_cards[DevCard::Knight.idx()]) * 80.0
            + ops.longest_road_length as f64 * 25.0
    });

    f64::from(ps.actual_victory_points) * 120_000.0
        + f64::from(ps.victory_points) * 35_000.0
        + f64::from(hidden_vps) * 95_000.0
        + weighted_production * 24_000.0
        + production * 9_000.0
        + variety * 3_500.0
        + city_engine * 18_000.0
        + expansion_engine * 8_500.0
        + city_ready * 11_000.0
        + settlement_ready * 7_500.0
        + dev_ready * 6_000.0
        + buildable_nodes.min(8.0) * 1_600.0
        + port_value * 2_200.0
        + road_pressure * 900.0
        + army_pressure * 2_200.0
        + f64::from(hand_devs) * 2_600.0
        + hand_cards.min(7.0) * 350.0
        - discard_risk
        - opponent_score
}

pub fn champion_value(game: &Game, color: Color) -> f64 {
    let base = base_value(game, color, ValueWeights::default());
    if game.winning_color() == Some(color) {
        return base + 1.0e18;
    }
    if game.winning_color().is_some() {
        return base - 1.0e18;
    }

    let ps = game.state.player_state(color);
    let hidden_vps = ps.dev_cards[DevCard::VictoryPoint.idx()];
    let knight_count = ps.played_dev_cards[DevCard::Knight.idx()];
    let army_gap_bonus = if ps.has_army {
        8.0e10
    } else if knight_count >= 2 {
        2.5e10
    } else {
        f64::from(knight_count) * 1.0e9
    };
    let opponent_actual_vps = max_opponent_metric(game, color, |opponent| {
        f64::from(game.state.player_state(opponent).actual_victory_points)
    });
    let lead = f64::from(ps.actual_victory_points) - opponent_actual_vps;

    base + f64::from(hidden_vps) * 2.75e14
        + lead * 2.0e12
        + army_gap_bonus
        + ps.longest_road_length as f64 * 1.0e6
}

pub fn value_function_action<R: Rng + ?Sized>(
    game: &Game,
    color: Color,
    actions: &[Action],
    rng: &mut R,
    epsilon_percent: Option<u8>,
) -> Option<Action> {
    if actions.len() == 1 {
        return actions.first().cloned();
    }
    if epsilon_percent.is_some_and(|epsilon| rng.gen_range(0..100) < epsilon) {
        return actions.choose(rng).cloned();
    }

    let mut best_value = f64::NEG_INFINITY;
    let mut best_action = None;
    for action in actions {
        let mut game_copy = game.clone();
        if game_copy.execute(action.clone(), false, None).is_err() {
            continue;
        }
        let value = base_value(&game_copy, color, ValueWeights::default());
        if value > best_value {
            best_value = value;
            best_action = Some(action.clone());
        }
    }
    best_action.or_else(|| actions.first().cloned())
}

pub fn strategic_value_action<R: Rng + ?Sized>(
    game: &Game,
    color: Color,
    actions: &[Action],
    rng: &mut R,
    epsilon_percent: Option<u8>,
) -> Option<Action> {
    if actions.len() == 1 {
        return actions.first().cloned();
    }
    if epsilon_percent.is_some_and(|epsilon| rng.gen_range(0..100) < epsilon) {
        return actions.choose(rng).cloned();
    }

    let candidates = if actions.len() > 12 {
        list_pruned_actions(game)
    } else {
        actions.to_vec()
    };
    let mut best_value = f64::NEG_INFINITY;
    let mut best_actions = Vec::new();
    for action in candidates.iter().filter(|action| actions.contains(action)) {
        let outcomes = execute_spectrum(game, action);
        let value = outcomes
            .into_iter()
            .map(|(outcome, probability)| probability * strategic_value(&outcome, color))
            .sum::<f64>();
        match value.total_cmp(&best_value) {
            Ordering::Greater => {
                best_value = value;
                best_actions.clear();
                best_actions.push(action.clone());
            }
            Ordering::Equal => best_actions.push(action.clone()),
            Ordering::Less => {}
        }
    }
    best_actions
        .choose(rng)
        .cloned()
        .or_else(|| actions.first().cloned())
}

fn random_playout_winner<R: Rng + ?Sized>(
    game: &Game,
    rng: &mut R,
    turn_limit: usize,
) -> Option<Color> {
    let mut game_copy = game.clone();
    while game_copy.winning_color().is_none() && game_copy.state.num_turns < turn_limit {
        let action = game_copy.playable_actions.choose(rng).cloned()?;
        if game_copy.execute(action, true, None).is_err() {
            return None;
        }
    }
    game_copy.winning_color()
}

pub fn playout_action<R: Rng + ?Sized>(
    game: &Game,
    color: Color,
    actions: &[Action],
    rng: &mut R,
    playouts_per_action: u16,
) -> Option<Action> {
    if actions.len() == 1 {
        return actions.first().cloned();
    }

    const PLAYOUT_TURN_LIMIT: usize = 300;
    const MAX_PLAYOUT_CANDIDATES: usize = 8;
    let playouts = usize::from(playouts_per_action.max(1));
    let mut candidates = if actions.len() > MAX_PLAYOUT_CANDIDATES {
        list_pruned_actions(game)
    } else {
        actions.to_vec()
    };
    if candidates.is_empty() {
        candidates = actions.to_vec();
    }
    if candidates.len() > MAX_PLAYOUT_CANDIDATES {
        candidates.sort_by(|a, b| {
            let mut a_game = game.clone();
            let mut b_game = game.clone();
            let _ = a_game.execute(a.clone(), false, None);
            let _ = b_game.execute(b.clone(), false, None);
            let a_value = base_value(&a_game, color, ValueWeights::default());
            let b_value = base_value(&b_game, color, ValueWeights::default());
            b_value.total_cmp(&a_value)
        });
        candidates.truncate(MAX_PLAYOUT_CANDIDATES);
    }

    let mut best_wins = i32::MIN;
    let mut best_actions = Vec::new();
    for action in &candidates {
        let mut action_game = game.clone();
        if action_game.execute(action.clone(), false, None).is_err() {
            continue;
        }

        let mut wins = if action_game.winning_color() == Some(color) {
            playouts as i32
        } else {
            0
        };
        if wins == 0 {
            for _ in 0..playouts {
                if random_playout_winner(&action_game, rng, PLAYOUT_TURN_LIMIT) == Some(color) {
                    wins += 1;
                }
            }
        }

        match wins.cmp(&best_wins) {
            std::cmp::Ordering::Greater => {
                best_wins = wins;
                best_actions.clear();
                best_actions.push(action.clone());
            }
            std::cmp::Ordering::Equal => best_actions.push(action.clone()),
            std::cmp::Ordering::Less => {}
        }
    }
    best_actions.choose(rng).cloned()
}

pub fn strategic_playout_action<R: Rng + ?Sized>(
    game: &Game,
    color: Color,
    actions: &[Action],
    rng: &mut R,
    playouts_per_action: u16,
) -> Option<Action> {
    if actions.len() == 1 {
        return actions.first().cloned();
    }

    const PLAYOUT_TURN_LIMIT: usize = 300;
    const MAX_STRATEGIC_CANDIDATES: usize = 6;
    let playouts = usize::from(playouts_per_action.max(1));
    let mut scored = actions
        .iter()
        .filter_map(|action| {
            let outcomes = execute_spectrum(game, action);
            if outcomes
                .iter()
                .any(|(outcome, _)| outcome.winning_color() == Some(color))
            {
                return Some((action.clone(), f64::INFINITY));
            }
            let value = outcomes
                .into_iter()
                .map(|(outcome, probability)| probability * strategic_value(&outcome, color))
                .sum::<f64>();
            Some((action.clone(), value))
        })
        .collect::<Vec<_>>();
    scored.sort_by(|a, b| b.1.total_cmp(&a.1));
    scored.truncate(MAX_STRATEGIC_CANDIDATES.min(scored.len()));
    if scored.is_empty() {
        return actions.first().cloned();
    }
    if scored[0].1.is_infinite() {
        return Some(scored[0].0.clone());
    }

    let baseline = scored
        .iter()
        .map(|(_, value)| *value)
        .fold(f64::INFINITY, f64::min);
    let span = (scored
        .iter()
        .map(|(_, value)| *value)
        .fold(f64::NEG_INFINITY, f64::max)
        - baseline)
        .max(1.0);

    let mut best_score = f64::NEG_INFINITY;
    let mut best_actions = Vec::new();
    for (action, heuristic_value) in &scored {
        let mut action_game = game.clone();
        if action_game.execute(action.clone(), false, None).is_err() {
            continue;
        }
        let mut wins = if action_game.winning_color() == Some(color) {
            playouts
        } else {
            0
        };
        if wins == 0 {
            for _ in 0..playouts {
                if random_playout_winner(&action_game, rng, PLAYOUT_TURN_LIMIT) == Some(color) {
                    wins += 1;
                }
            }
        }
        let heuristic_bonus = (*heuristic_value - baseline) / span;
        let score = wins as f64 + heuristic_bonus * 0.35;
        match score.total_cmp(&best_score) {
            Ordering::Greater => {
                best_score = score;
                best_actions.clear();
                best_actions.push(action.clone());
            }
            Ordering::Equal => best_actions.push(action.clone()),
            Ordering::Less => {}
        }
    }
    best_actions.choose(rng).cloned()
}

pub fn ensemble_action<R: Rng + ?Sized>(
    game: &Game,
    color: Color,
    actions: &[Action],
    rng: &mut R,
) -> Option<Action> {
    if actions.len() == 1 {
        return actions.first().cloned();
    }

    let mut candidates = Vec::new();
    if let Some(action) = playout_action(game, color, actions, rng, 100) {
        candidates.push(action);
    }
    if let Some(action) = alpha_beta_action(game, color, 2, false, false, rng, None) {
        candidates.push(action);
    }
    if let Some(action) = strategic_alpha_beta_action(game, color, 3, false, true, rng, None) {
        candidates.push(action);
    }
    if let Some(action) = strategic_value_action(game, color, actions, rng, None) {
        candidates.push(action);
    }
    candidates.sort_by(action_cmp);
    candidates.dedup();

    let mut best_value = f64::NEG_INFINITY;
    let mut best_actions = Vec::new();
    for action in candidates.iter().filter(|action| actions.contains(action)) {
        let outcomes = execute_spectrum(game, action);
        if outcomes
            .iter()
            .any(|(outcome, _)| outcome.winning_color() == Some(color))
        {
            return Some(action.clone());
        }
        let value = outcomes
            .into_iter()
            .map(|(outcome, probability)| probability * strategic_value(&outcome, color))
            .sum::<f64>();
        match value.total_cmp(&best_value) {
            Ordering::Greater => {
                best_value = value;
                best_actions.clear();
                best_actions.push(action.clone());
            }
            Ordering::Equal => best_actions.push(action.clone()),
            Ordering::Less => {}
        }
    }
    best_actions
        .choose(rng)
        .cloned()
        .or_else(|| actions.first().cloned())
}

fn deterministic_action_type(action_type: ActionType) -> bool {
    matches!(
        action_type,
        ActionType::EndTurn
            | ActionType::BuildSettlement
            | ActionType::BuildRoad
            | ActionType::BuildCity
            | ActionType::PlayKnightCard
            | ActionType::PlayYearOfPlenty
            | ActionType::PlayRoadBuilding
            | ActionType::MaritimeTrade
            | ActionType::DiscardResource
            | ActionType::PlayMonopoly
            | ActionType::OfferTrade
            | ActionType::AcceptTrade
            | ActionType::RejectTrade
            | ActionType::ConfirmTrade
            | ActionType::CancelTrade
    )
}

fn execute_deterministic_spectrum(game: &Game, action: &Action) -> Vec<(Game, f64)> {
    let mut copy = game.clone();
    let _ = copy.execute(action.clone(), false, None);
    vec![(copy, 1.0)]
}

pub fn execute_spectrum(game: &Game, action: &Action) -> Vec<(Game, f64)> {
    if deterministic_action_type(action.action_type) {
        return execute_deterministic_spectrum(game, action);
    }

    match action.action_type {
        ActionType::BuyDevelopmentCard => {
            let current_deck = &game.state.development_listdeck;
            if current_deck.is_empty() {
                return execute_deterministic_spectrum(game, action);
            }
            let mut counts = [0usize; DevCard::ALL.len()];
            for card in current_deck {
                counts[card.idx()] += 1;
            }
            let total = current_deck.len() as f64;
            let mut results = Vec::new();
            for card in DevCard::ALL {
                let count = counts[card.idx()];
                if count == 0 {
                    continue;
                }
                let mut option_game = game.clone();
                let _ =
                    option_game.execute(action.clone(), false, Some(ActionValue::DevCard(card)));
                results.push((option_game, count as f64 / total));
            }
            results
        }
        ActionType::Roll => {
            let mut results = Vec::new();
            for roll in 2..=12 {
                let d1 = roll / 2;
                let d2 = roll - d1;
                let mut option_game = game.clone();
                let _ = option_game.execute(
                    action.clone(),
                    false,
                    Some(ActionValue::Dice(d1 as u8, d2 as u8)),
                );
                results.push((option_game, number_probability(roll as u8)));
            }
            results
        }
        ActionType::MoveRobber => {
            let ActionValue::Robber(coordinate, Some(robbed_color)) = action.value else {
                return execute_deterministic_spectrum(game, action);
            };
            let victim_hand = game.state.player_state(robbed_color).resources;
            let total: u32 = victim_hand.iter().map(|count| *count as u32).sum();
            if total == 0 {
                // Nothing to steal: the robber still moves.
                let no_steal = Action::new(
                    action.color,
                    ActionType::MoveRobber,
                    ActionValue::Robber(coordinate, None),
                );
                return execute_deterministic_spectrum(game, &no_steal);
            }
            Resource::ALL
                .into_iter()
                .filter(|resource| victim_hand[resource.idx()] > 0)
                .map(|resource| {
                    let mut option_game = game.clone();
                    let _ = option_game.execute(
                        action.clone(),
                        false,
                        Some(ActionValue::Resource(resource)),
                    );
                    (
                        option_game,
                        victim_hand[resource.idx()] as f64 / total as f64,
                    )
                })
                .collect()
        }
        _ => execute_deterministic_spectrum(game, action),
    }
}

/// Probabilities of the `execute_spectrum(game, action)` outcomes, in the same
/// order, computed from the state alone without materializing any outcome game.
/// Must stay in lockstep with `execute_spectrum` (covered by unit tests).
pub fn spectrum_probabilities(game: &Game, action: &Action) -> Vec<f64> {
    if deterministic_action_type(action.action_type) {
        return vec![1.0];
    }
    match action.action_type {
        ActionType::BuyDevelopmentCard => {
            let deck = &game.state.development_listdeck;
            if deck.is_empty() {
                return vec![1.0];
            }
            let mut counts = [0usize; DevCard::ALL.len()];
            for card in deck {
                counts[card.idx()] += 1;
            }
            let total = deck.len() as f64;
            DevCard::ALL
                .into_iter()
                .filter(|card| counts[card.idx()] > 0)
                .map(|card| counts[card.idx()] as f64 / total)
                .collect()
        }
        ActionType::Roll => (2u8..=12).map(number_probability).collect(),
        ActionType::MoveRobber => {
            let ActionValue::Robber(_, Some(robbed_color)) = action.value else {
                return vec![1.0];
            };
            let victim_hand = game.state.player_state(robbed_color).resources;
            let total: u32 = victim_hand.iter().map(|count| *count as u32).sum();
            if total == 0 {
                return vec![1.0];
            }
            Resource::ALL
                .into_iter()
                .filter(|resource| victim_hand[resource.idx()] > 0)
                .map(|resource| victim_hand[resource.idx()] as f64 / total as f64)
                .collect()
        }
        _ => vec![1.0],
    }
}

/// Materialize chance outcomes of `action` as independent games in one pass.
/// `indices == None` yields every outcome in `execute_spectrum` order; explicit
/// indices (repeats allowed) yield exactly those outcomes.
pub fn execute_spectrum_games(
    game: &Game,
    action: &Action,
    indices: Option<&[usize]>,
) -> Result<Vec<Game>, String> {
    let outcomes = execute_spectrum(game, action);
    match indices {
        None => Ok(outcomes.into_iter().map(|(outcome, _)| outcome).collect()),
        Some(indices) => indices
            .iter()
            .map(|&index| {
                outcomes
                    .get(index)
                    .map(|(outcome, _)| outcome.clone())
                    .ok_or_else(|| {
                        format!("outcome index {index} out of range (0..{})", outcomes.len())
                    })
            })
            .collect(),
    }
}

/// One-round-trip expansion context for a decision node: every playable action
/// with its action-space index, its JSON encoding, and (for chance-typed
/// actions) the outcome probabilities in `execute_spectrum` order.
pub fn decision_context_json_value(
    game: &Game,
    player_colors: &[Color],
    map_kind: MapKind,
    include_spectrums: bool,
) -> Result<Value, String> {
    let action_space = ActionSpace::new(player_colors, map_kind);
    let mut actions = Vec::with_capacity(game.playable_actions.len());
    for action in &game.playable_actions {
        let index = action_space
            .index(action)
            .ok_or_else(|| format!("playable action missing from action space: {action:?}"))?;
        let mut entry = json!({
            "index": index,
            "action": action_to_json_value(action),
        });
        if include_spectrums && !deterministic_action_type(action.action_type) {
            entry["spectrum"] = json!(spectrum_probabilities(game, action));
        }
        actions.push(entry);
    }
    Ok(json!({
        "current_color": color_name(game.state.current_color()),
        "actions": actions,
    }))
}

pub fn expand_spectrum(game: &Game, actions: &[Action]) -> Vec<(Action, Vec<(Game, f64)>)> {
    actions
        .iter()
        .cloned()
        .map(|action| {
            let outcomes = execute_spectrum(game, &action);
            (action, outcomes)
        })
        .collect()
}

pub fn list_pruned_actions(game: &Game) -> Vec<Action> {
    let current_color = game.state.current_color();
    let mut actions = game.playable_actions.clone();
    if game.state.is_initial_build_phase
        && actions
            .iter()
            .any(|action| action.action_type == ActionType::BuildSettlement)
    {
        actions.retain(|action| match action.value {
            ActionValue::Node(node) => game
                .state
                .board
                .map
                .adjacent_tiles
                .get(&node)
                .map(|tiles| tiles.len() != 1)
                .unwrap_or(true),
            _ => true,
        });
    }

    if actions
        .iter()
        .any(|action| action.action_type == ActionType::MaritimeTrade)
    {
        let port_resources = game.state.board.get_player_port_resources(current_color);
        let has_three_to_one = port_resources.contains(&None);
        if has_three_to_one {
            actions.retain(|action| match action.value {
                ActionValue::MaritimeTrade(offering, _) => offering[3].is_none(),
                _ => true,
            });
        }
    }

    if actions
        .iter()
        .any(|action| action.action_type == ActionType::MoveRobber)
    {
        let robber_actions: Vec<_> = actions
            .iter()
            .filter(|action| action.action_type == ActionType::MoveRobber)
            .cloned()
            .collect();
        if let Some(best) = robber_actions.into_iter().max_by(|a, b| {
            let mut a_game = game.clone();
            let mut b_game = game.clone();
            let _ = a_game.execute(a.clone(), false, None);
            let _ = b_game.execute(b.clone(), false, None);
            let a_value = base_value(&a_game, current_color, ValueWeights::default());
            let b_value = base_value(&b_game, current_color, ValueWeights::default());
            a_value.total_cmp(&b_value)
        }) {
            actions
                .retain(|action| action.action_type != ActionType::MoveRobber || *action == best);
        }
    }

    if actions.is_empty() {
        game.playable_actions.clone()
    } else {
        actions
    }
}

pub fn alpha_beta_action<R: Rng + ?Sized>(
    game: &Game,
    color: Color,
    depth: u8,
    same_turn: bool,
    pruning: bool,
    rng: &mut R,
    epsilon_percent: Option<u8>,
) -> Option<Action> {
    let actions = if pruning {
        list_pruned_actions(game)
    } else {
        game.playable_actions.clone()
    };
    if actions.len() == 1 {
        return actions.first().cloned();
    }
    if epsilon_percent.is_some_and(|epsilon| rng.gen_range(0..100) < epsilon) {
        return actions.choose(rng).cloned();
    }
    let (action, _) = alphabeta_value(
        game.clone(),
        color,
        depth,
        f64::NEG_INFINITY,
        f64::INFINITY,
        same_turn,
        pruning,
        base_leaf_value,
    );
    action.or_else(|| actions.first().cloned())
}

pub fn strategic_alpha_beta_action<R: Rng + ?Sized>(
    game: &Game,
    color: Color,
    depth: u8,
    same_turn: bool,
    pruning: bool,
    rng: &mut R,
    epsilon_percent: Option<u8>,
) -> Option<Action> {
    let actions = if pruning {
        list_pruned_actions(game)
    } else {
        game.playable_actions.clone()
    };
    if actions.len() == 1 {
        return actions.first().cloned();
    }
    if epsilon_percent.is_some_and(|epsilon| rng.gen_range(0..100) < epsilon) {
        return actions.choose(rng).cloned();
    }
    let (action, _) = alphabeta_value(
        game.clone(),
        color,
        depth,
        f64::NEG_INFINITY,
        f64::INFINITY,
        same_turn,
        pruning,
        strategic_value,
    );
    action.or_else(|| actions.first().cloned())
}

pub fn champion_action<R: Rng + ?Sized>(
    game: &Game,
    color: Color,
    actions: &[Action],
    rng: &mut R,
) -> Option<Action> {
    if actions.len() == 1 {
        return actions.first().cloned();
    }
    let mut candidates = Vec::new();
    if let (Some(action), _) = alphabeta_value(
        game.clone(),
        color,
        2,
        f64::NEG_INFINITY,
        f64::INFINITY,
        false,
        false,
        champion_value,
    ) {
        candidates.push(action);
    }
    if let (Some(action), _) = alphabeta_value(
        game.clone(),
        color,
        6,
        f64::NEG_INFINITY,
        f64::INFINITY,
        true,
        true,
        champion_value,
    ) {
        candidates.push(action);
    }
    if let Some(action) = alpha_beta_action(game, color, 2, false, false, rng, None) {
        candidates.push(action);
    }
    candidates.sort_by(action_cmp);
    candidates.dedup();

    let mut best_value = f64::NEG_INFINITY;
    let mut best_actions = Vec::new();
    for action in candidates.iter().filter(|action| actions.contains(action)) {
        let value = execute_spectrum(game, action)
            .into_iter()
            .map(|(outcome, probability)| probability * champion_value(&outcome, color))
            .sum::<f64>();
        match value.total_cmp(&best_value) {
            Ordering::Greater => {
                best_value = value;
                best_actions.clear();
                best_actions.push(action.clone());
            }
            Ordering::Equal => best_actions.push(action.clone()),
            Ordering::Less => {}
        }
    }

    best_actions
        .choose(rng)
        .cloned()
        .or_else(|| actions.choose(rng).cloned())
}

fn base_leaf_value(game: &Game, color: Color) -> f64 {
    base_value(game, color, ValueWeights::default())
}

fn alphabeta_value(
    game: Game,
    color: Color,
    depth: u8,
    mut alpha: f64,
    mut beta: f64,
    same_turn: bool,
    pruning: bool,
    leaf_value: fn(&Game, Color) -> f64,
) -> (Option<Action>, f64) {
    if depth == 0
        || game.winning_color().is_some()
        || (same_turn && game.state.current_color() != color)
    {
        return (None, leaf_value(&game, color));
    }

    let actions = if pruning {
        list_pruned_actions(&game)
    } else {
        game.playable_actions.clone()
    };
    if actions.is_empty() {
        return (None, leaf_value(&game, color));
    }
    let action_outcomes = expand_spectrum(&game, &actions);
    let maximizing = game.state.current_color() == color;
    let mut best_action = None;
    let mut best_value = if maximizing {
        f64::NEG_INFINITY
    } else {
        f64::INFINITY
    };

    for (action, outcomes) in action_outcomes {
        let expected_value = outcomes
            .into_iter()
            .map(|(outcome, probability)| {
                let (_, value) = alphabeta_value(
                    outcome,
                    color,
                    depth - 1,
                    alpha,
                    beta,
                    same_turn,
                    pruning,
                    leaf_value,
                );
                probability * value
            })
            .sum::<f64>();

        if maximizing {
            if expected_value > best_value {
                best_value = expected_value;
                best_action = Some(action);
            }
            alpha = alpha.max(best_value);
            if alpha >= beta {
                break;
            }
        } else {
            if expected_value < best_value {
                best_value = expected_value;
                best_action = Some(action);
            }
            beta = beta.min(best_value);
            if beta <= alpha {
                break;
            }
        }
    }

    (best_action, best_value)
}

#[derive(Clone, Debug)]
pub struct Game {
    pub seed: Option<u64>,
    pub vps_to_win: i16,
    pub state: State,
    pub playable_actions: Vec<Action>,
    rng: SmallRng,
    record_actions: bool,
    materialize_playable_actions: bool,
}

impl Game {
    pub fn new(players: Vec<Player>, seed: Option<u64>) -> Self {
        Self::with_options(players, seed, 7, false, 10)
    }

    pub fn with_options(
        players: Vec<Player>,
        seed: Option<u64>,
        discard_limit: u8,
        friendly_robber: bool,
        vps_to_win: i16,
    ) -> Self {
        Self::with_options_and_map_kind(
            players,
            seed,
            discard_limit,
            friendly_robber,
            vps_to_win,
            MapKind::Base,
        )
    }

    pub fn with_options_and_map_kind(
        players: Vec<Player>,
        seed: Option<u64>,
        discard_limit: u8,
        friendly_robber: bool,
        vps_to_win: i16,
        map_kind: MapKind,
    ) -> Self {
        Self::with_options_and_map_options(
            players,
            seed,
            discard_limit,
            friendly_robber,
            vps_to_win,
            map_kind,
            NumberPlacement::OfficialSpiral,
        )
    }

    pub fn with_options_and_map_options(
        players: Vec<Player>,
        seed: Option<u64>,
        discard_limit: u8,
        friendly_robber: bool,
        vps_to_win: i16,
        map_kind: MapKind,
        number_placement: NumberPlacement,
    ) -> Self {
        let map = if map_kind == MapKind::Tournament {
            cached_tournament_map()
        } else {
            let mut board_rng: SmallRng = match seed {
                Some(seed) => SeedableRng::seed_from_u64(seed ^ 0xC47A_7A90_6D9E_3779),
                None => SeedableRng::from_entropy(),
            };
            CatanMap::from_template_with_rng(map_kind, number_placement, &mut board_rng)
        };
        let rng: SmallRng = match seed {
            Some(seed) => SeedableRng::seed_from_u64(seed ^ 0x9E37_79B9_7F4A_7C15),
            None => SeedableRng::from_entropy(),
        };
        let board = Board::new(map);
        let state = State::new(players, seed, discard_limit, friendly_robber, board);
        let playable_actions = generate_playable_actions(&state);
        Self {
            seed,
            vps_to_win,
            state,
            playable_actions,
            rng,
            record_actions: true,
            materialize_playable_actions: true,
        }
    }

    /// Return a root determinization sampled from public conservation constraints.
    ///
    /// The board, public counters, bank, action-history length, observer hand,
    /// and every other public field are preserved. Hidden resource cards are allocated
    /// from the public conservation pool according to each opponent's public hand
    /// size.  Face-down development cards and the remaining deck are resampled
    /// together from the base deck after subtracting public played cards and the
    /// observer's own known cards.  Opponent hidden victory points and
    /// `owned_at_start` are repaired to match the sampled hand.
    ///
    /// This is a conservation-based PIMC boundary, not a history-conditioned
    /// Bayesian posterior. The returned world is independent of authoritative
    /// hidden compositions, deck order, and hidden action-record payloads while
    /// remaining a rules-valid public-conservation sample. The authoritative
    /// game is never mutated.
    pub fn determinize_for_player(&self, observer: Color, seed: u64) -> Result<Game, String> {
        if !self.state.color_to_index.contains_key(&observer) {
            return Err("observer is not a player in this game".into());
        }
        if self.state.current_color() != observer {
            return Err("determinization observer must be the current player".into());
        }

        fn sample_freqdeck(
            pool: &mut FreqDeck,
            count: u8,
            rng: &mut SmallRng,
        ) -> Result<FreqDeck, String> {
            let mut sampled = [0_u8; 5];
            for _ in 0..count {
                let total = pool.iter().map(|value| usize::from(*value)).sum::<usize>();
                if total == 0 {
                    return Err("public resource conservation pool underflow".into());
                }
                let mut draw = rng.gen_range(0..total);
                let mut selected = None;
                for (index, value) in pool.iter().enumerate() {
                    if draw < usize::from(*value) {
                        selected = Some(index);
                        break;
                    }
                    draw -= usize::from(*value);
                }
                let index = selected.ok_or("failed to sample public resource pool")?;
                pool[index] -= 1;
                sampled[index] += 1;
            }
            Ok(sampled)
        }

        fn sample_dev_hand(
            pool: &mut [u8; 5],
            count: u8,
            max_hidden_vps: u8,
            rng: &mut SmallRng,
        ) -> Result<[u8; 5], String> {
            let mut sampled = [0_u8; 5];
            for _ in 0..count {
                let mut allowed = *pool;
                let vp_index = DevCard::VictoryPoint.idx();
                allowed[vp_index] =
                    allowed[vp_index].min(max_hidden_vps.saturating_sub(sampled[vp_index]));
                let total = allowed
                    .iter()
                    .map(|value| usize::from(*value))
                    .sum::<usize>();
                if total == 0 {
                    return Err(
                        "public development-card pool has no non-terminal allocation".into(),
                    );
                }
                let mut draw = rng.gen_range(0..total);
                let mut selected = None;
                for (index, value) in allowed.iter().enumerate() {
                    if draw < usize::from(*value) {
                        selected = Some(index);
                        break;
                    }
                    draw -= usize::from(*value);
                }
                let index = selected.ok_or("failed to sample public development-card pool")?;
                pool[index] -= 1;
                sampled[index] += 1;
            }
            Ok(sampled)
        }

        let mut rng = SmallRng::seed_from_u64(seed);
        let mut result = self.clone();
        let observer_state = self.state.player_state(observer);

        // Results can contain hidden robber steals, development draws, and
        // discarded resources. Preserve the public action sequence while
        // redacting every result and the two action payloads that themselves
        // carry a secret identity. This keeps meaningful public history usable
        // inside determinizations without allowing authoritative hidden truth
        // to cross the information-set boundary.
        result.state.action_records = self
            .state
            .action_records
            .iter()
            .map(|record| {
                let action = match record.action.action_type {
                    ActionType::BuyDevelopmentCard | ActionType::DiscardResource => Action::new(
                        record.action.color,
                        record.action.action_type,
                        ActionValue::None,
                    ),
                    _ => record.action.clone(),
                };
                ActionRecord {
                    action,
                    result: ActionValue::None,
                }
            })
            .collect();

        // In base Catan the bank starts with 19 cards of each resource.  The
        // bank composition and observer hand are public/known; the difference
        // is precisely the pool spread across the other hidden hands.
        let resource_bank = starting_resource_bank();
        let mut unseen_resources = [0_u8; 5];
        for index in 0..5 {
            let known = u16::from(self.state.resource_freqdeck[index])
                + u16::from(observer_state.resources[index]);
            if known > u16::from(resource_bank[index]) {
                return Err("public resource conservation exceeds base bank".into());
            }
            unseen_resources[index] = resource_bank[index] - known as u8;
        }
        let opponent_resource_total = self
            .state
            .colors
            .iter()
            .filter(|color| **color != observer)
            .map(|color| {
                self.state
                    .player_state(*color)
                    .resources
                    .iter()
                    .map(|value| usize::from(*value))
                    .sum::<usize>()
            })
            .sum::<usize>();
        if opponent_resource_total
            != unseen_resources
                .iter()
                .map(|value| usize::from(*value))
                .sum::<usize>()
        {
            return Err("public resource counts are not conservation-consistent".into());
        }

        let mut opponent_order = self
            .state
            .colors
            .iter()
            .copied()
            .filter(|color| *color != observer)
            .collect::<Vec<_>>();
        opponent_order.shuffle(&mut rng);
        for color in &opponent_order {
            let hand_size = self
                .state
                .player_state(*color)
                .resources
                .iter()
                .copied()
                .sum::<u8>();
            result.state.player_state_mut(*color).resources =
                sample_freqdeck(&mut unseen_resources, hand_size, &mut rng)?;
        }
        if unseen_resources.iter().any(|value| *value != 0) {
            return Err("public resource determinization left unallocated cards".into());
        }

        // Reconstruct the entire unknown development-card pool.  Opponent
        // face-down identities are not subtracted: they are sampled jointly
        // with the remaining deck, eliminating both true-deck weighting and
        // true-outcome-support leaks.
        let mut unseen_devs = [0_u8; 5];
        for card in starting_devcard_bank() {
            unseen_devs[card.idx()] += 1;
        }
        for color in &self.state.colors {
            let state = self.state.player_state(*color);
            for index in 0..5 {
                unseen_devs[index] = unseen_devs[index]
                    .checked_sub(state.played_dev_cards[index])
                    .ok_or("public played development cards exceed base deck")?;
            }
        }
        for index in 0..5 {
            unseen_devs[index] = unseen_devs[index]
                .checked_sub(observer_state.dev_cards[index])
                .ok_or("observer development cards exceed public remaining deck")?;
        }

        let expected_unknown = self.state.development_listdeck.len()
            + self
                .state
                .colors
                .iter()
                .filter(|color| **color != observer)
                .map(|color| {
                    self.state
                        .player_state(*color)
                        .dev_cards
                        .iter()
                        .map(|value| usize::from(*value))
                        .sum::<usize>()
                })
                .sum::<usize>();
        if expected_unknown
            != unseen_devs
                .iter()
                .map(|value| usize::from(*value))
                .sum::<usize>()
        {
            return Err("public development-card counts are not conservation-consistent".into());
        }

        for color in &opponent_order {
            let source = self.state.player_state(*color);
            let hand_size = source.dev_cards.iter().copied().sum::<u8>();
            // The public fact that the game is still running rules out a sampled
            // hidden-VP allocation that would already make an opponent a winner.
            let max_hidden_vps =
                (self.vps_to_win - 1 - source.victory_points).clamp(0, i16::from(u8::MAX)) as u8;
            let sampled = sample_dev_hand(&mut unseen_devs, hand_size, max_hidden_vps, &mut rng)?;
            let target = result.state.player_state_mut(*color);
            target.dev_cards = sampled;
            target.actual_victory_points =
                target.victory_points + i16::from(sampled[DevCard::VictoryPoint.idx()]);
            // Every opponent is between turns at a root owned by `observer`;
            // when its next turn begins the engine refreshes this field anyway.
            target.owned_at_start = sampled.map(|count| count > 0);
        }

        let mut sampled_deck = Vec::with_capacity(self.state.development_listdeck.len());
        for card in DevCard::ALL {
            sampled_deck.extend(std::iter::repeat_n(
                card,
                usize::from(unseen_devs[card.idx()]),
            ));
        }
        if sampled_deck.len() != self.state.development_listdeck.len() {
            return Err("public development-card determinization changed deck size".into());
        }
        sampled_deck.shuffle(&mut rng);
        result.state.development_listdeck = sampled_deck;
        result.rng = SmallRng::seed_from_u64(seed ^ 0x1F05_E7A5_5EED_0001);

        // Current-player legal actions are a function of public state and the
        // observer's own preserved hand.  Regenerate defensively and reject any
        // accidental drift at this correctness boundary.
        let expected_actions = self.playable_actions.clone();
        result.playable_actions = generate_playable_actions(&result.state);
        if result.playable_actions != expected_actions {
            return Err("public-belief determinization changed root legal actions".into());
        }
        Ok(result)
    }

    /// Materialize development-card draw successors from the observer's public
    /// information set, rather than from the authoritative hidden deck.
    ///
    /// Each returned game is independently sanitized with
    /// [`Game::determinize_for_player`] and then conditioned on drawing the
    /// requested card.  A requested card may therefore be materialized even
    /// when every authoritative copy is currently hidden in an opponent hand.
    /// Card identities in opponent hands and deck order cannot affect the
    /// result; only public played-card counts, public hand sizes, the observer's
    /// own cards, and `seed` do.  The caller supplies belief probabilities and
    /// receives children in exactly `cards` order.
    pub fn public_belief_development_draws(
        &self,
        observer: Color,
        action: &Action,
        cards: &[DevCard],
        seed: u64,
    ) -> Result<Vec<Game>, String> {
        if action.action_type != ActionType::BuyDevelopmentCard {
            return Err("public-belief development draw requires BUY_DEVELOPMENT_CARD".into());
        }
        if action.color != observer {
            return Err("public-belief development draw actor must be the observer".into());
        }

        // Public posterior support: base deck minus all publicly played cards
        // and the observer's own known, unplayed cards. Opponent face-down
        // cards remain exchangeable with the physical deck.
        let observer_state = self.state.player_state(observer);
        let mut public_unknown = [0_u8; 5];
        for card in starting_devcard_bank() {
            public_unknown[card.idx()] += 1;
        }
        for color in &self.state.colors {
            let state = self.state.player_state(*color);
            for index in 0..5 {
                public_unknown[index] = public_unknown[index]
                    .checked_sub(state.played_dev_cards[index])
                    .ok_or("public played development cards exceed base deck")?;
            }
        }
        for index in 0..5 {
            public_unknown[index] = public_unknown[index]
                .checked_sub(observer_state.dev_cards[index])
                .ok_or("observer development cards exceed public remaining deck")?;
        }

        let mut results = Vec::with_capacity(cards.len());
        for requested in cards {
            if public_unknown[requested.idx()] == 0 {
                return Err(format!(
                    "requested public-belief development card has zero support: {:?}",
                    requested
                ));
            }

            // Card-specific mixing keeps sibling hidden allocations independent
            // while remaining a pure function of public state and caller seed.
            let card_seed = seed ^ (requested.idx() as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15);
            let mut sampled = self.determinize_for_player(observer, card_seed)?;

            if !sampled.state.development_listdeck.contains(requested) {
                // The requested public-supported card landed in an opponent's
                // sampled hand. Exchange it with a deck card, preserving every
                // public hand/deck size and repairing hidden VP/playability.
                // Prefer a replacement that keeps the sampled world publicly
                // non-terminal.
                let donor = sampled
                    .state
                    .colors
                    .iter()
                    .copied()
                    .find(|color| {
                        *color != observer
                            && sampled.state.player_state(*color).dev_cards[requested.idx()] > 0
                    })
                    .ok_or(
                        "public-supported development card was neither in deck nor opponent hand",
                    )?;
                let replacement_position = sampled
                    .state
                    .development_listdeck
                    .iter()
                    .position(|replacement| {
                        let donor_state = sampled.state.player_state(donor);
                        let hidden_vps_after = donor_state.dev_cards[DevCard::VictoryPoint.idx()]
                            - u8::from(*requested == DevCard::VictoryPoint)
                            + u8::from(*replacement == DevCard::VictoryPoint);
                        donor_state.victory_points + i16::from(hidden_vps_after)
                            < sampled.vps_to_win
                    })
                    .ok_or(
                        "no non-terminal hidden allocation can condition on requested dev draw",
                    )?;
                let replacement = sampled
                    .state
                    .development_listdeck
                    .remove(replacement_position);
                {
                    let donor_state = sampled.state.player_state_mut(donor);
                    donor_state.dev_cards[requested.idx()] -= 1;
                    donor_state.dev_cards[replacement.idx()] += 1;
                    donor_state.actual_victory_points = donor_state.victory_points
                        + i16::from(donor_state.dev_cards[DevCard::VictoryPoint.idx()]);
                    donor_state.owned_at_start = donor_state.dev_cards.map(|count| count > 0);
                }
                sampled.state.development_listdeck.push(*requested);
            }

            sampled.execute(action.clone(), true, Some(ActionValue::DevCard(*requested)))?;
            results.push(sampled);
        }
        Ok(results)
    }

    pub fn set_record_actions(&mut self, record_actions: bool) {
        self.record_actions = record_actions;
    }

    pub fn set_materialize_playable_actions(&mut self, materialize_playable_actions: bool) {
        self.materialize_playable_actions = materialize_playable_actions;
    }

    pub fn play(&mut self) -> Option<Color> {
        const TURNS_LIMIT: usize = 1000;
        while self.winning_color().is_none() && self.state.num_turns < TURNS_LIMIT {
            self.play_tick().ok()?;
        }
        self.winning_color()
    }

    pub fn play_tick(&mut self) -> Result<ActionRecord, String> {
        let player = &self.state.players[self.state.current_player_index];
        let color = player.color;
        let kind = player.kind;
        if !self.materialize_playable_actions && kind == PlayerKind::Random {
            let action = random_sorted_playable_action(&self.state, &mut self.rng)
                .ok_or("no playable actions")?;
            return self.execute_known_playable(action);
        }
        let action = match kind {
            PlayerKind::Simple => self.playable_actions.first().cloned(),
            PlayerKind::Random => self.playable_actions.choose(&mut self.rng).cloned(),
            PlayerKind::WeightedRandom => {
                weighted_random_action(&self.playable_actions, &mut self.rng)
            }
            PlayerKind::VictoryPoint => {
                let playable_actions = self.playable_actions.clone();
                let mut rng = self.rng.clone();
                let action = victory_point_action(self, color, &playable_actions, &mut rng);
                self.rng = rng;
                action
            }
            PlayerKind::ValueFunction { epsilon } => {
                let playable_actions = self.playable_actions.clone();
                let mut rng = self.rng.clone();
                let action =
                    value_function_action(self, color, &playable_actions, &mut rng, epsilon);
                self.rng = rng;
                action
            }
            PlayerKind::StrategicValue { epsilon } => {
                let playable_actions = self.playable_actions.clone();
                let mut rng = self.rng.clone();
                let action =
                    strategic_value_action(self, color, &playable_actions, &mut rng, epsilon);
                self.rng = rng;
                action
            }
            PlayerKind::AlphaBeta {
                depth,
                same_turn,
                pruning,
                epsilon,
            } => {
                let mut rng = self.rng.clone();
                let action =
                    alpha_beta_action(self, color, depth, same_turn, pruning, &mut rng, epsilon);
                self.rng = rng;
                action
            }
            PlayerKind::StrategicAlphaBeta {
                depth,
                same_turn,
                pruning,
                epsilon,
            } => {
                let mut rng = self.rng.clone();
                let action = strategic_alpha_beta_action(
                    self, color, depth, same_turn, pruning, &mut rng, epsilon,
                );
                self.rng = rng;
                action
            }
            PlayerKind::Playout {
                playouts_per_action,
            } => {
                let playable_actions = self.playable_actions.clone();
                let mut rng = self.rng.clone();
                let action = playout_action(
                    self,
                    color,
                    &playable_actions,
                    &mut rng,
                    playouts_per_action,
                );
                self.rng = rng;
                action
            }
            PlayerKind::StrategicPlayout {
                playouts_per_action,
            } => {
                let playable_actions = self.playable_actions.clone();
                let mut rng = self.rng.clone();
                let action = strategic_playout_action(
                    self,
                    color,
                    &playable_actions,
                    &mut rng,
                    playouts_per_action,
                );
                self.rng = rng;
                action
            }
            PlayerKind::Champion => {
                let playable_actions = self.playable_actions.clone();
                let mut rng = self.rng.clone();
                let action = champion_action(self, color, &playable_actions, &mut rng);
                self.rng = rng;
                action
            }
        }
        .ok_or("no playable actions")?;
        self.execute_known_playable(action)
    }

    /// Return the pre-action legal width only when it is public information.
    ///
    /// Initial placement, robber movement, and free-road placement depend on
    /// the public board. Regular PLAY_TURN and DISCARD choices depend on the
    /// actor's hidden hand/development cards, so exposing even a sole-action
    /// bit for those prompts would leak private information.
    fn public_legal_action_count_before(&self) -> Option<usize> {
        match self.state.current_prompt {
            ActionPrompt::BuildInitialSettlement
            | ActionPrompt::BuildInitialRoad
            | ActionPrompt::MoveRobber => Some(count_sorted_playable_actions(&self.state)),
            ActionPrompt::PlayTurn if self.state.is_road_building => {
                Some(count_sorted_playable_actions(&self.state))
            }
            ActionPrompt::Discard
            | ActionPrompt::PlayTurn
            | ActionPrompt::DecideTrade
            | ActionPrompt::DecideAcceptees => None,
        }
    }

    fn set_latest_public_legal_action_count(&mut self, count: Option<usize>) {
        if !self.record_actions {
            return;
        }
        let Some(index) = self.state.action_records.len().checked_sub(1) else {
            return;
        };
        debug_assert_eq!(
            self.state.action_records.len(),
            self.state.action_public_legal_counts.len()
        );
        self.state.action_public_legal_counts[index] = count
            .and_then(|value| u16::try_from(value).ok())
            .filter(|value| *value > 0)
            .unwrap_or(0);
    }

    fn execute_known_playable(&mut self, action: Action) -> Result<ActionRecord, String> {
        let public_legal_action_count = self.public_legal_action_count_before();
        let record = apply_action_known_valid(
            &mut self.state,
            action,
            None,
            &mut self.rng,
            self.record_actions,
        )?;
        self.set_latest_public_legal_action_count(public_legal_action_count);
        if self.materialize_playable_actions {
            self.playable_actions = generate_playable_actions(&self.state);
        } else {
            self.playable_actions.clear();
        }
        Ok(record)
    }

    pub fn execute(
        &mut self,
        action: Action,
        validate: bool,
        replay_result: Option<ActionValue>,
    ) -> Result<ActionRecord, String> {
        if validate && !is_valid_action(&self.playable_actions, &self.state, &action) {
            return Err(format!("action not playable: {action:?}"));
        }
        let public_legal_action_count = self.public_legal_action_count_before();
        let record = apply_action(
            &mut self.state,
            action,
            replay_result,
            &mut self.rng,
            self.record_actions,
        )?;
        self.set_latest_public_legal_action_count(public_legal_action_count);
        if self.materialize_playable_actions {
            self.playable_actions = generate_playable_actions(&self.state);
        } else {
            self.playable_actions.clear();
        }
        Ok(record)
    }

    pub fn winning_color(&self) -> Option<Color> {
        self.state
            .colors
            .iter()
            .copied()
            .find(|color| self.state.player_state(*color).actual_victory_points >= self.vps_to_win)
    }
}

pub fn is_valid_action(playable_actions: &[Action], state: &State, action: &Action) -> bool {
    if action.action_type == ActionType::OfferTrade {
        let ActionValue::Trade(trade) = action.value else {
            return false;
        };
        let offering = freqdeck_from_trade_slice(&trade[..5]);
        return state.current_color() == action.color
            && state.current_prompt == ActionPrompt::PlayTurn
            && state.player_state(action.color).has_rolled
            && is_valid_trade(trade)
            && freqdeck_contains(state.player_state(action.color).resources, offering);
    }
    playable_actions.contains(action)
}

pub type FeatureMap = BTreeMap<String, f64>;

fn bool_feature(value: bool) -> f64 {
    if value { 1.0 } else { 0.0 }
}

fn resource_name(resource: Resource) -> &'static str {
    match resource {
        Resource::Wood => "WOOD",
        Resource::Brick => "BRICK",
        Resource::Sheep => "SHEEP",
        Resource::Wheat => "WHEAT",
        Resource::Ore => "ORE",
    }
}

fn dev_card_name(card: DevCard) -> &'static str {
    match card {
        DevCard::Knight => "KNIGHT",
        DevCard::YearOfPlenty => "YEAR_OF_PLENTY",
        DevCard::Monopoly => "MONOPOLY",
        DevCard::RoadBuilding => "ROAD_BUILDING",
        DevCard::VictoryPoint => "VICTORY_POINT",
    }
}

fn building_name(building: BuildingType) -> &'static str {
    match building {
        BuildingType::Settlement => "SETTLEMENT",
        BuildingType::City => "CITY",
        BuildingType::Road => "ROAD",
    }
}

fn color_name(color: Color) -> &'static str {
    match color {
        Color::Red => "RED",
        Color::Blue => "BLUE",
        Color::Orange => "ORANGE",
        Color::White => "WHITE",
    }
}

fn parse_color(value: &str) -> Result<Color, String> {
    match value {
        "RED" => Ok(Color::Red),
        "BLUE" => Ok(Color::Blue),
        "ORANGE" => Ok(Color::Orange),
        "WHITE" => Ok(Color::White),
        _ => Err(format!("unknown color: {value}")),
    }
}

fn parse_resource(value: &str) -> Result<Resource, String> {
    match value {
        "WOOD" => Ok(Resource::Wood),
        "BRICK" => Ok(Resource::Brick),
        "SHEEP" => Ok(Resource::Sheep),
        "WHEAT" => Ok(Resource::Wheat),
        "ORE" => Ok(Resource::Ore),
        _ => Err(format!("unknown resource: {value}")),
    }
}

fn parse_dev_card(value: &str) -> Result<DevCard, String> {
    match value {
        "KNIGHT" => Ok(DevCard::Knight),
        "YEAR_OF_PLENTY" => Ok(DevCard::YearOfPlenty),
        "MONOPOLY" => Ok(DevCard::Monopoly),
        "ROAD_BUILDING" => Ok(DevCard::RoadBuilding),
        "VICTORY_POINT" => Ok(DevCard::VictoryPoint),
        _ => Err(format!("unknown development card: {value}")),
    }
}

fn action_type_name(action_type: ActionType) -> &'static str {
    match action_type {
        ActionType::Roll => "ROLL",
        ActionType::MoveRobber => "MOVE_ROBBER",
        ActionType::DiscardResource => "DISCARD_RESOURCE",
        ActionType::BuildRoad => "BUILD_ROAD",
        ActionType::BuildSettlement => "BUILD_SETTLEMENT",
        ActionType::BuildCity => "BUILD_CITY",
        ActionType::BuyDevelopmentCard => "BUY_DEVELOPMENT_CARD",
        ActionType::PlayKnightCard => "PLAY_KNIGHT_CARD",
        ActionType::PlayYearOfPlenty => "PLAY_YEAR_OF_PLENTY",
        ActionType::PlayMonopoly => "PLAY_MONOPOLY",
        ActionType::PlayRoadBuilding => "PLAY_ROAD_BUILDING",
        ActionType::MaritimeTrade => "MARITIME_TRADE",
        ActionType::OfferTrade => "OFFER_TRADE",
        ActionType::AcceptTrade => "ACCEPT_TRADE",
        ActionType::RejectTrade => "REJECT_TRADE",
        ActionType::ConfirmTrade => "CONFIRM_TRADE",
        ActionType::CancelTrade => "CANCEL_TRADE",
        ActionType::EndTurn => "END_TURN",
    }
}

fn parse_action_type(value: &str) -> Result<ActionType, String> {
    match value {
        "ROLL" => Ok(ActionType::Roll),
        "MOVE_ROBBER" => Ok(ActionType::MoveRobber),
        "DISCARD_RESOURCE" => Ok(ActionType::DiscardResource),
        "BUILD_ROAD" => Ok(ActionType::BuildRoad),
        "BUILD_SETTLEMENT" => Ok(ActionType::BuildSettlement),
        "BUILD_CITY" => Ok(ActionType::BuildCity),
        "BUY_DEVELOPMENT_CARD" => Ok(ActionType::BuyDevelopmentCard),
        "PLAY_KNIGHT_CARD" => Ok(ActionType::PlayKnightCard),
        "PLAY_YEAR_OF_PLENTY" => Ok(ActionType::PlayYearOfPlenty),
        "PLAY_MONOPOLY" => Ok(ActionType::PlayMonopoly),
        "PLAY_ROAD_BUILDING" => Ok(ActionType::PlayRoadBuilding),
        "MARITIME_TRADE" => Ok(ActionType::MaritimeTrade),
        "OFFER_TRADE" => Ok(ActionType::OfferTrade),
        "ACCEPT_TRADE" => Ok(ActionType::AcceptTrade),
        "REJECT_TRADE" => Ok(ActionType::RejectTrade),
        "CONFIRM_TRADE" => Ok(ActionType::ConfirmTrade),
        "CANCEL_TRADE" => Ok(ActionType::CancelTrade),
        "END_TURN" => Ok(ActionType::EndTurn),
        _ => Err(format!("unknown action type: {value}")),
    }
}

fn action_prompt_name(prompt: ActionPrompt) -> &'static str {
    match prompt {
        ActionPrompt::BuildInitialSettlement => "BUILD_INITIAL_SETTLEMENT",
        ActionPrompt::BuildInitialRoad => "BUILD_INITIAL_ROAD",
        ActionPrompt::PlayTurn => "PLAY_TURN",
        ActionPrompt::Discard => "DISCARD",
        ActionPrompt::MoveRobber => "MOVE_ROBBER",
        ActionPrompt::DecideTrade => "DECIDE_TRADE",
        ActionPrompt::DecideAcceptees => "DECIDE_ACCEPTEES",
    }
}

fn direction_name(direction: Direction) -> &'static str {
    match direction {
        Direction::East => "EAST",
        Direction::Southeast => "SOUTHEAST",
        Direction::Southwest => "SOUTHWEST",
        Direction::West => "WEST",
        Direction::Northwest => "NORTHWEST",
        Direction::Northeast => "NORTHEAST",
    }
}

fn node_ref_name(node_ref: NodeRef) -> &'static str {
    match node_ref {
        NodeRef::North => "NORTH",
        NodeRef::Northeast => "NORTHEAST",
        NodeRef::Southeast => "SOUTHEAST",
        NodeRef::South => "SOUTH",
        NodeRef::Southwest => "SOUTHWEST",
        NodeRef::Northwest => "NORTHWEST",
    }
}

fn edge_ref_name(edge_ref: EdgeRef) -> &'static str {
    match edge_ref {
        EdgeRef::East => "EAST",
        EdgeRef::Southeast => "SOUTHEAST",
        EdgeRef::Southwest => "SOUTHWEST",
        EdgeRef::West => "WEST",
        EdgeRef::Northwest => "NORTHWEST",
        EdgeRef::Northeast => "NORTHEAST",
    }
}

fn coordinate_json(coordinate: Coordinate) -> Value {
    json!([coordinate.0, coordinate.1, coordinate.2])
}

fn edge_json(edge: Edge) -> Value {
    let edge = canonical_edge(edge);
    json!([edge.0, edge.1])
}

fn json_u64(value: &Value, context: &str) -> Result<u64, String> {
    value
        .as_u64()
        .ok_or_else(|| format!("{context} must be an unsigned integer"))
}

fn json_str<'a>(value: &'a Value, context: &str) -> Result<&'a str, String> {
    value
        .as_str()
        .ok_or_else(|| format!("{context} must be a string"))
}

fn parse_edge_value(value: &Value) -> Result<Edge, String> {
    let values = value
        .as_array()
        .ok_or_else(|| "edge must be a two-element array".to_string())?;
    if values.len() != 2 {
        return Err("edge must be a two-element array".to_string());
    }
    Ok(canonical_edge((
        json_u64(&values[0], "edge[0]")? as usize,
        json_u64(&values[1], "edge[1]")? as usize,
    )))
}

fn parse_coordinate_value(value: &Value) -> Result<Coordinate, String> {
    let values = value
        .as_array()
        .ok_or_else(|| "coordinate must be a three-element array".to_string())?;
    if values.len() != 3 {
        return Err("coordinate must be a three-element array".to_string());
    }
    Ok(Coordinate(
        values[0]
            .as_i64()
            .ok_or_else(|| "coordinate[0] must be an integer".to_string())? as i8,
        values[1]
            .as_i64()
            .ok_or_else(|| "coordinate[1] must be an integer".to_string())? as i8,
        values[2]
            .as_i64()
            .ok_or_else(|| "coordinate[2] must be an integer".to_string())? as i8,
    ))
}

fn action_value_to_json(value: &ActionValue) -> Value {
    match value {
        ActionValue::None => Value::Null,
        ActionValue::Node(node) => json!(node),
        ActionValue::Edge(edge) => edge_json(*edge),
        ActionValue::Resource(resource) => json!(resource_name(*resource)),
        ActionValue::Resources(resources) => {
            json!(
                resources
                    .iter()
                    .map(|r| resource_name(*r))
                    .collect::<Vec<_>>()
            )
        }
        ActionValue::Robber(coordinate, victim) => {
            json!([coordinate_json(*coordinate), victim.map(color_name)])
        }
        ActionValue::MaritimeTrade(offering, asking) => {
            let mut values: Vec<Value> = offering
                .iter()
                .map(|resource| resource.map(resource_name).map_or(Value::Null, Value::from))
                .collect();
            values.push(json!(resource_name(*asking)));
            Value::Array(values)
        }
        ActionValue::Trade(trade) => json!(trade),
        ActionValue::ConfirmTrade(trade, color) => {
            let mut values: Vec<Value> = trade.iter().map(|value| json!(value)).collect();
            values.push(json!(color_name(*color)));
            Value::Array(values)
        }
        ActionValue::Dice(a, b) => json!([a, b]),
        ActionValue::DevCard(card) => json!(dev_card_name(*card)),
    }
}

pub fn action_to_json_value(action: &Action) -> Value {
    json!([
        color_name(action.color),
        action_type_name(action.action_type),
        action_value_to_json(&action.value)
    ])
}

pub fn action_record_to_json_value(action_record: &ActionRecord) -> Value {
    json!({
        "action": action_to_json_value(&action_record.action),
        "result": action_value_to_json(&action_record.result),
    })
}

fn resource_deck_to_json_value(deck: FreqDeck) -> Value {
    Value::Object(
        Resource::ALL
            .iter()
            .map(|resource| {
                (
                    resource_name(*resource).to_string(),
                    json!(deck[resource.idx()]),
                )
            })
            .collect(),
    )
}

fn development_counts_to_json_value(counts: [u8; 5]) -> Value {
    Value::Object(
        DevCard::ALL
            .iter()
            .map(|card| (dev_card_name(*card).to_string(), json!(counts[card.idx()])))
            .collect(),
    )
}

/// Observer-scoped public card deductions for the two-player track.
///
/// Resource identities are exact in two-player Catan by conservation: every
/// card not in the public bank or the observer's known hand must be in the
/// sole opponent's hand. No opponent resource field is read to produce the
/// composition or total.
///
/// Face-down development-card identities are *not* deducible. For them this
/// returns the exact exchangeable unknown pool (base deck minus public plays
/// and the observer's known cards). The opponent face-down count is derived
/// from base-deck conservation and the physical-deck length. It never reads
/// opponent dev identities or deck order.
pub fn public_card_deductions_to_json_value(game: &Game, observer: Color) -> Result<Value, String> {
    if game.state.colors.len() != 2 {
        return Err("public card deductions v1 require exactly two players".into());
    }
    if !game.state.color_to_index.contains_key(&observer) {
        return Err("public card deduction observer is not a player".into());
    }
    if game.state.current_color() != observer {
        return Err("public card deduction observer must be the current player".into());
    }
    let opponent = game
        .state
        .colors
        .iter()
        .copied()
        .find(|color| *color != observer)
        .ok_or("public card deductions require one opponent")?;
    let observer_state = game.state.player_state(observer);

    let base_resources = starting_resource_bank();
    let mut opponent_resources = [0_u8; 5];
    for resource in Resource::ALL {
        let index = resource.idx();
        let known = u16::from(game.state.resource_freqdeck[index])
            + u16::from(observer_state.resources[index]);
        if known > u16::from(base_resources[index]) {
            return Err(format!(
                "public resource conservation exceeds base bank for {}",
                resource_name(resource)
            ));
        }
        opponent_resources[index] = base_resources[index] - known as u8;
    }
    let derived_opponent_total = opponent_resources.iter().copied().sum::<u8>();

    let mut unknown_development_pool = [0_u8; 5];
    for card in starting_devcard_bank() {
        unknown_development_pool[card.idx()] += 1;
    }
    let mut publicly_played = [0_u8; 5];
    for color in &game.state.colors {
        let player = game.state.player_state(*color);
        for index in 0..5 {
            publicly_played[index] = publicly_played[index]
                .checked_add(player.played_dev_cards[index])
                .ok_or("public played development-card count overflow")?;
            unknown_development_pool[index] = unknown_development_pool[index]
                .checked_sub(player.played_dev_cards[index])
                .ok_or("public played development cards exceed base deck")?;
        }
    }
    for index in 0..5 {
        unknown_development_pool[index] = unknown_development_pool[index]
            .checked_sub(observer_state.dev_cards[index])
            .ok_or("observer development cards exceed public unknown pool")?;
    }
    let observer_dev_count = observer_state.dev_cards.iter().copied().sum::<u8>();
    let publicly_played_count = publicly_played.iter().copied().sum::<u8>();
    let accounted_development_count = game
        .state
        .development_listdeck
        .len()
        .checked_add(usize::from(observer_dev_count))
        .and_then(|count| count.checked_add(usize::from(publicly_played_count)))
        .ok_or("public development-card count overflow")?;
    let opponent_dev_count = starting_devcard_bank()
        .len()
        .checked_sub(accounted_development_count)
        .ok_or("public development-card conservation exceeds base deck")?
        as u8;
    let unknown_pool_count = unknown_development_pool.iter().copied().sum::<u8>();
    let expected_unknown_count = opponent_dev_count
        .checked_add(game.state.development_listdeck.len() as u8)
        .ok_or("public unknown development-card count overflow")?;
    if unknown_pool_count != expected_unknown_count {
        return Err(format!(
            "public development-card conservation mismatch: pool {unknown_pool_count}, opponent hand + deck {expected_unknown_count}"
        ));
    }

    Ok(json!({
        "contract": "public_card_deductions_2p_v1",
        "observer": color_name(observer),
        "opponent": color_name(opponent),
        "resource_composition_exact": true,
        "observer_resources": resource_deck_to_json_value(observer_state.resources),
        "opponent_resources": resource_deck_to_json_value(opponent_resources),
        "opponent_resource_card_count": derived_opponent_total,
        "resource_bank": resource_deck_to_json_value(game.state.resource_freqdeck),
        "development_composition_exact": false,
        "observer_development_cards": development_counts_to_json_value(observer_state.dev_cards),
        "observer_development_card_count": observer_dev_count,
        "opponent_face_down_development_card_count": opponent_dev_count,
        "development_deck_count": game.state.development_listdeck.len(),
        "publicly_played_development_cards": development_counts_to_json_value(publicly_played),
        "unknown_development_pool": development_counts_to_json_value(unknown_development_pool),
        "unknown_development_pool_count": unknown_pool_count,
    }))
}

pub fn action_from_json_value(value: &Value) -> Result<Action, String> {
    let values = value
        .as_array()
        .ok_or_else(|| "action must be a three-element array".to_string())?;
    if values.len() != 3 {
        return Err("action must be a three-element array".to_string());
    }
    let color = parse_color(json_str(&values[0], "action color")?)?;
    let action_type = parse_action_type(json_str(&values[1], "action type")?)?;
    let raw_value = &values[2];
    let action_value = match action_type {
        ActionType::BuildRoad => ActionValue::Edge(parse_edge_value(raw_value)?),
        ActionType::BuildSettlement | ActionType::BuildCity => {
            ActionValue::Node(json_u64(raw_value, "node id")? as usize)
        }
        ActionType::PlayYearOfPlenty => {
            let resources = raw_value
                .as_array()
                .ok_or_else(|| "Year of Plenty action must have resources".to_string())?;
            if !matches!(resources.len(), 1 | 2) {
                return Err("Year of Plenty action must have 1 or 2 resources".to_string());
            }
            ActionValue::Resources(
                resources
                    .iter()
                    .map(|r| parse_resource(json_str(r, "resource")?))
                    .collect::<Result<Vec<_>, _>>()?,
            )
        }
        ActionType::MoveRobber => {
            let values = raw_value
                .as_array()
                .ok_or_else(|| "MOVE_ROBBER value must be [coordinate, victim]".to_string())?;
            if values.len() != 2 {
                return Err("MOVE_ROBBER value must be [coordinate, victim]".to_string());
            }
            let coordinate = parse_coordinate_value(&values[0])?;
            let victim = if values[1].is_null() {
                None
            } else {
                Some(parse_color(json_str(&values[1], "robber victim")?)?)
            };
            ActionValue::Robber(coordinate, victim)
        }
        ActionType::MaritimeTrade => {
            let values = raw_value
                .as_array()
                .ok_or_else(|| "MARITIME_TRADE value must be a five-element array".to_string())?;
            if values.len() != 5 {
                return Err("MARITIME_TRADE value must be a five-element array".to_string());
            }
            let mut offering = [None; 4];
            for (i, value) in values.iter().take(4).enumerate() {
                offering[i] = if value.is_null() {
                    None
                } else {
                    Some(parse_resource(json_str(
                        value,
                        "maritime offered resource",
                    )?)?)
                };
            }
            let asking = parse_resource(json_str(&values[4], "maritime requested resource")?)?;
            ActionValue::MaritimeTrade(offering, asking)
        }
        ActionType::DiscardResource | ActionType::PlayMonopoly => {
            ActionValue::Resource(parse_resource(json_str(raw_value, "resource")?)?)
        }
        ActionType::OfferTrade
        | ActionType::AcceptTrade
        | ActionType::RejectTrade
        | ActionType::ConfirmTrade => {
            let values = raw_value
                .as_array()
                .ok_or_else(|| "trade value must be an array".to_string())?;
            if action_type == ActionType::ConfirmTrade {
                if values.len() != 11 {
                    return Err("CONFIRM_TRADE value must have 11 entries".to_string());
                }
                let mut trade = [0; 10];
                for i in 0..10 {
                    trade[i] = json_u64(&values[i], "trade amount")? as u8;
                }
                ActionValue::ConfirmTrade(
                    trade,
                    parse_color(json_str(&values[10], "confirm trade color")?)?,
                )
            } else {
                if values.len() != 10 {
                    return Err("trade value must have 10 entries".to_string());
                }
                let mut trade = [0; 10];
                for i in 0..10 {
                    trade[i] = json_u64(&values[i], "trade amount")? as u8;
                }
                ActionValue::Trade(trade)
            }
        }
        ActionType::BuyDevelopmentCard => {
            if raw_value.is_null() {
                ActionValue::None
            } else {
                ActionValue::DevCard(parse_dev_card(json_str(raw_value, "development card")?)?)
            }
        }
        ActionType::Roll
        | ActionType::PlayKnightCard
        | ActionType::PlayRoadBuilding
        | ActionType::CancelTrade
        | ActionType::EndTurn => ActionValue::None,
    };
    Ok(Action::new(color, action_type, action_value))
}

fn tile_to_json_value(tile: &Tile) -> Value {
    match tile {
        Tile::Land(tile) if tile.resource.is_none() => json!({
            "id": tile.id,
            "type": "DESERT",
        }),
        Tile::Land(tile) => json!({
            "id": tile.id,
            "type": "RESOURCE_TILE",
            "resource": tile.resource.map(resource_name),
            "number": tile.number,
        }),
        Tile::Port(port) => json!({
            "id": port.id,
            "type": "PORT",
            "direction": direction_name(port.direction),
            "resource": port.resource.map(resource_name),
        }),
        Tile::Water { .. } => json!({ "type": "WATER" }),
    }
}

fn player_state_to_json_value(state: &PlayerState) -> Value {
    let resources = Resource::ALL
        .iter()
        .map(|resource| {
            (
                resource_name(*resource).to_string(),
                json!(state.resources[resource.idx()]),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let dev_cards = DevCard::ALL
        .iter()
        .map(|card| {
            (
                dev_card_name(*card).to_string(),
                json!(state.dev_cards[card.idx()]),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let played_dev_cards = DevCard::ALL
        .iter()
        .map(|card| {
            (
                dev_card_name(*card).to_string(),
                json!(state.played_dev_cards[card.idx()]),
            )
        })
        .collect::<serde_json::Map<_, _>>();

    json!({
        "victory_points": state.victory_points,
        "actual_victory_points": state.actual_victory_points,
        "roads_available": state.roads_available,
        "settlements_available": state.settlements_available,
        "cities_available": state.cities_available,
        "has_road": state.has_road,
        "has_army": state.has_army,
        "has_rolled": state.has_rolled,
        "has_played_development_card_in_turn": state.has_played_development_card_in_turn,
        "resources": resources,
        "dev_cards": dev_cards,
        "played_dev_cards": played_dev_cards,
        "longest_road_length": state.longest_road_length,
    })
}

pub fn longest_roads_by_player_json_value(state: &State) -> Value {
    let values = state
        .colors
        .iter()
        .map(|color| {
            (
                color_name(*color).to_string(),
                json!(state.player_state(*color).longest_road_length),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    Value::Object(values)
}

pub fn game_to_json_value(game: &Game) -> Value {
    let mut tiles: Vec<_> = game
        .state
        .board
        .map
        .tiles
        .iter()
        .map(|(&coordinate, tile)| {
            json!({
                "coordinate": coordinate_json(coordinate),
                "tile": tile_to_json_value(tile),
            })
        })
        .collect();
    tiles.sort_by_key(|value| value["coordinate"].to_string());

    let mut nodes = serde_json::Map::new();
    let mut edges: BTreeMap<Edge, Value> = BTreeMap::new();
    for (&coordinate, tile) in &game.state.board.map.tiles {
        for node_ref in NodeRef::ALL {
            let node_id = tile.nodes()[node_ref.idx()];
            let building = game.state.board.buildings.get(&node_id);
            nodes.insert(
                node_id.to_string(),
                json!({
                    "id": node_id,
                    "tile_coordinate": coordinate_json(coordinate),
                    "direction": node_ref_name(node_ref),
                    "building": building.map(|(_, b)| building_name(*b)),
                    "color": building.map(|(c, _)| color_name(*c)),
                }),
            );
        }
        for edge_ref in EdgeRef::ALL {
            let edge = canonical_edge(tile.edges()[edge_ref.idx()]);
            let color = game.state.board.roads.get(&edge).copied();
            edges.insert(
                edge,
                json!({
                    "id": edge_json(edge),
                    "tile_coordinate": coordinate_json(coordinate),
                    "direction": edge_ref_name(edge_ref),
                    "color": color.map(color_name),
                }),
            );
        }
    }

    json!({
        "tiles": tiles,
        "nodes": nodes,
        "edges": edges.into_values().collect::<Vec<_>>(),
        "action_records": game
            .state
            .action_records
            .iter()
            .map(action_record_to_json_value)
            .collect::<Vec<_>>(),
        // Zero means the old/private-dependent width is intentionally unknown.
        // Non-zero entries are aligned with action_records and derive only
        // from public board/prompt state.
        "action_public_legal_counts": game.state.action_public_legal_counts,
        "player_state": game
            .state
            .player_state
            .iter()
            .map(player_state_to_json_value)
            .collect::<Vec<_>>(),
        "colors": game
            .state
            .colors
            .iter()
            .map(|color| color_name(*color))
            .collect::<Vec<_>>(),
        "bot_colors": game
            .state
            .players
            .iter()
            .map(|player| color_name(player.color))
            .collect::<Vec<_>>(),
        "seed": game.seed,
        "vps_to_win": game.vps_to_win,
        "num_turns": game.state.num_turns,
        "current_player_index": game.state.current_player_index,
        "current_turn_index": game.state.current_turn_index,
        "is_initial_build_phase": game.state.is_initial_build_phase,
        "is_discarding": game.state.is_discarding,
        "is_moving_knight": game.state.is_moving_knight,
        "is_road_building": game.state.is_road_building,
        "free_roads_available": game.state.free_roads_available,
        "is_resolving_trade": game.state.is_resolving_trade,
        "current_trade": game.state.current_trade,
        "acceptees": game.state.acceptees,
        "discard_limit": game.state.discard_limit,
        "friendly_robber": game.state.friendly_robber,
        "resource_bank": Resource::ALL
            .iter()
            .map(|resource| {
                (
                    resource_name(*resource).to_string(),
                    json!(game.state.resource_freqdeck[resource.idx()]),
                )
            })
            .collect::<serde_json::Map<_, _>>(),
        "development_deck_count": game.state.development_listdeck.len(),
        "robber_coordinate": coordinate_json(game.state.board.robber_coordinate),
        "current_color": color_name(game.state.current_color()),
        "current_prompt": action_prompt_name(game.state.current_prompt),
        "current_discard_count": game.state.discard_counts[game.state.current_player_index],
        "current_playable_actions": game
            .playable_actions
            .iter()
            .map(action_to_json_value)
            .collect::<Vec<_>>(),
        "longest_roads_by_player": longest_roads_by_player_json_value(&game.state),
        "winning_color": game.winning_color().map(color_name),
        "state_index": game.state.action_records.len(),
    })
}

pub fn player_features(game: &Game, p0_color: Color) -> FeatureMap {
    let mut features = FeatureMap::new();
    for (i, color) in player_colors_from_perspective(&game.state.colors, p0_color) {
        let ps = game.state.player_state(color);
        if color == p0_color {
            features.insert("P0_ACTUAL_VPS".to_string(), ps.actual_victory_points as f64);
        }
        features.insert(format!("P{i}_PUBLIC_VPS"), ps.victory_points as f64);
        features.insert(format!("P{i}_HAS_ARMY"), bool_feature(ps.has_army));
        features.insert(format!("P{i}_HAS_ROAD"), bool_feature(ps.has_road));
        features.insert(format!("P{i}_ROADS_LEFT"), ps.roads_available as f64);
        features.insert(
            format!("P{i}_SETTLEMENTS_LEFT"),
            ps.settlements_available as f64,
        );
        features.insert(format!("P{i}_CITIES_LEFT"), ps.cities_available as f64);
        features.insert(format!("P{i}_HAS_ROLLED"), bool_feature(ps.has_rolled));
        features.insert(
            format!("P{i}_LONGEST_ROAD_LENGTH"),
            ps.longest_road_length as f64,
        );
    }
    features
}

pub fn resource_hand_features(game: &Game, p0_color: Color) -> FeatureMap {
    let mut features = FeatureMap::new();
    for (i, color) in player_colors_from_perspective(&game.state.colors, p0_color) {
        let ps = game.state.player_state(color);
        if color == p0_color {
            for resource in Resource::ALL {
                features.insert(
                    format!("P0_{}_IN_HAND", resource_name(resource)),
                    ps.resources[resource.idx()] as f64,
                );
            }
            for card in DevCard::ALL {
                features.insert(
                    format!("P0_{}_IN_HAND", dev_card_name(card)),
                    ps.dev_cards[card.idx()] as f64,
                );
            }
            features.insert(
                "P0_HAS_PLAYED_DEVELOPMENT_CARD_IN_TURN".to_string(),
                bool_feature(ps.has_played_development_card_in_turn),
            );
        }

        for card in DevCard::ALL {
            if card == DevCard::VictoryPoint {
                continue;
            }
            features.insert(
                format!("P{i}_{}_PLAYED", dev_card_name(card)),
                ps.played_dev_cards[card.idx()] as f64,
            );
        }
        features.insert(
            format!("P{i}_NUM_RESOURCES_IN_HAND"),
            ps.resources.iter().map(|value| *value as u16).sum::<u16>() as f64,
        );
        features.insert(
            format!("P{i}_NUM_DEVS_IN_HAND"),
            ps.dev_cards.iter().map(|value| *value as u16).sum::<u16>() as f64,
        );
    }
    features
}

pub fn tile_features(game: &Game) -> FeatureMap {
    let mut features = FeatureMap::new();
    let mut tiles: Vec<_> = game.state.board.map.land_tiles.values().collect();
    tiles.sort_by_key(|tile| tile.id);
    for tile in tiles {
        for resource in Resource::ALL {
            features.insert(
                format!("TILE{}_IS_{}", tile.id, resource_name(resource)),
                bool_feature(tile.resource == Some(resource)),
            );
        }
        features.insert(
            format!("TILE{}_IS_DESERT", tile.id),
            bool_feature(tile.resource.is_none()),
        );
        features.insert(
            format!("TILE{}_PROBA", tile.id),
            tile.number.map(number_probability).unwrap_or(0.0),
        );
        features.insert(
            format!("TILE{}_HAS_ROBBER", tile.id),
            bool_feature(tile.coordinate == game.state.board.robber_coordinate),
        );
    }
    features
}

pub fn port_features(game: &Game) -> FeatureMap {
    let mut features = FeatureMap::new();
    let mut ports: Vec<_> = game.state.board.map.ports_by_id.values().collect();
    ports.sort_by_key(|port| port.id);
    for port in ports {
        for resource in Resource::ALL {
            features.insert(
                format!("PORT{}_IS_{}", port.id, resource_name(resource)),
                bool_feature(port.resource == Some(resource)),
            );
        }
        features.insert(
            format!("PORT{}_IS_THREE_TO_ONE", port.id),
            bool_feature(port.resource.is_none()),
        );
    }
    features
}

pub fn graph_features(game: &Game, p0_color: Color) -> FeatureMap {
    let mut features = FeatureMap::new();
    for (i, color) in player_colors_from_perspective(&game.state.colors, p0_color) {
        for node_id in &game.state.board.map.land_nodes {
            for building in [BuildingType::Settlement, BuildingType::City] {
                features.insert(
                    format!("NODE{node_id}_P{i}_{}", building_name(building)),
                    0.0,
                );
            }
        }
        for edge in game.state.board.map.land_edges() {
            features.insert(format!("EDGE{edge:?}_P{i}_ROAD"), 0.0);
        }

        for (&node_id, &(owner, building)) in &game.state.board.buildings {
            if owner == color && matches!(building, BuildingType::Settlement | BuildingType::City) {
                features.insert(
                    format!("NODE{node_id}_P{i}_{}", building_name(building)),
                    1.0,
                );
            }
        }
        for (&edge, &owner) in &game.state.board.roads {
            if owner == color {
                features.insert(format!("EDGE{:?}_P{i}_ROAD", canonical_edge(edge)), 1.0);
            }
        }
    }
    features
}

pub fn production_features(game: &Game, p0_color: Color, consider_robber: bool) -> FeatureMap {
    let prefix = if consider_robber {
        "EFFECTIVE_"
    } else {
        "TOTAL_"
    };
    let mut features = FeatureMap::new();
    for resource in Resource::ALL {
        for (i, color) in player_colors_from_perspective(&game.state.colors, p0_color) {
            let mut production = 0.0;
            for (&node_id, &(owner, building)) in &game.state.board.buildings {
                if owner != color {
                    continue;
                }
                let multiplier = match building {
                    BuildingType::Settlement => 1.0,
                    BuildingType::City => 2.0,
                    BuildingType::Road => 0.0,
                };
                if multiplier == 0.0 {
                    continue;
                }
                production += multiplier
                    * node_resource_production(
                        &game.state.board.map,
                        node_id,
                        resource,
                        consider_robber.then_some(game.state.board.robber_coordinate),
                    );
            }
            features.insert(
                format!("{prefix}P{i}_{}_PRODUCTION", resource_name(resource)),
                production,
            );
        }
    }
    features
}

pub fn node_resource_production(
    catan_map: &CatanMap,
    node_id: NodeId,
    resource: Resource,
    robber_coordinate: Option<Coordinate>,
) -> f64 {
    catan_map
        .adjacent_tiles
        .get(&node_id)
        .into_iter()
        .flatten()
        .filter(|tile| {
            tile.resource == Some(resource)
                && robber_coordinate.is_none_or(|coordinate| tile.coordinate != coordinate)
        })
        .map(|tile| tile.number.map(number_probability).unwrap_or(0.0))
        .sum()
}

pub fn game_features(game: &Game) -> FeatureMap {
    let mut features = FeatureMap::new();
    features.insert(
        "BANK_DEV_CARDS".to_string(),
        game.state.development_listdeck.len() as f64,
    );
    features.insert(
        "IS_MOVING_ROBBER".to_string(),
        bool_feature(
            game.playable_actions
                .iter()
                .any(|action| action.action_type == ActionType::MoveRobber),
        ),
    );
    features.insert(
        "IS_DISCARDING".to_string(),
        bool_feature(
            game.playable_actions
                .iter()
                .any(|action| action.action_type == ActionType::DiscardResource),
        ),
    );
    for resource in Resource::ALL {
        features.insert(
            format!("BANK_{}", resource_name(resource)),
            game.state.resource_freqdeck[resource.idx()] as f64,
        );
    }
    features
}

pub fn create_sample(game: &Game, p0_color: Color) -> FeatureMap {
    let mut sample = FeatureMap::new();
    sample.extend(player_features(game, p0_color));
    sample.extend(resource_hand_features(game, p0_color));
    sample.extend(tile_features(game));
    sample.extend(port_features(game));
    sample.extend(graph_features(game, p0_color));
    sample.extend(game_features(game));
    sample
}

pub fn create_sample_with_production(
    game: &Game,
    p0_color: Color,
    consider_robber: bool,
) -> FeatureMap {
    let mut sample = create_sample(game, p0_color);
    sample.extend(production_features(game, p0_color, consider_robber));
    sample
}

pub fn create_sample_vector(game: &Game, p0_color: Color, ordering: Option<&[String]>) -> Vec<f64> {
    let sample = create_sample(game, p0_color);
    match ordering {
        Some(ordering) => ordering
            .iter()
            .filter_map(|feature| sample.get(feature).copied())
            .collect(),
        None => sample.values().copied().collect(),
    }
}

pub fn create_sample_vector_for_schema(
    game: &Game,
    p0_color: Color,
    schema: &[String],
) -> Vec<f64> {
    let sample = create_sample(game, p0_color);
    schema
        .iter()
        .map(|feature| sample.get(feature).copied().unwrap_or(0.0))
        .collect()
}

pub fn fill_sample_vector_batch_for_schema(
    samples: &[(&Game, Color)],
    schema: &[String],
    out: &mut [f32],
) -> Result<(usize, usize), String> {
    let width = schema.len();
    let expected = samples.len() * width;
    if out.len() != expected {
        return Err(format!(
            "sample vector batch output length mismatch: got {}, expected {expected}",
            out.len()
        ));
    }

    if width == 0 || samples.is_empty() {
        return Ok((samples.len(), width));
    }

    if samples.len() >= PARALLEL_SAMPLE_VECTOR_BATCH_MIN_ROWS {
        samples
            .par_iter()
            .zip(out.par_chunks_mut(width))
            .for_each(|((game, color), row)| {
                fill_sample_vector_row_for_schema(game, *color, schema, row);
            });
    } else {
        for ((game, color), row) in samples.iter().zip(out.chunks_mut(width)) {
            fill_sample_vector_row_for_schema(game, *color, schema, row);
        }
    }

    Ok((samples.len(), width))
}

fn fill_sample_vector_row_for_schema(
    game: &Game,
    p0_color: Color,
    schema: &[String],
    out: &mut [f32],
) {
    debug_assert_eq!(out.len(), schema.len());
    let sample = create_sample(game, p0_color);
    for (column, feature) in schema.iter().enumerate() {
        out[column] = sample.get(feature).copied().unwrap_or(0.0) as f32;
    }
}

pub fn fill_sample_vector_batch_for_schema_le_bytes(
    samples: &[(&Game, Color)],
    schema: &[String],
    out: &mut [u8],
) -> Result<(usize, usize), String> {
    let width = schema.len();
    let value_count = samples.len().checked_mul(width).ok_or_else(|| {
        format!(
            "sample vector batch output length overflow: {} rows * {width} columns",
            samples.len()
        )
    })?;
    let expected = value_count.checked_mul(4).ok_or_else(|| {
        format!("sample vector batch byte length overflow for {value_count} float32 values")
    })?;
    if out.len() != expected {
        return Err(format!(
            "sample vector batch output byte length mismatch: got {}, expected {expected}",
            out.len()
        ));
    }

    if width == 0 || samples.is_empty() {
        return Ok((samples.len(), width));
    }

    if samples.len() >= PARALLEL_SAMPLE_VECTOR_BATCH_MIN_ROWS {
        samples
            .par_iter()
            .zip(out.par_chunks_mut(width * 4))
            .for_each(|((game, color), row)| {
                fill_sample_vector_row_for_schema_le_bytes(game, *color, schema, row);
            });
    } else {
        for ((game, color), row) in samples.iter().zip(out.chunks_mut(width * 4)) {
            fill_sample_vector_row_for_schema_le_bytes(game, *color, schema, row);
        }
    }

    Ok((samples.len(), width))
}

fn fill_sample_vector_row_for_schema_le_bytes(
    game: &Game,
    p0_color: Color,
    schema: &[String],
    out: &mut [u8],
) {
    debug_assert_eq!(out.len(), schema.len() * 4);
    let sample = create_sample(game, p0_color);
    for (chunk, feature) in out.chunks_exact_mut(4).zip(schema.iter()) {
        let value = sample.get(feature).copied().unwrap_or(0.0) as f32;
        chunk.copy_from_slice(&value.to_le_bytes());
    }
}

pub fn feature_ordering(num_players: usize, map_kind: MapKind) -> Vec<String> {
    let players = Color::ALL
        .into_iter()
        .take(num_players)
        .map(Player::simple)
        .collect();
    let mut game = Game::new(players, Some(0));
    if map_kind != MapKind::Base {
        game.state.board = Board::new(CatanMap::from_template(
            map_kind,
            NumberPlacement::OfficialSpiral,
        ));
        game.playable_actions = generate_playable_actions(&game.state);
    }
    create_sample(&game, game.state.colors[0])
        .keys()
        .cloned()
        .collect()
}

pub fn stable_schema_hash_bytes(bytes: &[u8]) -> String {
    const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x00000100000001b3;

    let mut hash = FNV_OFFSET_BASIS;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    format!("fnv1a64:{hash:016x}")
}

pub fn stable_string_schema_hash(items: &[String]) -> String {
    const SEPARATOR: u8 = 0x1f;
    let mut bytes = Vec::new();
    for item in items {
        bytes.extend_from_slice(item.as_bytes());
        bytes.push(SEPARATOR);
    }
    stable_schema_hash_bytes(&bytes)
}

#[cfg(feature = "python")]
pub mod python_bindings {
    use super::*;
    use pyo3::exceptions::{PyBufferError, PyValueError};
    use pyo3::ffi;
    use pyo3::prelude::*;
    use pyo3::sync::with_critical_section;
    use pyo3::types::{PyByteArray, PyBytes, PyDict};
    use std::slice;

    type BatchStatsResult = (usize, usize, usize, Vec<(String, usize)>);
    type BatchEnvObservation = (
        Vec<f32>,
        (usize, usize, usize, usize),
        Vec<u8>,
        (usize, usize),
        Vec<f32>,
        Vec<bool>,
        Vec<Option<String>>,
        Vec<String>,
    );
    type BatchEnvBytesObservation = (
        Py<PyBytes>,
        (usize, usize, usize, usize),
        Py<PyBytes>,
        (usize, usize),
        Vec<f32>,
        Vec<bool>,
        Vec<Option<String>>,
        Vec<String>,
    );
    type BatchEnvFeatureVectors = (Vec<f32>, (usize, usize));
    type BatchEnvFeatureBytes = (Py<PyBytes>, (usize, usize));
    type BatchEnvBufferSpecs = (
        (usize, usize, usize, usize),
        usize,
        (usize, usize),
        usize,
        (usize, usize),
        usize,
    );
    type BatchEnvIntoObservation = (
        (usize, usize, usize, usize),
        (usize, usize),
        Vec<f32>,
        Vec<bool>,
        Vec<Option<String>>,
        Vec<String>,
    );

    fn py_err(error: String) -> PyErr {
        PyValueError::new_err(error)
    }

    fn require_bytearray_len(
        name: &str,
        buffer: &Bound<'_, PyByteArray>,
        expected: usize,
    ) -> PyResult<()> {
        let result = with_critical_section(buffer, || {
            let actual = buffer.len();
            if actual != expected {
                return Err(format!(
                    "{name} length must be {expected} bytes; got {actual}"
                ));
            }
            Ok(())
        });
        result.map_err(PyValueError::new_err)
    }

    struct WritableRawBuffer {
        view: ffi::Py_buffer,
    }

    impl WritableRawBuffer {
        fn get(name: &str, buffer: &Bound<'_, PyAny>, expected: usize) -> PyResult<Self> {
            let py = buffer.py();
            let mut view = ffi::Py_buffer::new();
            let flags = ffi::PyBUF_WRITABLE | ffi::PyBUF_C_CONTIGUOUS;
            let result = unsafe { ffi::PyObject_GetBuffer(buffer.as_ptr(), &mut view, flags) };
            if result != 0 {
                return Err(PyErr::fetch(py));
            }

            let raw_buffer = Self { view };
            if raw_buffer.view.len < 0 {
                return Err(PyBufferError::new_err(format!(
                    "{name} length must be non-negative; got {}",
                    raw_buffer.view.len
                )));
            }
            let actual = raw_buffer.view.len as usize;
            if actual != 0 && raw_buffer.view.buf.is_null() {
                return Err(PyBufferError::new_err(format!(
                    "{name} must expose a non-null writable buffer"
                )));
            }
            if raw_buffer.view.readonly != 0 {
                return Err(PyBufferError::new_err(format!("{name} must be writable")));
            }
            if actual != expected {
                return Err(PyValueError::new_err(format!(
                    "{name} length must be {expected} bytes; got {actual}"
                )));
            }
            Ok(raw_buffer)
        }

        fn as_mut_bytes(&mut self) -> &mut [u8] {
            let len = self.view.len as usize;
            if len == 0 {
                &mut []
            } else {
                unsafe { slice::from_raw_parts_mut(self.view.buf.cast::<u8>(), len) }
            }
        }
    }

    impl Drop for WritableRawBuffer {
        fn drop(&mut self) {
            unsafe {
                ffi::PyBuffer_Release(&mut self.view);
            }
        }
    }

    fn parse_player_kind(color: Color, kind: &str) -> PyResult<Player> {
        match kind {
            "simple" => Ok(Player::simple(color)),
            "random" => Ok(Player::random(color)),
            "weighted_random" => Ok(Player::weighted_random(color)),
            "victory_point" => Ok(Player::victory_point(color)),
            "value" | "value_function" => Ok(Player::value_function(color)),
            "strategic" | "strategic_value" => Ok(Player::strategic_value(color)),
            "alpha_beta" => Ok(Player::alpha_beta(color, 1)),
            "strategic_alpha_beta" => Ok(Player::strategic_alpha_beta(color, 2)),
            "same_turn_alpha_beta" => Ok(Player::same_turn_alpha_beta(color, 1)),
            "strategic_playout" | "hybrid_rollout" | "improved" => {
                Ok(Player::strategic_playout(color, 16))
            }
            "ensemble" | "champ" | "champion" => Ok(Player::champion(color)),
            "playout" | "greedy_playout" | "mcts" => Ok(Player::playout(color, 1)),
            _ => Err(PyValueError::new_err(format!(
                "unknown player kind: {kind}"
            ))),
        }
    }

    #[derive(Clone, Copy)]
    enum ParsedPlayerKind {
        Simple,
        Random,
        WeightedRandom,
        VictoryPoint,
        ValueFunction,
        StrategicValue,
        AlphaBeta,
        StrategicAlphaBeta,
        SameTurnAlphaBeta,
        StrategicPlayout,
        Champion,
        Playout,
    }

    impl ParsedPlayerKind {
        fn parse(kind: &str) -> PyResult<Self> {
            match kind {
                "simple" => Ok(Self::Simple),
                "random" => Ok(Self::Random),
                "weighted_random" => Ok(Self::WeightedRandom),
                "victory_point" => Ok(Self::VictoryPoint),
                "value" | "value_function" => Ok(Self::ValueFunction),
                "strategic" | "strategic_value" => Ok(Self::StrategicValue),
                "alpha_beta" => Ok(Self::AlphaBeta),
                "strategic_alpha_beta" => Ok(Self::StrategicAlphaBeta),
                "same_turn_alpha_beta" => Ok(Self::SameTurnAlphaBeta),
                "strategic_playout" | "hybrid_rollout" | "improved" => Ok(Self::StrategicPlayout),
                "ensemble" | "champ" | "champion" => Ok(Self::Champion),
                "playout" | "greedy_playout" | "mcts" => Ok(Self::Playout),
                _ => Err(PyValueError::new_err(format!(
                    "unknown player kind: {kind}"
                ))),
            }
        }

        fn player(self, color: Color) -> Player {
            match self {
                Self::Simple => Player::simple(color),
                Self::Random => Player::random(color),
                Self::WeightedRandom => Player::weighted_random(color),
                Self::VictoryPoint => Player::victory_point(color),
                Self::ValueFunction => Player::value_function(color),
                Self::StrategicValue => Player::strategic_value(color),
                Self::AlphaBeta => Player::alpha_beta(color, 1),
                Self::StrategicAlphaBeta => Player::strategic_alpha_beta(color, 2),
                Self::SameTurnAlphaBeta => Player::same_turn_alpha_beta(color, 1),
                Self::StrategicPlayout => Player::strategic_playout(color, 16),
                Self::Champion => Player::champion(color),
                Self::Playout => Player::playout(color, 1),
            }
        }
    }

    fn parse_players(colors: Vec<String>, kinds: Vec<String>) -> PyResult<Vec<Player>> {
        if colors.len() != kinds.len() {
            return Err(PyValueError::new_err(format!(
                "colors and player_kinds must have the same length; got {} colors and {} player_kinds",
                colors.len(),
                kinds.len()
            )));
        }
        colors
            .iter()
            .zip(kinds.iter())
            .map(|(color, kind)| {
                parse_color(color)
                    .map_err(py_err)
                    .and_then(|c| parse_player_kind(c, kind.as_str()))
            })
            .collect()
    }

    fn parse_map_kind(value: Option<&str>) -> PyResult<MapKind> {
        match value.unwrap_or("BASE") {
            "BASE" => Ok(MapKind::Base),
            "TOURNAMENT" => Ok(MapKind::Tournament),
            "MINI" => Ok(MapKind::Mini),
            other => Err(PyValueError::new_err(format!(
                "unsupported map kind for Rust bindings: {other}"
            ))),
        }
    }

    fn parse_number_placement(value: Option<&str>) -> PyResult<NumberPlacement> {
        match value.unwrap_or("official_spiral") {
            "official_spiral" => Ok(NumberPlacement::OfficialSpiral),
            "random" => Ok(NumberPlacement::Random),
            other => Err(PyValueError::new_err(format!(
                "unsupported number placement for Rust bindings: {other}"
            ))),
        }
    }

    #[pyclass(name = "Game")]
    #[derive(Clone)]
    pub struct PyGame {
        pub game: Game,
    }

    #[pymethods]
    impl PyGame {
        #[new]
        #[pyo3(signature = (colors=None, seed=None, player_kind=None, discard_limit=7, friendly_robber=false, vps_to_win=10, map_kind=None, player_kinds=None, number_placement=None))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            colors: Option<Vec<String>>,
            seed: Option<u64>,
            player_kind: Option<String>,
            discard_limit: u8,
            friendly_robber: bool,
            vps_to_win: i16,
            map_kind: Option<String>,
            player_kinds: Option<Vec<String>>,
            number_placement: Option<String>,
        ) -> PyResult<Self> {
            let colors = colors.unwrap_or_else(|| {
                ["RED", "BLUE", "WHITE", "ORANGE"]
                    .iter()
                    .map(|value| value.to_string())
                    .collect()
            });
            let kinds = player_kinds.unwrap_or_else(|| {
                vec![player_kind.unwrap_or_else(|| "random".to_string()); colors.len()]
            });
            let players = parse_players(colors, kinds)?;
            let map_kind = parse_map_kind(map_kind.as_deref())?;
            let number_placement = parse_number_placement(number_placement.as_deref())?;
            Ok(Self {
                game: Game::with_options_and_map_options(
                    players,
                    seed,
                    discard_limit,
                    friendly_robber,
                    vps_to_win,
                    map_kind,
                    number_placement,
                ),
            })
        }

        #[staticmethod]
        #[pyo3(signature = (colors=None, seed=None))]
        fn simple(colors: Option<Vec<String>>, seed: Option<u64>) -> PyResult<Self> {
            Self::new(
                colors,
                seed,
                Some("simple".to_string()),
                7,
                false,
                10,
                None,
                None,
                None,
            )
        }

        #[staticmethod]
        #[pyo3(signature = (colors=None, seed=None))]
        fn random(colors: Option<Vec<String>>, seed: Option<u64>) -> PyResult<Self> {
            Self::new(
                colors,
                seed,
                Some("random".to_string()),
                7,
                false,
                10,
                None,
                None,
                None,
            )
        }

        fn play(&mut self) -> Option<String> {
            self.game.play().map(|color| color_name(color).to_string())
        }

        fn play_tick(&mut self) -> PyResult<String> {
            let record = self.game.play_tick().map_err(py_err)?;
            serde_json::to_string(&action_record_to_json_value(&record)).map_err(|error| {
                PyValueError::new_err(format!("failed to serialize action record: {error}"))
            })
        }

        fn play_until_color(&mut self, target_color: &str, turn_limit: usize) -> PyResult<usize> {
            let target_color = parse_color(target_color).map_err(py_err)?;
            let mut ticks = 0;
            while self.game.winning_color().is_none()
                && self.game.state.current_color() != target_color
                && self.game.state.num_turns < turn_limit
            {
                self.game.play_tick().map_err(py_err)?;
                ticks += 1;
            }
            Ok(ticks)
        }

        fn execute_json(&mut self, action_json: &str) -> PyResult<String> {
            let value: Value = serde_json::from_str(action_json)
                .map_err(|error| PyValueError::new_err(format!("invalid action JSON: {error}")))?;
            let action = action_from_json_value(&value).map_err(py_err)?;
            let record = self.game.execute(action, true, None).map_err(py_err)?;
            serde_json::to_string(&action_record_to_json_value(&record)).map_err(|error| {
                PyValueError::new_err(format!("failed to serialize action record: {error}"))
            })
        }

        fn current_color(&self) -> String {
            color_name(self.game.state.current_color()).to_string()
        }

        fn winning_color(&self) -> Option<String> {
            self.game
                .winning_color()
                .map(|color| color_name(color).to_string())
        }

        fn num_turns(&self) -> usize {
            self.game.state.num_turns
        }

        fn state_index(&self) -> usize {
            self.game.state.action_records.len()
        }

        fn playable_actions_json(&self) -> PyResult<String> {
            let value = Value::Array(
                self.game
                    .playable_actions
                    .iter()
                    .map(action_to_json_value)
                    .collect(),
            );
            serde_json::to_string(&value).map_err(|error| {
                PyValueError::new_err(format!("failed to serialize playable actions: {error}"))
            })
        }

        fn playable_action_indices(
            &self,
            player_colors: Vec<String>,
            map_kind: Option<String>,
        ) -> PyResult<Vec<usize>> {
            let colors = player_colors
                .iter()
                .map(|color| parse_color(color).map_err(py_err))
                .collect::<PyResult<Vec<_>>>()?;
            let kind = parse_map_kind(map_kind.as_deref())?;
            // Build the ActionSpace ONCE per call and reuse `.index(action)`
            // per action (the `legal_action_indices_with_space` /
            // `decision_context_json_value` pattern). Going through the
            // free-standing `action_space_index` helper instead rebuilt the
            // full ActionSpace for EVERY playable action -- O(legal_actions
            // x action_space) per call, measured 4771us at a 54-wide
            // placement root on the per-leaf evaluator path.
            let action_space = ActionSpace::new(&colors, kind);
            self.game
                .playable_actions
                .iter()
                .map(|action| {
                    action_space.index(action).ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "action missing from Rust action space: {action:?}"
                        ))
                    })
                })
                .collect()
        }

        fn legal_action_mask(
            &self,
            player_colors: Vec<String>,
            map_kind: Option<String>,
        ) -> PyResult<Vec<u8>> {
            let colors = player_colors
                .iter()
                .map(|color| parse_color(color).map_err(py_err))
                .collect::<PyResult<Vec<_>>>()?;
            let kind = parse_map_kind(map_kind.as_deref())?;
            legal_action_mask(&self.game, &colors, kind).map_err(py_err)
        }

        fn json_snapshot(&self) -> PyResult<String> {
            serde_json::to_string(&game_to_json_value(&self.game)).map_err(|error| {
                PyValueError::new_err(format!("failed to serialize game snapshot: {error}"))
            })
        }

        /// Return the observer-scoped 2p card-counting deduction surface.
        /// Opponent resources are derived from conservation; opponent
        /// face-down development-card identities and deck order stay hidden.
        fn public_card_deductions_json(&self, observer: &str) -> PyResult<String> {
            let observer = parse_color(observer).map_err(py_err)?;
            let value = public_card_deductions_to_json_value(&self.game, observer)
                .map_err(PyValueError::new_err)?;
            serde_json::to_string(&value).map_err(|error| {
                PyValueError::new_err(format!(
                    "failed to serialize public card deductions: {error}"
                ))
            })
        }

        fn sample_json(&self, p0_color: &str) -> PyResult<String> {
            let color = parse_color(p0_color).map_err(py_err)?;
            serde_json::to_string(&create_sample(&self.game, color)).map_err(|error| {
                PyValueError::new_err(format!("failed to serialize feature sample: {error}"))
            })
        }

        fn sample_vector(&self, p0_color: &str) -> PyResult<Vec<f64>> {
            let color = parse_color(p0_color).map_err(py_err)?;
            Ok(create_sample_vector(&self.game, color, None))
        }

        fn sample_vector_ordered(
            &self,
            p0_color: &str,
            ordering: Vec<String>,
        ) -> PyResult<Vec<f64>> {
            let color = parse_color(p0_color).map_err(py_err)?;
            Ok(create_sample_vector(&self.game, color, Some(&ordering)))
        }

        fn board_tensor_flat(
            &self,
            p0_color: &str,
            channels_first: bool,
        ) -> PyResult<(Vec<f64>, (usize, usize, usize))> {
            let color = parse_color(p0_color).map_err(py_err)?;
            Ok(create_board_tensor_flat(&self.game, color, channels_first))
        }

        // ---- AlphaZero-style MCTS bindings (additive; expose existing Rust) ----

        /// Deep, independent clone of this game. The make-or-break primitive for
        /// MCTS tree branching: mutating the copy never touches the original.
        fn copy(&self) -> PyGame {
            PyGame {
                game: self.game.clone(),
            }
        }

        /// `copy.copy(game)` support — delegates to `copy()`.
        fn __copy__(&self) -> PyGame {
            self.copy()
        }

        /// `copy.deepcopy(game, memo)` support — delegates to `copy()` (already deep).
        #[pyo3(signature = (_memo=None))]
        fn __deepcopy__(&self, _memo: Option<Py<PyAny>>) -> PyGame {
            self.copy()
        }

        /// Enumerate the weighted chance outcomes of executing `action_json` at the
        /// current (chance) node, using the existing `execute_spectrum`. Returns a
        /// JSON list of `{"probability": f64, "snapshot": <game_json>}` where each
        /// snapshot is the full resulting game state (same schema as `json_snapshot`).
        ///
        /// For non-chance (deterministic) actions this yields a single outcome with
        /// probability 1.0. For ROLL it yields the 11 dice totals (2..=12) with their
        /// probabilities; for MoveRobber-with-victim the 5 stolen-resource outcomes;
        /// for BuyDevelopmentCard the distinct remaining card draws.
        fn spectrum_json(&self, action_json: &str) -> PyResult<String> {
            let value: Value = serde_json::from_str(action_json)
                .map_err(|error| PyValueError::new_err(format!("invalid action JSON: {error}")))?;
            let action = action_from_json_value(&value).map_err(py_err)?;
            let outcomes = execute_spectrum(&self.game, &action);
            let array = Value::Array(
                outcomes
                    .iter()
                    .map(|(outcome_game, probability)| {
                        json!({
                            "probability": probability,
                            "snapshot": game_to_json_value(outcome_game),
                        })
                    })
                    .collect(),
            );
            serde_json::to_string(&array).map_err(|error| {
                PyValueError::new_err(format!("failed to serialize spectrum: {error}"))
            })
        }

        /// Re-derive the `outcome_index`-th chance outcome of `action_json` as a new
        /// independent `Game`, without serializing every sibling outcome. Parallel to
        /// `spectrum_json` (same ordering) but lighter for the MCTS hot loop: call
        /// `spectrum_json` once for the (probability, index) weights, then materialize
        /// only the sampled child via this method.
        fn apply_chance_outcome(
            &self,
            action_json: &str,
            outcome_index: usize,
        ) -> PyResult<PyGame> {
            let value: Value = serde_json::from_str(action_json)
                .map_err(|error| PyValueError::new_err(format!("invalid action JSON: {error}")))?;
            let action = action_from_json_value(&value).map_err(py_err)?;
            let mut outcomes = execute_spectrum(&self.game, &action);
            if outcome_index >= outcomes.len() {
                return Err(PyValueError::new_err(format!(
                    "outcome_index {outcome_index} out of range (0..{})",
                    outcomes.len()
                )));
            }
            let (game, _probability) = outcomes.swap_remove(outcome_index);
            Ok(PyGame { game })
        }

        /// One-round-trip expansion context for the current decision node.
        /// Returns JSON `{"current_color": str, "actions": [{"index": int,
        /// "action": [color, type, value], "spectrum": [f64, ...]?}, ...]}` where
        /// `index` is the action-space index (same indexing as
        /// `playable_action_indices` / `legal_action_mask` / `execute_action_index`)
        /// and `spectrum` is present only for chance-typed actions: the outcome
        /// probabilities in `spectrum_json` / `apply_chance_outcome(s_batch)` order.
        /// Replaces the playable_action_indices + playable_actions_json +
        /// per-action spectrum_json call chain of a node expansion; probabilities
        /// are derived from the state without executing any outcome.
        #[pyo3(signature = (player_colors, map_kind=None, include_spectrums=true))]
        fn decision_context_json(
            &self,
            player_colors: Vec<String>,
            map_kind: Option<String>,
            include_spectrums: bool,
        ) -> PyResult<String> {
            let colors = player_colors
                .iter()
                .map(|color| parse_color(color).map_err(py_err))
                .collect::<PyResult<Vec<_>>>()?;
            let kind = parse_map_kind(map_kind.as_deref())?;
            let context = decision_context_json_value(&self.game, &colors, kind, include_spectrums)
                .map_err(py_err)?;
            serde_json::to_string(&context).map_err(|error| {
                PyValueError::new_err(format!("failed to serialize decision context: {error}"))
            })
        }

        /// Materialize chance outcomes of `action_json` as independent `Game`s in
        /// ONE call. `outcome_indices=None` returns every outcome in `spectrum_json`
        /// order (e.g. all 11 ROLL children); explicit indices (repeats allowed)
        /// return exactly those. Computes the spectrum once, versus once per child
        /// with repeated `apply_chance_outcome` calls.
        #[pyo3(signature = (action_json, outcome_indices=None))]
        fn apply_chance_outcomes_batch(
            &self,
            action_json: &str,
            outcome_indices: Option<Vec<usize>>,
        ) -> PyResult<Vec<PyGame>> {
            let value: Value = serde_json::from_str(action_json)
                .map_err(|error| PyValueError::new_err(format!("invalid action JSON: {error}")))?;
            let action = action_from_json_value(&value).map_err(py_err)?;
            let games = execute_spectrum_games(&self.game, &action, outcome_indices.as_deref())
                .map_err(py_err)?;
            Ok(games.into_iter().map(|game| PyGame { game }).collect())
        }

        /// Execute the playable action identified by its action-space `action_index`
        /// (the same indexing used by `playable_action_indices` / `legal_action_mask`),
        /// avoiding a JSON round-trip in the MCTS hot loop. The index is resolved
        /// against the current `playable_actions` via the shared `ActionSpace`, so the
        /// chosen action is always legal. Chance results are sampled by the engine RNG
        /// (use `spectrum_json` / `apply_chance_outcome` for explicit chance expansion).
        fn execute_action_index(
            &mut self,
            action_index: usize,
            player_colors: Vec<String>,
            map_kind: Option<String>,
        ) -> PyResult<()> {
            let colors = player_colors
                .iter()
                .map(|color| parse_color(color).map_err(py_err))
                .collect::<PyResult<Vec<_>>>()?;
            let kind = parse_map_kind(map_kind.as_deref())?;
            let action_space = ActionSpace::new(&colors, kind);
            let action = self
                .game
                .playable_actions
                .iter()
                .find(|candidate| action_space.index(candidate) == Some(action_index))
                .cloned()
                .ok_or_else(|| {
                    PyValueError::new_err(format!(
                        "action index {action_index} is not a currently playable action"
                    ))
                })?;
            self.game.execute(action, true, None).map_err(py_err)?;
            Ok(())
        }

        /// Expose a player's full state including the hidden hand (resources and
        /// development cards), as JSON, for IS-MCTS determinization. Resource and
        /// dev-card vectors are length-5, ordered `[Wood, Brick, Sheep, Wheat, Ore]`
        /// and `[Knight, YearOfPlenty, Monopoly, RoadBuilding, VictoryPoint]`.
        fn player_state_json(&self, color: &str) -> PyResult<String> {
            let color = parse_color(color).map_err(py_err)?;
            let ps = self.game.state.player_state(color);
            let value = json!({
                "color": color_name(color),
                "victory_points": ps.victory_points,
                "actual_victory_points": ps.actual_victory_points,
                "roads_available": ps.roads_available,
                "settlements_available": ps.settlements_available,
                "cities_available": ps.cities_available,
                "has_road": ps.has_road,
                "has_army": ps.has_army,
                "has_rolled": ps.has_rolled,
                "has_played_development_card_in_turn": ps.has_played_development_card_in_turn,
                "resources": ps.resources,
                "dev_cards": ps.dev_cards,
                "played_dev_cards": ps.played_dev_cards,
                "longest_road_length": ps.longest_road_length,
            });
            serde_json::to_string(&value).map_err(|error| {
                PyValueError::new_err(format!("failed to serialize player state: {error}"))
            })
        }

        /// Overwrite a player's hidden hand (resources + development cards) for
        /// IS-MCTS determinization. Both vectors must be length-5 with the same
        /// ordering as `player_state_json`. This only mutates the hidden hand
        /// counters; it does not recompute derived public state (e.g. largest army),
        /// which is not part of the hidden information being sampled.
        fn set_player_hand(
            &mut self,
            color: &str,
            resources: Vec<u8>,
            dev_cards: Vec<u8>,
        ) -> PyResult<()> {
            if resources.len() != 5 {
                return Err(PyValueError::new_err(format!(
                    "resources must have length 5, got {}",
                    resources.len()
                )));
            }
            if dev_cards.len() != 5 {
                return Err(PyValueError::new_err(format!(
                    "dev_cards must have length 5, got {}",
                    dev_cards.len()
                )));
            }
            let color = parse_color(color).map_err(py_err)?;
            let ps = self.game.state.player_state_mut(color);
            ps.resources = [
                resources[0],
                resources[1],
                resources[2],
                resources[3],
                resources[4],
            ];
            ps.dev_cards = [
                dev_cards[0],
                dev_cards[1],
                dev_cards[2],
                dev_cards[3],
                dev_cards[4],
            ];
            Ok(())
        }

        /// Sample a complete, rules-consistent hidden world using only the
        /// current player's information set.  Unlike `set_player_hand`, this
        /// atomically repairs resource conservation, face-down development-card
        /// allocation, remaining deck order, hidden victory points, and root
        /// legal actions.  The authoritative game is not mutated.
        fn determinize_for_player(&self, observer_color: &str, seed: u64) -> PyResult<PyGame> {
            let observer = parse_color(observer_color).map_err(py_err)?;
            self.game
                .determinize_for_player(observer, seed)
                .map(|game| PyGame { game })
                .map_err(py_err)
        }

        /// Materialize public-belief BUY_DEVELOPMENT_CARD successors in one
        /// round trip. ``card_names`` order is preserved. Unlike
        /// ``apply_chance_outcomes_batch``, support and child states are
        /// independent of the authoritative hidden deck/allocation.
        fn apply_public_belief_development_draws(
            &self,
            action_json: &str,
            observer_color: &str,
            card_names: Vec<String>,
            seed: u64,
        ) -> PyResult<Vec<PyGame>> {
            let value: Value = serde_json::from_str(action_json)
                .map_err(|error| PyValueError::new_err(format!("invalid action JSON: {error}")))?;
            let action = action_from_json_value(&value).map_err(py_err)?;
            let observer = parse_color(observer_color).map_err(py_err)?;
            let cards = card_names
                .iter()
                .map(|name| parse_dev_card(name).map_err(py_err))
                .collect::<PyResult<Vec<_>>>()?;
            self.game
                .public_belief_development_draws(observer, &action, &cards, seed)
                .map(|games| games.into_iter().map(|game| PyGame { game }).collect())
                .map_err(py_err)
        }
    }

    #[pyclass(name = "BatchEnv")]
    pub struct PyBatchEnv {
        games: Vec<Game>,
        player_colors: Vec<Color>,
        player_kinds: Vec<ParsedPlayerKind>,
        seed: Option<u64>,
        discard_limit: u8,
        friendly_robber: bool,
        vps_to_win: i16,
        map_kind: MapKind,
        number_placement: NumberPlacement,
        channels_first: bool,
        turn_limit: usize,
        action_space: ActionSpace,
        feature_schema: Vec<String>,
        rewards: Vec<f32>,
    }

    impl PyBatchEnv {
        fn build_game(&self, env_index: usize, seed: Option<u64>) -> Game {
            let players = self
                .player_colors
                .iter()
                .copied()
                .zip(self.player_kinds.iter().copied())
                .map(|(color, kind)| kind.player(color))
                .collect::<Vec<_>>();
            let game_seed =
                seed.or_else(|| self.seed.map(|base| base.wrapping_add(env_index as u64)));
            Game::with_options_and_map_options(
                players,
                game_seed,
                self.discard_limit,
                self.friendly_robber,
                self.vps_to_win,
                self.map_kind,
                self.number_placement,
            )
        }

        fn done_for(&self, game: &Game) -> bool {
            game.winning_color().is_some() || game.state.num_turns >= self.turn_limit
        }

        fn buffer_specs_inner(&self) -> BatchEnvBufferSpecs {
            let observation_shape = {
                let (a, b, c) = board_tensor_shape(self.player_colors.len(), self.channels_first);
                (self.games.len(), a, b, c)
            };
            let observation_nbytes =
                self.games.len() * board_tensor_flat_len(self.player_colors.len()) * 4;
            let mask_shape = (self.games.len(), self.action_space.len());
            let mask_nbytes = self.games.len() * self.action_space.len();
            let feature_shape = (self.games.len(), self.feature_schema.len());
            let feature_nbytes = self.games.len() * self.feature_schema.len() * 4;
            (
                observation_shape,
                observation_nbytes,
                mask_shape,
                mask_nbytes,
                feature_shape,
                feature_nbytes,
            )
        }

        fn validate_observation_mask_buffers(
            &self,
            observation_buffer: &Bound<'_, PyByteArray>,
            legal_mask_buffer: &Bound<'_, PyByteArray>,
        ) -> PyResult<()> {
            let (_, observation_nbytes, _, mask_nbytes, _, _) = self.buffer_specs_inner();
            require_bytearray_len("observation_buffer", observation_buffer, observation_nbytes)?;
            require_bytearray_len("legal_mask_buffer", legal_mask_buffer, mask_nbytes)?;
            Ok(())
        }

        fn fill_observation_mask_raw_buffers(
            &self,
            observation_out: &mut [u8],
            legal_mask_out: &mut [u8],
        ) -> PyResult<BatchEnvIntoObservation> {
            let (_, observation_nbytes, _, mask_nbytes, _, _) = self.buffer_specs_inner();
            if observation_out.len() != observation_nbytes {
                return Err(PyValueError::new_err(format!(
                    "observation_buffer length must be {observation_nbytes} bytes; got {}",
                    observation_out.len()
                )));
            }
            if legal_mask_out.len() != mask_nbytes {
                return Err(PyValueError::new_err(format!(
                    "legal_mask_buffer length must be {mask_nbytes} bytes; got {}",
                    legal_mask_out.len()
                )));
            }

            let action_space_size = self.action_space.len();
            let observation_shape = {
                let (a, b, c) = board_tensor_shape(self.player_colors.len(), self.channels_first);
                (self.games.len(), a, b, c)
            };
            let mask_shape = (self.games.len(), action_space_size);
            let mut dones = Vec::with_capacity(self.games.len());
            let mut winners = Vec::with_capacity(self.games.len());
            let mut current_colors = Vec::with_capacity(self.games.len());

            for game in &self.games {
                let done = self.done_for(game);
                dones.push(done);
                winners.push(
                    game.winning_color()
                        .map(|color| color_name(color).to_string()),
                );
                current_colors.push(color_name(game.state.current_color()).to_string());
                if !done {
                    validate_legal_action_mask_with_space(game, &self.action_space)
                        .map_err(py_err)?;
                }
            }

            let samples = self
                .games
                .iter()
                .map(|game| (game, game.state.current_color()))
                .collect::<Vec<_>>();
            fill_board_tensor_batch_flat_f32_le_bytes(
                &samples,
                self.channels_first,
                observation_out,
            )
            .map_err(py_err)?;

            legal_mask_out.fill(0);
            for (env_index, game) in self.games.iter().enumerate() {
                if !dones[env_index] {
                    let start = env_index * action_space_size;
                    let end = start + action_space_size;
                    fill_legal_action_mask_assume_zeroed_with_space(
                        game,
                        &self.action_space,
                        &mut legal_mask_out[start..end],
                    )
                    .map_err(py_err)?;
                }
            }

            Ok((
                observation_shape,
                mask_shape,
                self.rewards.clone(),
                dones,
                winners,
                current_colors,
            ))
        }

        fn observe_buffers_into_inner(
            &self,
            observation_buffer: &Bound<'_, PyAny>,
            legal_mask_buffer: &Bound<'_, PyAny>,
        ) -> PyResult<BatchEnvIntoObservation> {
            let (mut observation_out, mut legal_mask_out) =
                self.writable_observation_mask_buffers(observation_buffer, legal_mask_buffer)?;
            self.fill_observation_mask_raw_buffers(
                observation_out.as_mut_bytes(),
                legal_mask_out.as_mut_bytes(),
            )
        }

        fn writable_observation_mask_buffers(
            &self,
            observation_buffer: &Bound<'_, PyAny>,
            legal_mask_buffer: &Bound<'_, PyAny>,
        ) -> PyResult<(WritableRawBuffer, WritableRawBuffer)> {
            let (_, observation_nbytes, _, mask_nbytes, _, _) = self.buffer_specs_inner();
            let observation_out = WritableRawBuffer::get(
                "observation_buffer",
                observation_buffer,
                observation_nbytes,
            )?;
            let legal_mask_out =
                WritableRawBuffer::get("legal_mask_buffer", legal_mask_buffer, mask_nbytes)?;
            Ok((observation_out, legal_mask_out))
        }

        fn feature_vectors_buffer_into_inner(
            &self,
            feature_buffer: &Bound<'_, PyAny>,
        ) -> PyResult<(usize, usize)> {
            let (_, _, _, _, _, feature_nbytes) = self.buffer_specs_inner();
            let mut feature_out =
                WritableRawBuffer::get("feature_buffer", feature_buffer, feature_nbytes)?;
            let samples = self
                .games
                .iter()
                .map(|game| (game, game.state.current_color()))
                .collect::<Vec<_>>();
            fill_sample_vector_batch_for_schema_le_bytes(
                &samples,
                &self.feature_schema,
                feature_out.as_mut_bytes(),
            )
            .map_err(py_err)
        }

        fn observe_inner(&self) -> PyResult<BatchEnvObservation> {
            let (observations, observation_shape) = if self.games.is_empty() {
                let (a, b, c) = board_tensor_shape(self.player_colors.len(), self.channels_first);
                (Vec::new(), (0, a, b, c))
            } else {
                create_current_board_tensor_batch_flat_f32(&self.games, self.channels_first)
            };
            let action_space_size = self.action_space.len();
            let mut legal_masks = vec![0; self.games.len() * action_space_size];
            let mut dones = Vec::with_capacity(self.games.len());
            let mut winners = Vec::with_capacity(self.games.len());
            let mut current_colors = Vec::with_capacity(self.games.len());

            for (env_index, game) in self.games.iter().enumerate() {
                let done = self.done_for(game);
                dones.push(done);
                winners.push(
                    game.winning_color()
                        .map(|color| color_name(color).to_string()),
                );
                current_colors.push(color_name(game.state.current_color()).to_string());
                if !done {
                    let start = env_index * action_space_size;
                    let end = start + action_space_size;
                    fill_legal_action_mask_assume_zeroed_with_space(
                        game,
                        &self.action_space,
                        &mut legal_masks[start..end],
                    )
                    .map_err(py_err)?;
                }
            }

            Ok((
                observations,
                observation_shape,
                legal_masks,
                (self.games.len(), action_space_size),
                self.rewards.clone(),
                dones,
                winners,
                current_colors,
            ))
        }

        fn observe_bytes_into_inner(
            &self,
            observation_buffer: &Bound<'_, PyByteArray>,
            legal_mask_buffer: &Bound<'_, PyByteArray>,
        ) -> PyResult<BatchEnvIntoObservation> {
            self.validate_observation_mask_buffers(observation_buffer, legal_mask_buffer)?;
            let action_space_size = self.action_space.len();
            let observation_shape = {
                let (a, b, c) = board_tensor_shape(self.player_colors.len(), self.channels_first);
                (self.games.len(), a, b, c)
            };
            let mask_shape = (self.games.len(), action_space_size);
            let mut dones = Vec::with_capacity(self.games.len());
            let mut winners = Vec::with_capacity(self.games.len());
            let mut current_colors = Vec::with_capacity(self.games.len());

            for game in &self.games {
                let done = self.done_for(game);
                dones.push(done);
                winners.push(
                    game.winning_color()
                        .map(|color| color_name(color).to_string()),
                );
                current_colors.push(color_name(game.state.current_color()).to_string());
                if !done {
                    validate_legal_action_mask_with_space(game, &self.action_space)
                        .map_err(py_err)?;
                }
            }

            let samples = self
                .games
                .iter()
                .map(|game| (game, game.state.current_color()))
                .collect::<Vec<_>>();
            let observation_result = with_critical_section(observation_buffer, || {
                let (_, observation_nbytes, _, _, _, _) = self.buffer_specs_inner();
                let actual = observation_buffer.len();
                if actual != observation_nbytes {
                    return Err(format!(
                        "observation_buffer length must be {observation_nbytes} bytes; got {actual}"
                    ));
                }
                // SAFETY: length was validated before taking the slice; this block only
                // writes raw little-endian f32 bytes and does not invoke Python APIs.
                let out = unsafe { observation_buffer.as_bytes_mut() };
                fill_board_tensor_batch_flat_f32_le_bytes(&samples, self.channels_first, out)
                    .map(|_| ())
            });
            observation_result.map_err(PyValueError::new_err)?;

            let mask_result = with_critical_section(legal_mask_buffer, || {
                let (_, _, _, mask_nbytes, _, _) = self.buffer_specs_inner();
                let actual = legal_mask_buffer.len();
                if actual != mask_nbytes {
                    return Err(format!(
                        "legal_mask_buffer length must be {mask_nbytes} bytes; got {actual}"
                    ));
                }
                // SAFETY: length was validated before taking the slice; this block only
                // writes bytes and does not invoke Python APIs.
                let out = unsafe { legal_mask_buffer.as_bytes_mut() };
                out.fill(0);
                for (env_index, game) in self.games.iter().enumerate() {
                    if !dones[env_index] {
                        let start = env_index * action_space_size;
                        let end = start + action_space_size;
                        fill_legal_action_mask_assume_zeroed_with_space(
                            game,
                            &self.action_space,
                            &mut out[start..end],
                        )?;
                    }
                }
                Ok(())
            });
            mask_result.map_err(PyValueError::new_err)?;

            Ok((
                observation_shape,
                mask_shape,
                self.rewards.clone(),
                dones,
                winners,
                current_colors,
            ))
        }

        fn observe_bytes_inner(&self, py: Python<'_>) -> PyResult<BatchEnvBytesObservation> {
            let (_, observation_nbytes, _, mask_nbytes, _, _) = self.buffer_specs_inner();
            let mut observation_bytes = vec![0; observation_nbytes];
            let mut legal_masks = vec![0; mask_nbytes];
            let (observation_shape, mask_shape, rewards, dones, winners, current_colors) =
                self.fill_observation_mask_raw_buffers(&mut observation_bytes, &mut legal_masks)?;
            Ok((
                PyBytes::new(py, &observation_bytes).unbind(),
                observation_shape,
                PyBytes::new(py, &legal_masks).unbind(),
                mask_shape,
                rewards,
                dones,
                winners,
                current_colors,
            ))
        }

        fn feature_vectors_inner(&self) -> PyResult<BatchEnvFeatureVectors> {
            let samples = self
                .games
                .iter()
                .map(|game| (game, game.state.current_color()))
                .collect::<Vec<_>>();
            let mut features = vec![0.0_f32; samples.len() * self.feature_schema.len()];
            let shape =
                fill_sample_vector_batch_for_schema(&samples, &self.feature_schema, &mut features)
                    .map_err(py_err)?;
            Ok((features, shape))
        }

        fn feature_vectors_bytes_inner(&self, py: Python<'_>) -> PyResult<BatchEnvFeatureBytes> {
            let (features, shape) = self.feature_vectors_inner()?;
            let mut feature_bytes = Vec::with_capacity(features.len() * 4);
            for value in features {
                feature_bytes.extend_from_slice(&value.to_le_bytes());
            }
            Ok((PyBytes::new(py, &feature_bytes).unbind(), shape))
        }

        fn feature_vectors_bytes_into_inner(
            &self,
            feature_buffer: &Bound<'_, PyByteArray>,
        ) -> PyResult<(usize, usize)> {
            let (_, _, _, _, _, feature_nbytes) = self.buffer_specs_inner();
            let samples = self
                .games
                .iter()
                .map(|game| (game, game.state.current_color()))
                .collect::<Vec<_>>();
            let result = with_critical_section(feature_buffer, || {
                let actual = feature_buffer.len();
                if actual != feature_nbytes {
                    return Err(format!(
                        "feature_buffer length must be {feature_nbytes} bytes; got {actual}"
                    ));
                }
                // SAFETY: length is validated before taking the slice; this block only
                // writes raw little-endian f32 bytes and does not invoke Python APIs.
                let out = unsafe { feature_buffer.as_bytes_mut() };
                fill_sample_vector_batch_for_schema_le_bytes(&samples, &self.feature_schema, out)
            });
            result.map_err(PyValueError::new_err)
        }

        fn reset_games(&mut self, seeds: Option<Vec<u64>>) -> PyResult<()> {
            if let Some(seeds) = &seeds
                && seeds.len() != self.games.len()
            {
                return Err(PyValueError::new_err(format!(
                    "seeds length must equal num_envs; got {} seeds for {} envs",
                    seeds.len(),
                    self.games.len()
                )));
            }
            let seeds_ref = seeds.as_deref();
            self.games = (0..self.games.len())
                .map(|index| {
                    self.build_game(
                        index,
                        seeds_ref.and_then(|values| values.get(index).copied()),
                    )
                })
                .collect();
            self.rewards.fill(0.0);
            Ok(())
        }

        fn step_actions(&mut self, action_indices: Vec<usize>) -> PyResult<()> {
            if action_indices.len() != self.games.len() {
                return Err(PyValueError::new_err(format!(
                    "action_indices length must equal num_envs; got {} actions for {} envs",
                    action_indices.len(),
                    self.games.len()
                )));
            }
            self.rewards.fill(0.0);
            for (env_index, action_index) in action_indices.into_iter().enumerate() {
                if self.done_for(&self.games[env_index]) {
                    continue;
                }
                let actor = self.games[env_index].state.current_color();
                let action = self.games[env_index]
                    .playable_actions
                    .iter()
                    .find(|action| self.action_space.index(action) == Some(action_index))
                    .cloned()
                    .ok_or_else(|| {
                        PyValueError::new_err(format!(
                            "illegal action index {action_index} for env {env_index}"
                        ))
                    })?;
                self.games[env_index]
                    .execute(action, true, None)
                    .map_err(py_err)?;
                if let Some(winner) = self.games[env_index].winning_color() {
                    self.rewards[env_index] = if winner == actor { 1.0 } else { -1.0 };
                }
            }
            Ok(())
        }
    }

    #[pymethods]
    impl PyBatchEnv {
        #[new]
        #[pyo3(signature = (num_envs, colors=None, seed=None, player_kind=None, discard_limit=7, friendly_robber=false, vps_to_win=10, map_kind=None, player_kinds=None, number_placement=None, channels_first=false, turn_limit=1000))]
        #[allow(clippy::too_many_arguments)]
        fn new(
            num_envs: usize,
            colors: Option<Vec<String>>,
            seed: Option<u64>,
            player_kind: Option<String>,
            discard_limit: u8,
            friendly_robber: bool,
            vps_to_win: i16,
            map_kind: Option<String>,
            player_kinds: Option<Vec<String>>,
            number_placement: Option<String>,
            channels_first: bool,
            turn_limit: usize,
        ) -> PyResult<Self> {
            let color_names = colors.unwrap_or_else(|| {
                ["RED", "BLUE", "WHITE", "ORANGE"]
                    .iter()
                    .map(|value| value.to_string())
                    .collect()
            });
            let player_colors = color_names
                .iter()
                .map(|color| parse_color(color).map_err(py_err))
                .collect::<PyResult<Vec<_>>>()?;
            let kind_names = player_kinds.unwrap_or_else(|| {
                vec![player_kind.unwrap_or_else(|| "simple".to_string()); player_colors.len()]
            });
            if kind_names.len() != player_colors.len() {
                return Err(PyValueError::new_err(format!(
                    "colors and player_kinds must have the same length; got {} colors and {} player_kinds",
                    player_colors.len(),
                    kind_names.len()
                )));
            }
            let player_kinds = kind_names
                .iter()
                .map(|kind| ParsedPlayerKind::parse(kind))
                .collect::<PyResult<Vec<_>>>()?;
            let map_kind = parse_map_kind(map_kind.as_deref())?;
            let number_placement = parse_number_placement(number_placement.as_deref())?;
            let action_space = ActionSpace::new(&player_colors, map_kind);
            let feature_schema = feature_ordering(player_colors.len(), map_kind);
            let mut env = Self {
                games: Vec::new(),
                player_colors,
                player_kinds,
                seed,
                discard_limit,
                friendly_robber,
                vps_to_win,
                map_kind,
                number_placement,
                channels_first,
                turn_limit,
                action_space,
                feature_schema,
                rewards: vec![0.0; num_envs],
            };
            env.games = (0..num_envs)
                .map(|index| env.build_game(index, None))
                .collect();
            Ok(env)
        }

        fn action_space_len(&self) -> usize {
            self.action_space.len()
        }

        fn action_space_json(&self) -> PyResult<String> {
            serde_json::to_string(&self.action_space.json_value()).map_err(|error| {
                PyValueError::new_err(format!("failed to serialize action space: {error}"))
            })
        }

        fn action_space_hash(&self) -> PyResult<String> {
            let value = serde_json::to_vec(&self.action_space.json_value()).map_err(|error| {
                PyValueError::new_err(format!("failed to serialize action space: {error}"))
            })?;
            Ok(stable_schema_hash_bytes(&value))
        }

        fn feature_ordering(&self) -> Vec<String> {
            self.feature_schema.clone()
        }

        fn feature_schema_hash(&self) -> String {
            stable_string_schema_hash(&self.feature_schema)
        }

        fn feature_vectors(&self) -> PyResult<BatchEnvFeatureVectors> {
            self.feature_vectors_inner()
        }

        fn feature_vectors_bytes(&self, py: Python<'_>) -> PyResult<BatchEnvFeatureBytes> {
            self.feature_vectors_bytes_inner(py)
        }

        fn feature_vectors_bytes_into(
            &self,
            feature_buffer: &Bound<'_, PyByteArray>,
        ) -> PyResult<(usize, usize)> {
            self.feature_vectors_bytes_into_inner(feature_buffer)
        }

        fn feature_vectors_into_buffer(
            &self,
            feature_buffer: &Bound<'_, PyAny>,
        ) -> PyResult<(usize, usize)> {
            self.feature_vectors_buffer_into_inner(feature_buffer)
        }

        fn byte_buffer_layout(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
            let (
                observation_shape,
                observation_nbytes,
                mask_shape,
                mask_nbytes,
                feature_shape,
                feature_nbytes,
            ) = self.buffer_specs_inner();
            let layout = PyDict::new(py);
            layout.set_item("observation_shape", observation_shape)?;
            layout.set_item("observations_nbytes", observation_nbytes)?;
            layout.set_item("observation_dtype", "float32")?;
            layout.set_item("mask_shape", mask_shape)?;
            layout.set_item("legal_masks_nbytes", mask_nbytes)?;
            layout.set_item("mask_dtype", "uint8")?;
            layout.set_item("feature_shape", feature_shape)?;
            layout.set_item("features_nbytes", feature_nbytes)?;
            layout.set_item("feature_dtype", "float32")?;
            Ok(layout.unbind())
        }

        fn observe(&self) -> PyResult<BatchEnvObservation> {
            self.observe_inner()
        }

        fn observe_bytes(&self, py: Python<'_>) -> PyResult<BatchEnvBytesObservation> {
            self.observe_bytes_inner(py)
        }

        fn observe_bytes_into(
            &self,
            observation_buffer: &Bound<'_, PyByteArray>,
            legal_mask_buffer: &Bound<'_, PyByteArray>,
        ) -> PyResult<BatchEnvIntoObservation> {
            self.observe_bytes_into_inner(observation_buffer, legal_mask_buffer)
        }

        fn observe_into_buffer(
            &self,
            observation_buffer: &Bound<'_, PyAny>,
            legal_mask_buffer: &Bound<'_, PyAny>,
        ) -> PyResult<BatchEnvIntoObservation> {
            self.observe_buffers_into_inner(observation_buffer, legal_mask_buffer)
        }

        #[pyo3(signature = (seeds=None))]
        fn reset(&mut self, seeds: Option<Vec<u64>>) -> PyResult<BatchEnvObservation> {
            self.reset_games(seeds)?;
            self.observe_inner()
        }

        #[pyo3(signature = (seeds=None))]
        fn reset_bytes(
            &mut self,
            py: Python<'_>,
            seeds: Option<Vec<u64>>,
        ) -> PyResult<BatchEnvBytesObservation> {
            self.reset_games(seeds)?;
            self.observe_bytes_inner(py)
        }

        #[pyo3(signature = (observation_buffer, legal_mask_buffer, seeds=None))]
        fn reset_bytes_into(
            &mut self,
            observation_buffer: &Bound<'_, PyByteArray>,
            legal_mask_buffer: &Bound<'_, PyByteArray>,
            seeds: Option<Vec<u64>>,
        ) -> PyResult<BatchEnvIntoObservation> {
            self.validate_observation_mask_buffers(observation_buffer, legal_mask_buffer)?;
            self.reset_games(seeds)?;
            self.observe_bytes_into_inner(observation_buffer, legal_mask_buffer)
        }

        #[pyo3(signature = (observation_buffer, legal_mask_buffer, seeds=None))]
        fn reset_into_buffer(
            &mut self,
            observation_buffer: &Bound<'_, PyAny>,
            legal_mask_buffer: &Bound<'_, PyAny>,
            seeds: Option<Vec<u64>>,
        ) -> PyResult<BatchEnvIntoObservation> {
            let (mut observation_out, mut legal_mask_out) =
                self.writable_observation_mask_buffers(observation_buffer, legal_mask_buffer)?;
            self.reset_games(seeds)?;
            self.fill_observation_mask_raw_buffers(
                observation_out.as_mut_bytes(),
                legal_mask_out.as_mut_bytes(),
            )
        }

        fn step(&mut self, action_indices: Vec<usize>) -> PyResult<BatchEnvObservation> {
            self.step_actions(action_indices)?;
            self.observe_inner()
        }

        fn step_bytes(
            &mut self,
            py: Python<'_>,
            action_indices: Vec<usize>,
        ) -> PyResult<BatchEnvBytesObservation> {
            self.step_actions(action_indices)?;
            self.observe_bytes_inner(py)
        }

        fn step_bytes_into(
            &mut self,
            action_indices: Vec<usize>,
            observation_buffer: &Bound<'_, PyByteArray>,
            legal_mask_buffer: &Bound<'_, PyByteArray>,
        ) -> PyResult<BatchEnvIntoObservation> {
            self.validate_observation_mask_buffers(observation_buffer, legal_mask_buffer)?;
            self.step_actions(action_indices)?;
            self.observe_bytes_into_inner(observation_buffer, legal_mask_buffer)
        }

        fn step_into_buffer(
            &mut self,
            action_indices: Vec<usize>,
            observation_buffer: &Bound<'_, PyAny>,
            legal_mask_buffer: &Bound<'_, PyAny>,
        ) -> PyResult<BatchEnvIntoObservation> {
            let (mut observation_out, mut legal_mask_out) =
                self.writable_observation_mask_buffers(observation_buffer, legal_mask_buffer)?;
            self.step_actions(action_indices)?;
            self.fill_observation_mask_raw_buffers(
                observation_out.as_mut_bytes(),
                legal_mask_out.as_mut_bytes(),
            )
        }
    }

    #[pyfunction]
    fn action_from_json(action_json: &str) -> PyResult<String> {
        let value: Value = serde_json::from_str(action_json)
            .map_err(|error| PyValueError::new_err(format!("invalid action JSON: {error}")))?;
        let action = action_from_json_value(&value).map_err(py_err)?;
        serde_json::to_string(&action_to_json_value(&action))
            .map_err(|error| PyValueError::new_err(format!("failed to serialize action: {error}")))
    }

    #[pyfunction]
    fn feature_ordering_py(num_players: usize, map_kind: Option<String>) -> PyResult<Vec<String>> {
        let kind = parse_map_kind(map_kind.as_deref())?;
        Ok(feature_ordering(num_players, kind))
    }

    #[pyfunction]
    fn action_space_json(player_colors: Vec<String>, map_kind: Option<String>) -> PyResult<String> {
        let colors = player_colors
            .iter()
            .map(|color| parse_color(color).map_err(py_err))
            .collect::<PyResult<Vec<_>>>()?;
        let kind = parse_map_kind(map_kind.as_deref())?;
        serde_json::to_string(&action_space_json_value(&colors, kind)).map_err(|error| {
            PyValueError::new_err(format!("failed to serialize action space: {error}"))
        })
    }

    #[pyfunction]
    #[pyo3(signature = (num_games, colors, player_kinds, seed=None, discard_limit=7, friendly_robber=false, vps_to_win=10, map_kind=None, number_placement=None))]
    #[allow(clippy::too_many_arguments)]
    fn simulate_batch_stats(
        py: Python<'_>,
        num_games: usize,
        colors: Vec<String>,
        player_kinds: Vec<String>,
        seed: Option<u64>,
        discard_limit: u8,
        friendly_robber: bool,
        vps_to_win: i16,
        map_kind: Option<String>,
        number_placement: Option<String>,
    ) -> PyResult<BatchStatsResult> {
        if colors.len() != player_kinds.len() {
            return Err(PyValueError::new_err(format!(
                "colors and player_kinds must have the same length; got {} colors and {} player_kinds",
                colors.len(),
                player_kinds.len()
            )));
        }
        let colors = colors
            .iter()
            .map(|color| parse_color(color).map_err(py_err))
            .collect::<PyResult<Vec<_>>>()?;
        let player_kinds = player_kinds
            .iter()
            .map(|kind| ParsedPlayerKind::parse(kind))
            .collect::<PyResult<Vec<_>>>()?;
        let map_kind = parse_map_kind(map_kind.as_deref())?;
        let number_placement = parse_number_placement(number_placement.as_deref())?;

        let (wins, turns, wins_by_color) = py.detach(move || {
            (0..num_games)
                .into_par_iter()
                .map(|game_index| {
                    let players = colors
                        .iter()
                        .copied()
                        .zip(player_kinds.iter().copied())
                        .map(|(color, kind)| kind.player(color))
                        .collect::<Vec<_>>();
                    let game_seed = seed.map(|base| base.wrapping_add(game_index as u64));
                    let mut game = Game::with_options_and_map_options(
                        players,
                        game_seed,
                        discard_limit,
                        friendly_robber,
                        vps_to_win,
                        map_kind,
                        number_placement,
                    );
                    let winner = game.play();
                    let mut wins_by_color = BTreeMap::new();
                    if let Some(color) = winner {
                        wins_by_color.insert(color, 1usize);
                    }
                    (
                        usize::from(winner.is_some()),
                        game.state.num_turns,
                        wins_by_color,
                    )
                })
                .reduce(
                    || (0usize, 0usize, BTreeMap::<Color, usize>::new()),
                    |mut left, right| {
                        left.0 += right.0;
                        left.1 += right.1;
                        for (color, wins) in right.2 {
                            *left.2.entry(color).or_insert(0) += wins;
                        }
                        left
                    },
                )
        });
        let wins_by_color = wins_by_color
            .into_iter()
            .map(|(color, wins)| (color_name(color).to_string(), wins))
            .collect();
        Ok((num_games, wins, turns, wins_by_color))
    }

    // ---- Entity-token featurization (Rust port, task #81 phases 1-2) ----
    //
    // Bit-exact companion to `catan_zero.rl.entity_token_features.build_entity_token_features`
    // for the specific "Rust-MCTS adapter" payload shape produced by
    // `neural_rust_mcts._entity_payload_from_rust_snapshot`. Legacy calls keep
    // event tokens/masks all-zero; next-wave calls opt into the same bounded
    // meaningful-public-action history encoded from `state.action_records`.
    //
    // One known, pre-existing quirk of the Python reference this function
    // reproduces verbatim (do not fix here without also fixing the Python
    // reference + re-running the parity suite): the trade-action one-hot slots
    // of `legal_action_tokens` (cols 2..20,
    //      the ACTION_TYPES index) never fire for OFFER_TRADE/ACCEPT_TRADE/
    //      REJECT_TRADE/CANCEL_TRADE/CONFIRM_TRADE: Python's `ACTION_TYPES`
    //      tuple spells these lowercase ("offer_trade", ...) while the actual
    //      action-type string is uppercase, so the case-sensitive match always
    //      misses. Category classification is unaffected (separate substring
    //      check). `current_prompt` has an analogous mismatch: this engine's
    //      `DECIDE_TRADE`/`DECIDE_ACCEPTEES` prompt names never match any of
    //      Python's 8 `PROMPTS` substrings, so the prompt one-hot in
    //      `global_tokens` is all-zero whenever the game is in either prompt.
    //
    // Board/port TOPOLOGY (hex_vertex_ids/hex_edge_ids/edge_vertex_ids/
    // port_base_nodes, bundled as `EntityTopology`) is intentionally NOT
    // recomputed here: the Python reference derives it from a fixed lookup
    // keyed by hex coordinate (`entity_token_features._base_tile_topology`/
    // `_base_ports`, built once from a canonical 2p BASE-map environment), not
    // from the live board's own node/edge ids -- an existing quirk that is
    // only topologically correct for BASE-layout boards. Callers build ONE
    // `EntityTopology` per board/search and pass it by reference to every
    // per-leaf/per-batch call (the SAME arrays the Python topology cache
    // already produces and reuses across the whole game), so this Rust layer
    // reads only the LIVE, per-call resource/number/robber/building/road/
    // player/action state directly off each game, matching the Python
    // reference's split between "topology" (fixed lookup) and "tiles" (live
    // `tile_id`/`coordinate`/`resource`/`number`/`has_robber`, sourced from
    // the live snapshot in both implementations).
    //
    // NEITHER function below needs an `ActionSpace`/`legal_action_ids`
    // parameter: every production call site (`evaluate`/`evaluate_many`/
    // `evaluate_symmetry_averaged` in `neural_rust_mcts.py`, traced back
    // through `_fetch_legal_actions`/`_legal_action_indices`/
    // `decision_context_json` in `gumbel_chance_mcts.py`) always evaluates
    // the FULL native `playable_actions` set, in that Vec's own order, never
    // a filtered/reordered subset -- and `rust_policy_action_ids`'s output is
    // index-aligned to that same order (it's built by zipping
    // `playable_action_indices()`'s output, which itself preserves
    // `playable_actions`' iteration order, against `legal_actions`). So
    // `policy_action_ids[i]` is trusted to already correspond to
    // `game.playable_actions[i]` -- no rust-action-space id translation is
    // needed on this side of the boundary at all (confirmed with
    // speed-czar/team-lead before dropping it).
    #[pyclass(name = "EntityTopology")]
    #[derive(Clone)]
    pub struct PyEntityTopology {
        hex_vertex_ids: Vec<Vec<i16>>,
        hex_edge_ids: Vec<Vec<i16>>,
        edge_vertex_ids: Vec<Vec<i16>>,
        port_base_nodes: Vec<Vec<i16>>,
    }

    #[pymethods]
    impl PyEntityTopology {
        #[new]
        fn new(
            hex_vertex_ids: Vec<Vec<i16>>,
            hex_edge_ids: Vec<Vec<i16>>,
            edge_vertex_ids: Vec<Vec<i16>>,
            port_base_nodes: Vec<Vec<i16>>,
        ) -> PyResult<Self> {
            if hex_vertex_ids.len() != 19 || hex_edge_ids.len() != 19 {
                return Err(PyValueError::new_err(
                    "hex_vertex_ids/hex_edge_ids must have 19 rows",
                ));
            }
            Ok(Self {
                hex_vertex_ids,
                hex_edge_ids,
                edge_vertex_ids,
                port_base_nodes,
            })
        }
    }

    const ENTITY_HEX_FEATURE_SIZE: usize = 13;
    const ENTITY_VERTEX_FEATURE_SIZE: usize = 24;
    const ENTITY_EDGE_FEATURE_SIZE: usize = 8;
    const ENTITY_PLAYER_FEATURE_SIZE: usize = 31;
    const ENTITY_GLOBAL_FEATURE_SIZE: usize = 43;
    const ENTITY_LEGAL_ACTION_FEATURE_SIZE: usize = 50;
    const ENTITY_EVENT_FEATURE_SIZE: usize = 41;
    const ENTITY_EVENT_HISTORY_LIMIT: usize = 64;
    const ENTITY_MEANINGFUL_PUBLIC_HISTORY_LIMIT: usize = 32;
    const ENTITY_ADAPTER_V2: &str = "rust_entity_adapter_v2_land_topology_ports_maritime";
    const ENTITY_ADAPTER_V3: &str = "rust_entity_adapter_v3_structured_action_resources";
    const ENTITY_PLAYERS_ORDER: [&str; 4] = ["BLUE", "RED", "ORANGE", "WHITE"];
    const ENTITY_PROMPTS: [&str; 8] = [
        "BUILD_INITIAL_SETTLEMENT",
        "BUILD_INITIAL_ROAD",
        "ROLL",
        "PLAY_TURN",
        "DISCARD",
        "MOVE_ROBBER",
        "RESPOND_TO_TRADE",
        "CONFIRM_TRADE",
    ];
    const ENTITY_ACTION_TYPES: [&str; 18] = [
        "BUILD_SETTLEMENT",
        "BUILD_ROAD",
        "BUILD_CITY",
        "BUY_DEVELOPMENT_CARD",
        "MARITIME_TRADE",
        "offer_trade",
        "accept_trade",
        "reject_trade",
        "cancel_trade",
        "confirm_trade",
        "MOVE_ROBBER",
        "DISCARD_RESOURCE",
        "PLAY_KNIGHT_CARD",
        "PLAY_YEAR_OF_PLENTY",
        "PLAY_MONOPOLY",
        "PLAY_ROAD_BUILDING",
        "ROLL",
        "END_TURN",
    ];

    fn entity_adapter_encodes_action_resources(version: &str) -> PyResult<bool> {
        match version {
            ENTITY_ADAPTER_V2 => Ok(false),
            ENTITY_ADAPTER_V3 => Ok(true),
            _ => Err(PyValueError::new_err(format!(
                "unknown entity feature adapter version {version:?}; expected {ENTITY_ADAPTER_V2:?} or {ENTITY_ADAPTER_V3:?}"
            ))),
        }
    }

    fn entity_action_resource_bundle(action: &Action, enabled: bool) -> [f64; 5] {
        let mut bundle = [0.0; 5];
        if !enabled {
            return bundle;
        }
        match (&action.action_type, &action.value) {
            (
                ActionType::DiscardResource | ActionType::PlayMonopoly,
                ActionValue::Resource(resource),
            ) => bundle[resource.idx()] = 0.5,
            (ActionType::PlayYearOfPlenty, ActionValue::Resources(resources)) => {
                for resource in resources {
                    bundle[resource.idx()] = (bundle[resource.idx()] + 0.5f64).min(1.0);
                }
            }
            _ => {}
        }
        bundle
    }
    const ENTITY_CATEGORIES: [&str; 5] = ["build", "trade", "development", "robber", "turn"];

    fn entity_player_index(name: &str) -> Option<usize> {
        ENTITY_PLAYERS_ORDER.iter().position(|&p| p == name)
    }
    fn entity_scale(value: i64, denominator: f64) -> f64 {
        let denom = denominator.max(1.0);
        (value as f64 / denom).clamp(0.0, 1.0)
    }
    fn entity_dice_pips(number: Option<u8>) -> i64 {
        match number {
            None => 0,
            Some(7) => 0,
            Some(n) => (6 - (n as i64 - 7).abs()).max(0),
        }
    }
    fn entity_action_category(action_type: &str) -> &'static str {
        if matches!(
            action_type,
            "BUILD_SETTLEMENT" | "BUILD_ROAD" | "BUILD_CITY"
        ) {
            "build"
        } else if action_type.contains("TRADE") {
            "trade"
        } else if action_type.contains("ROBBER") || action_type == "PLAY_KNIGHT_CARD" {
            "robber"
        } else if action_type.contains("DEVELOPMENT") || action_type.starts_with("PLAY_") {
            "development"
        } else {
            "turn"
        }
    }
    fn entity_action_priority(action_type: &str) -> f64 {
        match action_type {
            "BUILD_CITY" => 1.00,
            "BUILD_SETTLEMENT" => 0.90,
            "BUILD_ROAD" => 0.70,
            "BUY_DEVELOPMENT_CARD" => 0.60,
            "MOVE_ROBBER" => 0.35,
            "END_TURN" => 0.0,
            _ => 0.5,
        }
    }

    fn entity_is_meaningful_public_action(action_type: ActionType) -> bool {
        matches!(
            action_type,
            ActionType::BuildSettlement
                | ActionType::BuildRoad
                | ActionType::BuildCity
                | ActionType::BuyDevelopmentCard
                | ActionType::MaritimeTrade
                | ActionType::MoveRobber
                | ActionType::DiscardResource
                | ActionType::PlayKnightCard
                | ActionType::PlayYearOfPlenty
                | ActionType::PlayMonopoly
                | ActionType::PlayRoadBuilding
        )
    }

    /// Per-game entity-feature arrays, unpadded (`legal_action_*` widths vary
    /// with `n_legal`; everything else is fixed-size). Pure Rust, no PyO3
    /// types -- safe to build in parallel (rayon) or sequentially alike.
    struct EntityFeatureArrays {
        hex_tokens: Vec<f64>,
        vertex_tokens: Vec<f64>,
        edge_tokens: Vec<f64>,
        player_tokens: Vec<f64>,
        global_tokens: Vec<f64>,
        legal_action_tokens: Vec<f64>,
        legal_action_target_ids: Vec<i64>,
        event_tokens: Vec<f64>,
        event_target_ids: Vec<i64>,
        hex_mask: Vec<bool>,
        vertex_mask: Vec<bool>,
        edge_mask: Vec<bool>,
        player_mask: Vec<bool>,
        legal_action_mask: Vec<bool>,
        event_mask: Vec<bool>,
        n_legal: usize,
        event_history_limit: usize,
    }

    /// Core per-game builder shared by the single-item and batch pyfunctions.
    /// `colors` is the CALLER's fixed-order color tuple -- deliberately not
    /// re-derived from `game.state.colors`, because `State::new` shuffles
    /// player order at game creation, so the live internal order can differ
    /// from the order every caller already uses to key `policy_action_ids`
    /// and the `players` payload (`_resolve_entity_adapter`'s
    /// `states_by_color`). `policy_action_ids` must be index-aligned to
    /// `game.playable_actions`'s own order (see the module-level doc comment
    /// above for why no separate action-id translation happens here).
    fn build_entity_feature_arrays(
        game: &Game,
        colors: &[Color],
        policy_action_ids: &[i64],
        action_size: i64,
        topology: &PyEntityTopology,
        public_observation: bool,
        meaningful_public_history: bool,
        requested_event_history_limit: usize,
        encode_structured_action_resources: bool,
    ) -> PyResult<EntityFeatureArrays> {
        let state = &game.state;
        let actions = &game.playable_actions;
        if actions.len() != policy_action_ids.len() {
            return Err(PyValueError::new_err(format!(
                "policy_action_ids length ({}) must match game.playable_actions length ({})",
                policy_action_ids.len(),
                actions.len()
            )));
        }
        let n_legal = actions.len();
        let perspective_color = state.current_color();
        if requested_event_history_limit > ENTITY_EVENT_HISTORY_LIMIT {
            return Err(PyValueError::new_err(format!(
                "event_history_limit must be <= {ENTITY_EVENT_HISTORY_LIMIT}, got {requested_event_history_limit}"
            )));
        }
        let event_history_limit = if meaningful_public_history {
            requested_event_history_limit.min(ENTITY_MEANINGFUL_PUBLIC_HISTORY_LIMIT)
        } else {
            requested_event_history_limit
        };

        // ---- live per-coordinate tile id lookup (mirrors Python's
        // `_build_topology`'s `coordinate_to_hex`, sourced from THIS game's
        // own tile_id/coordinate, not the fixed base topology). ----
        let mut tiles_by_id: Vec<(Coordinate, &LandTile)> = state
            .board
            .map
            .land_tiles
            .iter()
            .map(|(coordinate, tile)| (*coordinate, tile))
            .filter(|(_, tile)| tile.id < 19)
            .collect();
        tiles_by_id.sort_by_key(|(_, tile)| tile.id);
        let mut coordinate_to_hex: HashMap<Coordinate, i64> = HashMap::new();
        for (coordinate, tile) in &tiles_by_id {
            coordinate_to_hex.insert(*coordinate, tile.id as i64);
        }
        let robber_tile_id: Option<usize> = state
            .board
            .map
            .land_tiles
            .get(&state.board.robber_coordinate)
            .map(|tile| tile.id)
            .filter(|&id| id < 19);

        // ---- hex_tokens (19, ENTITY_HEX_FEATURE_SIZE) ----
        let mut hex_tokens = vec![0.0f64; 19 * ENTITY_HEX_FEATURE_SIZE];
        for (coordinate, tile) in &tiles_by_id {
            let base = tile.id * ENTITY_HEX_FEATURE_SIZE;
            hex_tokens[base] = 1.0;
            hex_tokens[base + 1] = coordinate.0 as f64 / 4.0;
            hex_tokens[base + 2] = coordinate.1 as f64 / 4.0;
            hex_tokens[base + 3] = coordinate.2 as f64 / 4.0;
            match tile.resource {
                None => hex_tokens[base + 9] = 1.0,
                Some(resource) => hex_tokens[base + 4 + resource.idx()] = 1.0,
            }
            hex_tokens[base + 10] = entity_scale(tile.number.unwrap_or(0) as i64, 12.0);
            hex_tokens[base + 11] = entity_scale(entity_dice_pips(tile.number), 5.0);
            hex_tokens[base + 12] = if Some(tile.id) == robber_tile_id {
                1.0
            } else {
                0.0
            };
        }

        // ---- port_by_node: fixed port->node topology (input) joined with
        // this game's LIVE per-port resource assignment. ----
        let mut port_by_node: HashMap<usize, Option<Resource>> = HashMap::new();
        for (&port_id, port) in state.board.map.ports_by_id.iter() {
            if let Some(nodes) = topology.port_base_nodes.get(port_id) {
                for &node in nodes {
                    if node >= 0 {
                        port_by_node.insert(node as usize, port.resource);
                    }
                }
            }
        }

        // ---- vertex_tokens (54, ENTITY_VERTEX_FEATURE_SIZE) ----
        let mut vertex_tokens = vec![0.0f64; 54 * ENTITY_VERTEX_FEATURE_SIZE];
        for node in 0..54usize {
            let base = node * ENTITY_VERTEX_FEATURE_SIZE;
            vertex_tokens[base] = 1.0;
            let building = state.board.buildings.get(&node);
            let owner = building.map(|(color, _)| *color);
            match owner {
                None => vertex_tokens[base + 1] = 1.0,
                Some(color) => {
                    if let Some(idx) = entity_player_index(color_name(color)) {
                        vertex_tokens[base + 2 + idx] = 1.0;
                    }
                }
            }
            match building.map(|(_, building_type)| *building_type) {
                Some(BuildingType::Settlement) => vertex_tokens[base + 7] = 1.0,
                Some(BuildingType::City) => vertex_tokens[base + 8] = 1.0,
                _ => vertex_tokens[base + 6] = 1.0,
            }
            let mut pips_by_resource = [0i64; 5];
            for (_, tile) in &tiles_by_id {
                if topology.hex_vertex_ids[tile.id]
                    .iter()
                    .any(|&n| n as i64 == node as i64)
                    && let Some(resource) = tile.resource
                {
                    pips_by_resource[resource.idx()] += entity_dice_pips(tile.number);
                }
            }
            let total_pips: i64 = pips_by_resource.iter().sum();
            vertex_tokens[base + 9] = entity_scale(total_pips, 18.0);
            for (idx, &pips) in pips_by_resource.iter().enumerate() {
                vertex_tokens[base + 10 + idx] = entity_scale(pips, 10.0);
            }
            let adjacent_robber = robber_tile_id
                .map(|tile_id| {
                    topology.hex_vertex_ids[tile_id]
                        .iter()
                        .any(|&n| n as i64 == node as i64)
                })
                .unwrap_or(false);
            vertex_tokens[base + 15] = if adjacent_robber { 1.0 } else { 0.0 };
            match port_by_node.get(&node) {
                None => vertex_tokens[base + 16] = 1.0,
                Some(None) => vertex_tokens[base + 17] = 1.0,
                Some(Some(resource)) => vertex_tokens[base + 18 + resource.idx()] = 1.0,
            }
            let is_actor = owner
                .map(|color| color == perspective_color)
                .unwrap_or(false);
            vertex_tokens[base + 23] = if is_actor { 1.0 } else { 0.0 };
        }

        // ---- edge_tokens (72, ENTITY_EDGE_FEATURE_SIZE); edge_to_id is the
        // reverse of the fixed `edge_vertex_ids` topology (input). ----
        let mut edge_to_id: HashMap<(i16, i16), usize> = HashMap::new();
        for (edge_id, pair) in topology.edge_vertex_ids.iter().enumerate() {
            if pair.len() == 2 && pair[0] >= 0 && pair[1] >= 0 {
                let a = pair[0].min(pair[1]);
                let b = pair[0].max(pair[1]);
                edge_to_id.insert((a, b), edge_id);
            }
        }
        let mut edge_tokens = vec![0.0f64; 72 * ENTITY_EDGE_FEATURE_SIZE];
        for (&(a, b), &edge_id) in edge_to_id.iter() {
            if edge_id >= 72 {
                continue;
            }
            let base = edge_id * ENTITY_EDGE_FEATURE_SIZE;
            edge_tokens[base] = 1.0;
            let owner = state.board.roads.get(&(a as usize, b as usize)).copied();
            match owner {
                None => edge_tokens[base + 1] = 1.0,
                Some(color) => {
                    if let Some(idx) = entity_player_index(color_name(color)) {
                        edge_tokens[base + 2 + idx] = 1.0;
                    }
                }
            }
            let adjacent_hex_count = topology
                .hex_edge_ids
                .iter()
                .flatten()
                .filter(|&&e| e as i64 == edge_id as i64)
                .count();
            edge_tokens[base + 6] = entity_scale(adjacent_hex_count as i64, 2.0);
            let is_actor = owner
                .map(|color| color == perspective_color)
                .unwrap_or(false);
            edge_tokens[base + 7] = if is_actor { 1.0 } else { 0.0 };
        }

        // ---- player_tokens (4, ENTITY_PLAYER_FEATURE_SIZE) ----
        let mut player_tokens = vec![0.0f64; 4 * ENTITY_PLAYER_FEATURE_SIZE];
        for &color in colors {
            let idx = match entity_player_index(color_name(color)) {
                Some(idx) => idx,
                None => continue,
            };
            let ps = state.player_state(color);
            let base = idx * ENTITY_PLAYER_FEATURE_SIZE;
            let masked = public_observation && color != perspective_color;
            player_tokens[base] = 1.0;
            player_tokens[base + 1] = if color == perspective_color { 1.0 } else { 0.0 };
            player_tokens[base + 2] = if color == state.current_color() {
                1.0
            } else {
                0.0
            };
            player_tokens[base + 3] = entity_scale(ps.victory_points as i64, 10.0);
            if !masked {
                player_tokens[base + 4] = 1.0;
                player_tokens[base + 5] = entity_scale(ps.actual_victory_points as i64, 10.0);
            }
            let resource_card_count: i64 = ps.resources.iter().map(|&count| count as i64).sum();
            let dev_card_count: i64 = ps.dev_cards.iter().map(|&count| count as i64).sum();
            player_tokens[base + 6] = entity_scale(resource_card_count, 20.0);
            player_tokens[base + 7] = entity_scale(dev_card_count, 10.0);
            player_tokens[base + 8] = entity_scale(ps.roads_available as i64, 15.0);
            player_tokens[base + 9] = entity_scale(ps.settlements_available as i64, 5.0);
            player_tokens[base + 10] = entity_scale(ps.cities_available as i64, 4.0);
            player_tokens[base + 11] = if ps.has_army { 1.0 } else { 0.0 };
            // Longest-road ownership is public and authoritative in PlayerState.
            player_tokens[base + 12] = if ps.has_road { 1.0 } else { 0.0 };
            player_tokens[base + 13] = if ps.has_rolled { 1.0 } else { 0.0 };
            player_tokens[base + 14] = entity_scale(ps.longest_road_length as i64, 15.0);
            if !masked {
                player_tokens[base + 15] = 1.0;
                for (offset, &count) in ps.resources.iter().enumerate() {
                    player_tokens[base + 16 + offset] = entity_scale(count as i64, 10.0);
                }
            }
            if !masked {
                player_tokens[base + 21] = 1.0;
                for offset in 0..5 {
                    player_tokens[base + 22 + offset] =
                        entity_scale(ps.dev_cards[offset] as i64, 5.0);
                }
            }
            for offset in 0..4 {
                player_tokens[base + 27 + offset] =
                    entity_scale(ps.played_dev_cards[offset] as i64, 5.0);
            }
        }

        // ---- global_tokens (1, ENTITY_GLOBAL_FEATURE_SIZE) ----
        let mut global_tokens = vec![0.0f64; ENTITY_GLOBAL_FEATURE_SIZE];
        let prompt_name = action_prompt_name(state.current_prompt);
        for (idx, known) in ENTITY_PROMPTS.iter().enumerate() {
            global_tokens[idx] = if prompt_name.contains(known) {
                1.0
            } else {
                0.0
            };
        }
        if let Some(idx) = entity_player_index(color_name(state.current_color())) {
            global_tokens[16 + idx] = 1.0;
        }
        if let Some(idx) = entity_player_index(color_name(perspective_color)) {
            global_tokens[20 + idx] = 1.0;
        }
        global_tokens[24] = entity_scale(n_legal as i64, 607.0);
        global_tokens[25] = entity_scale(state.action_records.len() as i64, 512.0);
        for (offset, &count) in state.resource_freqdeck.iter().enumerate() {
            global_tokens[26 + offset] = entity_scale(count as i64, 19.0);
        }
        global_tokens[31] = entity_scale(state.development_listdeck.len() as i64, 25.0);
        global_tokens[32] = entity_scale(0, 3.0); // trade_panel.offers_remaining: hardcoded 0 in the adapter
        global_tokens[33] = 0.0; // trade_panel.current_offer: hardcoded None in the adapter
        global_tokens[34] = if state.is_resolving_trade { 1.0 } else { 0.0 };
        for (offset, &name) in ENTITY_PLAYERS_ORDER.iter().enumerate() {
            let present = colors.iter().any(|&color| color_name(color) == name);
            global_tokens[35 + offset] = if present { 1.0 } else { 0.0 };
        }
        let player_count = colors.len();
        if (2..=4).contains(&player_count) {
            global_tokens[39 + (player_count - 2)] = 1.0;
        }

        // ---- legal_action_tokens (n_legal, ENTITY_LEGAL_ACTION_FEATURE_SIZE)
        // and legal_action_target_ids (n_legal, 4) ----
        let is_initial_prompt = prompt_name.contains("INITIAL");
        let mut legal_action_tokens = vec![0.0f64; n_legal * ENTITY_LEGAL_ACTION_FEATURE_SIZE];
        let mut legal_action_target_ids = vec![-1i64; n_legal * 4];
        for (row, action) in actions.iter().enumerate() {
            let base = row * ENTITY_LEGAL_ACTION_FEATURE_SIZE;
            let target_base = row * 4;
            legal_action_tokens[base] = 1.0;
            legal_action_tokens[base + 1] =
                entity_scale(policy_action_ids[row], action_size.max(1) as f64);
            let type_name = action_type_name(action.action_type);
            if let Some(type_index) = ENTITY_ACTION_TYPES
                .iter()
                .position(|&entry| entry == type_name)
            {
                legal_action_tokens[base + 2 + type_index] = 1.0;
            }
            let category = entity_action_category(type_name);
            if let Some(cat_index) = ENTITY_CATEGORIES
                .iter()
                .position(|&entry| entry == category)
            {
                legal_action_tokens[base + 20 + cat_index] = 1.0;
            }
            // target_kind priority (matches Python's `_target_kind`):
            // tile_coordinate > node > edge > victim/target > resource > none.
            let kind_index = match (action.action_type, &action.value) {
                (ActionType::MoveRobber, ActionValue::Robber(coordinate, victim)) => {
                    if let Some(&tile_id) = coordinate_to_hex.get(coordinate) {
                        legal_action_target_ids[target_base] = tile_id;
                    }
                    if let Some(color) = victim
                        && let Some(pidx) = entity_player_index(color_name(*color))
                    {
                        legal_action_target_ids[target_base + 3] = pidx as i64;
                    }
                    1 // "hex"
                }
                (ActionType::BuildSettlement, ActionValue::Node(node))
                | (ActionType::BuildCity, ActionValue::Node(node)) => {
                    legal_action_target_ids[target_base + 1] = *node as i64;
                    2 // "vertex"
                }
                (ActionType::BuildRoad, ActionValue::Edge(edge)) => {
                    let (a, b) = canonical_edge(*edge);
                    if let Some(&edge_id) = edge_to_id.get(&(a as i16, b as i16)) {
                        legal_action_target_ids[target_base + 2] = edge_id as i64;
                    }
                    3 // "edge"
                }
                (ActionType::DiscardResource, ActionValue::Resource(_))
                | (ActionType::PlayMonopoly, ActionValue::Resource(_)) => 5, // "resource"
                (ActionType::PlayYearOfPlenty, ActionValue::Resources(_))
                    if encode_structured_action_resources =>
                {
                    5
                }
                _ => 0, // "none" -- includes MARITIME_TRADE: `_target_kind` never
                        // inspects "give"/"want", only "node"/"edge"/"tile_coordinate"/
                        // "victim"/"resource"/"resources", none of which
                        // `_structured_action` sets for MARITIME_TRADE.
            };
            legal_action_tokens[base + 25 + kind_index] = 1.0;
            for (offset, value) in
                entity_action_resource_bundle(action, encode_structured_action_resources)
                    .iter()
                    .enumerate()
            {
                legal_action_tokens[base + 31 + offset] = *value;
            }
            if let ActionValue::MaritimeTrade(offering, asking) = &action.value {
                for resource in offering.iter().flatten() {
                    let slot = base + 36 + resource.idx();
                    legal_action_tokens[slot] = (legal_action_tokens[slot] + 0.25).min(1.0);
                }
                legal_action_tokens[base + 41 + asking.idx()] = 0.25;
            }
            legal_action_tokens[base + 46] = entity_action_priority(type_name);
            legal_action_tokens[base + 47] = if type_name == "END_TURN" { 1.0 } else { 0.0 };
            legal_action_tokens[base + 48] = if is_initial_prompt { 1.0 } else { 0.0 };
            legal_action_tokens[base + 49] = entity_scale(0, 3.0); // trade_panel.offers_remaining: hardcoded 0
        }

        // ---- event tokens/masks: optional bounded meaningful PUBLIC history.
        // Default-off preserves the historical all-zero 64-token surface.
        // The enabled path reads only action type/color and the public robber
        // victim; it never serializes discard resources, bought dev identity,
        // stolen resource, or any other ActionRecord result/value secret. ----
        let mut event_tokens = vec![0.0f64; event_history_limit * ENTITY_EVENT_FEATURE_SIZE];
        let mut event_target_ids = vec![-1i64; event_history_limit * 4];
        let mut event_mask = vec![false; event_history_limit];
        if meaningful_public_history && event_history_limit > 0 {
            let meaningful = state
                .action_records
                .iter()
                .enumerate()
                .filter(|(index, record)| {
                    entity_is_meaningful_public_action(record.action.action_type)
                        // Only a known-public width of one suppresses an event.
                        // Missing/zero metadata is private or legacy and must
                        // remain in history rather than leak a hidden-hand bit.
                        && state
                            .action_public_legal_counts
                            .get(*index)
                            .copied()
                            .unwrap_or(0)
                            != 1
                })
                .map(|(_, record)| record)
                .collect::<Vec<_>>();
            let retained = meaningful
                .len()
                .min(event_history_limit)
                .min(ENTITY_MEANINGFUL_PUBLIC_HISTORY_LIMIT);
            let source_start = meaningful.len().saturating_sub(retained);
            let row_start = event_history_limit - retained;
            for (index, record) in meaningful[source_start..].iter().enumerate() {
                let row = row_start + index;
                let base = row * ENTITY_EVENT_FEATURE_SIZE;
                event_mask[row] = true;
                event_tokens[base] = 1.0;
                event_tokens[base + 1] =
                    entity_scale((retained - index) as i64, event_history_limit as f64);
                // EVENT_TYPES[1] == "board_action" in the Python reference.
                event_tokens[base + 3] = 1.0;
                if let Some(actor_index) = entity_player_index(color_name(record.action.color)) {
                    event_tokens[base + 10 + actor_index] = 1.0;
                }
                let type_name = action_type_name(record.action.action_type);
                if let Some(type_index) = ENTITY_ACTION_TYPES
                    .iter()
                    .position(|&entry| entry == type_name)
                {
                    event_tokens[base + 17 + type_index] = 1.0;
                }
                let target_base = row * 4;
                match &record.action.value {
                    ActionValue::Node(node)
                        if matches!(
                            record.action.action_type,
                            ActionType::BuildSettlement | ActionType::BuildCity
                        ) =>
                    {
                        event_target_ids[target_base + 1] = *node as i64;
                    }
                    ActionValue::Edge(edge)
                        if record.action.action_type == ActionType::BuildRoad =>
                    {
                        let (a, b) = canonical_edge(*edge);
                        if let Some(&edge_id) = edge_to_id.get(&(a as i16, b as i16)) {
                            event_target_ids[target_base + 2] = edge_id as i64;
                        }
                    }
                    ActionValue::Robber(coordinate, victim)
                        if record.action.action_type == ActionType::MoveRobber =>
                    {
                        if let Some(&tile_id) = coordinate_to_hex.get(coordinate) {
                            event_target_ids[target_base] = tile_id;
                        }
                        if let Some(victim) = victim
                            && let Some(target_index) = entity_player_index(color_name(*victim))
                        {
                            event_target_ids[target_base + 3] = target_index as i64;
                        }
                    }
                    _ => {}
                }
                if event_target_ids[target_base] >= 0 {
                    event_tokens[base + 14] = entity_scale(event_target_ids[target_base], 19.0);
                } else if event_target_ids[target_base + 1] >= 0 {
                    event_tokens[base + 14] = entity_scale(event_target_ids[target_base + 1], 54.0);
                } else if event_target_ids[target_base + 2] >= 0 {
                    event_tokens[base + 14] = entity_scale(event_target_ids[target_base + 2], 72.0);
                }
                if let ActionValue::Robber(_, Some(victim)) = &record.action.value
                    && let Some(target_index) = entity_player_index(color_name(*victim))
                {
                    event_tokens[base + 36 + target_index] = 1.0;
                }
            }
        }

        // ---- masks ----
        let hex_mask = vec![true; 19];
        let vertex_mask = vec![true; 54];
        let edge_mask = vec![true; 72];
        let mut player_mask = vec![false; 4];
        for &color in colors {
            if let Some(idx) = entity_player_index(color_name(color)) {
                player_mask[idx] = true;
            }
        }
        let legal_action_mask = vec![true; n_legal];

        Ok(EntityFeatureArrays {
            hex_tokens,
            vertex_tokens,
            edge_tokens,
            player_tokens,
            global_tokens,
            legal_action_tokens,
            legal_action_target_ids,
            event_tokens,
            event_target_ids,
            hex_mask,
            vertex_mask,
            edge_mask,
            player_mask,
            legal_action_mask,
            event_mask,
            n_legal,
            event_history_limit,
        })
    }

    // ---- byte-buffer marshalling (task #81 "numpy-reshape" lever) ----
    //
    // Returning `Vec<f64>`/`Vec<i64>` directly makes PyO3 build a Python
    // LIST, boxing every element into its own fresh `PyFloat`/`PyLong`
    // object one at a time -- measured as roughly half the wall-clock cost
    // of the Python-wrapped call (see the entity/context bench reports).
    // Converting to raw little-endian bytes (a Python `bytes` object, ONE
    // bulk copy, no per-element object allocation) and reconstructing via
    // `np.frombuffer(..., dtype="<f8"/"<i8").reshape(...)` on the Python
    // side is numerically IDENTICAL -- same f64/i64 values, same eventual
    // `.astype(float16/int16)` cast -- so this cannot change any output
    // value, only how it's marshalled. (Explicit little-endian, not a raw
    // pointer/native-endianness reinterpret: safe with no `unsafe` code, and
    // every host this crate runs on today -- Apple Silicon dev machines,
    // B200/A100 GPU hosts -- is little-endian anyway.) Masks stay as
    // `Vec<bool>`: Python `bool` is a singleton (`True`/`False`), so
    // converting those to a list is already a cheap refcount bump per
    // element, not a fresh allocation -- no benefit to bytes there.
    fn f64_vec_to_le_bytes(values: Vec<f64>) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    }
    fn i64_vec_to_le_bytes(values: Vec<i64>) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(values.len() * 8);
        for v in values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        bytes
    }

    fn entity_feature_arrays_to_pydict(
        py: Python<'_>,
        arrays: EntityFeatureArrays,
    ) -> PyResult<Py<PyDict>> {
        let n_legal = arrays.n_legal;
        let event_history_limit = arrays.event_history_limit;
        let dict = PyDict::new(py);
        dict.set_item(
            "hex_tokens",
            (
                f64_vec_to_le_bytes(arrays.hex_tokens),
                (19usize, ENTITY_HEX_FEATURE_SIZE),
            ),
        )?;
        dict.set_item(
            "vertex_tokens",
            (
                f64_vec_to_le_bytes(arrays.vertex_tokens),
                (54usize, ENTITY_VERTEX_FEATURE_SIZE),
            ),
        )?;
        dict.set_item(
            "edge_tokens",
            (
                f64_vec_to_le_bytes(arrays.edge_tokens),
                (72usize, ENTITY_EDGE_FEATURE_SIZE),
            ),
        )?;
        dict.set_item(
            "player_tokens",
            (
                f64_vec_to_le_bytes(arrays.player_tokens),
                (4usize, ENTITY_PLAYER_FEATURE_SIZE),
            ),
        )?;
        dict.set_item(
            "global_tokens",
            (
                f64_vec_to_le_bytes(arrays.global_tokens),
                (1usize, ENTITY_GLOBAL_FEATURE_SIZE),
            ),
        )?;
        dict.set_item(
            "legal_action_tokens",
            (
                f64_vec_to_le_bytes(arrays.legal_action_tokens),
                (n_legal, ENTITY_LEGAL_ACTION_FEATURE_SIZE),
            ),
        )?;
        dict.set_item(
            "legal_action_target_ids",
            (
                i64_vec_to_le_bytes(arrays.legal_action_target_ids),
                (n_legal, 4usize),
            ),
        )?;
        dict.set_item(
            "event_tokens",
            (
                f64_vec_to_le_bytes(arrays.event_tokens),
                (event_history_limit, ENTITY_EVENT_FEATURE_SIZE),
            ),
        )?;
        dict.set_item(
            "event_target_ids",
            (
                i64_vec_to_le_bytes(arrays.event_target_ids),
                (event_history_limit, 4usize),
            ),
        )?;
        dict.set_item("hex_mask", arrays.hex_mask)?;
        dict.set_item("vertex_mask", arrays.vertex_mask)?;
        dict.set_item("edge_mask", arrays.edge_mask)?;
        dict.set_item("player_mask", arrays.player_mask)?;
        dict.set_item("legal_action_mask", arrays.legal_action_mask)?;
        dict.set_item("event_mask", arrays.event_mask)?;
        Ok(dict.into())
    }

    fn parse_colors_list(colors: &[String]) -> PyResult<Vec<Color>> {
        colors
            .iter()
            .map(|value| parse_color(value).map_err(py_err))
            .collect()
    }

    #[pyfunction]
    #[pyo3(signature = (game, colors, policy_action_ids, action_size, topology, public_observation=false, meaningful_public_history=false, event_history_limit=64, entity_feature_adapter_version=ENTITY_ADAPTER_V3))]
    fn build_entity_features_flat(
        py: Python<'_>,
        game: &PyGame,
        colors: Vec<String>,
        policy_action_ids: Vec<i64>,
        action_size: i64,
        topology: &PyEntityTopology,
        public_observation: bool,
        meaningful_public_history: bool,
        event_history_limit: usize,
        entity_feature_adapter_version: &str,
    ) -> PyResult<Py<PyDict>> {
        let colors = parse_colors_list(&colors)?;
        let encode_structured_action_resources =
            entity_adapter_encodes_action_resources(entity_feature_adapter_version)?;
        let arrays = build_entity_feature_arrays(
            &game.game,
            &colors,
            &policy_action_ids,
            action_size,
            topology,
            public_observation,
            meaningful_public_history,
            event_history_limit,
            encode_structured_action_resources,
        )?;
        entity_feature_arrays_to_pydict(py, arrays)
    }

    fn pad_and_stack_f64(
        rows: &[Vec<f64>],
        widths: &[usize],
        feature_size: usize,
        max_width: usize,
    ) -> Vec<f64> {
        let mut out = vec![0.0f64; rows.len() * max_width * feature_size];
        for (game_index, (row, &width)) in rows.iter().zip(widths.iter()).enumerate() {
            let dest_base = game_index * max_width * feature_size;
            let copy_len = width * feature_size;
            out[dest_base..dest_base + copy_len].copy_from_slice(&row[..copy_len]);
        }
        out
    }

    fn pad_and_stack_i64(
        rows: &[Vec<i64>],
        widths: &[usize],
        feature_size: usize,
        max_width: usize,
    ) -> Vec<i64> {
        let mut out = vec![-1i64; rows.len() * max_width * feature_size];
        for (game_index, (row, &width)) in rows.iter().zip(widths.iter()).enumerate() {
            let dest_base = game_index * max_width * feature_size;
            let copy_len = width * feature_size;
            out[dest_base..dest_base + copy_len].copy_from_slice(&row[..copy_len]);
        }
        out
    }

    fn pad_and_stack_bool(rows: &[Vec<bool>], widths: &[usize], max_width: usize) -> Vec<bool> {
        let mut out = vec![false; rows.len() * max_width];
        for (game_index, (row, &width)) in rows.iter().zip(widths.iter()).enumerate() {
            let dest_base = game_index * max_width;
            out[dest_base..dest_base + width].copy_from_slice(&row[..width]);
        }
        out
    }

    /// Batched companion to `build_entity_features_flat`: one call builds
    /// entity features for MANY games sharing the same board/colors/action
    /// catalog (a Gumbel Sequential-Halving wave, a chance-node ROLL/robber/
    /// dev-card expansion, ...), stacking fixed-size arrays with a leading
    /// batch dimension and padding the ragged `legal_action_*` arrays to the
    /// batch's own max width -- so the Python side does zero re-stacking,
    /// which is where wave-batching gets its win (see speed-czar/team-lead
    /// batch-API discussion, task #81). Returns the same keys as
    /// `build_entity_features_flat` (now with a leading batch dim) plus
    /// `widths`: the true (unpadded) legal-action count per game, needed to
    /// mask off the padding downstream.
    ///
    /// `parallel` is OPT-IN and OFF by default: production self-play workers
    /// are single-core-pinned processes today, and rayon inside each would
    /// oversubscribe. It becomes valuable once evaluation moves to one
    /// process per GPU with a shared model (the phase-2/3 thread
    /// architecture) -- plumbed now, default sequential until then.
    #[pyfunction]
    #[pyo3(signature = (games, colors, policy_action_ids, action_size, topology, public_observation=false, parallel=false, meaningful_public_history=false, event_history_limit=64, entity_feature_adapter_version=ENTITY_ADAPTER_V3))]
    #[allow(clippy::too_many_arguments)]
    fn build_entity_features_batch(
        py: Python<'_>,
        games: Vec<Py<PyGame>>,
        colors: Vec<String>,
        policy_action_ids: Vec<Vec<i64>>,
        action_size: i64,
        topology: &PyEntityTopology,
        public_observation: bool,
        parallel: bool,
        meaningful_public_history: bool,
        event_history_limit: usize,
        entity_feature_adapter_version: &str,
    ) -> PyResult<Py<PyDict>> {
        if games.len() != policy_action_ids.len() {
            return Err(PyValueError::new_err(format!(
                "games and policy_action_ids must be the same length (got {} and {})",
                games.len(),
                policy_action_ids.len()
            )));
        }
        let colors = parse_colors_list(&colors)?;
        let encode_structured_action_resources =
            entity_adapter_encodes_action_resources(entity_feature_adapter_version)?;
        let batch_size = games.len();

        // Borrow every game's inner `Game` up front (needs the GIL) so the
        // (optionally parallel) compute loop below never touches PyO3 types.
        let borrowed: Vec<PyRef<'_, PyGame>> = games.iter().map(|g| g.borrow(py)).collect();
        let game_refs: Vec<&Game> = borrowed.iter().map(|g| &g.game).collect();

        let arrays: Vec<EntityFeatureArrays> = if parallel {
            game_refs
                .par_iter()
                .zip(policy_action_ids.par_iter())
                .map(|(game, policy_ids)| {
                    build_entity_feature_arrays(
                        game,
                        &colors,
                        policy_ids,
                        action_size,
                        topology,
                        public_observation,
                        meaningful_public_history,
                        event_history_limit,
                        encode_structured_action_resources,
                    )
                })
                .collect::<PyResult<Vec<_>>>()?
        } else {
            game_refs
                .iter()
                .zip(policy_action_ids.iter())
                .map(|(game, policy_ids)| {
                    build_entity_feature_arrays(
                        game,
                        &colors,
                        policy_ids,
                        action_size,
                        topology,
                        public_observation,
                        meaningful_public_history,
                        event_history_limit,
                        encode_structured_action_resources,
                    )
                })
                .collect::<PyResult<Vec<_>>>()?
        };

        // Consume `arrays` BY VALUE (`.into_iter()`) and move each field
        // exactly once via `.extend()` -- an earlier draft cloned every
        // fixed-size array field of every game before concatenating, which
        // measured SLOWER than N individual single-item calls (the clone
        // overhead ate the entire win from removing per-call PyDict
        // construction). No behavior change, just removes the redundant
        // heap copies.
        let batch_len = arrays.len();
        let effective_event_history_limit = arrays
            .first()
            .map(|arrays| arrays.event_history_limit)
            .unwrap_or_else(|| {
                if meaningful_public_history {
                    event_history_limit.min(ENTITY_MEANINGFUL_PUBLIC_HISTORY_LIMIT)
                } else {
                    event_history_limit
                }
            });
        let mut widths: Vec<usize> = Vec::with_capacity(batch_len);
        let mut hex_tokens = Vec::with_capacity(batch_len * 19 * ENTITY_HEX_FEATURE_SIZE);
        let mut vertex_tokens = Vec::with_capacity(batch_len * 54 * ENTITY_VERTEX_FEATURE_SIZE);
        let mut edge_tokens = Vec::with_capacity(batch_len * 72 * ENTITY_EDGE_FEATURE_SIZE);
        let mut player_tokens = Vec::with_capacity(batch_len * 4 * ENTITY_PLAYER_FEATURE_SIZE);
        let mut global_tokens = Vec::with_capacity(batch_len * ENTITY_GLOBAL_FEATURE_SIZE);
        let mut event_tokens = Vec::with_capacity(
            batch_len * effective_event_history_limit * ENTITY_EVENT_FEATURE_SIZE,
        );
        let mut event_target_ids =
            Vec::with_capacity(batch_len * effective_event_history_limit * 4);
        let mut hex_mask = Vec::with_capacity(batch_len * 19);
        let mut vertex_mask = Vec::with_capacity(batch_len * 54);
        let mut edge_mask = Vec::with_capacity(batch_len * 72);
        let mut player_mask = Vec::with_capacity(batch_len * 4);
        let mut event_mask = Vec::with_capacity(batch_len * effective_event_history_limit);
        let mut legal_action_tokens_rows: Vec<Vec<f64>> = Vec::with_capacity(batch_len);
        let mut legal_action_target_ids_rows: Vec<Vec<i64>> = Vec::with_capacity(batch_len);
        let mut legal_action_mask_rows: Vec<Vec<bool>> = Vec::with_capacity(batch_len);

        for a in arrays.into_iter() {
            widths.push(a.n_legal);
            hex_tokens.extend(a.hex_tokens);
            vertex_tokens.extend(a.vertex_tokens);
            edge_tokens.extend(a.edge_tokens);
            player_tokens.extend(a.player_tokens);
            global_tokens.extend(a.global_tokens);
            event_tokens.extend(a.event_tokens);
            event_target_ids.extend(a.event_target_ids);
            hex_mask.extend(a.hex_mask);
            vertex_mask.extend(a.vertex_mask);
            edge_mask.extend(a.edge_mask);
            player_mask.extend(a.player_mask);
            event_mask.extend(a.event_mask);
            legal_action_tokens_rows.push(a.legal_action_tokens);
            legal_action_target_ids_rows.push(a.legal_action_target_ids);
            legal_action_mask_rows.push(a.legal_action_mask);
        }
        let max_width = widths.iter().copied().max().unwrap_or(0);

        let legal_action_tokens = pad_and_stack_f64(
            &legal_action_tokens_rows,
            &widths,
            ENTITY_LEGAL_ACTION_FEATURE_SIZE,
            max_width,
        );
        let legal_action_target_ids =
            pad_and_stack_i64(&legal_action_target_ids_rows, &widths, 4, max_width);
        let legal_action_mask = pad_and_stack_bool(&legal_action_mask_rows, &widths, max_width);

        let dict = PyDict::new(py);
        dict.set_item(
            "hex_tokens",
            (
                f64_vec_to_le_bytes(hex_tokens),
                (batch_size, 19usize, ENTITY_HEX_FEATURE_SIZE),
            ),
        )?;
        dict.set_item(
            "vertex_tokens",
            (
                f64_vec_to_le_bytes(vertex_tokens),
                (batch_size, 54usize, ENTITY_VERTEX_FEATURE_SIZE),
            ),
        )?;
        dict.set_item(
            "edge_tokens",
            (
                f64_vec_to_le_bytes(edge_tokens),
                (batch_size, 72usize, ENTITY_EDGE_FEATURE_SIZE),
            ),
        )?;
        dict.set_item(
            "player_tokens",
            (
                f64_vec_to_le_bytes(player_tokens),
                (batch_size, 4usize, ENTITY_PLAYER_FEATURE_SIZE),
            ),
        )?;
        dict.set_item(
            "global_tokens",
            (
                f64_vec_to_le_bytes(global_tokens),
                (batch_size, 1usize, ENTITY_GLOBAL_FEATURE_SIZE),
            ),
        )?;
        dict.set_item(
            "legal_action_tokens",
            (
                f64_vec_to_le_bytes(legal_action_tokens),
                (batch_size, max_width, ENTITY_LEGAL_ACTION_FEATURE_SIZE),
            ),
        )?;
        dict.set_item(
            "legal_action_target_ids",
            (
                i64_vec_to_le_bytes(legal_action_target_ids),
                (batch_size, max_width, 4usize),
            ),
        )?;
        dict.set_item(
            "event_tokens",
            (
                f64_vec_to_le_bytes(event_tokens),
                (
                    batch_size,
                    effective_event_history_limit,
                    ENTITY_EVENT_FEATURE_SIZE,
                ),
            ),
        )?;
        dict.set_item(
            "event_target_ids",
            (
                i64_vec_to_le_bytes(event_target_ids),
                (batch_size, effective_event_history_limit, 4usize),
            ),
        )?;
        dict.set_item("hex_mask", (hex_mask, (batch_size, 19usize)))?;
        dict.set_item("vertex_mask", (vertex_mask, (batch_size, 54usize)))?;
        dict.set_item("edge_mask", (edge_mask, (batch_size, 72usize)))?;
        dict.set_item("player_mask", (player_mask, (batch_size, 4usize)))?;
        dict.set_item(
            "legal_action_mask",
            (legal_action_mask, (batch_size, max_width)),
        )?;
        dict.set_item(
            "event_mask",
            (event_mask, (batch_size, effective_event_history_limit)),
        )?;
        dict.set_item("widths", widths)?;
        Ok(dict.into())
    }

    // ---- Action-context featurization (Rust port, task #81 "context lever") ----
    //
    // Bit-exact companion to `catan_zero.rl.action_features._context_vector`,
    // for the specific per-leaf call site `neural_rust_mcts.rust_action_context_batch`
    // always uses (`valid=True` unconditionally -- this is NOT the general
    // `build_action_context_feature_table` used for the full action-space
    // table, only the "one row per already-legal action" path). Reads game
    // state directly off this game -- no `json_snapshot`/`playable_actions_json`
    // round trip -- reusing the SAME `EntityTopology` object the entity
    // featurizer already takes (board-fixed lookup, see that function's doc
    // comment for the BASE-map-only caveat).
    //
    // Needs neither `colors` nor `action_size`: `actor_public_vp` is always
    // the ACTING player's own public VP (`game.current_color()`), and the one
    // other per-player lookup (`MOVE_ROBBER`'s victim) is by that action's own
    // `victim` color, never the full player list.
    //
    // FOUR more pre-existing, INTENTIONAL quirks of the Python reference this
    // reproduces verbatim (in addition to the entity-side quirks already
    // documented above them -- do not "fix" without re-running parity):
    //   1. `_PRIORITY`'s five trade-verb keys ("offer_trade", "accept_trade",
    //      "reject_trade", "confirm_trade", "cancel_trade") are lowercase,
    //      but `action_type_name()` always returns uppercase -- so priority
    //      (feature slot 1) is always 0 for those five action types, the
    //      same case-sensitivity miss as `ACTION_TYPES` on the entity side.
    //   2. `_trade_totals` only ever returns nonzero give/receive totals for
    //      MARITIME_TRADE: the other trade-verb branches read
    //      `payload["trade_panel"]["current_board_trade"]`, a key
    //      `_entity_payload_from_rust_snapshot`'s `trade_panel` dict never
    //      sets (it only sets `offers_remaining`/`current_offer`/
    //      `is_resolving`) -- and `offer_trade`'s own branch reads
    //      `args.get("give")`/`args.get("want")`, which `_structured_action`
    //      never populates for `offer_trade` (only for `MARITIME_TRADE`).
    //      So give/receive totals (slots 7-9) are always 0 for every
    //      trade-verb action type except MARITIME_TRADE.
    //   3. `_offers_remaining_score` (slot 17) reads
    //      `panel.get("offers_remaining_this_turn", 0)`, but the adapter's
    //      `trade_panel` dict key is `"offers_remaining"` (no
    //      `"_this_turn"` suffix) -- so this always falls through to the
    //      `0` default. Slot 17 is always 0.0 in this adapter.
    //   4. `_resource_total` on a LIST (which is what `_structured_action`
    //      always produces for MARITIME_TRADE's give/want) counts LIST
    //      LENGTH, not resource-weighted quantity: `give_total` = count of
    //      non-null offered slots (0-4), `receive_total` is always exactly
    //      1 (the single asked-for resource is never null in
    //      `ActionValue::MaritimeTrade`).
    const CONTEXT_ACTION_FEATURE_SIZE: usize = 18;

    fn context_action_priority(action_type: &str) -> f64 {
        match action_type {
            "BUILD_CITY" => 100.0,
            "BUILD_SETTLEMENT" => 90.0,
            "BUILD_ROAD" => 70.0,
            "BUY_DEVELOPMENT_CARD" => 60.0,
            "PLAY_KNIGHT_CARD" => 55.0,
            "PLAY_YEAR_OF_PLENTY" => 54.0,
            "PLAY_MONOPOLY" => 53.0,
            "PLAY_ROAD_BUILDING" => 52.0,
            // "offer_trade"/"accept_trade"/"reject_trade"/"confirm_trade"/
            // "cancel_trade" are deliberately OMITTED: the Python reference
            // spells them lowercase and `action_type_name()` is always
            // uppercase, so they never match here either (quirk 1 above) --
            // matching that miss means NOT adding uppercase entries for them.
            "MARITIME_TRADE" => 42.0,
            "ROLL" => 40.0,
            "MOVE_ROBBER" => 35.0,
            "DISCARD_RESOURCE" => 30.0,
            "END_TURN" => 0.0,
            _ => 0.0,
        }
    }

    struct ContextFeatureArrays {
        context_tokens: Vec<f64>,
        n_legal: usize,
    }

    fn build_action_context_arrays(
        game: &Game,
        topology: &PyEntityTopology,
    ) -> ContextFeatureArrays {
        let state = &game.state;
        let actions = &game.playable_actions;
        let n_legal = actions.len();
        let perspective_color = state.current_color();
        let prompt_name = action_prompt_name(state.current_prompt);
        let is_initial_prompt = prompt_name.contains("INITIAL");
        let actor_public_vp = state.player_state(perspective_color).victory_points as f64;
        let actor_vp_score = (actor_public_vp / 10.0).clamp(0.0, 1.0);

        let mut tiles_by_id: Vec<(Coordinate, &LandTile)> = state
            .board
            .map
            .land_tiles
            .iter()
            .map(|(coordinate, tile)| (*coordinate, tile))
            .filter(|(_, tile)| tile.id < 19)
            .collect();
        tiles_by_id.sort_by_key(|(_, tile)| tile.id);

        let mut port_by_node: HashMap<usize, Option<Resource>> = HashMap::new();
        for (&port_id, port) in state.board.map.ports_by_id.iter() {
            if let Some(nodes) = topology.port_base_nodes.get(port_id) {
                for &node in nodes {
                    if node >= 0 {
                        port_by_node.insert(node as usize, port.resource);
                    }
                }
            }
        }

        let occupied_nodes: std::collections::HashSet<usize> =
            state.board.buildings.keys().copied().collect();

        // Fixed-topology node adjacency (mirrors `_neighbor_nodes`'s final
        // SET -- the per-tile dedup in the Python reference doesn't change
        // which nodes end up adjacent, only how many times each is visited
        // while building the list).
        let node_neighbors = |node: usize| -> Vec<usize> {
            let mut neighbors = Vec::new();
            for pair in topology.edge_vertex_ids.iter() {
                if pair.len() != 2 || pair[0] < 0 || pair[1] < 0 {
                    continue;
                }
                let (a, b) = (pair[0] as usize, pair[1] as usize);
                if a == node {
                    neighbors.push(b);
                } else if b == node {
                    neighbors.push(a);
                }
            }
            neighbors
        };

        let node_total_pips = |node: usize| -> i64 {
            let mut total = 0i64;
            for (_, tile) in &tiles_by_id {
                if topology.hex_vertex_ids[tile.id]
                    .iter()
                    .any(|&n| n as i64 == node as i64)
                {
                    total += entity_dice_pips(tile.number);
                }
            }
            total
        };
        let scaled_production = |node: usize| -> f64 { entity_scale(node_total_pips(node), 18.0) };

        let mut context_tokens = vec![0.0f64; n_legal * CONTEXT_ACTION_FEATURE_SIZE];
        for (row, action) in actions.iter().enumerate() {
            let base = row * CONTEXT_ACTION_FEATURE_SIZE;
            let type_name = action_type_name(action.action_type);
            context_tokens[base] = 1.0; // "valid" -- hardcoded true at this call site
            context_tokens[base + 1] = context_action_priority(type_name) / 100.0;
            context_tokens[base + 10] = actor_vp_score;
            context_tokens[base + 12] = if is_initial_prompt { 1.0 } else { 0.0 };
            context_tokens[base + 11] = if type_name == "END_TURN" { 1.0 } else { 0.0 };
            // Slot 17 (offers_remaining): the Python reference reads a
            // "_this_turn"-suffixed key the adapter payload never sets --
            // see quirk 3 above. Always 0.0 for bit-exact parity.

            match (action.action_type, &action.value) {
                (ActionType::BuildSettlement, ActionValue::Node(node))
                | (ActionType::BuildCity, ActionValue::Node(node)) => {
                    let node = *node;
                    context_tokens[base + 2] = scaled_production(node);
                    context_tokens[base + 13] = if port_by_node.contains_key(&node) {
                        1.0
                    } else {
                        0.0
                    };
                    context_tokens[base + 14] = match port_by_node.get(&node) {
                        None => 0.0,
                        Some(None) => 0.5,
                        Some(Some(_)) => 1.0,
                    };
                    // Python dedupes neighbors into a `set` before scoring
                    // (`set(_neighbor_nodes(...))`) and divides by ITS
                    // length, not the raw (possibly-repeated) list length.
                    let neighbor_set: std::collections::HashSet<usize> =
                        node_neighbors(node).into_iter().collect();
                    context_tokens[base + 15] = if neighbor_set.is_empty() {
                        0.0
                    } else {
                        let occupied_neighbor_count = neighbor_set
                            .iter()
                            .filter(|n| occupied_nodes.contains(*n))
                            .count();
                        occupied_neighbor_count as f64 / neighbor_set.len() as f64
                    };
                }
                (ActionType::BuildRoad, ActionValue::Edge(edge)) => {
                    let (a, b) = canonical_edge(*edge);
                    let productions = [scaled_production(a), scaled_production(b)];
                    context_tokens[base + 3] = productions.iter().copied().fold(f64::MIN, f64::max);
                    context_tokens[base + 4] =
                        productions.iter().sum::<f64>() / productions.len() as f64;
                    let expansion_scores: Vec<f64> = [a, b]
                        .iter()
                        .map(|&node| {
                            if occupied_nodes.contains(&node) {
                                0.0
                            } else {
                                scaled_production(node)
                            }
                        })
                        .collect();
                    context_tokens[base + 16] =
                        expansion_scores.iter().copied().fold(0.0, f64::max);
                }
                (ActionType::MoveRobber, ActionValue::Robber(_, Some(color))) => {
                    let vp = state.player_state(*color).victory_points as f64;
                    context_tokens[base + 5] = (vp / 10.0).clamp(0.0, 1.0);
                }
                (ActionType::DiscardResource, ActionValue::Resource(resource)) => {
                    context_tokens[base + 6] = resource.idx() as f64 / 4.0;
                }
                _ => {}
            }

            let (give_total, receive_total) = match &action.value {
                ActionValue::MaritimeTrade(offering, _asking) => {
                    let give = offering.iter().filter(|o| o.is_some()).count() as f64;
                    (give, 1.0) // receive_total always 1: `asking` is never absent
                }
                _ => (0.0, 0.0), // every other trade-verb type: see quirk 2 above
            };
            context_tokens[base + 7] = give_total / 4.0;
            context_tokens[base + 8] = receive_total / 4.0;
            context_tokens[base + 9] = (receive_total - give_total) / 4.0;
        }

        ContextFeatureArrays {
            context_tokens,
            n_legal,
        }
    }

    #[pyfunction]
    fn build_action_context_flat(
        game: &PyGame,
        topology: &PyEntityTopology,
    ) -> PyResult<(Vec<u8>, (usize, usize))> {
        let arrays = build_action_context_arrays(&game.game, topology);
        Ok((
            f64_vec_to_le_bytes(arrays.context_tokens),
            (arrays.n_legal, CONTEXT_ACTION_FEATURE_SIZE),
        ))
    }

    /// Batched companion to `build_action_context_flat` -- same padding/
    /// stacking/`parallel` conventions as `build_entity_features_batch` (see
    /// its doc comment). Returns `{"context_tokens": (flat, (B, max_width,
    /// CONTEXT_ACTION_FEATURE_SIZE)), "widths": [...]}`.
    #[pyfunction]
    #[pyo3(signature = (games, topology, parallel=false))]
    fn build_action_context_batch(
        py: Python<'_>,
        games: Vec<Py<PyGame>>,
        topology: &PyEntityTopology,
        parallel: bool,
    ) -> PyResult<Py<PyDict>> {
        let borrowed: Vec<PyRef<'_, PyGame>> = games.iter().map(|g| g.borrow(py)).collect();
        let game_refs: Vec<&Game> = borrowed.iter().map(|g| &g.game).collect();

        let arrays: Vec<ContextFeatureArrays> = if parallel {
            game_refs
                .par_iter()
                .map(|game| build_action_context_arrays(game, topology))
                .collect()
        } else {
            game_refs
                .iter()
                .map(|game| build_action_context_arrays(game, topology))
                .collect()
        };

        let batch_len = arrays.len();
        let mut widths: Vec<usize> = Vec::with_capacity(batch_len);
        let mut context_rows: Vec<Vec<f64>> = Vec::with_capacity(batch_len);
        for a in arrays.into_iter() {
            widths.push(a.n_legal);
            context_rows.push(a.context_tokens);
        }
        let max_width = widths.iter().copied().max().unwrap_or(0);
        let context_tokens = pad_and_stack_f64(
            &context_rows,
            &widths,
            CONTEXT_ACTION_FEATURE_SIZE,
            max_width,
        );

        let dict = PyDict::new(py);
        dict.set_item(
            "context_tokens",
            (
                f64_vec_to_le_bytes(context_tokens),
                (batch_len, max_width, CONTEXT_ACTION_FEATURE_SIZE),
            ),
        )?;
        dict.set_item("widths", widths)?;
        Ok(dict.into())
    }

    /// Extract a clone of the inner Game from a PyGame Python object.
    /// Used by the gumbel_mcts crate to hold Game state natively.
    pub fn extract_game(py_obj: &Bound<'_, PyAny>) -> PyResult<Game> {
        let py_game = py_obj.cast::<PyGame>()?;
        Ok(py_game.borrow().game.clone())
    }

    pub fn catanatron_rs(module: &Bound<'_, PyModule>) -> PyResult<()> {
        module.add_class::<PyGame>()?;
        module.add_class::<PyBatchEnv>()?;
        module.add_class::<PyEntityTopology>()?;
        module.add_function(wrap_pyfunction!(action_from_json, module)?)?;
        module.add_function(wrap_pyfunction!(feature_ordering_py, module)?)?;
        module.add_function(wrap_pyfunction!(build_entity_features_flat, module)?)?;
        module.add_function(wrap_pyfunction!(build_entity_features_batch, module)?)?;
        module.add_function(wrap_pyfunction!(build_action_context_flat, module)?)?;
        module.add_function(wrap_pyfunction!(build_action_context_batch, module)?)?;
        module.add_function(wrap_pyfunction!(action_space_json, module)?)?;
        module.add_function(wrap_pyfunction!(simulate_batch_stats, module)?)?;
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn first_legal_actions_from_mask(mask: &[u8], rows: usize, width: usize) -> Vec<usize> {
            mask.chunks_exact(width)
                .take(rows)
                .map(|row| row.iter().position(|value| *value == 1).unwrap())
                .collect()
        }

        fn layout_usize(layout: &Py<PyDict>, py: Python<'_>, key: &str) -> usize {
            layout
                .bind(py)
                .get_item(key)
                .unwrap()
                .unwrap()
                .extract::<usize>()
                .unwrap()
        }

        #[test]
        fn py_game_smoke_works() {
            Python::attach(|_| {
                let mut game =
                    PyGame::simple(Some(vec!["RED".to_string(), "BLUE".to_string()]), Some(1))
                        .unwrap();
                assert!(["RED", "BLUE"].contains(&game.current_color().as_str()));
                let playable = game.playable_actions_json().unwrap();
                assert!(playable.contains("BUILD_SETTLEMENT"));
                let record = game.play_tick().unwrap();
                assert!(record.contains("BUILD_SETTLEMENT"));
                assert_eq!(game.state_index(), 1);
                assert!(
                    game.json_snapshot()
                        .unwrap()
                        .contains("current_playable_actions")
                );
                let actor = game.current_color();
                let public_cards = game.public_card_deductions_json(&actor).unwrap();
                assert!(public_cards.contains("public_card_deductions_2p_v1"));
                let (tensor, shape) = game.board_tensor_flat("RED", false).unwrap();
                assert_eq!(shape, (21, 11, 16));
                assert_eq!(tensor.len(), 21 * 11 * 16);
                let action_space = action_space_json(
                    vec!["BLUE".to_string(), "RED".to_string()],
                    Some("BASE".to_string()),
                )
                .unwrap();
                assert!(action_space.contains("MOVE_ROBBER"));
            });
        }

        #[test]
        fn entity_player_tokens_preserve_public_awards_when_hidden_hands_are_masked() {
            let mut py_game =
                PyGame::simple(Some(vec!["RED".to_string(), "BLUE".to_string()]), Some(31))
                    .unwrap();
            let actor = py_game.game.state.current_color();
            let opponent = py_game
                .game
                .state
                .colors
                .iter()
                .copied()
                .find(|color| *color != actor)
                .unwrap();
            py_game.game.state.player_state_mut(actor).has_army = true;
            py_game.game.state.player_state_mut(opponent).has_road = true;

            // Player award slots do not depend on topology, but the shared
            // builder validates/visits its fixed-size topology arrays while
            // constructing the other feature families.
            let topology = PyEntityTopology {
                hex_vertex_ids: vec![vec![-1; 6]; 19],
                hex_edge_ids: vec![vec![-1; 6]; 19],
                edge_vertex_ids: vec![vec![-1; 2]; 72],
                port_base_nodes: vec![vec![-1; 2]; 16],
            };
            let colors = py_game.game.state.colors.clone();
            let policy_action_ids = vec![0; py_game.game.playable_actions.len()];
            let arrays = build_entity_feature_arrays(
                &py_game.game,
                &colors,
                &policy_action_ids,
                400,
                &topology,
                true,
                false,
                ENTITY_EVENT_HISTORY_LIMIT,
                false,
            )
            .unwrap();
            let actor_base =
                entity_player_index(color_name(actor)).unwrap() * ENTITY_PLAYER_FEATURE_SIZE;
            let opponent_base =
                entity_player_index(color_name(opponent)).unwrap() * ENTITY_PLAYER_FEATURE_SIZE;

            assert_eq!(arrays.player_tokens[actor_base + 11], 1.0);
            assert_eq!(arrays.player_tokens[opponent_base + 12], 1.0);
            assert_eq!(arrays.player_tokens[opponent_base + 4], 0.0);
            assert_eq!(arrays.player_tokens[opponent_base + 15], 0.0);
            assert_eq!(arrays.player_tokens[opponent_base + 21], 0.0);
        }

        #[test]
        fn meaningful_public_history_filters_plumbing_and_caps_at_32() {
            let mut py_game =
                PyGame::simple(Some(vec!["RED".to_string(), "BLUE".to_string()]), Some(32))
                    .unwrap();
            let actor = py_game.game.state.current_color();
            py_game.game.state.action_records.clear();
            py_game.game.state.action_public_legal_counts.clear();
            for _ in 0..35 {
                py_game.game.state.action_records.push(ActionRecord {
                    action: Action::new(actor, ActionType::BuildRoad, ActionValue::Edge((0, 1))),
                    result: ActionValue::None,
                });
                py_game.game.state.action_public_legal_counts.push(3);
            }
            py_game.game.state.action_records.push(ActionRecord {
                action: Action::new(actor, ActionType::Roll, ActionValue::Dice(3, 4)),
                result: ActionValue::Dice(3, 4),
            });
            py_game.game.state.action_public_legal_counts.push(1);
            py_game.game.state.action_records.push(ActionRecord {
                action: Action::new(actor, ActionType::EndTurn, ActionValue::None),
                result: ActionValue::None,
            });
            py_game.game.state.action_public_legal_counts.push(1);
            // A strategic action is still omitted when its sole-choice status
            // came from a publicly reconstructible prompt.
            py_game.game.state.action_records.push(ActionRecord {
                action: Action::new(actor, ActionType::BuildCity, ActionValue::Node(7)),
                result: ActionValue::None,
            });
            py_game.game.state.action_public_legal_counts.push(1);
            py_game.game.state.action_records.push(ActionRecord {
                action: Action::new(
                    actor,
                    ActionType::DiscardResource,
                    ActionValue::Resource(Resource::Ore),
                ),
                result: ActionValue::Resource(Resource::Ore),
            });
            // DISCARD width is hidden-hand-dependent and therefore unknown.
            py_game.game.state.action_public_legal_counts.push(0);

            let mut topology = PyEntityTopology {
                hex_vertex_ids: vec![vec![-1; 6]; 19],
                hex_edge_ids: vec![vec![-1; 6]; 19],
                edge_vertex_ids: vec![vec![-1; 2]; 72],
                port_base_nodes: vec![vec![-1; 2]; 16],
            };
            topology.edge_vertex_ids[19] = vec![0, 1];
            let colors = py_game.game.state.colors.clone();
            let policy_action_ids = vec![0; py_game.game.playable_actions.len()];
            let enabled = build_entity_feature_arrays(
                &py_game.game,
                &colors,
                &policy_action_ids,
                400,
                &topology,
                true,
                true,
                ENTITY_EVENT_HISTORY_LIMIT,
                true,
            )
            .unwrap();
            assert_eq!(enabled.event_history_limit, 32);
            assert_eq!(
                enabled.event_mask.iter().filter(|&&value| value).count(),
                32
            );
            // The newest retained event is DISCARD_RESOURCE, not ROLL/END_TURN.
            let last = (enabled.event_history_limit - 1) * ENTITY_EVENT_FEATURE_SIZE;
            let discard_index = ENTITY_ACTION_TYPES
                .iter()
                .position(|&name| name == "DISCARD_RESOURCE")
                .unwrap();
            assert_eq!(enabled.event_tokens[last + 17 + discard_index], 1.0);
            assert_eq!(enabled.event_tokens[last + 35], 0.0);
            assert_eq!(enabled.event_target_ids[2], 19);
            assert_eq!(enabled.event_tokens[14], 19.0 / 72.0);
            assert_eq!(
                enabled.event_target_ids[(enabled.event_history_limit - 1) * 4 + 2],
                -1
            );

            let legacy = build_entity_feature_arrays(
                &py_game.game,
                &colors,
                &policy_action_ids,
                400,
                &topology,
                true,
                false,
                ENTITY_EVENT_HISTORY_LIMIT,
                false,
            )
            .unwrap();
            assert_eq!(legacy.event_history_limit, 64);
            assert!(legacy.event_mask.iter().all(|value| !value));
            assert!(legacy.event_tokens.iter().all(|value| *value == 0.0));
        }

        #[test]
        fn structured_action_resource_bundle_is_versioned_and_semantic() {
            let monopoly_wood = Action::new(
                Color::Red,
                ActionType::PlayMonopoly,
                ActionValue::Resource(Resource::Wood),
            );
            let discard_ore = Action::new(
                Color::Red,
                ActionType::DiscardResource,
                ActionValue::Resource(Resource::Ore),
            );
            let plenty = Action::new(
                Color::Red,
                ActionType::PlayYearOfPlenty,
                ActionValue::Resources(vec![Resource::Wood, Resource::Ore]),
            );

            assert_eq!(
                entity_action_resource_bundle(&monopoly_wood, false),
                [0.0; 5]
            );
            assert_eq!(entity_action_resource_bundle(&discard_ore, false), [0.0; 5]);
            assert_eq!(entity_action_resource_bundle(&plenty, false), [0.0; 5]);
            assert_eq!(
                entity_action_resource_bundle(&monopoly_wood, true),
                [0.5, 0.0, 0.0, 0.0, 0.0]
            );
            assert_eq!(
                entity_action_resource_bundle(&discard_ore, true),
                [0.0, 0.0, 0.0, 0.0, 0.5]
            );
            assert_eq!(
                entity_action_resource_bundle(&plenty, true),
                [0.5, 0.0, 0.0, 0.0, 0.5]
            );
        }

        #[test]
        fn entity_action_tokens_encode_v3_resources_but_preserve_v2() {
            let mut py_game =
                PyGame::simple(Some(vec!["RED".to_string(), "BLUE".to_string()]), Some(33))
                    .unwrap();
            py_game.game.playable_actions = vec![
                Action::new(
                    Color::Red,
                    ActionType::PlayYearOfPlenty,
                    ActionValue::Resources(vec![Resource::Wood, Resource::Ore]),
                ),
                Action::new(
                    Color::Red,
                    ActionType::PlayMonopoly,
                    ActionValue::Resource(Resource::Wood),
                ),
                Action::new(
                    Color::Red,
                    ActionType::DiscardResource,
                    ActionValue::Resource(Resource::Ore),
                ),
            ];
            let topology = PyEntityTopology {
                hex_vertex_ids: vec![vec![-1; 6]; 19],
                hex_edge_ids: vec![vec![-1; 6]; 19],
                edge_vertex_ids: vec![vec![-1; 2]; 72],
                port_base_nodes: vec![vec![-1; 2]; 16],
            };
            let colors = py_game.game.state.colors.clone();
            let ids = vec![311, 305, 185];

            let v2 = build_entity_feature_arrays(
                &py_game.game,
                &colors,
                &ids,
                567,
                &topology,
                false,
                false,
                ENTITY_EVENT_HISTORY_LIMIT,
                false,
            )
            .unwrap();
            let v3 = build_entity_feature_arrays(
                &py_game.game,
                &colors,
                &ids,
                567,
                &topology,
                false,
                false,
                ENTITY_EVENT_HISTORY_LIMIT,
                true,
            )
            .unwrap();

            for row in 0..3 {
                let base = row * ENTITY_LEGAL_ACTION_FEATURE_SIZE;
                assert_eq!(&v2.legal_action_tokens[base + 31..base + 36], &[0.0; 5]);
                assert_eq!(v3.legal_action_tokens[base + 30], 1.0);
            }
            assert_eq!(v2.legal_action_tokens[25], 1.0);
            assert_eq!(v2.legal_action_tokens[30], 0.0);
            assert_eq!(&v3.legal_action_tokens[31..36], &[0.5, 0.0, 0.0, 0.0, 0.5]);
            assert_eq!(
                &v3.legal_action_tokens
                    [ENTITY_LEGAL_ACTION_FEATURE_SIZE + 31..ENTITY_LEGAL_ACTION_FEATURE_SIZE + 36],
                &[0.5, 0.0, 0.0, 0.0, 0.0]
            );
            assert_eq!(
                &v3.legal_action_tokens[2 * ENTITY_LEGAL_ACTION_FEATURE_SIZE + 31
                    ..2 * ENTITY_LEGAL_ACTION_FEATURE_SIZE + 36],
                &[0.0, 0.0, 0.0, 0.0, 0.5]
            );
        }

        #[test]
        fn public_legal_width_sidecar_never_records_private_prompt_widths() {
            let players = vec![Player::simple(Color::Red), Player::simple(Color::Blue)];
            let mut placement = Game::new(players.clone(), Some(91));
            let public_width = placement.playable_actions.len();
            let action = placement.playable_actions[0].clone();
            placement.execute(action, true, None).unwrap();
            assert_eq!(
                placement.state.action_public_legal_counts,
                vec![u16::try_from(public_width).unwrap()]
            );

            let mut private_turn = Game::new(players, Some(92));
            private_turn.state.current_prompt = ActionPrompt::PlayTurn;
            private_turn.state.is_initial_build_phase = false;
            let actor = private_turn.state.current_color();
            private_turn.state.player_state_mut(actor).has_rolled = false;
            private_turn.playable_actions = generate_playable_actions(&private_turn.state);
            assert_eq!(private_turn.playable_actions.len(), 1);
            let roll = private_turn.playable_actions[0].clone();
            private_turn
                .execute(roll, true, Some(ActionValue::Dice(3, 4)))
                .unwrap();
            assert_eq!(
                private_turn.state.action_public_legal_counts,
                vec![0],
                "even a sole regular-turn action is private-dependent metadata",
            );
        }

        #[test]
        fn py_batch_env_smoke_works() {
            Python::attach(|_| {
                let mut env = PyBatchEnv::new(
                    2,
                    Some(vec!["RED".to_string(), "BLUE".to_string()]),
                    Some(1),
                    Some("simple".to_string()),
                    7,
                    false,
                    10,
                    None,
                    None,
                    None,
                    false,
                    1000,
                )
                .unwrap();
                let (obs, obs_shape, masks, mask_shape, rewards, dones, winners, colors) =
                    env.observe().unwrap();
                assert_eq!(obs_shape, (2, 21, 11, 16));
                assert_eq!(obs.len(), 2 * 21 * 11 * 16);
                assert_eq!(mask_shape, (2, env.action_space_len()));
                assert_eq!(masks.len(), 2 * env.action_space_len());
                assert_eq!(rewards, vec![0.0, 0.0]);
                assert_eq!(dones, vec![false, false]);
                assert_eq!(winners, vec![None, None]);
                assert_eq!(colors.len(), 2);
                let feature_ordering = env.feature_ordering();
                let feature_hash = env.feature_schema_hash();
                let action_hash = env.action_space_hash().unwrap();
                assert!(feature_hash.starts_with("fnv1a64:"));
                assert!(action_hash.starts_with("fnv1a64:"));
                let (features, feature_shape) = env.feature_vectors().unwrap();
                assert!(!feature_ordering.is_empty());
                assert_eq!(feature_shape, (2, feature_ordering.len()));
                assert_eq!(features.len(), 2 * feature_ordering.len());

                let first_actions = masks
                    .chunks(env.action_space_len())
                    .map(|mask| mask.iter().position(|value| *value == 1).unwrap())
                    .collect::<Vec<_>>();
                let (_, _, next_masks, next_mask_shape, next_rewards, next_dones, _, _) =
                    env.step(first_actions).unwrap();
                assert_eq!(next_mask_shape, (2, env.action_space_len()));
                assert_eq!(next_masks.len(), 2 * env.action_space_len());
                assert_eq!(next_rewards.len(), 2);
                assert_eq!(next_dones.len(), 2);

                let reset = env.reset(None).unwrap();
                assert_eq!(reset.1, (2, 21, 11, 16));
                let (_, reset_feature_shape) = env.feature_vectors().unwrap();
                assert_eq!(reset_feature_shape, feature_shape);
            });
        }

        #[test]
        fn py_batch_env_bytes_match_list_observation() {
            Python::attach(|py| {
                let env = PyBatchEnv::new(
                    2,
                    Some(vec!["RED".to_string(), "BLUE".to_string()]),
                    Some(3),
                    Some("simple".to_string()),
                    7,
                    false,
                    10,
                    None,
                    None,
                    None,
                    false,
                    1000,
                )
                .unwrap();
                let (obs, obs_shape, masks, mask_shape, rewards, dones, winners, colors) =
                    env.observe().unwrap();
                let (
                    obs_bytes,
                    bytes_obs_shape,
                    mask_bytes,
                    bytes_mask_shape,
                    bytes_rewards,
                    bytes_dones,
                    bytes_winners,
                    bytes_colors,
                ) = env.observe_bytes(py).unwrap();
                assert_eq!(bytes_obs_shape, obs_shape);
                assert_eq!(bytes_mask_shape, mask_shape);
                assert_eq!(bytes_rewards, rewards);
                assert_eq!(bytes_dones, dones);
                assert_eq!(bytes_winners, winners);
                assert_eq!(bytes_colors, colors);

                let obs_bytes = obs_bytes.bind(py).as_bytes();
                assert_eq!(obs_bytes.len(), obs.len() * 4);
                for (chunk, expected) in obs_bytes.chunks_exact(4).zip(obs) {
                    let actual = f32::from_le_bytes(chunk.try_into().unwrap());
                    assert_eq!(actual, expected);
                }
                assert_eq!(mask_bytes.bind(py).as_bytes(), masks.as_slice());
            });
        }

        #[test]
        fn py_batch_env_bytearray_observation_matches_bytes() {
            Python::attach(|py| {
                let env = PyBatchEnv::new(
                    2,
                    Some(vec!["RED".to_string(), "BLUE".to_string()]),
                    Some(4),
                    Some("simple".to_string()),
                    7,
                    false,
                    10,
                    None,
                    None,
                    None,
                    false,
                    1000,
                )
                .unwrap();
                let (obs_bytes, obs_shape, mask_bytes, mask_shape, rewards, dones, winners, colors) =
                    env.observe_bytes(py).unwrap();
                let layout = env.byte_buffer_layout(py).unwrap();
                assert_eq!(
                    layout
                        .bind(py)
                        .get_item("observations_nbytes")
                        .unwrap()
                        .unwrap()
                        .extract::<usize>()
                        .unwrap(),
                    obs_bytes.bind(py).as_bytes().len()
                );
                assert_eq!(
                    layout
                        .bind(py)
                        .get_item("legal_masks_nbytes")
                        .unwrap()
                        .unwrap()
                        .extract::<usize>()
                        .unwrap(),
                    mask_bytes.bind(py).as_bytes().len()
                );

                let obs_buf = PyByteArray::new(py, &vec![0; obs_bytes.bind(py).as_bytes().len()]);
                let mask_buf = PyByteArray::new(py, &vec![0; mask_bytes.bind(py).as_bytes().len()]);
                let (
                    into_obs_shape,
                    into_mask_shape,
                    into_rewards,
                    into_dones,
                    into_winners,
                    into_colors,
                ) = env.observe_bytes_into(&obs_buf, &mask_buf).unwrap();
                assert_eq!(into_obs_shape, obs_shape);
                assert_eq!(into_mask_shape, mask_shape);
                assert_eq!(into_rewards, rewards);
                assert_eq!(into_dones, dones);
                assert_eq!(into_winners, winners);
                assert_eq!(into_colors, colors);
                assert_eq!(unsafe { obs_buf.as_bytes() }, obs_bytes.bind(py).as_bytes());
                assert_eq!(
                    unsafe { mask_buf.as_bytes() },
                    mask_bytes.bind(py).as_bytes()
                );
            });
        }

        #[test]
        fn py_batch_env_bytearray_observation_matches_bytes_for_layout_matrix() {
            Python::attach(|py| {
                for (num_envs, colors, channels_first) in [
                    (0, vec!["RED".to_string(), "BLUE".to_string()], false),
                    (0, vec!["RED".to_string(), "BLUE".to_string()], true),
                    (2, vec!["RED".to_string(), "BLUE".to_string()], false),
                    (
                        3,
                        vec![
                            "RED".to_string(),
                            "BLUE".to_string(),
                            "WHITE".to_string(),
                            "ORANGE".to_string(),
                        ],
                        true,
                    ),
                ] {
                    let env = PyBatchEnv::new(
                        num_envs,
                        Some(colors),
                        Some(15),
                        Some("simple".to_string()),
                        7,
                        false,
                        10,
                        None,
                        None,
                        None,
                        channels_first,
                        1000,
                    )
                    .unwrap();
                    let (
                        obs_bytes,
                        obs_shape,
                        mask_bytes,
                        mask_shape,
                        rewards,
                        dones,
                        winners,
                        colors,
                    ) = env.observe_bytes(py).unwrap();
                    let layout = env.byte_buffer_layout(py).unwrap();
                    let obs_nbytes = layout
                        .bind(py)
                        .get_item("observations_nbytes")
                        .unwrap()
                        .unwrap()
                        .extract::<usize>()
                        .unwrap();
                    let mask_nbytes = layout
                        .bind(py)
                        .get_item("legal_masks_nbytes")
                        .unwrap()
                        .unwrap()
                        .extract::<usize>()
                        .unwrap();
                    let obs_buf = PyByteArray::new(py, &vec![0; obs_nbytes]);
                    let mask_buf = PyByteArray::new(py, &vec![0; mask_nbytes]);

                    let (
                        into_obs_shape,
                        into_mask_shape,
                        into_rewards,
                        into_dones,
                        into_winners,
                        into_colors,
                    ) = env.observe_bytes_into(&obs_buf, &mask_buf).unwrap();

                    assert_eq!(into_obs_shape, obs_shape);
                    assert_eq!(into_mask_shape, mask_shape);
                    assert_eq!(into_rewards, rewards);
                    assert_eq!(into_dones, dones);
                    assert_eq!(into_winners, winners);
                    assert_eq!(into_colors, colors);
                    assert_eq!(unsafe { obs_buf.as_bytes() }, obs_bytes.bind(py).as_bytes());
                    assert_eq!(
                        unsafe { mask_buf.as_bytes() },
                        mask_bytes.bind(py).as_bytes()
                    );
                    if num_envs == 0 && channels_first {
                        let channels = board_tensor_channels(2);
                        assert_eq!(
                            into_obs_shape,
                            (0, channels, BOARD_TENSOR_WIDTH, BOARD_TENSOR_HEIGHT)
                        );
                    }
                }
            });
        }

        #[test]
        fn py_batch_env_memoryview_buffers_match_bytes() {
            Python::attach(|py| {
                let env = PyBatchEnv::new(
                    3,
                    Some(vec![
                        "RED".to_string(),
                        "BLUE".to_string(),
                        "WHITE".to_string(),
                    ]),
                    Some(18),
                    Some("simple".to_string()),
                    7,
                    false,
                    10,
                    None,
                    None,
                    None,
                    true,
                    1000,
                )
                .unwrap();
                let (obs_bytes, obs_shape, mask_bytes, mask_shape, rewards, dones, winners, colors) =
                    env.observe_bytes(py).unwrap();
                let (feature_bytes, feature_shape) = env.feature_vectors_bytes(py).unwrap();
                let layout = env.byte_buffer_layout(py).unwrap();
                let obs_nbytes = layout_usize(&layout, py, "observations_nbytes");
                let mask_nbytes = layout_usize(&layout, py, "legal_masks_nbytes");
                let feature_nbytes = layout_usize(&layout, py, "features_nbytes");
                let obs_buf = PyByteArray::new(py, &vec![0; obs_nbytes]);
                let mask_buf = PyByteArray::new(py, &vec![0; mask_nbytes]);
                let feature_buf = PyByteArray::new(py, &vec![0; feature_nbytes]);
                let memoryview = py
                    .import("builtins")
                    .unwrap()
                    .getattr("memoryview")
                    .unwrap();
                let obs_view = memoryview.call1((obs_buf.as_any(),)).unwrap();
                let mask_view = memoryview.call1((mask_buf.as_any(),)).unwrap();
                let feature_view = memoryview.call1((feature_buf.as_any(),)).unwrap();

                let (
                    into_obs_shape,
                    into_mask_shape,
                    into_rewards,
                    into_dones,
                    into_winners,
                    into_colors,
                ) = env.observe_into_buffer(&obs_view, &mask_view).unwrap();
                let into_feature_shape = env.feature_vectors_into_buffer(&feature_view).unwrap();

                assert_eq!(into_obs_shape, obs_shape);
                assert_eq!(into_mask_shape, mask_shape);
                assert_eq!(into_rewards, rewards);
                assert_eq!(into_dones, dones);
                assert_eq!(into_winners, winners);
                assert_eq!(into_colors, colors);
                assert_eq!(into_feature_shape, feature_shape);
                assert_eq!(unsafe { obs_buf.as_bytes() }, obs_bytes.bind(py).as_bytes());
                assert_eq!(
                    unsafe { mask_buf.as_bytes() },
                    mask_bytes.bind(py).as_bytes()
                );
                assert_eq!(
                    unsafe { feature_buf.as_bytes() },
                    feature_bytes.bind(py).as_bytes()
                );
            });
        }

        #[test]
        fn py_batch_env_reset_and_step_memoryview_buffers_match_bytes() {
            Python::attach(|py| {
                let mut bytes_env = PyBatchEnv::new(
                    2,
                    Some(vec!["RED".to_string(), "BLUE".to_string()]),
                    Some(22),
                    Some("simple".to_string()),
                    7,
                    false,
                    10,
                    None,
                    None,
                    None,
                    false,
                    1000,
                )
                .unwrap();
                let mut buffer_env = PyBatchEnv::new(
                    2,
                    Some(vec!["RED".to_string(), "BLUE".to_string()]),
                    Some(22),
                    Some("simple".to_string()),
                    7,
                    false,
                    10,
                    None,
                    None,
                    None,
                    false,
                    1000,
                )
                .unwrap();
                let layout = buffer_env.byte_buffer_layout(py).unwrap();
                let obs_buf = PyByteArray::new(
                    py,
                    &vec![0; layout_usize(&layout, py, "observations_nbytes")],
                );
                let mask_buf = PyByteArray::new(
                    py,
                    &vec![0; layout_usize(&layout, py, "legal_masks_nbytes")],
                );
                let memoryview = py
                    .import("builtins")
                    .unwrap()
                    .getattr("memoryview")
                    .unwrap();
                let obs_view = memoryview.call1((obs_buf.as_any(),)).unwrap();
                let mask_view = memoryview.call1((mask_buf.as_any(),)).unwrap();

                let (
                    reset_obs,
                    reset_obs_shape,
                    reset_mask,
                    reset_mask_shape,
                    reset_rewards,
                    reset_dones,
                    reset_winners,
                    reset_colors,
                ) = bytes_env.reset_bytes(py, Some(vec![301, 302])).unwrap();
                let (
                    into_obs_shape,
                    into_mask_shape,
                    into_rewards,
                    into_dones,
                    into_winners,
                    into_colors,
                ) = buffer_env
                    .reset_into_buffer(&obs_view, &mask_view, Some(vec![301, 302]))
                    .unwrap();
                assert_eq!(into_obs_shape, reset_obs_shape);
                assert_eq!(into_mask_shape, reset_mask_shape);
                assert_eq!(into_rewards, reset_rewards);
                assert_eq!(into_dones, reset_dones);
                assert_eq!(into_winners, reset_winners);
                assert_eq!(into_colors, reset_colors);
                assert_eq!(unsafe { obs_buf.as_bytes() }, reset_obs.bind(py).as_bytes());
                assert_eq!(
                    unsafe { mask_buf.as_bytes() },
                    reset_mask.bind(py).as_bytes()
                );

                let actions = first_legal_actions_from_mask(
                    reset_mask.bind(py).as_bytes(),
                    2,
                    buffer_env.action_space_len(),
                );
                let (
                    step_obs,
                    step_obs_shape,
                    step_mask,
                    step_mask_shape,
                    step_rewards,
                    step_dones,
                    step_winners,
                    step_colors,
                ) = bytes_env.step_bytes(py, actions.clone()).unwrap();
                let (
                    into_step_obs_shape,
                    into_step_mask_shape,
                    into_step_rewards,
                    into_step_dones,
                    into_step_winners,
                    into_step_colors,
                ) = buffer_env
                    .step_into_buffer(actions, &obs_view, &mask_view)
                    .unwrap();
                assert_eq!(into_step_obs_shape, step_obs_shape);
                assert_eq!(into_step_mask_shape, step_mask_shape);
                assert_eq!(into_step_rewards, step_rewards);
                assert_eq!(into_step_dones, step_dones);
                assert_eq!(into_step_winners, step_winners);
                assert_eq!(into_step_colors, step_colors);
                assert_eq!(unsafe { obs_buf.as_bytes() }, step_obs.bind(py).as_bytes());
                assert_eq!(
                    unsafe { mask_buf.as_bytes() },
                    step_mask.bind(py).as_bytes()
                );
            });
        }

        #[test]
        fn py_batch_env_generic_buffer_errors_reject_before_mutation() {
            Python::attach(|py| {
                let mut env = PyBatchEnv::new(
                    2,
                    Some(vec!["RED".to_string(), "BLUE".to_string()]),
                    Some(23),
                    Some("simple".to_string()),
                    7,
                    false,
                    10,
                    None,
                    None,
                    None,
                    false,
                    1000,
                )
                .unwrap();
                let layout = env.byte_buffer_layout(py).unwrap();
                let obs_nbytes = layout_usize(&layout, py, "observations_nbytes");
                let mask_nbytes = layout_usize(&layout, py, "legal_masks_nbytes");
                let feature_nbytes = layout_usize(&layout, py, "features_nbytes");
                let obs_buf = PyByteArray::new(py, &vec![0xA5; obs_nbytes]);
                let short_mask_buf =
                    PyByteArray::new(py, &vec![0x5A; mask_nbytes.saturating_sub(1)]);
                let feature_buf = PyByteArray::new(py, &vec![0x3C; feature_nbytes]);
                let memoryview = py
                    .import("builtins")
                    .unwrap()
                    .getattr("memoryview")
                    .unwrap();
                let obs_view = memoryview.call1((obs_buf.as_any(),)).unwrap();
                let short_mask_view = memoryview.call1((short_mask_buf.as_any(),)).unwrap();
                let feature_view = memoryview.call1((feature_buf.as_any(),)).unwrap();
                let readonly_obs = PyBytes::new(py, &vec![0; obs_nbytes]);
                let readonly_feature = PyBytes::new(py, &vec![0; feature_nbytes]);

                let turns_before = env
                    .games
                    .iter()
                    .map(|game| game.state.action_records.len())
                    .collect::<Vec<_>>();
                let error = env
                    .reset_into_buffer(&obs_view, &short_mask_view, Some(vec![401, 402]))
                    .unwrap_err();
                assert!(error.to_string().contains("legal_mask_buffer length"));
                assert_eq!(
                    turns_before,
                    env.games
                        .iter()
                        .map(|game| game.state.action_records.len())
                        .collect::<Vec<_>>()
                );
                assert!(
                    unsafe { obs_buf.as_bytes() }
                        .iter()
                        .all(|value| *value == 0xA5)
                );
                assert!(
                    unsafe { short_mask_buf.as_bytes() }
                        .iter()
                        .all(|value| *value == 0x5A)
                );

                let (_, _, masks, _, _, _, _, _) = env.observe().unwrap();
                let actions = masks
                    .chunks(env.action_space_len())
                    .map(|mask| mask.iter().position(|value| *value == 1).unwrap())
                    .collect::<Vec<_>>();
                let records_before = env
                    .games
                    .iter()
                    .map(|game| game.state.action_records.len())
                    .collect::<Vec<_>>();
                let error = env
                    .step_into_buffer(actions, &obs_view, &short_mask_view)
                    .unwrap_err();
                assert!(error.to_string().contains("legal_mask_buffer length"));
                assert_eq!(
                    records_before,
                    env.games
                        .iter()
                        .map(|game| game.state.action_records.len())
                        .collect::<Vec<_>>()
                );

                assert!(
                    env.observe_into_buffer(readonly_obs.as_any(), &short_mask_view)
                        .unwrap_err()
                        .to_string()
                        .contains("writable")
                );
                assert!(
                    env.feature_vectors_into_buffer(readonly_feature.as_any())
                        .unwrap_err()
                        .to_string()
                        .contains("writable")
                );
                let feature_shape = env.feature_vectors_into_buffer(&feature_view).unwrap();
                assert_eq!(feature_shape, (2, env.feature_ordering().len()));
                assert!(
                    unsafe { feature_buf.as_bytes() }
                        .iter()
                        .any(|value| *value != 0x3C)
                );
            });
        }

        #[test]
        fn py_batch_env_observe_bytes_into_matches_list_observation_directly() {
            Python::attach(|py| {
                let env = PyBatchEnv::new(
                    2,
                    Some(vec!["RED".to_string(), "BLUE".to_string()]),
                    Some(14),
                    Some("simple".to_string()),
                    7,
                    false,
                    10,
                    None,
                    None,
                    None,
                    true,
                    1000,
                )
                .unwrap();
                let (obs, obs_shape, masks, mask_shape, rewards, dones, winners, colors) =
                    env.observe().unwrap();
                let layout = env.byte_buffer_layout(py).unwrap();
                let obs_nbytes = layout
                    .bind(py)
                    .get_item("observations_nbytes")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap();
                let mask_nbytes = layout
                    .bind(py)
                    .get_item("legal_masks_nbytes")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap();
                let obs_buf = PyByteArray::new(py, &vec![0; obs_nbytes]);
                let mask_buf = PyByteArray::new(py, &vec![0; mask_nbytes]);

                let (
                    into_obs_shape,
                    into_mask_shape,
                    into_rewards,
                    into_dones,
                    into_winners,
                    into_colors,
                ) = env.observe_bytes_into(&obs_buf, &mask_buf).unwrap();

                assert_eq!(into_obs_shape, obs_shape);
                assert_eq!(into_mask_shape, mask_shape);
                assert_eq!(into_rewards, rewards);
                assert_eq!(into_dones, dones);
                assert_eq!(into_winners, winners);
                assert_eq!(into_colors, colors);
                for (chunk, expected) in unsafe { obs_buf.as_bytes() }.chunks_exact(4).zip(obs) {
                    assert_eq!(f32::from_le_bytes(chunk.try_into().unwrap()), expected);
                }
                assert_eq!(unsafe { mask_buf.as_bytes() }, masks.as_slice());
            });
        }

        #[test]
        fn py_batch_env_reset_and_step_bytes_into_match_bytes() {
            Python::attach(|py| {
                let mut bytes_env = PyBatchEnv::new(
                    2,
                    Some(vec!["RED".to_string(), "BLUE".to_string()]),
                    Some(21),
                    Some("simple".to_string()),
                    7,
                    false,
                    10,
                    None,
                    None,
                    None,
                    false,
                    1000,
                )
                .unwrap();
                let mut into_env = PyBatchEnv::new(
                    2,
                    Some(vec!["RED".to_string(), "BLUE".to_string()]),
                    Some(21),
                    Some("simple".to_string()),
                    7,
                    false,
                    10,
                    None,
                    None,
                    None,
                    false,
                    1000,
                )
                .unwrap();
                let layout = into_env.byte_buffer_layout(py).unwrap();
                let obs_nbytes = layout
                    .bind(py)
                    .get_item("observations_nbytes")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap();
                let mask_nbytes = layout
                    .bind(py)
                    .get_item("legal_masks_nbytes")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap();
                let obs_buf = PyByteArray::new(py, &vec![0; obs_nbytes]);
                let mask_buf = PyByteArray::new(py, &vec![0; mask_nbytes]);

                let (
                    reset_obs,
                    reset_obs_shape,
                    reset_mask,
                    reset_mask_shape,
                    reset_rewards,
                    reset_dones,
                    reset_winners,
                    reset_colors,
                ) = bytes_env.reset_bytes(py, Some(vec![101, 202])).unwrap();
                let (
                    into_obs_shape,
                    into_mask_shape,
                    into_rewards,
                    into_dones,
                    into_winners,
                    into_colors,
                ) = into_env
                    .reset_bytes_into(&obs_buf, &mask_buf, Some(vec![101, 202]))
                    .unwrap();
                assert_eq!(into_obs_shape, reset_obs_shape);
                assert_eq!(into_mask_shape, reset_mask_shape);
                assert_eq!(into_rewards, reset_rewards);
                assert_eq!(into_dones, reset_dones);
                assert_eq!(into_winners, reset_winners);
                assert_eq!(into_colors, reset_colors);
                assert_eq!(unsafe { obs_buf.as_bytes() }, reset_obs.bind(py).as_bytes());
                assert_eq!(
                    unsafe { mask_buf.as_bytes() },
                    reset_mask.bind(py).as_bytes()
                );

                let actions = first_legal_actions_from_mask(
                    reset_mask.bind(py).as_bytes(),
                    2,
                    into_env.action_space_len(),
                );
                let (
                    step_obs,
                    step_obs_shape,
                    step_mask,
                    step_mask_shape,
                    step_rewards,
                    step_dones,
                    step_winners,
                    step_colors,
                ) = bytes_env.step_bytes(py, actions.clone()).unwrap();
                let (
                    into_step_obs_shape,
                    into_step_mask_shape,
                    into_step_rewards,
                    into_step_dones,
                    into_step_winners,
                    into_step_colors,
                ) = into_env
                    .step_bytes_into(actions, &obs_buf, &mask_buf)
                    .unwrap();
                assert_eq!(into_step_obs_shape, step_obs_shape);
                assert_eq!(into_step_mask_shape, step_mask_shape);
                assert_eq!(into_step_rewards, step_rewards);
                assert_eq!(into_step_dones, step_dones);
                assert_eq!(into_step_winners, step_winners);
                assert_eq!(into_step_colors, step_colors);
                assert_eq!(unsafe { obs_buf.as_bytes() }, step_obs.bind(py).as_bytes());
                assert_eq!(
                    unsafe { mask_buf.as_bytes() },
                    step_mask.bind(py).as_bytes()
                );
            });
        }

        #[test]
        fn py_batch_env_bytearray_errors_do_not_partially_write_or_step() {
            Python::attach(|py| {
                let mut env = PyBatchEnv::new(
                    2,
                    Some(vec!["RED".to_string(), "BLUE".to_string()]),
                    Some(5),
                    Some("simple".to_string()),
                    7,
                    false,
                    10,
                    None,
                    None,
                    None,
                    false,
                    1000,
                )
                .unwrap();
                let (obs_bytes, _, mask_bytes, _, _, _, _, _) = env.observe_bytes(py).unwrap();
                let obs_len = obs_bytes.bind(py).as_bytes().len();
                let mask_len = mask_bytes.bind(py).as_bytes().len();
                let obs_buf = PyByteArray::new(py, &vec![7; obs_len]);
                let short_mask_buf = PyByteArray::new(py, &vec![0; mask_len.saturating_sub(1)]);
                assert!(
                    env.observe_bytes_into(&obs_buf, &short_mask_buf)
                        .unwrap_err()
                        .to_string()
                        .contains("legal_mask_buffer length")
                );
                assert!(
                    unsafe { obs_buf.as_bytes() }
                        .iter()
                        .all(|value| *value == 7)
                );

                let (_, _, masks, _, _, _, _, _) = env.observe().unwrap();
                let actions = masks
                    .chunks(env.action_space_len())
                    .map(|mask| mask.iter().position(|value| *value == 1).unwrap())
                    .collect::<Vec<_>>();
                let turns_before = env.games[0].state.num_turns;
                assert!(
                    env.step_bytes_into(actions, &obs_buf, &short_mask_buf)
                        .unwrap_err()
                        .to_string()
                        .contains("legal_mask_buffer length")
                );
                assert_eq!(env.games[0].state.num_turns, turns_before);
            });
        }

        #[test]
        fn py_batch_env_observe_bytes_into_internal_mask_error_does_not_write() {
            Python::attach(|py| {
                let mut env = PyBatchEnv::new(
                    2,
                    Some(vec!["RED".to_string(), "BLUE".to_string()]),
                    Some(17),
                    Some("simple".to_string()),
                    7,
                    false,
                    10,
                    None,
                    None,
                    None,
                    false,
                    1000,
                )
                .unwrap();
                let layout = env.byte_buffer_layout(py).unwrap();
                let obs_nbytes = layout
                    .bind(py)
                    .get_item("observations_nbytes")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap();
                env.games[0].playable_actions.push(Action::new(
                    Color::Red,
                    ActionType::BuildSettlement,
                    ActionValue::Node(usize::MAX),
                ));
                let mask_nbytes = layout
                    .bind(py)
                    .get_item("legal_masks_nbytes")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap();
                let obs_buf = PyByteArray::new(py, &vec![0xA5; obs_nbytes]);
                let mask_buf = PyByteArray::new(py, &vec![0x5A; mask_nbytes]);

                let error = env.observe_bytes_into(&obs_buf, &mask_buf).unwrap_err();

                assert!(
                    error
                        .to_string()
                        .contains("playable action missing from action space")
                );
                assert!(
                    unsafe { obs_buf.as_bytes() }
                        .iter()
                        .all(|value| *value == 0xA5)
                );
                assert!(
                    unsafe { mask_buf.as_bytes() }
                        .iter()
                        .all(|value| *value == 0x5A)
                );
            });
        }

        #[test]
        fn py_batch_env_feature_bytes_match_list_vectors() {
            Python::attach(|py| {
                let mut env = PyBatchEnv::new(
                    3,
                    Some(vec![
                        "RED".to_string(),
                        "BLUE".to_string(),
                        "WHITE".to_string(),
                    ]),
                    Some(9),
                    Some("simple".to_string()),
                    7,
                    false,
                    10,
                    None,
                    None,
                    None,
                    false,
                    1000,
                )
                .unwrap();
                let ordering = env.feature_ordering();
                let (features, shape) = env.feature_vectors().unwrap();
                let (feature_bytes, byte_shape) = env.feature_vectors_bytes(py).unwrap();
                assert_eq!(byte_shape, shape);
                assert_eq!(shape, (3, ordering.len()));
                assert_eq!(features.len(), 3 * ordering.len());

                let feature_bytes = feature_bytes.bind(py).as_bytes();
                assert_eq!(feature_bytes.len(), features.len() * 4);
                for (chunk, expected) in feature_bytes.chunks_exact(4).zip(features) {
                    let actual = f32::from_le_bytes(chunk.try_into().unwrap());
                    assert_eq!(actual, expected);
                }

                let (obs, _, masks, _, _, _, _, _) = env.observe().unwrap();
                let actions = masks
                    .chunks(env.action_space_len())
                    .map(|mask| mask.iter().position(|value| *value == 1).unwrap())
                    .collect::<Vec<_>>();
                env.step(actions).unwrap();
                let (next_features, next_shape) = env.feature_vectors().unwrap();
                assert_eq!(next_shape, shape);
                assert_eq!(next_features.len(), 3 * ordering.len());
                assert_eq!(obs.len(), 3 * 21 * 11 * 18);
            });
        }

        #[test]
        fn py_batch_env_feature_bytearray_matches_bytes() {
            Python::attach(|py| {
                let env = PyBatchEnv::new(
                    3,
                    Some(vec![
                        "RED".to_string(),
                        "BLUE".to_string(),
                        "WHITE".to_string(),
                    ]),
                    Some(10),
                    Some("simple".to_string()),
                    7,
                    false,
                    10,
                    None,
                    None,
                    None,
                    false,
                    1000,
                )
                .unwrap();
                let (feature_bytes, shape) = env.feature_vectors_bytes(py).unwrap();
                let feature_buf =
                    PyByteArray::new(py, &vec![0; feature_bytes.bind(py).as_bytes().len()]);
                let into_shape = env.feature_vectors_bytes_into(&feature_buf).unwrap();
                assert_eq!(into_shape, shape);
                assert_eq!(
                    unsafe { feature_buf.as_bytes() },
                    feature_bytes.bind(py).as_bytes()
                );
            });
        }

        #[test]
        fn py_batch_env_done_rows_clear_stale_masks() {
            Python::attach(|py| {
                let mut env = PyBatchEnv::new(
                    2,
                    Some(vec!["RED".to_string(), "BLUE".to_string()]),
                    Some(16),
                    Some("simple".to_string()),
                    7,
                    false,
                    10,
                    None,
                    None,
                    None,
                    false,
                    0,
                )
                .unwrap();
                let layout = env.byte_buffer_layout(py).unwrap();
                let obs_nbytes = layout
                    .bind(py)
                    .get_item("observations_nbytes")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap();
                let mask_nbytes = layout
                    .bind(py)
                    .get_item("legal_masks_nbytes")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap();
                let obs_buf = PyByteArray::new(py, &vec![3; obs_nbytes]);
                let mask_buf = PyByteArray::new(py, &vec![7; mask_nbytes]);

                let (_, _, _, dones, _, _) = env.observe_bytes_into(&obs_buf, &mask_buf).unwrap();
                assert_eq!(dones, vec![true, true]);
                assert!(
                    unsafe { mask_buf.as_bytes() }
                        .iter()
                        .all(|value| *value == 0)
                );

                unsafe { mask_buf.as_bytes_mut() }.fill(7);
                let (_, _, _, reset_dones, _, _) =
                    env.reset_bytes_into(&obs_buf, &mask_buf, None).unwrap();
                assert_eq!(reset_dones, vec![true, true]);
                assert!(
                    unsafe { mask_buf.as_bytes() }
                        .iter()
                        .all(|value| *value == 0)
                );

                unsafe { mask_buf.as_bytes_mut() }.fill(7);
                let (_, _, _, step_dones, _, _) = env
                    .step_bytes_into(vec![0, 0], &obs_buf, &mask_buf)
                    .unwrap();
                assert_eq!(step_dones, vec![true, true]);
                assert!(
                    unsafe { mask_buf.as_bytes() }
                        .iter()
                        .all(|value| *value == 0)
                );
            });
        }

        #[test]
        fn py_batch_env_feature_bytearray_validates_length_without_write() {
            Python::attach(|py| {
                let env = PyBatchEnv::new(
                    2,
                    Some(vec!["RED".to_string(), "BLUE".to_string()]),
                    Some(12),
                    Some("simple".to_string()),
                    7,
                    false,
                    10,
                    None,
                    None,
                    None,
                    false,
                    1000,
                )
                .unwrap();
                let layout = env.byte_buffer_layout(py).unwrap();
                let feature_nbytes = layout
                    .bind(py)
                    .get_item("features_nbytes")
                    .unwrap()
                    .unwrap()
                    .extract::<usize>()
                    .unwrap();
                let short_buf = PyByteArray::new(py, &vec![7; feature_nbytes - 1]);

                let error = env.feature_vectors_bytes_into(&short_buf).unwrap_err();

                assert!(error.to_string().contains("feature_buffer length"));
                assert!(
                    unsafe { short_buf.as_bytes() }
                        .iter()
                        .all(|value| *value == 7)
                );
            });
        }

        #[test]
        fn py_batch_env_empty_feature_bytearray_preserves_shape() {
            Python::attach(|py| {
                let env = PyBatchEnv::new(
                    0,
                    Some(vec!["RED".to_string(), "BLUE".to_string()]),
                    Some(13),
                    Some("simple".to_string()),
                    7,
                    false,
                    10,
                    None,
                    None,
                    None,
                    false,
                    1000,
                )
                .unwrap();
                let feature_buf = PyByteArray::new(py, &[]);

                let shape = env.feature_vectors_bytes_into(&feature_buf).unwrap();

                assert_eq!(shape, (0, env.feature_ordering().len()));
                assert!(unsafe { feature_buf.as_bytes() }.is_empty());
            });
        }

        #[test]
        fn py_batch_env_empty_batch_preserves_shapes() {
            Python::attach(|_| {
                let env = PyBatchEnv::new(
                    0,
                    Some(vec!["RED".to_string(), "BLUE".to_string()]),
                    Some(1),
                    Some("simple".to_string()),
                    7,
                    false,
                    10,
                    None,
                    None,
                    None,
                    false,
                    1000,
                )
                .unwrap();
                let (obs, obs_shape, masks, mask_shape, rewards, dones, winners, colors) =
                    env.observe().unwrap();
                assert_eq!(obs_shape, (0, 21, 11, 16));
                assert!(obs.is_empty());
                assert_eq!(mask_shape, (0, env.action_space_len()));
                assert!(masks.is_empty());
                assert!(rewards.is_empty());
                assert!(dones.is_empty());
                assert!(winners.is_empty());
                assert!(colors.is_empty());
                let (features, feature_shape) = env.feature_vectors().unwrap();
                assert_eq!(feature_shape, (0, env.feature_ordering().len()));
                assert!(features.is_empty());
            });
        }

        #[test]
        fn py_batch_env_feature_vectors_work_on_mini_and_tournament() {
            Python::attach(|py| {
                for map_kind in ["MINI", "TOURNAMENT"] {
                    let env = PyBatchEnv::new(
                        2,
                        Some(vec![
                            "RED".to_string(),
                            "BLUE".to_string(),
                            "WHITE".to_string(),
                            "ORANGE".to_string(),
                        ]),
                        Some(11),
                        Some("simple".to_string()),
                        7,
                        false,
                        10,
                        Some(map_kind.to_string()),
                        None,
                        Some("random".to_string()),
                        true,
                        1000,
                    )
                    .unwrap();
                    let (features, shape) = env.feature_vectors().unwrap();
                    let (feature_bytes, byte_shape) = env.feature_vectors_bytes(py).unwrap();
                    assert_eq!(shape, (2, env.feature_ordering().len()));
                    assert_eq!(byte_shape, shape);
                    assert_eq!(feature_bytes.bind(py).as_bytes().len(), features.len() * 4);
                    assert_ne!(env.feature_schema_hash(), "");
                    assert_ne!(env.action_space_hash().unwrap(), "");
                }
            });
        }

        #[test]
        fn py_action_from_json_normalizes_action() {
            Python::attach(|_| {
                let action = action_from_json(r#"["BLUE","BUILD_ROAD",[1,0]]"#).unwrap();
                assert_eq!(action, r#"["BLUE","BUILD_ROAD",[0,1]]"#);
            });
        }
    }
}

#[cfg(feature = "python")]
#[cfg(feature = "python")]
pub use python_bindings::extract_game;

#[cfg(feature = "python")]
pub fn init_python_module(module: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    python_bindings::catanatron_rs(module)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn color_strategy() -> BoxedStrategy<Color> {
        prop_oneof![
            Just(Color::Red),
            Just(Color::Blue),
            Just(Color::Orange),
            Just(Color::White),
        ]
        .boxed()
    }

    fn resource_strategy() -> BoxedStrategy<Resource> {
        prop_oneof![
            Just(Resource::Wood),
            Just(Resource::Brick),
            Just(Resource::Sheep),
            Just(Resource::Wheat),
            Just(Resource::Ore),
        ]
        .boxed()
    }

    fn dev_card_strategy() -> BoxedStrategy<DevCard> {
        prop_oneof![
            Just(DevCard::Knight),
            Just(DevCard::YearOfPlenty),
            Just(DevCard::Monopoly),
            Just(DevCard::RoadBuilding),
            Just(DevCard::VictoryPoint),
        ]
        .boxed()
    }

    fn coordinate_strategy() -> BoxedStrategy<Coordinate> {
        (-3i8..=3, -3i8..=3, -3i8..=3)
            .prop_map(|(x, y, z)| Coordinate(x, y, z))
            .boxed()
    }

    fn optional_resource_strategy() -> BoxedStrategy<Option<Resource>> {
        prop_oneof![Just(None), resource_strategy().prop_map(Some)].boxed()
    }

    fn optional_color_strategy() -> BoxedStrategy<Option<Color>> {
        prop_oneof![Just(None), color_strategy().prop_map(Some)].boxed()
    }

    fn parseable_action_strategy() -> BoxedStrategy<Action> {
        let color = color_strategy();
        prop_oneof![
            color
                .clone()
                .prop_map(|color| Action::new(color, ActionType::Roll, ActionValue::None)),
            (color.clone(), 0usize..100).prop_map(|(color, node)| Action::new(
                color,
                ActionType::BuildSettlement,
                ActionValue::Node(node)
            )),
            (color.clone(), 0usize..100).prop_map(|(color, node)| Action::new(
                color,
                ActionType::BuildCity,
                ActionValue::Node(node)
            )),
            (color.clone(), 0usize..100, 0usize..100).prop_map(|(color, a, b)| Action::new(
                color,
                ActionType::BuildRoad,
                ActionValue::Edge(canonical_edge((a, b)))
            )),
            (color.clone(), resource_strategy()).prop_map(|(color, resource)| Action::new(
                color,
                ActionType::DiscardResource,
                ActionValue::Resource(resource)
            )),
            (
                color.clone(),
                coordinate_strategy(),
                optional_color_strategy()
            )
                .prop_map(|(color, coordinate, victim)| Action::new(
                    color,
                    ActionType::MoveRobber,
                    ActionValue::Robber(coordinate, victim)
                )),
            color.clone().prop_map(|color| Action::new(
                color,
                ActionType::BuyDevelopmentCard,
                ActionValue::None
            )),
            (color.clone(), dev_card_strategy()).prop_map(|(color, card)| Action::new(
                color,
                ActionType::BuyDevelopmentCard,
                ActionValue::DevCard(card)
            )),
            color.clone().prop_map(|color| Action::new(
                color,
                ActionType::PlayKnightCard,
                ActionValue::None
            )),
            (
                color.clone(),
                prop::collection::vec(resource_strategy(), 1..=2)
            )
                .prop_map(|(color, resources)| Action::new(
                    color,
                    ActionType::PlayYearOfPlenty,
                    ActionValue::Resources(resources)
                )),
            (color.clone(), resource_strategy()).prop_map(|(color, resource)| Action::new(
                color,
                ActionType::PlayMonopoly,
                ActionValue::Resource(resource)
            )),
            color.clone().prop_map(|color| Action::new(
                color,
                ActionType::PlayRoadBuilding,
                ActionValue::None
            )),
            (
                color.clone(),
                prop::array::uniform4(optional_resource_strategy()),
                resource_strategy()
            )
                .prop_map(|(color, offering, asking)| Action::new(
                    color,
                    ActionType::MaritimeTrade,
                    ActionValue::MaritimeTrade(offering, asking)
                )),
            (color.clone(), prop::array::uniform10(0u8..=3)).prop_map(|(color, trade)| {
                Action::new(color, ActionType::OfferTrade, ActionValue::Trade(trade))
            }),
            (color.clone(), prop::array::uniform10(0u8..=3)).prop_map(|(color, trade)| {
                Action::new(color, ActionType::AcceptTrade, ActionValue::Trade(trade))
            }),
            (color.clone(), prop::array::uniform10(0u8..=3)).prop_map(|(color, trade)| {
                Action::new(color, ActionType::RejectTrade, ActionValue::Trade(trade))
            }),
            (
                color.clone(),
                prop::array::uniform10(0u8..=3),
                color_strategy()
            )
                .prop_map(|(color, trade, target)| Action::new(
                    color,
                    ActionType::ConfirmTrade,
                    ActionValue::ConfirmTrade(trade, target)
                )),
            color.clone().prop_map(|color| Action::new(
                color,
                ActionType::CancelTrade,
                ActionValue::None
            )),
            color.prop_map(|color| Action::new(color, ActionType::EndTurn, ActionValue::None)),
        ]
        .boxed()
    }

    #[test]
    fn base_map_has_expected_shape() {
        let map = CatanMap::base();
        assert_eq!(map.land_tiles.len(), 19);
        assert_eq!(map.land_nodes.len(), 54);
        assert_eq!(map.land_edges().len(), 72);
        for resource in Resource::ALL {
            assert_eq!(map.port_nodes[&Some(resource)].len(), 2);
        }
        assert_eq!(map.port_nodes[&None].len(), 8);
    }

    #[test]
    fn tournament_map_matches_python_fixed_template() {
        let map = CatanMap::from_template(MapKind::Tournament, NumberPlacement::Random);

        assert_eq!(map.land_tiles[&Coordinate(0, 0, 0)].resource, None);
        assert_eq!(
            map.land_tiles[&Coordinate(1, -1, 0)].resource,
            Some(Resource::Ore)
        );
        assert_eq!(map.land_tiles[&Coordinate(1, -1, 0)].number, Some(6));
        assert_eq!(
            map.land_tiles[&Coordinate(-2, 0, 2)].resource,
            Some(Resource::Brick)
        );
        assert_eq!(map.land_tiles[&Coordinate(-2, 0, 2)].number, Some(8));

        let port = match &map.tiles[&Coordinate(1, -3, 2)] {
            Tile::Port(port) => port,
            _ => panic!("expected tournament port"),
        };
        assert_eq!(port.resource, Some(Resource::Brick));
        assert_eq!(port.direction, Direction::Northwest);
    }

    #[test]
    fn tournament_games_use_fixed_template_map() {
        let players = vec![Player::random(Color::Red), Player::random(Color::Blue)];
        let game = Game::with_options_and_map_options(
            players,
            Some(123),
            7,
            false,
            10,
            MapKind::Tournament,
            NumberPlacement::Random,
        );
        let expected = CatanMap::from_template(MapKind::Tournament, NumberPlacement::Random);

        assert_eq!(game.state.board.map.tiles, expected.tiles);
    }

    #[test]
    fn random_number_placement_preserves_base_number_multiset() {
        let map = CatanMap::from_template(MapKind::Base, NumberPlacement::Random);
        let mut numbers = map
            .land_tiles
            .values()
            .filter_map(|tile| tile.number)
            .collect::<Vec<_>>();
        let mut expected = base_numbers();
        numbers.sort_unstable();
        expected.sort_unstable();

        assert_eq!(numbers, expected);
    }

    #[test]
    fn seeded_games_reproduce_random_map_layout() {
        let players = || vec![Player::random(Color::Red), Player::random(Color::Blue)];
        let mut first = Game::with_options_and_map_options(
            players(),
            Some(123),
            7,
            false,
            10,
            MapKind::Base,
            NumberPlacement::Random,
        );
        let second = Game::with_options_and_map_options(
            players(),
            Some(123),
            7,
            false,
            10,
            MapKind::Base,
            NumberPlacement::Random,
        );
        let third = Game::with_options_and_map_options(
            players(),
            Some(124),
            7,
            false,
            10,
            MapKind::Base,
            NumberPlacement::Random,
        );

        first.play_tick().unwrap();

        assert_eq!(map_signature(&first), map_signature(&second));
        assert_ne!(map_signature(&first), map_signature(&third));
    }

    #[test]
    fn seeded_games_reproduce_full_random_play_history() {
        let players = || vec![Player::random(Color::Red), Player::random(Color::Blue)];
        let mut first = Game::with_options_and_map_options(
            players(),
            Some(321),
            7,
            false,
            3,
            MapKind::Base,
            NumberPlacement::Random,
        );
        let mut second = Game::with_options_and_map_options(
            players(),
            Some(321),
            7,
            false,
            3,
            MapKind::Base,
            NumberPlacement::Random,
        );

        first.play();
        second.play();

        assert_eq!(first.state.action_records, second.state.action_records);
        assert_eq!(first.winning_color(), second.winning_color());
    }

    fn normalized_components(board: &Board) -> BTreeMap<Color, Vec<Vec<NodeId>>> {
        let mut out = BTreeMap::new();
        for color in Color::ALL {
            let mut components: Vec<_> = board
                .connected_components
                .get(&color)
                .into_iter()
                .flatten()
                .map(|component| {
                    let mut nodes: Vec<_> = component.iter().copied().collect();
                    nodes.sort_unstable();
                    nodes
                })
                .collect();
            components.sort_unstable();
            if !components.is_empty() {
                out.insert(color, components);
            }
        }
        out
    }

    #[test]
    fn scoped_road_recompute_matches_full_recompute_during_play() {
        for seed in 400..406 {
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

            for _ in 0..300 {
                if game.winning_color().is_some() {
                    break;
                }
                game.play_tick().unwrap();

                let mut full = game.state.board.clone();
                full.recompute_components_and_roads();

                assert_eq!(game.state.board.road_color, full.road_color);
                assert_eq!(game.state.board.road_length, full.road_length);
                assert_eq!(game.state.board.road_lengths, full.road_lengths);
                assert_eq!(
                    normalized_components(&game.state.board),
                    normalized_components(&full)
                );
                for color in Color::ALL {
                    assert_eq!(
                        game.state.board.buildable_edges(color),
                        full.buildable_edges(color)
                    );
                    assert_eq!(
                        game.state.board.buildable_node_ids(color, false),
                        full.buildable_node_ids(color, false)
                    );
                }
            }
        }
    }

    #[test]
    fn generated_playable_actions_stay_sorted_during_random_games() {
        for seed in 500..506 {
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

            for tick in 0..300 {
                let mut sorted = game.playable_actions.clone();
                sort_actions(&mut sorted);
                assert_eq!(
                    game.playable_actions, sorted,
                    "generated actions changed sorted order at seed {seed}, tick {tick}, prompt {:?}",
                    game.state.current_prompt
                );
                if game.winning_color().is_some() {
                    break;
                }
                game.play_tick().unwrap();
            }
        }
    }

    #[test]
    fn random_fast_path_matches_materialized_random_games() {
        for seed in 600..606 {
            let players = || {
                vec![
                    Player::random(Color::Red),
                    Player::random(Color::Blue),
                    Player::random(Color::White),
                    Player::random(Color::Orange),
                ]
            };
            let mut materialized = Game::with_options(players(), Some(seed), 7, seed % 2 == 0, 5);
            let mut fast = Game::with_options(players(), Some(seed), 7, seed % 2 == 0, 5);
            fast.set_materialize_playable_actions(false);

            materialized.play();
            fast.play();

            assert_eq!(materialized.state.action_records, fast.state.action_records);
            assert_eq!(materialized.winning_color(), fast.winning_color());
            assert_eq!(materialized.state.num_turns, fast.state.num_turns);
        }
    }

    #[test]
    fn official_spiral_number_placement_follows_spiral_order() {
        let map = CatanMap::from_template(MapKind::Base, NumberPlacement::OfficialSpiral);
        let numbers = spiral_land_coordinates(&map.tiles, Coordinate(2, -2, 0))
            .into_iter()
            .filter_map(|coordinate| match map.tiles.get(&coordinate).unwrap() {
                Tile::Land(tile) => tile.resource.and(tile.number),
                _ => None,
            })
            .collect::<Vec<_>>();

        assert_eq!(numbers, official_spiral_numbers());
    }

    fn map_signature(game: &Game) -> Vec<(Coordinate, Option<Resource>, Option<u8>)> {
        let mut signature = game
            .state
            .board
            .map
            .land_tiles
            .iter()
            .map(|(coordinate, tile)| (*coordinate, tile.resource, tile.number))
            .collect::<Vec<_>>();
        signature.sort_unstable();
        signature
    }

    #[test]
    fn node_distances_match_python_contract() {
        let distances = node_distances();
        assert_eq!(distances[&2][&3], 1);
        assert_eq!(distances[&0][&3], 3);
        assert_eq!(distances[&3][&0], 3);
        assert_eq!(distances[&3][&9], 2);
        assert_eq!(distances[&3][&29], 4);
        assert_eq!(distances[&34][&32], 2);
        assert_eq!(distances[&31][&45], 11);
    }

    #[test]
    fn board_tensor_maps_match_python_contract() {
        let (node_map, edge_map) = board_tensor_node_edge_maps();
        assert_eq!(node_map[&82], (0, 0));
        assert_eq!(node_map[&81], (2, 0));
        assert_eq!(node_map[&93], (20, 0));
        assert_eq!(node_map[&79], (0, 2));
        assert_eq!(node_map[&43], (4, 2));
        assert_eq!(node_map[&72], (0, 10));
        assert_eq!(node_map[&60], (20, 10));

        assert_eq!(edge_map[&canonical_edge((82, 81))], (1, 0));
        assert_eq!(edge_map[&canonical_edge((81, 47))], (3, 0));
        assert_eq!(edge_map[&canonical_edge((92, 93))], (19, 0));
        assert_eq!(edge_map[&canonical_edge((82, 79))], (0, 1));
        assert_eq!(edge_map[&canonical_edge((47, 43))], (4, 1));
        assert_eq!(edge_map[&canonical_edge((53, 94))], (19, 2));
        assert_eq!(edge_map[&canonical_edge((44, 40))], (2, 3));
        assert_eq!(edge_map[&canonical_edge((21, 16))], (6, 3));
        assert_eq!(edge_map[&canonical_edge((24, 53))], (18, 3));
        assert_eq!(edge_map[&canonical_edge((72, 71))], (1, 10));
        assert_eq!(edge_map[&canonical_edge((60, 61))], (19, 10));

        let board = Board::default();
        for node_id in &board.map.land_nodes {
            assert!(node_map.contains_key(node_id));
        }
        for edge in board.map.land_edges() {
            assert!(edge_map.contains_key(&edge));
        }
    }

    #[test]
    fn board_tensor_tile_map_matches_python_contract() {
        let tile_map = board_tensor_tile_coordinate_map();
        assert_eq!(tile_map[&Coordinate(-1, 3, -2)], (0, 0));
        assert_eq!(tile_map[&Coordinate(0, 2, -2)], (0, 4));
        assert_eq!(tile_map[&Coordinate(-2, 2, 0)], (4, 0));
        assert_eq!(tile_map[&Coordinate(-1, 2, -1)], (2, 2));
        assert_eq!(tile_map[&Coordinate(0, 0, 0)], (4, 8));
        assert_eq!(tile_map[&Coordinate(0, -2, 2)], (8, 12));

        for coordinate in Board::default().map.land_tiles.keys() {
            assert!(tile_map.contains_key(coordinate));
        }
    }

    #[test]
    fn board_tensor_flat_marks_buildings_roads_robber_and_ports() {
        let players = vec![
            Player::simple(Color::Red),
            Player::simple(Color::Blue),
            Player::simple(Color::White),
            Player::simple(Color::Orange),
        ];
        let mut game = Game::new(players, Some(1));
        game.state
            .board
            .build_settlement(Color::Red, 3, true)
            .unwrap();
        game.state.board.build_road(Color::Red, (3, 4)).unwrap();

        let (tensor, shape) = create_board_tensor_flat(&game, Color::Red, false);
        assert_eq!(shape, (21, 11, 20));
        let idx = |x: usize, y: usize, channel: usize| (x * 11 + y) * 20 + channel;
        assert_eq!(tensor[idx(10, 6, 0)], 1.0);
        assert_eq!(tensor[idx(9, 6, 1)], 1.0);

        let robber_channel = 13;
        let mut robber_sum = 0.0;
        for x in 0..21 {
            for y in 0..11 {
                robber_sum += tensor[idx(x, y, robber_channel)];
            }
        }
        assert_eq!(robber_sum, 6.0);

        let mut port_sum = 0.0;
        for x in 0..21 {
            for y in 0..11 {
                for channel in 14..20 {
                    port_sum += tensor[idx(x, y, channel)];
                }
            }
        }
        assert_eq!(port_sum, 18.0);

        let (channels_first, cf_shape) = create_board_tensor_flat(&game, Color::Red, true);
        assert_eq!(cf_shape, (20, 21, 11));
        assert_eq!(channels_first[board_tensor_plane_index(0, 10, 6)], 1.0);
        assert_eq!(channels_first[board_tensor_plane_index(1, 9, 6)], 1.0);
    }

    #[test]
    fn board_tensor_shape_uses_player_count() {
        assert_eq!(board_tensor_channels(2), 16);
        assert_eq!(board_tensor_channels(4), 20);
        assert_eq!(board_tensor_flat_len(4), 21 * 11 * 20);
        assert_eq!(board_tensor_shape(4, false), (21, 11, 20));
        assert_eq!(board_tensor_shape(4, true), (20, 21, 11));
    }

    #[test]
    fn board_tensor_fill_matches_create_for_both_layouts() {
        let mut game = Game::new(
            vec![
                Player::simple(Color::Red),
                Player::simple(Color::Blue),
                Player::simple(Color::White),
                Player::simple(Color::Orange),
            ],
            Some(13),
        );
        for _ in 0..12 {
            game.play_tick().unwrap();
        }

        for channels_first in [false, true] {
            let (created, shape) = create_board_tensor_flat(&game, Color::Red, channels_first);
            let mut filled = vec![99.0; board_tensor_flat_len(game.state.colors.len())];
            let filled_shape =
                fill_board_tensor_flat(&game, Color::Red, channels_first, &mut filled).unwrap();
            assert_eq!(filled_shape, shape);
            assert_eq!(filled, created);

            let mut filled_f32 = vec![99.0_f32; board_tensor_flat_len(game.state.colors.len())];
            let filled_f32_shape =
                fill_board_tensor_flat_f32(&game, Color::Red, channels_first, &mut filled_f32)
                    .unwrap();
            assert_eq!(filled_f32_shape, shape);
            for (actual, expected) in filled_f32.iter().zip(created.iter()) {
                assert!((*actual - *expected as f32).abs() < 1.0e-6);
            }
        }

        let mut too_short = vec![0.0; board_tensor_flat_len(game.state.colors.len()) - 1];
        assert!(fill_board_tensor_flat(&game, Color::Red, false, &mut too_short).is_err());
    }

    #[test]
    fn current_board_tensor_batch_matches_individual_chunks() {
        let games = vec![
            Game::new(
                vec![Player::simple(Color::Red), Player::simple(Color::Blue)],
                Some(1),
            ),
            Game::new(
                vec![Player::simple(Color::Red), Player::simple(Color::Blue)],
                Some(2),
            ),
        ];
        let (batch, shape) = create_current_board_tensor_batch_flat(&games, false);
        assert_eq!(shape, (2, 21, 11, 16));

        let row_len = 21 * 11 * 16;
        for (index, game) in games.iter().enumerate() {
            let (single, single_shape) =
                create_board_tensor_flat(game, game.state.current_color(), false);
            assert_eq!(single_shape, (21, 11, 16));
            assert_eq!(&batch[index * row_len..(index + 1) * row_len], &single);
        }

        let samples = games
            .iter()
            .map(|game| (game, game.state.current_color()))
            .collect::<Vec<_>>();
        let mut filled = vec![0.0; row_len * samples.len()];
        let filled_shape = fill_board_tensor_batch_flat(&samples, false, &mut filled).unwrap();
        assert_eq!(filled_shape, shape);
        assert_eq!(filled, batch);

        for channels_first in [false, true] {
            let (batch_f64, batch_shape) = create_board_tensor_batch_flat(&samples, channels_first);
            let mut filled_f32 = vec![0.0_f32; row_len * samples.len()];
            let filled_f32_shape =
                fill_board_tensor_batch_flat_f32(&samples, channels_first, &mut filled_f32)
                    .unwrap();
            assert_eq!(filled_f32_shape, batch_shape);
            for (actual, expected) in filled_f32.iter().zip(batch_f64.iter()) {
                assert!((*actual - *expected as f32).abs() < 1.0e-6);
            }
        }
    }

    #[test]
    fn f32_board_tensor_batch_handles_empty_and_mixed_player_counts() {
        for channels_first in [false, true] {
            let mut empty_out = Vec::new();
            let shape =
                fill_board_tensor_batch_flat_f32(&[], channels_first, &mut empty_out).unwrap();
            let expected = if channels_first {
                (0, 0, BOARD_TENSOR_WIDTH, BOARD_TENSOR_HEIGHT)
            } else {
                (0, BOARD_TENSOR_WIDTH, BOARD_TENSOR_HEIGHT, 0)
            };
            assert_eq!(shape, expected);
            assert_eq!(
                create_board_tensor_batch_flat_f32(&[], channels_first),
                (Vec::new(), expected)
            );
        }

        let g2 = Game::new(
            vec![Player::simple(Color::Red), Player::simple(Color::Blue)],
            Some(1),
        );
        let g3 = Game::new(
            vec![
                Player::simple(Color::Red),
                Player::simple(Color::Blue),
                Player::simple(Color::White),
            ],
            Some(2),
        );
        let samples = [
            (&g2, g2.state.current_color()),
            (&g3, g3.state.current_color()),
        ];
        let mut out = vec![7.0_f32; board_tensor_flat_len(2) * samples.len()];
        let error = fill_board_tensor_batch_flat_f32(&samples, false, &mut out).unwrap_err();
        assert!(error.contains("same player count"));
        assert!(out.iter().all(|value| *value == 7.0));
    }

    #[test]
    fn f32_board_tensor_batch_parallel_matches_individual_chunks() {
        let games = (0..16)
            .map(|seed| {
                let mut game = Game::new(
                    vec![
                        Player::simple(Color::Red),
                        Player::simple(Color::Blue),
                        Player::simple(Color::White),
                        Player::simple(Color::Orange),
                    ],
                    Some(seed),
                );
                for _ in 0..8 {
                    game.play_tick().unwrap();
                }
                game
            })
            .collect::<Vec<_>>();
        let samples = games
            .iter()
            .map(|game| (game, game.state.current_color()))
            .collect::<Vec<_>>();
        let row_len = board_tensor_flat_len(4);

        for channels_first in [false, true] {
            let (batch, shape) = create_board_tensor_batch_flat_f32(&samples, channels_first);
            assert_eq!(batch.len(), row_len * samples.len());
            assert_eq!(shape, {
                let (a, b, c) = board_tensor_shape(4, channels_first);
                (samples.len(), a, b, c)
            });

            for (index, (game, color)) in samples.iter().enumerate() {
                let (single, single_shape) = create_board_tensor_flat(game, *color, channels_first);
                assert_eq!(single_shape, board_tensor_shape(4, channels_first));
                let row = &batch[index * row_len..(index + 1) * row_len];
                for (actual, expected) in row.iter().zip(single.iter()) {
                    assert!((*actual - *expected as f32).abs() < 1.0e-6);
                }
            }
        }
    }

    #[test]
    fn f32_board_tensor_batch_le_bytes_matches_f32_for_serial_and_parallel() {
        for batch_size in [4usize, 16] {
            let games = (0..batch_size)
                .map(|seed| {
                    let mut game = Game::new(
                        vec![
                            Player::simple(Color::Red),
                            Player::simple(Color::Blue),
                            Player::simple(Color::White),
                            Player::simple(Color::Orange),
                        ],
                        Some(seed as u64),
                    );
                    for _ in 0..(seed % 7) {
                        game.play_tick().unwrap();
                    }
                    game
                })
                .collect::<Vec<_>>();
            let samples = games
                .iter()
                .map(|game| (game, game.state.current_color()))
                .collect::<Vec<_>>();
            let row_len = board_tensor_flat_len(4);

            for channels_first in [false, true] {
                let mut f32_out = vec![0.0_f32; row_len * samples.len()];
                let mut byte_out = vec![0_u8; f32_out.len() * 4];
                let f32_shape =
                    fill_board_tensor_batch_flat_f32(&samples, channels_first, &mut f32_out)
                        .unwrap();
                let byte_shape = fill_board_tensor_batch_flat_f32_le_bytes(
                    &samples,
                    channels_first,
                    &mut byte_out,
                )
                .unwrap();

                assert_eq!(byte_shape, f32_shape);
                for (chunk, expected) in byte_out.chunks_exact(4).zip(f32_out) {
                    assert_eq!(chunk, expected.to_le_bytes());
                }
            }
        }
    }

    #[test]
    fn f32_board_tensor_batch_le_bytes_validates_before_write() {
        let empty_shape = fill_board_tensor_batch_flat_f32_le_bytes(&[], false, &mut []).unwrap();
        assert_eq!(empty_shape, (0, BOARD_TENSOR_WIDTH, BOARD_TENSOR_HEIGHT, 0));
        let mut non_empty = vec![7_u8; 4];
        let error =
            fill_board_tensor_batch_flat_f32_le_bytes(&[], false, &mut non_empty).unwrap_err();
        assert!(error.contains("expected 0"));
        assert!(non_empty.iter().all(|value| *value == 7));

        let g2 = Game::new(
            vec![Player::simple(Color::Red), Player::simple(Color::Blue)],
            Some(1),
        );
        let g3 = Game::new(
            vec![
                Player::simple(Color::Red),
                Player::simple(Color::Blue),
                Player::simple(Color::White),
            ],
            Some(2),
        );
        let samples = [
            (&g2, g2.state.current_color()),
            (&g3, g3.state.current_color()),
        ];
        let mut out = vec![9_u8; board_tensor_flat_len(2) * samples.len() * 4];
        let error =
            fill_board_tensor_batch_flat_f32_le_bytes(&samples, false, &mut out).unwrap_err();
        assert!(error.contains("same player count"));
        assert!(out.iter().all(|value| *value == 9));
    }

    #[test]
    fn action_space_json_matches_python_contract_shape() {
        let base = action_space_json_value(&[Color::Blue, Color::Red], MapKind::Base);
        let mini = action_space_json_value(&[Color::Blue, Color::Red], MapKind::Mini);
        let base_actions = base.as_array().unwrap();
        let mini_actions = mini.as_array().unwrap();

        assert_eq!(base_actions.len(), 338);
        assert_eq!(mini_actions.len(), 200);
        assert!(base_actions.contains(&json!(["BUILD_SETTLEMENT", 10])));
        assert!(base_actions.contains(&json!(["BUILD_ROAD", [0, 1]])));
        assert!(base_actions.contains(&json!(["PLAY_YEAR_OF_PLENTY", ["SHEEP", "WHEAT"]])));
        assert!(base_actions.contains(&json!(["MOVE_ROBBER", [[-1, 0, 1], "BLUE"]])));
        assert!(base_actions.contains(&json!(["MOVE_ROBBER", [[-1, 0, 1], null]])));
        assert!(base_actions.contains(&json!([
            "MARITIME_TRADE",
            ["ORE", "ORE", null, null, "WHEAT"]
        ])));
        assert!(base_actions.contains(&json!(["OFFER_TRADE", null])));
        assert!(base_actions.contains(&json!(["ACCEPT_TRADE", null])));
        assert!(base_actions.contains(&json!(["REJECT_TRADE", null])));
        assert!(base_actions.contains(&json!(["CONFIRM_TRADE", "BLUE"])));
        assert!(base_actions.contains(&json!(["CONFIRM_TRADE", "RED"])));
        assert!(base_actions.contains(&json!(["CANCEL_TRADE", null])));
        assert!(!mini_actions.contains(&json!(["MOVE_ROBBER", [[0, 2, -2], "BLUE"]])));
    }

    #[test]
    fn action_space_json_is_deterministic() {
        let colors = [Color::Red, Color::Blue, Color::White, Color::Orange];
        let first = action_space_json_value(&colors, MapKind::Base);
        for _ in 0..10 {
            assert_eq!(action_space_json_value(&colors, MapKind::Base), first);
        }
    }

    #[test]
    fn typed_action_space_keys_match_json_entries() {
        for map_kind in [MapKind::Base, MapKind::Mini, MapKind::Tournament] {
            for player_count in [2usize, 3, 4] {
                let colors = Color::ALL[..player_count].to_vec();
                let entries = action_space_json_value(&colors, map_kind)
                    .as_array()
                    .unwrap()
                    .clone();
                let action_space = ActionSpace::new(&colors, map_kind);

                assert_eq!(action_space.len(), entries.len());
                for (index, entry) in entries.iter().enumerate() {
                    let key = action_space_key_from_value(entry).unwrap();
                    assert_eq!(action_space.index_by_key.get(&key), Some(&index));
                }
            }
        }
    }

    #[test]
    fn typed_action_space_lookup_matches_legacy_json_string_lookup() {
        for map_kind in [MapKind::Base, MapKind::Mini, MapKind::Tournament] {
            for player_count in [2usize, 3, 4] {
                let colors = Color::ALL[..player_count].to_vec();
                let players = colors.iter().copied().map(Player::random).collect();
                let mut game =
                    Game::with_options_and_map_kind(players, Some(21), 7, false, 5, map_kind);
                let action_space = ActionSpace::new(&colors, map_kind);
                let legacy_index_by_key = action_space
                    .entries
                    .iter()
                    .enumerate()
                    .map(|(index, value)| (serde_json::to_string(value).unwrap(), index))
                    .collect::<HashMap<_, _>>();

                for _ in 0..80 {
                    for action in &game.playable_actions {
                        let typed = action_space.index(action);
                        let legacy = action_space_legacy_string_key(action)
                            .ok()
                            .and_then(|key| legacy_index_by_key.get(&key).copied());
                        assert_eq!(
                            typed, legacy,
                            "typed action-space lookup changed id for {action:?} on {map_kind:?}/{player_count}p"
                        );
                    }
                    if game.winning_color().is_some() {
                        break;
                    }
                    game.play_tick().unwrap();
                }
            }
        }
    }

    #[test]
    fn legal_action_mask_matches_indices() {
        let colors = [Color::Red, Color::Blue];
        let action_space = ActionSpace::new(&colors, MapKind::Base);
        let game = Game::new(
            vec![Player::simple(Color::Red), Player::simple(Color::Blue)],
            Some(14),
        );
        let indices = legal_action_indices_with_space(&game, &action_space).unwrap();
        let mask = legal_action_mask_with_space(&game, &action_space).unwrap();
        let mut filled_mask = vec![9; action_space.len()];
        fill_legal_action_mask_with_space(&game, &action_space, &mut filled_mask).unwrap();

        assert_eq!(mask.len(), action_space.len());
        assert_eq!(filled_mask, mask);
        assert_eq!(
            mask.iter().filter(|value| **value == 1).count(),
            indices.len()
        );
        for index in indices {
            assert_eq!(mask[index], 1);
        }
        assert!(
            fill_legal_action_mask_with_space(&game, &action_space, &mut filled_mask[..1]).is_err()
        );
    }

    #[test]
    fn playable_actions_have_action_space_indices_across_maps_and_player_counts() {
        for map_kind in [MapKind::Base, MapKind::Mini, MapKind::Tournament] {
            for player_count in [2usize, 3, 4] {
                let colors = Color::ALL[..player_count].to_vec();
                let players = colors.iter().copied().map(Player::random).collect();
                let mut game =
                    Game::with_options_and_map_kind(players, Some(20), 7, false, 5, map_kind);
                let action_space = ActionSpace::new(&colors, map_kind);
                for _ in 0..50 {
                    for action in &game.playable_actions {
                        assert!(
                            action_space.index(action).is_some(),
                            "missing action index for {action:?} on {map_kind:?}/{player_count}p"
                        );
                    }
                    if game.winning_color().is_some() {
                        break;
                    }
                    game.play_tick().unwrap();
                }
            }
        }
    }

    #[test]
    fn action_space_index_normalizes_logged_action_values() {
        let colors = [Color::Red, Color::Blue, Color::White, Color::Orange];
        let roll = Action::new(Color::Red, ActionType::Roll, ActionValue::Dice(3, 4));
        let settlement = Action::new(
            Color::Red,
            ActionType::BuildSettlement,
            ActionValue::Node(0),
        );
        let reversed_road =
            Action::new(Color::Red, ActionType::BuildRoad, ActionValue::Edge((1, 0)));
        let canonical_road =
            Action::new(Color::Red, ActionType::BuildRoad, ActionValue::Edge((0, 1)));
        let trade = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0];
        let other_trade = [0, 1, 0, 0, 0, 1, 0, 0, 0, 0];
        let offer_trade = Action::new(
            Color::Red,
            ActionType::OfferTrade,
            ActionValue::Trade(trade),
        );
        let accept_trade = Action::new(
            Color::Blue,
            ActionType::AcceptTrade,
            ActionValue::Trade(trade),
        );
        let reject_trade = Action::new(
            Color::Blue,
            ActionType::RejectTrade,
            ActionValue::Trade(trade),
        );
        let confirm_trade = Action::new(
            Color::Red,
            ActionType::ConfirmTrade,
            ActionValue::ConfirmTrade(trade, Color::Blue),
        );
        let confirm_other_trade_same_target = Action::new(
            Color::Red,
            ActionType::ConfirmTrade,
            ActionValue::ConfirmTrade(other_trade, Color::Blue),
        );
        let confirm_other_target = Action::new(
            Color::Red,
            ActionType::ConfirmTrade,
            ActionValue::ConfirmTrade(trade, Color::White),
        );
        let cancel_trade = Action::new(Color::Red, ActionType::CancelTrade, ActionValue::None);
        let year_of_plenty = Action::new(
            Color::Red,
            ActionType::PlayYearOfPlenty,
            ActionValue::Resources(vec![Resource::Wood, Resource::Brick]),
        );
        let reversed_year_of_plenty = Action::new(
            Color::Red,
            ActionType::PlayYearOfPlenty,
            ActionValue::Resources(vec![Resource::Brick, Resource::Wood]),
        );
        let packed_maritime = Action::new(
            Color::Red,
            ActionType::MaritimeTrade,
            ActionValue::MaritimeTrade(
                [Some(Resource::Wood), Some(Resource::Wood), None, None],
                Resource::Brick,
            ),
        );
        let unpacked_maritime = Action::new(
            Color::Red,
            ActionType::MaritimeTrade,
            ActionValue::MaritimeTrade(
                [Some(Resource::Wood), None, Some(Resource::Wood), None],
                Resource::Brick,
            ),
        );

        assert_eq!(action_space_index(&roll, &colors, MapKind::Base), Some(0));
        for card in DevCard::ALL {
            let bought_card = Action::new(
                Color::Red,
                ActionType::BuyDevelopmentCard,
                ActionValue::DevCard(card),
            );
            assert_eq!(
                action_space_index(&bought_card, &colors, MapKind::Base),
                action_space_index(
                    &Action::new(
                        Color::Red,
                        ActionType::BuyDevelopmentCard,
                        ActionValue::None
                    ),
                    &colors,
                    MapKind::Base
                )
            );
        }
        assert!(action_space_index(&settlement, &colors, MapKind::Base).is_some());
        assert_eq!(
            action_space_index(&reversed_road, &colors, MapKind::Base),
            action_space_index(&canonical_road, &colors, MapKind::Base)
        );
        assert!(action_space_index(&offer_trade, &colors, MapKind::Base).is_some());
        assert_eq!(
            action_space_index(&accept_trade, &colors, MapKind::Base),
            action_space_index(
                &Action::new(Color::Blue, ActionType::AcceptTrade, ActionValue::None),
                &colors,
                MapKind::Base
            )
        );
        assert_eq!(
            action_space_index(&reject_trade, &colors, MapKind::Base),
            action_space_index(
                &Action::new(Color::Blue, ActionType::RejectTrade, ActionValue::None),
                &colors,
                MapKind::Base
            )
        );
        assert_eq!(
            action_space_index(&confirm_trade, &colors, MapKind::Base),
            action_space_index(&confirm_other_trade_same_target, &colors, MapKind::Base)
        );
        assert_ne!(
            action_space_index(&confirm_trade, &colors, MapKind::Base),
            action_space_index(&confirm_other_target, &colors, MapKind::Base)
        );
        assert!(action_space_index(&year_of_plenty, &colors, MapKind::Base).is_some());
        assert!(action_space_index(&reversed_year_of_plenty, &colors, MapKind::Base).is_none());
        assert!(action_space_index(&packed_maritime, &colors, MapKind::Base).is_some());
        assert!(action_space_index(&unpacked_maritime, &colors, MapKind::Base).is_none());
        assert!(action_space_index(&cancel_trade, &colors, MapKind::Base).is_some());
        assert_eq!(action_type_index(ActionType::EndTurn), 17);
    }

    #[test]
    fn board_build_rules_work() {
        let mut board = Board::default();
        assert!(board.build_settlement(Color::Red, 3, false).is_err());
        assert!(board.build_road(Color::Red, (3, 2)).is_err());
        board.build_settlement(Color::Red, 3, true).unwrap();
        assert!(board.build_road(Color::Red, (2, 1)).is_err());
        board.build_road(Color::Red, (3, 2)).unwrap();
        board.build_road(Color::Red, (2, 1)).unwrap();
        assert!(board.build_settlement(Color::Blue, 4, true).is_err());
        board.build_settlement(Color::Blue, 1, true).unwrap();
    }

    #[test]
    fn buildable_nodes_and_edges_match_core_contract() {
        let mut board = Board::default();
        assert_eq!(board.buildable_node_ids(Color::Red, false).len(), 0);
        assert_eq!(board.buildable_node_ids(Color::Red, true).len(), 54);

        board.build_settlement(Color::Red, 3, true).unwrap();
        assert_eq!(board.buildable_node_ids(Color::Red, true).len(), 50);
        assert_eq!(board.buildable_edges(Color::Red).len(), 3);

        board.build_road(Color::Red, (3, 4)).unwrap();
        assert_eq!(board.buildable_node_ids(Color::Red, false).len(), 0);

        board.build_road(Color::Red, (4, 5)).unwrap();
        assert_eq!(board.buildable_node_ids(Color::Red, false), vec![5]);
    }

    #[test]
    fn enemy_settlement_blocks_road_expansion() {
        let mut board = Board::default();
        board.build_settlement(Color::Red, 3, true).unwrap();
        board.build_road(Color::Red, (3, 2)).unwrap();
        board.build_road(Color::Red, (2, 1)).unwrap();
        board.build_settlement(Color::Blue, 1, true).unwrap();

        let red_edges = board.buildable_edges(Color::Red);
        let road_into_blocked_node = canonical_edge((1, 2));
        for neighbor in board.adjacency[&1].iter().copied() {
            let edge = canonical_edge((1, neighbor));
            if edge != road_into_blocked_node && board.land_edges.contains(&edge) {
                assert!(
                    !red_edges.contains(&edge),
                    "red should not build through blue settlement at node 1 onto edge {edge:?}"
                );
            }
        }
    }

    #[test]
    fn mini_map_has_expected_buildable_shape() {
        let mut board = Board::new(CatanMap::mini());
        assert_eq!(board.buildable_node_ids(Color::Red, true).len(), 24);
        board.build_settlement(Color::Red, 19, true).unwrap();
        assert_eq!(board.buildable_edges(Color::Red).len(), 2);
    }

    #[test]
    fn city_requires_existing_settlement() {
        let mut board = Board::default();
        assert!(board.build_city(Color::Red, 3).is_err());
        board.build_settlement(Color::Red, 3, true).unwrap();
        board.build_city(Color::Red, 3).unwrap();
        assert_eq!(
            board.buildings.get(&3),
            Some(&(Color::Red, BuildingType::City))
        );
    }

    #[test]
    fn valid_trade_rules_match_python() {
        assert!(!is_valid_trade([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]));
        assert!(!is_valid_trade([1, 0, 0, 0, 0, 1, 0, 0, 0, 0]));
        assert!(is_valid_trade([1, 0, 0, 0, 0, 0, 1, 0, 0, 0]));
    }

    #[test]
    fn domestic_trade_requires_resources_acceptance_and_matching_confirmation() {
        let players = vec![
            Player::simple(Color::Red),
            Player::simple(Color::Blue),
            Player::simple(Color::White),
        ];
        let mut game = Game::new(players, Some(13));
        game.state.current_prompt = ActionPrompt::PlayTurn;
        game.state.is_initial_build_phase = false;
        game.state.current_turn_index = game.state.current_player_index;
        let offerer = game.state.current_color();
        let taker = game
            .state
            .colors
            .iter()
            .copied()
            .find(|color| *color != offerer)
            .unwrap();
        game.state.player_state_mut(offerer).has_rolled = true;
        game.playable_actions = generate_playable_actions(&game.state);

        let trade = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0];
        let offer = Action::new(offerer, ActionType::OfferTrade, ActionValue::Trade(trade));
        assert!(game.execute(offer.clone(), true, None).is_err());

        player_deck_replenish(&mut game.state, offerer, Resource::Wood, 1);
        player_deck_replenish(&mut game.state, taker, Resource::Brick, 1);
        game.playable_actions = generate_playable_actions(&game.state);
        game.execute(offer, true, None).unwrap();
        assert_eq!(game.state.current_prompt, ActionPrompt::DecideTrade);
        assert_eq!(game.state.current_color(), taker);
        assert!(game.state.is_resolving_trade);

        let accept = game
            .playable_actions
            .iter()
            .find(|action| action.action_type == ActionType::AcceptTrade)
            .cloned()
            .expect("taker with requested resource can accept");
        game.execute(accept, true, None).unwrap();
        assert_eq!(game.state.current_prompt, ActionPrompt::DecideTrade);

        while game.state.current_prompt == ActionPrompt::DecideTrade {
            let reject = game
                .playable_actions
                .iter()
                .find(|action| action.action_type == ActionType::RejectTrade)
                .cloned()
                .expect("non-accepting players can reject");
            game.execute(reject, true, None).unwrap();
        }
        assert_eq!(game.state.current_prompt, ActionPrompt::DecideAcceptees);

        let mut wrong_trade = trade;
        wrong_trade[0] = 2;
        let wrong_confirm = Action::new(
            offerer,
            ActionType::ConfirmTrade,
            ActionValue::ConfirmTrade(wrong_trade, taker),
        );
        assert!(game.execute(wrong_confirm, false, None).is_err());

        let unaccepted = game
            .state
            .colors
            .iter()
            .copied()
            .find(|color| *color != offerer && *color != taker)
            .unwrap();
        let unaccepted_confirm = Action::new(
            offerer,
            ActionType::ConfirmTrade,
            ActionValue::ConfirmTrade(trade, unaccepted),
        );
        assert!(game.execute(unaccepted_confirm, false, None).is_err());

        let confirm = game
            .playable_actions
            .iter()
            .find(|action| action.action_type == ActionType::ConfirmTrade)
            .cloned()
            .expect("accepted trade can be confirmed");
        game.execute(confirm, true, None).unwrap();
        assert_eq!(game.state.current_prompt, ActionPrompt::PlayTurn);
        assert!(!game.state.is_resolving_trade);
        assert_eq!(
            game.state.player_state(offerer).resources[Resource::Wood.idx()],
            0
        );
        assert_eq!(
            game.state.player_state(offerer).resources[Resource::Brick.idx()],
            1
        );
        assert_eq!(
            game.state.player_state(taker).resources[Resource::Wood.idx()],
            1
        );
        assert_eq!(
            game.state.player_state(taker).resources[Resource::Brick.idx()],
            0
        );
    }

    #[test]
    fn domestic_offer_trade_is_valid_but_not_dense_mask_reachable() {
        let players = vec![Player::simple(Color::Red), Player::simple(Color::Blue)];
        let mut game = Game::new(players, Some(31));
        while !game
            .playable_actions
            .iter()
            .any(|action| action.action_type == ActionType::Roll)
        {
            game.play_tick().unwrap();
        }

        let offerer = game.state.current_color();
        game.state.player_state_mut(offerer).has_rolled = true;
        player_deck_replenish(&mut game.state, offerer, Resource::Wood, 1);
        game.playable_actions = generate_playable_actions(&game.state);

        let trade = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0];
        let offer = Action::new(offerer, ActionType::OfferTrade, ActionValue::Trade(trade));
        assert!(is_valid_action(&game.playable_actions, &game.state, &offer));
        assert!(
            !game
                .playable_actions
                .iter()
                .any(|action| action.action_type == ActionType::OfferTrade)
        );

        let colors = game.state.colors.clone();
        let action_space = ActionSpace::new(&colors, MapKind::Base);
        let offer_index = action_space.index(&offer).unwrap();
        let mask = legal_action_mask_with_space(&game, &action_space).unwrap();
        assert_eq!(mask[offer_index], 0);
    }

    #[test]
    fn maritime_trade_rates_use_ports() {
        let hand = [4, 3, 2, 0, 0];
        let bank = [19, 19, 19, 19, 19];
        let no_ports = HashSet::new();
        let no_port_trades = inner_maritime_trade_possibilities(hand, bank, &no_ports);
        assert!(no_port_trades.iter().any(|(offer, ask)| {
            *ask == Resource::Brick
                && offer.iter().filter(|r| **r == Some(Resource::Wood)).count() == 4
        }));
        assert!(!no_port_trades.iter().any(|(offer, _)| {
            offer
                .iter()
                .filter(|r| **r == Some(Resource::Sheep))
                .count()
                == 2
        }));

        let mut ports = HashSet::new();
        ports.insert(Some(Resource::Sheep));
        let port_trades = inner_maritime_trade_possibilities(hand, bank, &ports);
        assert!(port_trades.iter().any(|(offer, ask)| {
            *ask == Resource::Ore
                && offer
                    .iter()
                    .filter(|r| **r == Some(Resource::Sheep))
                    .count()
                    == 2
        }));
    }

    #[test]
    fn initial_build_phase_reaches_play_turn() {
        let players = vec![Player::simple(Color::Red), Player::simple(Color::Blue)];
        let mut game = Game::new(players, Some(1));
        while !game
            .playable_actions
            .iter()
            .any(|a| a.action_type == ActionType::Roll)
        {
            game.play_tick().unwrap();
        }
        assert_eq!(game.state.current_prompt, ActionPrompt::PlayTurn);
        assert_eq!(game.state.player_state[0].actual_victory_points, 2);
        assert_eq!(game.state.player_state[1].actual_victory_points, 2);
        assert_eq!(game.state.board.buildings.len(), 4);
    }

    #[test]
    fn roll_seven_triggers_discard() {
        let players = vec![Player::simple(Color::Red), Player::simple(Color::Blue)];
        let mut game = Game::new(players, Some(2));
        while !game
            .playable_actions
            .iter()
            .any(|a| a.action_type == ActionType::Roll)
        {
            game.play_tick().unwrap();
        }
        let p1 = game.state.colors[1];
        let current = player_num_resource_cards(&game.state, p1, None);
        player_deck_replenish(&mut game.state, p1, Resource::Wheat, 9 - current);
        let p0 = game.state.current_color();
        game.execute(
            Action::new(p0, ActionType::Roll, ActionValue::None),
            true,
            Some(ActionValue::Dice(1, 6)),
        )
        .unwrap();
        assert_eq!(game.state.current_color(), p1);
        assert_eq!(game.state.current_prompt, ActionPrompt::Discard);
        assert!(
            game.playable_actions
                .iter()
                .all(|a| a.action_type == ActionType::DiscardResource)
        );
    }

    #[test]
    fn play_road_building_uses_two_free_roads() {
        let players = vec![Player::simple(Color::Red), Player::simple(Color::Blue)];
        let mut game = Game::new(players, Some(4));
        let p0 = game.state.colors[0];
        {
            let ps = game.state.player_state_mut(p0);
            ps.dev_cards[DevCard::RoadBuilding.idx()] = 1;
            ps.owned_at_start[DevCard::RoadBuilding.idx()] = true;
        }
        while !game
            .playable_actions
            .iter()
            .any(|a| a.action_type == ActionType::Roll)
        {
            game.play_tick().unwrap();
        }
        game.execute(
            Action::new(p0, ActionType::PlayRoadBuilding, ActionValue::None),
            false,
            None,
        )
        .unwrap();
        assert!(game.state.is_road_building);
        assert_eq!(game.state.free_roads_available, 2);
        game.play_tick().unwrap();
        assert_eq!(game.state.free_roads_available, 1);
        game.play_tick().unwrap();
        assert!(!game.state.is_road_building);
    }

    #[test]
    fn random_players_can_complete_or_hit_turn_limit() {
        let players = vec![
            Player::random(Color::Red),
            Player::random(Color::Blue),
            Player::random(Color::White),
            Player::random(Color::Orange),
        ];
        let mut game = Game::with_options(players, Some(5), 7, false, 6);
        let _ = game.play();
        assert!(game.state.num_turns <= 1000);
    }

    #[test]
    fn can_play_simple_game_ticks() {
        let players = vec![Player::simple(Color::Red), Player::simple(Color::Blue)];
        let mut game = Game::new(players, Some(3));
        for _ in 0..20 {
            game.play_tick().unwrap();
        }
    }

    #[test]
    fn feature_sample_contains_core_python_compatible_names() {
        let players = vec![Player::simple(Color::Red), Player::simple(Color::Blue)];
        let game = Game::new(players, Some(6));
        let p0 = game.state.colors[0];
        let sample = create_sample(&game, p0);

        assert_eq!(sample["P0_ACTUAL_VPS"], 0.0);
        assert_eq!(sample["P0_PUBLIC_VPS"], 0.0);
        assert_eq!(sample["P0_ROADS_LEFT"], 15.0);
        assert_eq!(sample["P0_WOOD_IN_HAND"], 0.0);
        assert_eq!(sample["BANK_DEV_CARDS"], 25.0);
        assert_eq!(sample["BANK_WOOD"], 19.0);
        assert!(sample.contains_key("TILE0_IS_WOOD"));
        assert!(sample.contains_key("TILE0_PROBA"));
        assert!(sample.contains_key("NODE0_P0_SETTLEMENT"));
        assert!(sample.contains_key("EDGE(0, 1)_P0_ROAD"));
    }

    #[test]
    fn feature_vector_follows_explicit_ordering() {
        let players = vec![Player::simple(Color::Red), Player::simple(Color::Blue)];
        let game = Game::new(players, Some(7));
        let p0 = game.state.colors[0];
        let ordering = vec![
            "BANK_DEV_CARDS".to_string(),
            "P0_ROADS_LEFT".to_string(),
            "P0_WOOD_IN_HAND".to_string(),
        ];

        let vector = create_sample_vector(&game, p0, Some(&ordering));
        assert_eq!(vector, vec![25.0, 15.0, 0.0]);
    }

    #[test]
    fn sample_vector_for_schema_preserves_width_and_zero_fills_missing_features() {
        let players = vec![Player::simple(Color::Red), Player::simple(Color::Blue)];
        let game = Game::new(players, Some(7));
        let p0 = game.state.colors[0];
        let schema = vec![
            "BANK_DEV_CARDS".to_string(),
            "MISSING_FEATURE".to_string(),
            "P0_ROADS_LEFT".to_string(),
        ];

        assert_eq!(
            create_sample_vector_for_schema(&game, p0, &schema),
            vec![25.0, 0.0, 15.0]
        );
        assert_eq!(
            create_sample_vector(&game, p0, Some(&schema)),
            vec![25.0, 15.0]
        );
    }

    #[test]
    fn sample_vector_batch_for_schema_writes_row_major_dense_rows() {
        let g0 = Game::new(
            vec![Player::simple(Color::Red), Player::simple(Color::Blue)],
            Some(1),
        );
        let g1 = Game::new(
            vec![Player::simple(Color::Red), Player::simple(Color::Blue)],
            Some(2),
        );
        let schema = feature_ordering(2, MapKind::Base);
        let samples = [
            (&g0, g0.state.current_color()),
            (&g1, g1.state.current_color()),
        ];
        let mut out = vec![0.0_f32; samples.len() * schema.len()];

        let shape = fill_sample_vector_batch_for_schema(&samples, &schema, &mut out).unwrap();

        assert_eq!(shape, (2, schema.len()));
        for (row, (game, color)) in samples.iter().enumerate() {
            let expected = create_sample_vector_for_schema(game, *color, &schema);
            let actual = &out[row * schema.len()..(row + 1) * schema.len()];
            assert_eq!(
                actual,
                expected
                    .iter()
                    .map(|value| *value as f32)
                    .collect::<Vec<_>>()
            );
        }
    }

    #[test]
    fn sample_vector_batch_le_bytes_matches_f32_rows() {
        let games = (0..16)
            .map(|seed| {
                let mut game = Game::new(
                    vec![
                        Player::simple(Color::Red),
                        Player::simple(Color::Blue),
                        Player::simple(Color::White),
                    ],
                    Some(seed),
                );
                for _ in 0..(seed as usize % 5) {
                    let _ = game.play_tick();
                }
                game
            })
            .collect::<Vec<_>>();
        let samples = games
            .iter()
            .map(|game| (game, game.state.current_color()))
            .collect::<Vec<_>>();
        let schema = feature_ordering(3, MapKind::Base);
        let mut rows = vec![0.0_f32; samples.len() * schema.len()];
        let mut bytes = vec![0_u8; rows.len() * 4];

        let row_shape = fill_sample_vector_batch_for_schema(&samples, &schema, &mut rows).unwrap();
        let byte_shape =
            fill_sample_vector_batch_for_schema_le_bytes(&samples, &schema, &mut bytes).unwrap();

        assert_eq!(byte_shape, row_shape);
        for (chunk, expected) in bytes.chunks_exact(4).zip(rows) {
            assert_eq!(f32::from_le_bytes(chunk.try_into().unwrap()), expected);
        }
    }

    #[test]
    fn sample_vector_batch_for_schema_rejects_wrong_output_length() {
        let game = Game::new(
            vec![Player::simple(Color::Red), Player::simple(Color::Blue)],
            Some(1),
        );
        let schema = feature_ordering(2, MapKind::Base);
        let samples = [(&game, game.state.current_color())];
        let mut out = vec![0.0_f32; schema.len() - 1];

        let error = fill_sample_vector_batch_for_schema(&samples, &schema, &mut out).unwrap_err();

        assert!(error.contains("sample vector batch output length mismatch"));
    }

    #[test]
    fn sample_vector_batch_le_bytes_rejects_wrong_output_length() {
        let game = Game::new(
            vec![Player::simple(Color::Red), Player::simple(Color::Blue)],
            Some(1),
        );
        let schema = feature_ordering(2, MapKind::Base);
        let samples = [(&game, game.state.current_color())];
        let mut out = vec![0_u8; schema.len() * 4 - 1];

        let error =
            fill_sample_vector_batch_for_schema_le_bytes(&samples, &schema, &mut out).unwrap_err();

        assert!(error.contains("sample vector batch output byte length mismatch"));
    }

    #[test]
    fn graph_features_mark_settlements_and_roads() {
        let players = vec![Player::simple(Color::Red), Player::simple(Color::Blue)];
        let mut game = Game::new(players, Some(8));
        let p0 = game.state.colors[0];
        let settlement = game.playable_actions[0].clone();
        let ActionValue::Node(node) = settlement.value else {
            panic!("expected initial settlement");
        };
        game.execute(settlement, true, None).unwrap();
        let road = game.playable_actions[0].clone();
        let ActionValue::Edge(edge) = road.value else {
            panic!("expected initial road");
        };
        game.execute(road, true, None).unwrap();

        let graph = graph_features(&game, p0);
        assert_eq!(graph[&format!("NODE{node}_P0_SETTLEMENT")], 1.0);
        assert_eq!(
            graph[&format!("EDGE{:?}_P0_ROAD", canonical_edge(edge))],
            1.0
        );
    }

    #[test]
    fn production_features_respect_robber_tile_only() {
        let players = vec![Player::simple(Color::Red), Player::simple(Color::Blue)];
        let mut game = Game::new(players, Some(9));
        let p0 = game.state.colors[0];

        let candidate = game
            .state
            .board
            .map
            .adjacent_tiles
            .iter()
            .find_map(|(&node, tiles)| {
                let resource = tiles.iter().find_map(|tile| tile.resource)?;
                let total = node_resource_production(&game.state.board.map, node, resource, None);
                let robbed_tile = tiles
                    .iter()
                    .find(|tile| tile.resource == Some(resource))
                    .map(|tile| tile.coordinate)?;
                let effective = node_resource_production(
                    &game.state.board.map,
                    node,
                    resource,
                    Some(robbed_tile),
                );
                (total >= effective).then_some((node, resource, robbed_tile, total, effective))
            })
            .expect("base map should have productive nodes");

        let (node, resource, robbed_tile, total, effective) = candidate;
        game.state.board.build_settlement(p0, node, true).unwrap();
        build_settlement_state(&mut game.state, p0, node, true);
        game.state.board.robber_coordinate = robbed_tile;

        let total_features = production_features(&game, p0, false);
        let effective_features = production_features(&game, p0, true);
        assert_eq!(
            total_features[&format!("TOTAL_P0_{}_PRODUCTION", resource_name(resource))],
            total
        );
        assert_eq!(
            effective_features[&format!("EFFECTIVE_P0_{}_PRODUCTION", resource_name(resource))],
            effective
        );
        assert!(total >= effective);
    }

    #[test]
    fn feature_ordering_is_stable_and_nonempty_for_base_and_mini() {
        let base = feature_ordering(2, MapKind::Base);
        let mini = feature_ordering(2, MapKind::Mini);
        assert!(base.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(mini.windows(2).all(|pair| pair[0] <= pair[1]));
        assert!(base.len() > mini.len());
        assert!(base.contains(&"BANK_DEV_CARDS".to_string()));
        assert!(mini.contains(&"BANK_DEV_CARDS".to_string()));
    }

    #[test]
    fn action_from_json_matches_python_cases() {
        let build_road = action_from_json_value(&json!(["BLUE", "BUILD_ROAD", [0, 1]])).unwrap();
        assert_eq!(
            build_road,
            Action::new(
                Color::Blue,
                ActionType::BuildRoad,
                ActionValue::Edge((0, 1))
            )
        );

        let yop = action_from_json_value(&json!(["RED", "PLAY_YEAR_OF_PLENTY", ["WOOD", "BRICK"]]))
            .unwrap();
        assert_eq!(
            yop,
            Action::new(
                Color::Red,
                ActionType::PlayYearOfPlenty,
                ActionValue::Resources(vec![Resource::Wood, Resource::Brick])
            )
        );

        let robber =
            action_from_json_value(&json!(["ORANGE", "MOVE_ROBBER", [[0, 0, 0], "RED"]])).unwrap();
        assert_eq!(
            robber,
            Action::new(
                Color::Orange,
                ActionType::MoveRobber,
                ActionValue::Robber(Coordinate(0, 0, 0), Some(Color::Red))
            )
        );

        let maritime = action_from_json_value(&json!([
            "RED",
            "MARITIME_TRADE",
            ["SHEEP", "SHEEP", "SHEEP", "SHEEP", "ORE"]
        ]))
        .unwrap();
        assert_eq!(
            maritime,
            Action::new(
                Color::Red,
                ActionType::MaritimeTrade,
                ActionValue::MaritimeTrade(
                    [
                        Some(Resource::Sheep),
                        Some(Resource::Sheep),
                        Some(Resource::Sheep),
                        Some(Resource::Sheep)
                    ],
                    Resource::Ore
                )
            )
        );
    }

    #[test]
    fn action_from_json_rejects_invalid_year_of_plenty() {
        let err = action_from_json_value(&json!([
            "WHITE",
            "PLAY_YEAR_OF_PLENTY",
            ["WOOD", "BRICK", "SHEEP"]
        ]))
        .unwrap_err();
        assert!(err.contains("Year of Plenty action must have 1 or 2 resources"));
    }

    #[test]
    fn action_json_round_trips_typed_values() {
        let actions = vec![
            Action::new(Color::Red, ActionType::BuildRoad, ActionValue::Edge((3, 4))),
            Action::new(
                Color::Blue,
                ActionType::MoveRobber,
                ActionValue::Robber(Coordinate(1, -1, 0), None),
            ),
            Action::new(
                Color::White,
                ActionType::PlayMonopoly,
                ActionValue::Resource(Resource::Ore),
            ),
            Action::new(
                Color::Orange,
                ActionType::OfferTrade,
                ActionValue::Trade([1, 0, 0, 0, 0, 0, 1, 0, 0, 0]),
            ),
        ];

        for action in actions {
            let json_value = action_to_json_value(&action);
            let parsed = action_from_json_value(&json_value).unwrap();
            assert_eq!(parsed, action);
        }
    }

    #[test]
    fn game_to_json_exposes_python_encoder_shape() {
        let players = vec![
            Player::simple(Color::Red),
            Player::simple(Color::Blue),
            Player::simple(Color::White),
            Player::simple(Color::Orange),
        ];
        let mut game = Game::new(players, Some(10));
        game.play_tick().unwrap();
        let value = game_to_json_value(&game);

        assert!(value["robber_coordinate"].is_array());
        assert!(value["tiles"].as_array().unwrap().len() >= 19);
        assert!(value["edges"].as_array().unwrap().len() >= 72);
        assert!(value["nodes"].is_object());
        assert!(value["action_records"].is_array());
        assert!(value["action_public_legal_counts"].is_array());
        assert_eq!(
            value["action_public_legal_counts"]
                .as_array()
                .unwrap()
                .len(),
            value["action_records"].as_array().unwrap().len(),
        );
        assert!(value["player_state"].is_array());
        assert!(value["colors"].is_array());
        assert!(value["current_playable_actions"].is_array());
        assert!(value["resource_bank"].is_object());
        assert!(value["current_trade"].is_array());
        assert!(value["acceptees"].is_array());
        assert!(value["development_deck_count"].as_u64().unwrap() <= 25);
        assert_eq!(value["seed"], json!(10));
        assert_eq!(value["vps_to_win"], json!(10));
        assert_eq!(value["num_turns"], json!(0));
        assert_eq!(value["current_player_index"], json!(0));
        assert_eq!(value["current_turn_index"], json!(0));
        assert_eq!(value["is_resolving_trade"], json!(false));
        assert_eq!(value["discard_limit"], json!(7));
        assert_eq!(value["friendly_robber"], json!(false));
        assert_eq!(value["state_index"], json!(1));
        assert_eq!(value["current_prompt"], json!("BUILD_INITIAL_ROAD"));
    }

    #[test]
    fn weighted_random_prefers_high_weight_actions_but_returns_playable() {
        let actions = vec![
            Action::new(Color::Red, ActionType::EndTurn, ActionValue::None),
            Action::new(Color::Red, ActionType::BuildCity, ActionValue::Node(3)),
        ];
        let mut rng = SmallRng::seed_from_u64(11);
        for _ in 0..20 {
            let action = weighted_random_action(&actions, &mut rng).unwrap();
            assert!(actions.contains(&action));
        }
    }

    #[test]
    fn roll_spectrum_has_dice_outcomes_and_probabilities() {
        let players = vec![Player::simple(Color::Red), Player::simple(Color::Blue)];
        let mut game = Game::new(players, Some(12));
        while !game
            .playable_actions
            .iter()
            .any(|a| a.action_type == ActionType::Roll)
        {
            game.play_tick().unwrap();
        }
        let roll = game
            .playable_actions
            .iter()
            .find(|action| action.action_type == ActionType::Roll)
            .unwrap()
            .clone();
        let outcomes = execute_spectrum(&game, &roll);
        let probability_sum: f64 = outcomes.iter().map(|(_, p)| *p).sum();
        assert_eq!(outcomes.len(), 11);
        assert!((probability_sum - 1.0).abs() < 1e-9);
        assert!(
            outcomes
                .iter()
                .all(|(outcome, _)| outcome.state.player_state(roll.color).has_rolled)
        );
    }

    #[test]
    fn value_function_action_returns_playable_action() {
        let players = vec![Player::simple(Color::Red), Player::simple(Color::Blue)];
        let mut game = Game::new(players, Some(13));
        while !game
            .playable_actions
            .iter()
            .any(|a| a.action_type == ActionType::Roll)
        {
            game.play_tick().unwrap();
        }
        let color = game.state.current_color();
        let mut rng = SmallRng::seed_from_u64(14);
        let action =
            value_function_action(&game, color, &game.playable_actions, &mut rng, None).unwrap();
        assert!(game.playable_actions.contains(&action));
        assert!(base_value(&game, color, ValueWeights::default()).is_finite());
    }

    #[test]
    fn alpha_beta_action_returns_playable_action() {
        let players = vec![Player::simple(Color::Red), Player::simple(Color::Blue)];
        let mut game = Game::new(players, Some(15));
        while !game
            .playable_actions
            .iter()
            .any(|a| a.action_type == ActionType::Roll)
        {
            game.play_tick().unwrap();
        }
        let color = game.state.current_color();
        let mut rng = SmallRng::seed_from_u64(16);
        let action = alpha_beta_action(&game, color, 1, false, false, &mut rng, None).unwrap();
        assert!(game.playable_actions.contains(&action));
    }

    #[test]
    fn playout_action_returns_playable_action() {
        let players = vec![Player::simple(Color::Red), Player::simple(Color::Blue)];
        let mut game = Game::new(players, Some(18));
        while !game
            .playable_actions
            .iter()
            .any(|a| a.action_type == ActionType::Roll)
        {
            game.play_tick().unwrap();
        }
        let color = game.state.current_color();
        let mut rng = SmallRng::seed_from_u64(19);
        let action = playout_action(&game, color, &game.playable_actions, &mut rng, 1).unwrap();
        assert!(game.playable_actions.contains(&action));
    }

    #[test]
    fn advanced_player_kinds_can_play_ticks() {
        let players = vec![
            Player::weighted_random(Color::Red),
            Player::victory_point(Color::Blue),
            Player::value_function(Color::White),
            Player::playout(Color::Orange, 1),
        ];
        let mut game = Game::with_options(players, Some(17), 7, false, 6);
        for _ in 0..40 {
            if game.winning_color().is_some() {
                break;
            }
            game.play_tick().unwrap();
        }
        assert!(!game.state.action_records.is_empty());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(128))]

        #[test]
        fn generated_action_json_round_trips(action in parseable_action_strategy()) {
            let encoded = action_to_json_value(&action);
            let decoded = action_from_json_value(&encoded).unwrap();
            prop_assert_eq!(decoded, action);
        }

        #[test]
        fn generated_action_sort_is_ordered_and_idempotent(mut actions in prop::collection::vec(parseable_action_strategy(), 0..80)) {
            sort_actions(&mut actions);
            prop_assert!(actions.windows(2).all(|pair| action_cmp(&pair[0], &pair[1]) != Ordering::Greater));

            let sorted_once = actions.clone();
            sort_actions(&mut actions);
            prop_assert_eq!(actions, sorted_once);
        }
    }

    fn find_simple_node_path(board: &Board, target_len: usize) -> Vec<NodeId> {
        fn dfs(board: &Board, path: &mut Vec<NodeId>, target: usize) -> bool {
            if path.len() == target {
                return true;
            }
            let last = *path.last().unwrap();
            let neighbors = board.adjacency.get(&last).cloned().unwrap_or_default();
            for neighbor in neighbors {
                if path.contains(&neighbor) {
                    continue;
                }
                path.push(neighbor);
                if dfs(board, path, target) {
                    return true;
                }
                path.pop();
            }
            false
        }
        let mut starts: Vec<NodeId> = board.adjacency.keys().copied().collect();
        starts.sort_unstable();
        for start in starts {
            let mut path = vec![start];
            if dfs(board, &mut path, target_len) {
                return path;
            }
        }
        panic!("no simple path of {target_len} nodes found");
    }

    #[test]
    fn longest_road_tie_keeps_incumbent() {
        let mut board = Board::default();
        let path = find_simple_node_path(&board, 13);
        for i in 0..5 {
            board.build_road_known_valid(Color::Blue, (path[i], path[i + 1]));
        }
        assert_eq!(board.road_color, Some(Color::Blue));
        assert_eq!(board.road_length, 5);
        for i in 6..11 {
            board.build_road_known_valid(Color::Red, (path[i], path[i + 1]));
        }
        assert_eq!(board.road_lengths.get(&Color::Red), Some(&5));
        assert_eq!(
            board.road_color,
            Some(Color::Blue),
            "incumbent must keep the award on a tie"
        );
        board.build_road_known_valid(Color::Red, (path[11], path[12]));
        assert_eq!(
            board.road_color,
            Some(Color::Red),
            "strictly longer road must take the award"
        );
        assert_eq!(board.road_length, 6);
    }

    #[test]
    fn settlement_cut_below_five_clears_award_on_board() {
        let mut board = Board::default();
        let path = find_simple_node_path(&board, 6);
        for i in 0..5 {
            board.build_road_known_valid(Color::Blue, (path[i], path[i + 1]));
        }
        assert_eq!(board.road_color, Some(Color::Blue));
        let (previous, road_color, _lengths) =
            board.build_settlement_known_valid(Color::Red, path[2], false);
        assert_eq!(previous, Some(Color::Blue));
        assert_eq!(road_color, None);
        assert_eq!(board.road_color, None);
        assert_eq!(board.road_length, 0);
    }

    #[test]
    fn maintain_longest_road_revokes_points_when_no_qualifier_remains() {
        let players = vec![Player::simple(Color::Red), Player::simple(Color::Blue)];
        let mut game = Game::new(players, Some(1));
        {
            let ps = game.state.player_state_mut(Color::Blue);
            ps.has_road = true;
            ps.victory_points += 2;
            ps.actual_victory_points += 2;
        }
        let vp_before = game.state.player_state(Color::Blue).victory_points;
        let lengths = HashMap::from([(Color::Blue, 3usize)]);
        maintain_longest_road(&mut game.state, Some(Color::Blue), None, &lengths);
        let ps = game.state.player_state(Color::Blue);
        assert!(
            !ps.has_road,
            "award must be revoked when holder drops below 5"
        );
        assert_eq!(ps.victory_points, vp_before - 2);
        assert_eq!(ps.longest_road_length, 3);
    }

    #[test]
    fn road_capped_by_enemy_buildings_at_both_ends_counts_all_edges() {
        let mut board = Board::default();
        let path = find_simple_node_path(&board, 7);
        for i in 0..6 {
            board.build_road_known_valid(Color::Red, (path[i], path[i + 1]));
        }
        assert_eq!(board.road_lengths.get(&Color::Red), Some(&6));
        board.build_settlement_known_valid(Color::Blue, path[0], false);
        assert_eq!(
            board.road_lengths.get(&Color::Red),
            Some(&6),
            "single enemy cap must not shorten the road"
        );
        board.build_settlement_known_valid(Color::Blue, path[6], false);
        assert_eq!(
            board.road_lengths.get(&Color::Red),
            Some(&6),
            "both-ends enemy caps must not shorten the road"
        );
        assert_eq!(board.road_color, Some(Color::Red));
    }

    #[test]
    fn robber_steal_spectrum_weighted_by_victim_hand() {
        let players = vec![Player::simple(Color::Red), Player::simple(Color::Blue)];
        let mut game = Game::new(players, Some(3));
        game.state.player_state_mut(Color::Blue).resources = [2, 0, 1, 0, 0];
        let coordinate = game
            .state
            .board
            .map
            .land_tiles
            .keys()
            .copied()
            .find(|c| *c != game.state.board.robber_coordinate)
            .unwrap();
        let action = Action::new(
            Color::Red,
            ActionType::MoveRobber,
            ActionValue::Robber(coordinate, Some(Color::Blue)),
        );
        let outcomes = execute_spectrum(&game, &action);
        assert_eq!(outcomes.len(), 2, "zero-count resources must be omitted");
        let total: f64 = outcomes.iter().map(|(_, p)| *p).sum();
        assert!((total - 1.0).abs() < 1e-9);
        for (outcome, _) in &outcomes {
            assert_eq!(outcome.state.board.robber_coordinate, coordinate);
        }
        let (wood_game, wood_probability) = outcomes
            .iter()
            .find(|(g, _)| g.state.player_state(Color::Blue).resources[Resource::Wood.idx()] == 1)
            .expect("wood steal outcome");
        assert!((wood_probability - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(
            wood_game.state.player_state(Color::Red).resources[Resource::Wood.idx()],
            1
        );
        let (sheep_game, sheep_probability) = outcomes
            .iter()
            .find(|(g, _)| g.state.player_state(Color::Blue).resources[Resource::Sheep.idx()] == 0)
            .expect("sheep steal outcome");
        assert!((sheep_probability - 1.0 / 3.0).abs() < 1e-9);
        assert_eq!(
            sheep_game.state.player_state(Color::Red).resources[Resource::Sheep.idx()],
            1
        );
    }

    #[test]
    fn robber_steal_spectrum_empty_hand_moves_robber_without_steal() {
        let players = vec![Player::simple(Color::Red), Player::simple(Color::Blue)];
        let mut game = Game::new(players, Some(3));
        game.state.player_state_mut(Color::Blue).resources = [0; 5];
        let red_before = game.state.player_state(Color::Red).resources;
        let coordinate = game
            .state
            .board
            .map
            .land_tiles
            .keys()
            .copied()
            .find(|c| *c != game.state.board.robber_coordinate)
            .unwrap();
        let action = Action::new(
            Color::Red,
            ActionType::MoveRobber,
            ActionValue::Robber(coordinate, Some(Color::Blue)),
        );
        let outcomes = execute_spectrum(&game, &action);
        assert_eq!(outcomes.len(), 1);
        assert!((outcomes[0].1 - 1.0).abs() < 1e-9);
        assert_eq!(
            outcomes[0].0.state.board.robber_coordinate, coordinate,
            "robber must still move when the victim has no cards"
        );
        assert_eq!(
            outcomes[0].0.state.player_state(Color::Red).resources,
            red_before
        );
        assert_eq!(
            outcomes[0].0.state.player_state(Color::Blue).resources,
            [0; 5]
        );
    }

    #[test]
    fn dev_card_spectrum_matches_remaining_deck_exactly() {
        let players = vec![Player::simple(Color::Red), Player::simple(Color::Blue)];
        let mut game = Game::new(players, Some(5));
        game.state.player_state_mut(Color::Red).resources = [0, 0, 1, 1, 1];
        // Opponent hand cards must not leak into the deck spectrum.
        game.state.player_state_mut(Color::Blue).dev_cards[DevCard::VictoryPoint.idx()] = 2;
        game.state.development_listdeck = vec![DevCard::Knight, DevCard::Monopoly, DevCard::Knight];
        let action = Action::new(
            Color::Red,
            ActionType::BuyDevelopmentCard,
            ActionValue::None,
        );
        let outcomes = execute_spectrum(&game, &action);
        assert_eq!(
            outcomes.len(),
            2,
            "exactly the distinct cards remaining in the deck"
        );
        let total: f64 = outcomes.iter().map(|(_, p)| *p).sum();
        assert!(
            (total - 1.0).abs() < 1e-9,
            "probabilities must sum to 1, got {total}"
        );
        for (outcome, probability) in &outcomes {
            let red = outcome.state.player_state(Color::Red);
            let drawn: Vec<DevCard> = DevCard::ALL
                .into_iter()
                .filter(|card| red.dev_cards[card.idx()] > 0)
                .collect();
            assert_eq!(drawn.len(), 1, "every outcome must actually draw a card");
            assert_eq!(red.dev_cards[drawn[0].idx()], 1);
            assert_eq!(
                outcome.state.development_listdeck.len(),
                2,
                "every outcome must decrement the deck"
            );
            assert_eq!(red.resources, [0; 5], "outcome must pay the card cost");
            match drawn[0] {
                DevCard::Knight => assert!((probability - 2.0 / 3.0).abs() < 1e-9),
                DevCard::Monopoly => assert!((probability - 1.0 / 3.0).abs() < 1e-9),
                other => panic!("unexpected card {other:?} in spectrum"),
            }
        }
    }

    #[test]
    fn dev_card_spectrum_single_card_deck() {
        let players = vec![Player::simple(Color::Red), Player::simple(Color::Blue)];
        let mut game = Game::new(players, Some(5));
        game.state.player_state_mut(Color::Red).resources = [0, 0, 1, 1, 1];
        game.state.development_listdeck = vec![DevCard::YearOfPlenty];
        let action = Action::new(
            Color::Red,
            ActionType::BuyDevelopmentCard,
            ActionValue::None,
        );
        let outcomes = execute_spectrum(&game, &action);
        assert_eq!(outcomes.len(), 1);
        assert!((outcomes[0].1 - 1.0).abs() < 1e-9);
        let red = outcomes[0].0.state.player_state(Color::Red);
        assert_eq!(red.dev_cards[DevCard::YearOfPlenty.idx()], 1);
        assert!(outcomes[0].0.state.development_listdeck.is_empty());
    }

    #[test]
    fn a15_parity_settlement_split_revokes_card_and_points() {
        let players = vec![Player::simple(Color::Red), Player::simple(Color::Blue)];
        let mut game = Game::new(players, Some(2));
        let path = find_simple_node_path(&game.state.board, 7);
        for i in 0..6 {
            let (previous, road_color, lengths) = game
                .state
                .board
                .build_road_known_valid(Color::Red, (path[i], path[i + 1]));
            maintain_longest_road(&mut game.state, previous, road_color, &lengths);
        }
        let red = game.state.player_state(Color::Red);
        assert!(red.has_road);
        assert_eq!(red.longest_road_length, 6);
        let vp_with_award = red.victory_points;

        // BLUE settlement mid-chain splits the road 3/3: below 5 with no
        // other qualifier, so the card and both points must be revoked.
        let (previous, road_color, lengths) =
            game.state
                .board
                .build_settlement_known_valid(Color::Blue, path[3], false);
        maintain_longest_road(&mut game.state, previous, road_color, &lengths);
        let red = game.state.player_state(Color::Red);
        assert_eq!(red.longest_road_length, 3);
        assert!(!red.has_road, "card must be revoked after the 3/3 split");
        assert_eq!(red.victory_points, vp_with_award - 2);
    }

    #[test]
    fn a16_parity_double_cut_leaves_both_ends_capped_middle_segment() {
        let mut board = Board::default();
        let path = find_simple_node_path(&board, 12);
        for i in 0..11 {
            board.build_road_known_valid(Color::Red, (path[i], path[i + 1]));
        }
        assert_eq!(board.road_lengths.get(&Color::Red), Some(&11));
        // Two interior cuts leave segments of 3, 6 (capped at both ends by
        // the enemy settlements) and 2; the middle one is the longest.
        board.build_settlement_known_valid(Color::Blue, path[3], false);
        board.build_settlement_known_valid(Color::Blue, path[9], false);
        assert_eq!(
            board.road_lengths.get(&Color::Red),
            Some(&6),
            "both-ends-capped middle segment must count all 6 edges"
        );
        assert_eq!(board.road_color, Some(Color::Red));
        assert_eq!(board.road_length, 6);
    }

    #[test]
    fn spectrum_probabilities_match_execute_spectrum_across_playouts() {
        for seed in [21u64, 22, 23] {
            let players = vec![Player::simple(Color::Red), Player::simple(Color::Blue)];
            let mut game = Game::new(players, Some(seed));
            for _ in 0..300 {
                if game.winning_color().is_some() {
                    break;
                }
                for action in game.playable_actions.clone() {
                    if deterministic_action_type(action.action_type) {
                        continue;
                    }
                    let expected: Vec<f64> = execute_spectrum(&game, &action)
                        .iter()
                        .map(|(_, probability)| *probability)
                        .collect();
                    let fast = spectrum_probabilities(&game, &action);
                    assert_eq!(
                        fast.len(),
                        expected.len(),
                        "outcome count mismatch for {action:?}"
                    );
                    for (a, b) in fast.iter().zip(expected.iter()) {
                        assert!((a - b).abs() < 1e-12, "probability mismatch for {action:?}");
                    }
                }
                game.play_tick().unwrap();
            }
        }
    }

    #[test]
    fn spectrum_probabilities_known_cases() {
        let players = vec![Player::simple(Color::Red), Player::simple(Color::Blue)];
        let mut game = Game::new(players, Some(7));
        // deterministic action: single certain outcome
        let end_turn = Action::new(Color::Red, ActionType::EndTurn, ActionValue::None);
        assert_eq!(spectrum_probabilities(&game, &end_turn), vec![1.0]);
        // roll: the 11 dice totals
        let roll = Action::new(Color::Red, ActionType::Roll, ActionValue::None);
        let roll_probabilities = spectrum_probabilities(&game, &roll);
        assert_eq!(roll_probabilities.len(), 11);
        assert!((roll_probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-9);
        // steal weighted by the victim hand
        game.state.player_state_mut(Color::Blue).resources = [2, 0, 1, 0, 0];
        let coordinate = game
            .state
            .board
            .map
            .land_tiles
            .keys()
            .copied()
            .find(|c| *c != game.state.board.robber_coordinate)
            .unwrap();
        let rob = Action::new(
            Color::Red,
            ActionType::MoveRobber,
            ActionValue::Robber(coordinate, Some(Color::Blue)),
        );
        assert_eq!(
            spectrum_probabilities(&game, &rob),
            vec![2.0 / 3.0, 1.0 / 3.0]
        );
        // empty hand: single no-steal outcome
        game.state.player_state_mut(Color::Blue).resources = [0; 5];
        assert_eq!(spectrum_probabilities(&game, &rob), vec![1.0]);
        // dev deck composition
        game.state.development_listdeck = vec![DevCard::Knight, DevCard::Monopoly, DevCard::Knight];
        let buy = Action::new(
            Color::Red,
            ActionType::BuyDevelopmentCard,
            ActionValue::None,
        );
        assert_eq!(
            spectrum_probabilities(&game, &buy),
            vec![2.0 / 3.0, 1.0 / 3.0]
        );
    }

    #[test]
    fn decision_context_lists_indices_actions_and_chance_spectrums() {
        let players = vec![Player::simple(Color::Red), Player::simple(Color::Blue)];
        let mut game = Game::new(players, Some(12));
        while !game
            .playable_actions
            .iter()
            .any(|a| a.action_type == ActionType::Roll)
        {
            game.play_tick().unwrap();
        }
        let colors = [Color::Red, Color::Blue];
        let context = decision_context_json_value(&game, &colors, MapKind::Base, true).unwrap();
        assert_eq!(
            context["current_color"],
            json!(color_name(game.state.current_color()))
        );
        let actions = context["actions"].as_array().unwrap();
        assert_eq!(actions.len(), game.playable_actions.len());
        let space = ActionSpace::new(&colors, MapKind::Base);
        for (entry, action) in actions.iter().zip(game.playable_actions.iter()) {
            assert_eq!(
                entry["index"].as_u64().unwrap() as usize,
                space.index(action).unwrap()
            );
            assert_eq!(entry["action"], action_to_json_value(action));
            if deterministic_action_type(action.action_type) {
                assert!(entry.get("spectrum").is_none());
            } else {
                let spectrum = entry["spectrum"].as_array().unwrap();
                let total: f64 = spectrum.iter().map(|p| p.as_f64().unwrap()).sum();
                assert!((total - 1.0).abs() < 1e-9);
                assert_eq!(
                    spectrum.len(),
                    execute_spectrum(&game, action).len(),
                    "spectrum length must match execute_spectrum for {action:?}"
                );
            }
        }
        // spectrums can be skipped entirely
        let bare = decision_context_json_value(&game, &colors, MapKind::Base, false).unwrap();
        for entry in bare["actions"].as_array().unwrap() {
            assert!(entry.get("spectrum").is_none());
        }
    }

    #[test]
    fn execute_spectrum_games_matches_per_outcome_application() {
        let players = vec![Player::simple(Color::Red), Player::simple(Color::Blue)];
        let mut game = Game::new(players, Some(12));
        while !game
            .playable_actions
            .iter()
            .any(|a| a.action_type == ActionType::Roll)
        {
            game.play_tick().unwrap();
        }
        let roll = game
            .playable_actions
            .iter()
            .find(|action| action.action_type == ActionType::Roll)
            .unwrap()
            .clone();
        let reference = execute_spectrum(&game, &roll);
        // all outcomes, spectrum order
        let all = execute_spectrum_games(&game, &roll, None).unwrap();
        assert_eq!(all.len(), reference.len());
        for (batch_game, (reference_game, _)) in all.iter().zip(reference.iter()) {
            assert_eq!(
                game_to_json_value(batch_game),
                game_to_json_value(reference_game)
            );
        }
        // an explicit subset, including a repeated index
        let subset = execute_spectrum_games(&game, &roll, Some(&[10, 0, 0])).unwrap();
        assert_eq!(subset.len(), 3);
        assert_eq!(
            game_to_json_value(&subset[0]),
            game_to_json_value(&reference[10].0)
        );
        assert_eq!(
            game_to_json_value(&subset[1]),
            game_to_json_value(&reference[0].0)
        );
        assert_eq!(
            game_to_json_value(&subset[2]),
            game_to_json_value(&reference[0].0)
        );
        // out of range must error, not panic
        assert!(execute_spectrum_games(&game, &roll, Some(&[11])).is_err());
    }
}
#[cfg(test)]
mod public_card_deduction_tests {
    use super::*;

    fn two_player_public_fixture() -> (Game, Color, Color) {
        let mut game = Game::new(
            vec![Player::simple(Color::Red), Player::simple(Color::Blue)],
            Some(91),
        );
        let observer = game.state.current_color();
        let opponent = game
            .state
            .colors
            .iter()
            .copied()
            .find(|color| *color != observer)
            .unwrap();

        let observer_resources = [1, 2, 0, 1, 0];
        let opponent_resources = [3, 0, 2, 1, 4];
        game.state.player_state_mut(observer).resources = observer_resources;
        game.state.player_state_mut(opponent).resources = opponent_resources;
        for index in 0..5 {
            game.state.resource_freqdeck[index] = starting_resource_bank()[index]
                - observer_resources[index]
                - opponent_resources[index];
        }

        game.state.player_state_mut(observer).dev_cards[DevCard::Knight.idx()] = 1;
        game.state.player_state_mut(observer).played_dev_cards[DevCard::YearOfPlenty.idx()] = 1;
        game.state.player_state_mut(opponent).played_dev_cards[DevCard::Knight.idx()] = 2;
        game.state.player_state_mut(opponent).dev_cards[DevCard::Monopoly.idx()] = 1;
        game.state.player_state_mut(opponent).dev_cards[DevCard::VictoryPoint.idx()] = 1;
        // 25 base - 3 public plays - 1 observer-known - 2 opponent-hidden.
        // Only the length is public; identities/order must not be consumed.
        game.state.development_listdeck = vec![DevCard::RoadBuilding; 19];
        (game, observer, opponent)
    }

    #[test]
    fn two_player_public_card_deductions_are_exact_for_resources_only() {
        let (game, observer, opponent) = two_player_public_fixture();
        let value = public_card_deductions_to_json_value(&game, observer).unwrap();

        assert_eq!(value["contract"], json!("public_card_deductions_2p_v1"));
        assert_eq!(value["opponent"], json!(color_name(opponent)));
        assert_eq!(value["resource_composition_exact"], json!(true));
        assert_eq!(value["opponent_resource_card_count"], json!(10));
        assert_eq!(value["opponent_resources"]["WOOD"], json!(3));
        assert_eq!(value["opponent_resources"]["BRICK"], json!(0));
        assert_eq!(value["opponent_resources"]["SHEEP"], json!(2));
        assert_eq!(value["opponent_resources"]["WHEAT"], json!(1));
        assert_eq!(value["opponent_resources"]["ORE"], json!(4));

        assert_eq!(value["development_composition_exact"], json!(false));
        assert_eq!(value["opponent_face_down_development_card_count"], json!(2));
        assert_eq!(value["development_deck_count"], json!(19));
        assert_eq!(value["unknown_development_pool_count"], json!(21));
        assert!(value.get("opponent_development_cards").is_none());
    }

    #[test]
    fn public_card_deductions_ignore_authoritative_hidden_identities() {
        let (first, observer, opponent) = two_player_public_fixture();
        let mut second = first.clone();

        // Preserve every public aggregate while permuting authoritative hidden
        // resource and development identities and the physical deck order.
        // The observer-scoped deduction surface must be byte-identical.
        second.state.player_state_mut(opponent).resources = [0, 4, 1, 2, 3];
        second.state.player_state_mut(opponent).dev_cards = [0; 5];
        second.state.player_state_mut(opponent).dev_cards[DevCard::Knight.idx()] = 2;
        second.state.development_listdeck = vec![DevCard::VictoryPoint; 19];

        assert_eq!(
            public_card_deductions_to_json_value(&first, observer).unwrap(),
            public_card_deductions_to_json_value(&second, observer).unwrap(),
        );
    }

    #[test]
    fn public_card_deductions_fail_closed_outside_two_player_track() {
        let game = Game::new(
            vec![
                Player::simple(Color::Red),
                Player::simple(Color::Blue),
                Player::simple(Color::Orange),
            ],
            Some(92),
        );
        let error =
            public_card_deductions_to_json_value(&game, game.state.current_color()).unwrap_err();
        assert!(error.contains("exactly two players"));
    }
}

#[cfg(test)]
mod public_belief_determinization_tests {
    use super::*;

    fn simple_three_player_game(seed: u64) -> Game {
        Game::new(
            vec![
                Player::simple(Color::Red),
                Player::simple(Color::Blue),
                Player::simple(Color::Orange),
            ],
            Some(seed),
        )
    }

    fn opponents(game: &Game, observer: Color) -> Vec<Color> {
        game.state
            .colors
            .iter()
            .copied()
            .filter(|color| *color != observer)
            .collect()
    }

    fn assert_public_conservation(game: &Game) {
        let resource_bank = starting_resource_bank();
        for resource in Resource::ALL {
            let index = resource.idx();
            let held = game
                .state
                .colors
                .iter()
                .map(|color| u16::from(game.state.player_state(*color).resources[index]))
                .sum::<u16>();
            assert_eq!(
                u16::from(game.state.resource_freqdeck[index]) + held,
                u16::from(resource_bank[index]),
                "resource {resource:?} must be conserved",
            );
        }

        let mut expected_devs = [0_u16; 5];
        for card in starting_devcard_bank() {
            expected_devs[card.idx()] += 1;
        }
        let mut observed_devs = [0_u16; 5];
        for card in &game.state.development_listdeck {
            observed_devs[card.idx()] += 1;
        }
        for color in &game.state.colors {
            let player = game.state.player_state(*color);
            for index in 0..5 {
                observed_devs[index] += u16::from(player.dev_cards[index]);
                observed_devs[index] += u16::from(player.played_dev_cards[index]);
            }
        }
        assert_eq!(
            observed_devs, expected_devs,
            "development cards must be conserved"
        );
    }

    #[test]
    fn public_belief_determinization_ignores_authoritative_hidden_allocation() {
        let mut first = simple_three_player_game(17);
        let observer = first.state.current_color();
        let opponents = opponents(&first, observer);
        assert_eq!(opponents.len(), 2);

        // Exercise preservation of a non-empty observer hand as well as the
        // opponent-hidden allocation. The observer's known knight must be
        // removed from the unknown deck before sampling.
        first.state.player_state_mut(observer).resources = [1, 0, 1, 0, 1];
        first.state.player_state_mut(observer).dev_cards[DevCard::Knight.idx()] = 1;
        first.state.player_state_mut(observer).owned_at_start[DevCard::Knight.idx()] = true;
        for index in 0..5 {
            first.state.resource_freqdeck[index] -=
                first.state.player_state(observer).resources[index];
        }

        // Two authoritative worlds with the same public state/counts but a
        // different hidden assignment of one knight and one victory-point card.
        first.state.player_state_mut(opponents[0]).dev_cards[DevCard::Knight.idx()] = 1;
        first
            .state
            .player_state_mut(opponents[0])
            .actual_victory_points = 0;
        first.state.player_state_mut(opponents[1]).dev_cards[DevCard::VictoryPoint.idx()] = 1;
        first
            .state
            .player_state_mut(opponents[1])
            .actual_victory_points = 1;
        let mut deck = starting_devcard_bank();
        deck.remove(
            deck.iter()
                .position(|card| *card == DevCard::Knight)
                .unwrap(),
        );
        deck.remove(
            deck.iter()
                .position(|card| *card == DevCard::Knight)
                .unwrap(),
        );
        deck.remove(
            deck.iter()
                .position(|card| *card == DevCard::VictoryPoint)
                .unwrap(),
        );
        first.state.development_listdeck = deck;

        let mut second = first.clone();
        second.state.player_state_mut(opponents[0]).dev_cards = [0; 5];
        second.state.player_state_mut(opponents[0]).dev_cards[DevCard::VictoryPoint.idx()] = 1;
        second
            .state
            .player_state_mut(opponents[0])
            .actual_victory_points = 1;
        second.state.player_state_mut(opponents[1]).dev_cards = [0; 5];
        second.state.player_state_mut(opponents[1]).dev_cards[DevCard::Knight.idx()] = 1;
        second
            .state
            .player_state_mut(opponents[1])
            .actual_victory_points = 0;

        // Authoritative histories may encode a hidden dev draw/result. The
        // determinized snapshot must redact that payload while retaining public
        // actor/action sequence and history length.
        first.state.action_records.push(ActionRecord {
            action: Action::new(observer, ActionType::BuyDevelopmentCard, ActionValue::None),
            result: ActionValue::DevCard(DevCard::Knight),
        });
        second.state.action_records.push(ActionRecord {
            action: Action::new(observer, ActionType::BuyDevelopmentCard, ActionValue::None),
            result: ActionValue::DevCard(DevCard::VictoryPoint),
        });

        let first_before = format!("{first:?}");
        let second_before = format!("{second:?}");
        let sampled_first = first.determinize_for_player(observer, 123_456).unwrap();
        let sampled_second = second.determinize_for_player(observer, 123_456).unwrap();
        assert_eq!(
            format!("{first:?}"),
            first_before,
            "authoritative game mutated"
        );
        assert_eq!(
            format!("{second:?}"),
            second_before,
            "authoritative comparison world mutated",
        );
        assert_eq!(
            game_to_json_value(&sampled_first),
            game_to_json_value(&sampled_second),
            "same information set + seed must yield the same sampled world",
        );
        let public_draw = sampled_first.state.action_records.last().unwrap();
        assert_eq!(
            public_draw.action.action_type,
            ActionType::BuyDevelopmentCard,
            "determinization must preserve the public action taxonomy",
        );
        assert_eq!(public_draw.action.value, ActionValue::None);
        assert_eq!(public_draw.result, ActionValue::None);
        assert_eq!(sampled_first.playable_actions, first.playable_actions);
        assert_eq!(
            sampled_first.state.player_state(observer).dev_cards,
            first.state.player_state(observer).dev_cards,
        );
        assert_eq!(
            sampled_first.state.player_state(observer).resources,
            first.state.player_state(observer).resources,
        );
        assert_eq!(
            sampled_first.state.player_state(observer).owned_at_start,
            first.state.player_state(observer).owned_at_start,
        );
        for color in &opponents {
            let player = sampled_first.state.player_state(*color);
            assert_eq!(
                player.actual_victory_points,
                player.victory_points + i16::from(player.dev_cards[DevCard::VictoryPoint.idx()]),
                "sampled hidden victory points must repair actual VP",
            );
            assert_eq!(
                player.owned_at_start,
                player.dev_cards.map(|count| count > 0),
                "sampled opponent playability metadata must match its hand",
            );
        }
        let hidden_and_deck = sampled_first.state.development_listdeck.len()
            + opponents
                .iter()
                .map(|color| {
                    sampled_first
                        .state
                        .player_state(*color)
                        .dev_cards
                        .iter()
                        .map(|value| usize::from(*value))
                        .sum::<usize>()
                })
                .sum::<usize>();
        assert_eq!(hidden_and_deck, 24);
        assert_public_conservation(&sampled_first);
    }

    #[test]
    fn resource_determinization_depends_only_on_public_counts_and_seed() {
        let mut first = simple_three_player_game(71);
        let observer = first.state.current_color();
        let opponents = opponents(&first, observer);
        assert_eq!(opponents.len(), 2);

        let observer_hand = [1, 0, 1, 0, 1];
        let first_hands = [[4, 1, 0, 0, 0], [0, 2, 1, 1, 0]];
        let second_hands = [[1, 2, 1, 1, 0], [3, 1, 0, 0, 0]];
        let aggregate_opponent = [4, 3, 1, 1, 0];
        first.state.player_state_mut(observer).resources = observer_hand;
        for (color, hand) in opponents.iter().zip(first_hands) {
            first.state.player_state_mut(*color).resources = hand;
        }
        for index in 0..5 {
            first.state.resource_freqdeck[index] =
                starting_resource_bank()[index] - observer_hand[index] - aggregate_opponent[index];
        }

        let mut second = first.clone();
        for (color, hand) in opponents.iter().zip(second_hands) {
            second.state.player_state_mut(*color).resources = hand;
        }

        let first_before = format!("{first:?}");
        let second_before = format!("{second:?}");
        let sampled_first = first.determinize_for_player(observer, 9_991).unwrap();
        let sampled_second = second.determinize_for_player(observer, 9_991).unwrap();

        assert_eq!(
            game_to_json_value(&sampled_first),
            game_to_json_value(&sampled_second),
            "same public resource counts + seed must ignore true allocation",
        );
        assert_eq!(
            format!("{first:?}"),
            first_before,
            "authoritative game mutated"
        );
        assert_eq!(
            format!("{second:?}"),
            second_before,
            "comparison game mutated"
        );
        assert_eq!(
            sampled_first.state.player_state(observer).resources,
            observer_hand,
            "observer resource hand must be preserved exactly",
        );
        for (color, original_hand) in opponents.iter().zip(first_hands) {
            assert_eq!(
                sampled_first
                    .state
                    .player_state(*color)
                    .resources
                    .iter()
                    .copied()
                    .sum::<u8>(),
                original_hand.iter().copied().sum::<u8>(),
                "opponent public hand size must be preserved",
            );
        }
        assert_public_conservation(&sampled_first);

        let baseline = opponents
            .iter()
            .map(|color| sampled_first.state.player_state(*color).resources)
            .collect::<Vec<_>>();
        let has_diverse_seed = (0_u64..64).any(|seed| {
            let sampled = first.determinize_for_player(observer, seed).unwrap();
            opponents
                .iter()
                .map(|color| sampled.state.player_state(*color).resources)
                .collect::<Vec<_>>()
                != baseline
        });
        assert!(
            has_diverse_seed,
            "a non-degenerate hidden pool must produce particle diversity across seeds",
        );
    }

    fn two_player_buy_game(hidden_cards: [u8; 5]) -> (Game, Color, Action) {
        let mut game = Game::new(
            vec![Player::simple(Color::Red), Player::simple(Color::Blue)],
            Some(79),
        );
        let observer = game.state.current_color();
        let opponent = game
            .state
            .colors
            .iter()
            .copied()
            .find(|color| *color != observer)
            .unwrap();

        game.state.is_initial_build_phase = false;
        game.state.current_prompt = ActionPrompt::PlayTurn;
        game.state.player_state_mut(observer).has_rolled = true;
        let buy_resources = [0, 0, 1, 1, 1];
        game.state.player_state_mut(observer).resources = buy_resources;
        for index in 0..5 {
            game.state.resource_freqdeck[index] -= buy_resources[index];
        }

        {
            let opponent_state = game.state.player_state_mut(opponent);
            opponent_state.dev_cards = hidden_cards;
            opponent_state.owned_at_start = hidden_cards.map(|count| count > 0);
            opponent_state.actual_victory_points = opponent_state.victory_points
                + i16::from(hidden_cards[DevCard::VictoryPoint.idx()]);
        }
        let mut deck = starting_devcard_bank();
        for card in DevCard::ALL {
            for _ in 0..hidden_cards[card.idx()] {
                let position = deck
                    .iter()
                    .position(|candidate| *candidate == card)
                    .unwrap();
                deck.remove(position);
            }
        }
        game.state.development_listdeck = deck;
        game.playable_actions = generate_playable_actions(&game.state);
        let action = game
            .playable_actions
            .iter()
            .find(|action| action.action_type == ActionType::BuyDevelopmentCard)
            .cloned()
            .expect("buy-development-card action must be legal in fixture");
        (game, observer, action)
    }

    #[test]
    fn public_belief_dev_draws_cover_support_and_ignore_authoritative_deck() {
        // In `first`, every ROAD_BUILDING card is authoritatively hidden in the
        // opponent hand. In `second`, every MONOPOLY is hidden instead. Public
        // hand/deck sizes are identical, so all five conditioned successors
        // must still exist and match exactly for the same seed.
        let mut road_hidden = [0_u8; 5];
        road_hidden[DevCard::RoadBuilding.idx()] = 2;
        let mut monopoly_hidden = [0_u8; 5];
        monopoly_hidden[DevCard::Monopoly.idx()] = 2;
        let (first, observer, action) = two_player_buy_game(road_hidden);
        let mut second = first.clone();
        let opponent = second
            .state
            .colors
            .iter()
            .copied()
            .find(|color| *color != observer)
            .unwrap();
        {
            let opponent_state = second.state.player_state_mut(opponent);
            opponent_state.dev_cards = monopoly_hidden;
            opponent_state.owned_at_start = monopoly_hidden.map(|count| count > 0);
            opponent_state.actual_victory_points = opponent_state.victory_points;
        }
        let mut second_deck = starting_devcard_bank();
        for _ in 0..2 {
            let position = second_deck
                .iter()
                .position(|candidate| *candidate == DevCard::Monopoly)
                .unwrap();
            second_deck.remove(position);
        }
        second.state.development_listdeck = second_deck;

        let first_children = first
            .public_belief_development_draws(observer, &action, &DevCard::ALL, 45_901)
            .unwrap();
        let second_children = second
            .public_belief_development_draws(observer, &action, &DevCard::ALL, 45_901)
            .unwrap();
        assert_eq!(first_children.len(), DevCard::ALL.len());
        assert_eq!(second_children.len(), DevCard::ALL.len());

        for (card, (first_child, second_child)) in DevCard::ALL
            .iter()
            .zip(first_children.iter().zip(second_children.iter()))
        {
            let first_json = game_to_json_value(first_child);
            let second_json = game_to_json_value(second_child);
            assert_eq!(
                first_json, second_json,
                "conditioned {card:?} child leaked authoritative hidden allocation",
            );
            assert_eq!(
                first_child.state.player_state(observer).dev_cards[card.idx()],
                1,
                "observer must receive the requested {card:?}",
            );
            assert_eq!(first_child.state.development_listdeck.len(), 22);
            assert_public_conservation(first_child);
        }
    }

    #[test]
    fn determinization_fails_closed_on_inconsistent_public_counts() {
        let mut bad_resources = simple_three_player_game(81);
        let observer = bad_resources.state.current_color();
        bad_resources.state.player_state_mut(observer).resources[Resource::Wood.idx()] = 1;
        let error = bad_resources
            .determinize_for_player(observer, 1)
            .unwrap_err();
        assert!(
            error.contains("public resource conservation exceeds base bank"),
            "unexpected resource-conservation error: {error}",
        );

        let mut bad_devs = simple_three_player_game(82);
        let observer = bad_devs.state.current_color();
        let opponent = opponents(&bad_devs, observer)[0];
        bad_devs.state.player_state_mut(opponent).dev_cards[DevCard::Knight.idx()] = 1;
        let error = bad_devs.determinize_for_player(observer, 1).unwrap_err();
        assert!(
            error.contains("public development-card counts are not conservation-consistent"),
            "unexpected development-card conservation error: {error}",
        );
    }
}
