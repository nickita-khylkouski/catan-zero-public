import pytest

from catanatron.models.map import MINI_MAP_TEMPLATE, CatanMap
from catanatron.models.enums import RESOURCES
from catanatron.models.board import Board, get_node_distances, longest_acyclic_path
from catanatron.models.player import Color


def test_initial_build_phase_bypasses_restrictions():
    board = Board()
    with pytest.raises(ValueError):  # not connected and not initial-placement
        board.build_settlement(Color.RED, 3)
    with pytest.raises(ValueError):  # not connected to settlement
        board.build_road(Color.RED, (3, 2))

    board.build_settlement(Color.RED, 3, initial_build_phase=True)


def test_roads_must_always_be_connected():
    board = Board()
    board.build_settlement(Color.RED, 3, initial_build_phase=True)

    with pytest.raises(ValueError):  # not connected to settlement
        board.build_road(Color.RED, (2, 1))
    board.build_road(Color.RED, (3, 2))
    board.build_road(Color.RED, (2, 1))
    board.build_road(Color.RED, (3, 4))


def test_must_build_distance_two():
    board = Board()
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 2))

    with pytest.raises(ValueError):  # distance less than 2
        board.build_settlement(Color.BLUE, 4, initial_build_phase=True)
    board.build_settlement(Color.BLUE, 1, initial_build_phase=True)


def test_placements_must_be_connected():
    board = Board()
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 2))

    with pytest.raises(ValueError):  # distance less than 2 (even if connected)
        board.build_settlement(Color.RED, 2)
    with pytest.raises(ValueError):  # not connected
        board.build_settlement(Color.RED, 1)

    board.build_road(Color.RED, (2, 1))
    board.build_settlement(Color.RED, 1)


def test_city_requires_settlement_first():
    board = Board()
    with pytest.raises(ValueError):  # no settlement there
        board.build_city(Color.RED, 3)

    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_city(Color.RED, 3)


def test_calling_the_edge_differently_is_not_a_problem():
    """Tests building on (0,0,0), East is the same as (1,-1,0), West"""
    pass


def test_get_ports():
    board = Board()
    ports = board.map.port_nodes
    for resource in RESOURCES:
        assert len(ports[resource]) == 2
    assert len(ports[None]) == 8


def test_node_distances():
    node_distances = get_node_distances()
    assert node_distances[2][3] == 1

    # Test are symmetric
    assert node_distances[0][3] == 3
    assert node_distances[3][0] == 3

    assert node_distances[3][9] == 2
    assert node_distances[3][29] == 4

    assert node_distances[34][32] == 2
    assert node_distances[31][45] == 11


# ===== Buildable nodes
def test_buildable_nodes():
    board = Board()
    nodes = board.buildable_node_ids(Color.RED)
    assert len(nodes) == 0
    nodes = board.buildable_node_ids(Color.RED, initial_build_phase=True)
    assert len(nodes) == 54


def test_buildable_nodes_in_mini_map():
    board = Board(catan_map=CatanMap.from_template(MINI_MAP_TEMPLATE, "random"))
    nodes = board.buildable_node_ids(Color.RED)
    assert len(nodes) == 0
    nodes = board.buildable_node_ids(Color.RED, initial_build_phase=True)
    assert len(nodes) == 24


def test_placing_settlement_removes_four_buildable_nodes():
    board = Board()
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    nodes = board.buildable_node_ids(Color.RED)
    assert len(nodes) == 0
    nodes = board.buildable_node_ids(Color.RED, initial_build_phase=True)
    assert len(nodes) == 50
    nodes = board.buildable_node_ids(Color.BLUE, initial_build_phase=True)
    assert len(nodes) == 50


def test_buildable_nodes_respects_distance_two():
    board = Board()
    board.build_settlement(Color.RED, 3, initial_build_phase=True)

    board.build_road(Color.RED, (3, 4))
    nodes = board.buildable_node_ids(Color.RED)
    assert len(nodes) == 0

    board.build_road(Color.RED, (4, 5))
    nodes = board.buildable_node_ids(Color.RED)
    assert len(nodes) == 1
    assert nodes.pop() == 5


def test_cant_use_enemy_roads_to_connect():
    board = Board()
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 2))

    board.build_settlement(Color.BLUE, 1, initial_build_phase=True)
    board.build_road(Color.BLUE, (1, 2))
    board.build_road(Color.BLUE, (0, 1))
    board.build_road(Color.BLUE, (0, 20))  # north out of center tile

    nodes = board.buildable_node_ids(Color.RED)
    assert len(nodes) == 0

    nodes = board.buildable_node_ids(Color.BLUE)
    assert len(nodes) == 1


# ===== Buildable edges
def test_buildable_edges_simple():
    board = Board()
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    buildable = board.buildable_edges(Color.RED)
    assert len(buildable) == 3


def test_buildable_edges_in_mini():
    board = Board(catan_map=CatanMap.from_template(MINI_MAP_TEMPLATE, "random"))
    board.build_settlement(Color.RED, 19, initial_build_phase=True)
    buildable = board.buildable_edges(Color.RED)
    assert len(buildable) == 2


def test_buildable_edges():
    board = Board()
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 4))
    buildable = board.buildable_edges(Color.RED)
    assert len(buildable) == 4


def test_water_edge_is_not_buildable():
    board = Board()
    top_left_north_edge = 45
    board.build_settlement(Color.RED, top_left_north_edge, initial_build_phase=True)
    buildable = board.buildable_edges(Color.RED)
    assert len(buildable) == 2


# ===== Find connected components
def test_connected_components_empty_board():
    board = Board()
    components = board.find_connected_components(Color.RED)
    assert len(components) == 0


def test_one_connected_component():
    board = Board()
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 2))
    board.build_settlement(Color.RED, 1, initial_build_phase=True)
    board.build_road(Color.RED, (1, 2))
    components = board.find_connected_components(Color.RED)
    assert len(components) == 1

    board.build_road(Color.RED, (0, 1))
    components = board.find_connected_components(Color.RED)
    assert len(components) == 1


def test_two_connected_components():
    board = Board()
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 4))
    components = board.find_connected_components(Color.RED)
    assert len(components) == 1

    board.build_settlement(Color.RED, 1, initial_build_phase=True)
    board.build_road(Color.RED, (0, 1))
    components = board.find_connected_components(Color.RED)
    assert len(components) == 2


def test_three_connected_components_bc_enemy_cut_road():
    board = Board()
    # Initial Building Phase of 2 players:
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 4))

    board.build_settlement(Color.BLUE, 15, initial_build_phase=True)
    board.build_road(Color.BLUE, (15, 4))
    board.build_settlement(Color.BLUE, 34, initial_build_phase=True)
    board.build_road(Color.BLUE, (34, 13))

    board.build_settlement(Color.RED, 1, initial_build_phase=True)
    board.build_road(Color.RED, (0, 1))

    # Extend road in a risky way
    board.build_road(Color.RED, (5, 0))
    board.build_road(Color.RED, (5, 16))

    # Plow </3
    board.build_road(Color.BLUE, (5, 4))
    board.build_settlement(Color.BLUE, 5)

    components = board.find_connected_components(Color.RED)
    assert len(components) == 3


def test_connected_components():
    board = Board()
    assert board.find_connected_components(Color.RED) == []

    # Simple test: roads stay at component, disconnected settlement creates new
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 2))
    assert len(board.find_connected_components(Color.RED)) == 1
    assert len(board.find_connected_components(Color.RED)[0]) == 2

    # This is just to be realistic
    board.build_settlement(Color.BLUE, 13, initial_build_phase=True)
    board.build_road(Color.BLUE, (13, 14))
    board.build_settlement(Color.BLUE, 37, initial_build_phase=True)
    board.build_road(Color.BLUE, (37, 14))

    board.build_settlement(Color.RED, 0, initial_build_phase=True)
    board.build_road(Color.RED, (0, 1))
    assert len(board.find_connected_components(Color.RED)) == 2
    assert len(board.find_connected_components(Color.RED)[0]) == 2
    assert len(board.find_connected_components(Color.RED)[1]) == 2

    # Merging subcomponents
    board.build_road(Color.RED, (1, 2))
    assert len(board.find_connected_components(Color.RED)) == 1
    assert len(board.find_connected_components(Color.RED)[0]) == 4

    board.build_road(Color.RED, (3, 4))
    board.build_road(Color.RED, (4, 15))
    board.build_road(Color.RED, (15, 17))
    assert len(board.find_connected_components(Color.RED)) == 1

    # Enemy cutoff
    board.build_road(Color.BLUE, (14, 15))
    board.build_settlement(Color.BLUE, 15)
    assert len(board.find_connected_components(Color.RED)) == 2


def test_building_road_to_enemy_works_well():
    board = Board()

    board.build_settlement(Color.BLUE, 0, initial_build_phase=True)
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 2))
    board.build_road(Color.RED, (2, 1))
    board.build_road(Color.RED, (1, 0))

    # Test building towards enemy works well.
    assert len(board.find_connected_components(Color.RED)) == 1
    assert len(board.find_connected_components(Color.RED)[0]) == 3


def test_building_into_enemy_doesnt_merge_components():
    board = Board()

    board.build_settlement(Color.BLUE, 0, initial_build_phase=True)
    board.build_settlement(Color.RED, 16, initial_build_phase=True)
    board.build_settlement(Color.RED, 6, initial_build_phase=True)
    board.build_road(Color.RED, (16, 5))
    board.build_road(Color.RED, (5, 0))
    board.build_road(Color.RED, (6, 1))
    board.build_road(Color.RED, (1, 0))
    assert len(board.find_connected_components(Color.RED)) == 2


def test_enemy_edge_not_buildable():
    board = Board()
    board.build_settlement(Color.BLUE, 0, initial_build_phase=True)
    board.build_road(Color.BLUE, (0, 1))

    board.build_settlement(Color.RED, 2, initial_build_phase=True)
    board.build_road(Color.RED, (2, 1))
    buildable_edges = board.buildable_edges(Color.RED)
    assert len(buildable_edges) == 3


def test_many_buildings():
    board = Board()
    board.build_settlement(Color.ORANGE, 7, True)
    board.build_settlement(Color.ORANGE, 12, True)
    board.build_road(Color.ORANGE, (6, 7))
    board.build_road(Color.ORANGE, (7, 8))
    board.build_road(Color.ORANGE, (8, 9))
    board.build_road(Color.ORANGE, (8, 27))
    board.build_road(Color.ORANGE, (26, 27))
    board.build_road(Color.ORANGE, (9, 10))
    board.build_road(Color.ORANGE, (10, 11))
    board.build_road(Color.ORANGE, (11, 12))
    board.build_road(Color.ORANGE, (12, 13))
    board.build_road(Color.ORANGE, (13, 34))
    assert len(board.find_connected_components(Color.ORANGE)) == 1

    board.build_settlement(Color.WHITE, 30, True)
    board.build_road(Color.WHITE, (29, 30))
    board.build_road(Color.WHITE, (10, 29))
    board.build_road(Color.WHITE, (28, 29))
    board.build_road(Color.WHITE, (27, 28))
    board.build_settlement(Color.WHITE, 10)  # cut
    board.build_road(Color.WHITE, (30, 31))
    board.build_road(Color.WHITE, (31, 32))
    board.build_settlement(Color.WHITE, 32)
    board.build_road(Color.WHITE, (11, 32))
    board.build_road(Color.WHITE, (32, 33))
    board.build_road(Color.WHITE, (33, 34))
    board.build_settlement(Color.WHITE, 34)
    board.build_road(Color.WHITE, (34, 35))
    board.build_road(Color.WHITE, (35, 36))

    board.build_settlement(Color.WHITE, 41, True)
    board.build_city(Color.WHITE, 41)
    board.build_road(Color.WHITE, (41, 42))
    board.build_road(Color.WHITE, (40, 42))
    board.build_settlement(Color.WHITE, 27)  # cut

    assert len(board.find_connected_components(Color.WHITE)) == 2
    assert len(board.find_connected_components(Color.ORANGE)) == 3


# TODO: Test super long road, cut at many places, to yield 5+ component graph


# ===== A15/A16/A17: rules-fixes regression tests (longest-road / buildable-edges
# enemy-boundary handling). See docs/catan_ai_bug_audit_20260702.md.


def test_a15_settlement_cut_below_five_revokes_longest_road():
    """FIX A15: the settlement-split recompute (build_settlement) must mirror build_road's
    >=5 minimum-length gate. A 7-road network cut into 4+3 (both under 5) must REVOKE the
    Longest Road card entirely, not keep it at the reduced (sub-5) length."""
    board = Board()
    board.build_settlement(Color.RED, 31, initial_build_phase=True)
    chain = [31, 30, 29, 28, 27, 26, 25, 24]
    for a, b in zip(chain, chain[1:]):
        board.build_road(Color.RED, (a, b))
    assert board.road_color == Color.RED
    assert board.road_length == 7

    # BLUE approaches the interior junction node 27 via its own separate road (through 9, 8)
    # and settles there, cutting RED's 7-road chain into a 4-piece and a 3-piece.
    board.build_settlement(Color.BLUE, 9, initial_build_phase=True)
    board.build_road(Color.BLUE, (9, 8))
    board.build_road(Color.BLUE, (8, 27))
    board.build_settlement(Color.BLUE, 27)

    assert board.road_lengths[Color.RED] == 4
    assert board.road_color is None
    assert board.road_length == 0


def test_a15_settlement_cut_leaving_a_qualifying_piece_keeps_the_card():
    """An 11-road network cut into a 7-piece and a 4-piece: the surviving 7-piece still
    qualifies (>=5), so the card must be KEPT (at the new, shorter length), not revoked."""
    board = Board()
    board.build_settlement(Color.RED, 31, initial_build_phase=True)
    chain = [31, 30, 29, 28, 27, 26, 25, 24, 53, 52, 51, 50]
    for a, b in zip(chain, chain[1:]):
        board.build_road(Color.RED, (a, b))
    assert board.road_color == Color.RED
    assert board.road_length == 11

    # BLUE approaches junction node 24 via its own separate road (through 6, 7) and settles
    # there, cutting RED's network into a 7-piece (31..24) and a 4-piece (24..50).
    board.build_settlement(Color.BLUE, 6, initial_build_phase=True)
    board.build_road(Color.BLUE, (6, 7))
    board.build_road(Color.BLUE, (7, 24))
    board.build_settlement(Color.BLUE, 24)

    assert board.road_lengths[Color.RED] == 7
    assert board.road_color == Color.RED
    assert board.road_length == 7


def test_a16_longest_acyclic_path_counts_both_enemy_capped_ends():
    """FIX A16 (matches upstream issue #378 / PR #379): a road network capped by an enemy
    settlement at BOTH ends must count every road segment, including the final edge touching
    each enemy node -- not undercount by 1 at whichever end is approached from the interior."""
    board = Board()
    board.build_settlement(Color.RED, 31, initial_build_phase=True)
    chain = [31, 30, 29, 28, 27, 26, 25, 24, 53, 52, 51, 50, 49, 48]
    for a, b in zip(chain, chain[1:]):
        board.build_road(Color.RED, (a, b))
    assert board.road_length == 13

    # BLUE caps one end at junction node 29 (approached via 11, 10) ...
    board.build_settlement(Color.BLUE, 11, initial_build_phase=True)
    board.build_road(Color.BLUE, (11, 10))
    board.build_road(Color.BLUE, (10, 29))
    board.build_settlement(Color.BLUE, 29)

    # ... and the OTHER end at junction node 49 (approached via 20, 22).
    board.build_settlement(Color.BLUE, 20, initial_build_phase=True)
    board.build_road(Color.BLUE, (20, 22))
    board.build_road(Color.BLUE, (22, 49))
    board.build_settlement(Color.BLUE, 49)

    # The surviving middle piece is 29-28-27-26-25-24-53-52-51-50-49: 10 edges, both
    # boundary (enemy) nodes' final segments included. Pre-fix this undercounted to 9.
    assert board.road_lengths[Color.RED] == 10
    assert board.road_color == Color.RED
    assert board.road_length == 10


def test_a16_longest_acyclic_path_single_enemy_cap_is_unaffected():
    """Sanity: a single-capped network (the case the pre-fix bug happened to mask via an
    alternate DFS start) must still report its true, un-undercounted length."""
    board = Board()
    board.build_settlement(Color.BLUE, 0, initial_build_phase=True)
    board.build_settlement(Color.RED, 3, initial_build_phase=True)
    board.build_road(Color.RED, (3, 2))
    board.build_road(Color.RED, (2, 1))
    board.build_road(Color.RED, (1, 0))

    component = board.find_connected_components(Color.RED)[0]
    path = longest_acyclic_path(board, component, Color.RED)
    assert len(path) == 3


def test_a17_buildable_edges_excludes_placements_beyond_enemy_node():
    """FIX A17 (matches upstream issue #376, same root cause as A16): a road may legally end
    AT an enemy-owned node, but must not be offered as a buildable ANCHOR for a new edge
    reaching further out -- that would mean building through the enemy settlement."""
    board = Board()
    board.build_settlement(Color.BLUE, 0, initial_build_phase=True)
    board.build_settlement(Color.RED, 2, initial_build_phase=True)
    # Directly install the post-cut component shape (node 0 -- the enemy boundary node -- is
    # a member of RED's connected component, exactly as a real settlement-cut's dfs_walk would
    # leave it) so this test exercises buildable_edges() in isolation. Node 2 (RED's own
    # settlement) anchors the component so it isn't discarded as an unanchored orphan.
    board.roads[(2, 1)] = Color.RED
    board.roads[(1, 2)] = Color.RED
    board.roads[(1, 0)] = Color.RED
    board.roads[(0, 1)] = Color.RED
    board.connected_components[Color.RED] = [{0, 1, 2}]

    buildable = board.buildable_edges(Color.RED)

    assert (0, 5) not in buildable and (5, 0) not in buildable
    assert (0, 20) not in buildable and (20, 0) not in buildable
    # Edges anchored at the legitimate (non-enemy) node 1 remain buildable.
    assert (1, 2) not in buildable  # already a road, not a NEW candidate
    assert (1, 6) in buildable
    assert (2, 9) in buildable


def test_a17_buildable_edges_after_through_node_cut_still_anchors_the_far_side():
    """CORRECTED by FIX A24 (was originally the "A17 refinement" test, asserting
    the opposite): RED's road reaches a THROUGH node N before anyone settles
    there; when BLUE later settles at N, the {28, 29} piece beyond N survives
    in connected_components (roads already built there still count toward
    RED's longest-road LENGTH) -- and this test used to also assert it could
    no longer anchor NEW construction (node 28 -> 27).

    That assertion was wrong: node 28 has its own real, still-existing RED
    road (29, 28) and is a legitimate frontier in its own right, exactly like
    node 5 in the A24/seed-97 shape (see test_a24_... above) -- it is
    reachable via the shared boundary node 29 from RED's genuinely anchored
    piece. Re-verified directly against the real Rust engine via the full
    equivalence harness (both the seed this test was originally modeled on,
    seed=5, AND seed=97 now complete with zero divergence under the A24 fix).
    What the ORIGINAL (still correct, non-refinement) A17 fix protects
    against is different and distinct: anchoring a candidate edge material
    AT the enemy node itself and extending FURTHER OUT from it -- see
    test_a17_buildable_edges_excludes_placements_beyond_enemy_node above,
    which is unaffected by this correction (node 28 is not node 29, the
    enemy node; it's RED's own pre-existing frontier one hop away)."""
    board = Board()
    board.build_settlement(Color.RED, 31, initial_build_phase=True)
    board.build_road(Color.RED, (31, 30))
    board.build_road(Color.RED, (30, 29))
    # RED already extends past node 29 before anyone settles there -- 29 becomes a
    # THROUGH node for RED (two red edges) once BLUE later settles there.
    board.build_road(Color.RED, (29, 28))

    board.build_settlement(Color.BLUE, 11, initial_build_phase=True)
    board.build_road(Color.BLUE, (11, 10))
    board.build_road(Color.BLUE, (10, 29))
    board.build_settlement(Color.BLUE, 29)

    # The {28, 29} piece survives in connected_components (roads already built there still
    # count toward RED's longest-road length)...
    assert {28, 29} in board.connected_components[Color.RED]
    # ...and (per A24) DOES still anchor new buildable edges via its shared
    # boundary node (29) with RED's real, settlement-anchored piece.
    buildable = board.buildable_edges(Color.RED)
    assert (28, 27) in buildable or (27, 28) in buildable
    # The still-anchored piece (containing RED's settlement at 31) remains buildable.
    assert (31, 32) in buildable


def test_a24_buildable_edges_reaches_through_a_shared_boundary_node_after_a_settlement_cut():
    """FIX A24 (live equivalence-scan regression, 8/1000 games incl. seed=97): a
    settlement-triggered cut (build_settlement's split, see A17 refinement above)
    correctly keeps the cut node in BOTH resulting pieces as their shared
    boundary -- but the A17 per-component is_friendly_node filter treated each
    piece as independently anchored-or-not, so a piece with no RED building of
    its own (here {4, 5}) was discarded ENTIRELY even though it shares boundary
    node 4 with the other, genuinely anchored piece ({4, 37, 14, 15}). Node 5 is
    real, reachable, un-severed RED network -- it must still anchor new
    construction. Reproduces the exact seed-97 dump shape:
    connected_components[RED] == [{4, 5}, {4, 37, 38, 39, 41, 14, 15, 17}]
    (node 4 duplicated across two entries)."""
    board = Board()
    # RED's real network: settlement at 37, chain 37-14-15-4, then a stub 4-5
    # with no RED building at the far end -- node 4 ends up with EXACTLY 2 RED
    # edges, (15, 4) and (4, 5), satisfying the settlement-cut precondition
    # (build_settlement only splits when exactly 2 of the cut node's edges
    # belong to the same color).
    board.build_settlement(Color.RED, 37, initial_build_phase=True)
    board.build_road(Color.RED, (37, 14))
    board.build_road(Color.RED, (14, 15))
    board.build_road(Color.RED, (15, 4))
    board.build_road(Color.RED, (4, 5))

    # BLUE reaches node 4 via the one edge RED left open, (3, 4) -- routed
    # through node 2 (not node 3) so BLUE's own settlement isn't adjacent to
    # node 4 itself (the distance-two rule would otherwise block it).
    board.build_settlement(Color.BLUE, 2, initial_build_phase=True)
    board.build_road(Color.BLUE, (2, 3))
    board.build_road(Color.BLUE, (3, 4))
    board.build_settlement(Color.BLUE, 4)  # non-initial: triggers the cut

    components = board.find_connected_components(Color.RED)
    assert len(components) == 2
    assert sum(1 for component in components if 4 in component) == 2
    assert {4, 5} in components

    buildable = board.buildable_edges(Color.RED)
    # Node 5 is only a member of the {4, 5} piece, which alone has no RED
    # building -- but it shares boundary node 4 with the friendly
    # {4, 37, 14, 15} piece, so it IS part of the real anchored network.
    assert (0, 5) in buildable or (5, 0) in buildable
    assert (5, 16) in buildable or (16, 5) in buildable


def test_a26_settlement_cut_tie_with_another_color_keeps_the_incumbent():
    """FIX A26 (live equivalence-scan regression, winner-flip: seeds 26, 179,
    194, 388, 331): the settlement-split recompute picked the GLOBAL max
    across ALL colors' road_lengths and unconditionally reassigned
    road_color/road_length to it -- but Python's max() resolves ties to
    whichever key was inserted first, not necessarily the CURRENT incumbent.
    build_road already has the correct rule (a challenger must STRICTLY
    exceed the incumbent's length to take the card, mirrored in its
    `candidate_length > self.road_length` guard); the settlement-cut path
    lacked that guard entirely, so a mere TIE (or even an unrelated,
    unchanged color's length) could silently steal the card and its 2VP from
    the actual incumbent -- in the worst case (seed 331) flipping who wins
    the game."""
    board = Board()
    # BLUE builds a length-5 network first and holds the card (5 >= 5).
    board.build_settlement(Color.BLUE, 3, initial_build_phase=True)
    for a, b in zip([3, 4, 15, 14, 37], [4, 15, 14, 37, 36]):
        board.build_road(Color.BLUE, (a, b))
    assert board.road_color == Color.BLUE and board.road_length == 5

    # RED builds a length-7 network and correctly takes the card (7 > 5) --
    # ordinary build_road behavior, unaffected by this fix.
    board.build_settlement(Color.RED, 31, initial_build_phase=True)
    chain_red = [31, 30, 29, 28, 27, 26, 25, 24]
    for a, b in zip(chain_red, chain_red[1:]):
        board.build_road(Color.RED, (a, b))
    assert board.road_color == Color.RED and board.road_length == 7

    # ORANGE reaches the interior junction node 29 via its own separate road
    # (through 33, 32, 11, 10) and settles there, cutting RED's chain into a
    # 2-piece (31-30-29) and a 5-piece (29-28-27-26-25-24) -- RED's
    # recomputed length (5) now exactly TIES BLUE's already-held 5.
    board.build_settlement(Color.ORANGE, 33, initial_build_phase=True)
    board.build_road(Color.ORANGE, (33, 32))
    board.build_road(Color.ORANGE, (32, 11))
    board.build_road(Color.ORANGE, (11, 10))
    board.build_road(Color.ORANGE, (10, 29))
    board.build_settlement(Color.ORANGE, 29)

    assert board.road_lengths[Color.RED] == 5
    assert board.road_lengths[Color.BLUE] == 5
    # RED is the current incumbent and still qualifies (>=5) after the cut --
    # a mere TIE with BLUE's unchanged network must NOT take the card away.
    assert board.road_color == Color.RED
    assert board.road_length == 5


def test_a26_settlement_cut_below_five_transfers_to_the_only_remaining_qualifier():
    """Companion to the tie test above: when the cut DOES drop the incumbent
    below 5 (losing the card outright), and exactly one other color still
    qualifies (>=5), the card correctly transfers to them -- distinguishing
    "no strict-exceed guard needed" (this case, no ambiguity) from the tie
    case above (guard required)."""
    board = Board()
    board.build_settlement(Color.BLUE, 3, initial_build_phase=True)
    for a, b in zip([3, 4, 15, 14, 37], [4, 15, 14, 37, 36]):
        board.build_road(Color.BLUE, (a, b))
    assert board.road_color == Color.BLUE and board.road_length == 5

    board.build_settlement(Color.RED, 31, initial_build_phase=True)
    chain_red = [31, 30, 29, 28, 27, 26, 25, 24]
    for a, b in zip(chain_red, chain_red[1:]):
        board.build_road(Color.RED, (a, b))
    assert board.road_color == Color.RED and board.road_length == 7

    # ORANGE cuts at node 27 instead (via 8, 9 -- distinct from node 29's
    # approach above), splitting RED into a 4-piece and a 3-piece: BOTH
    # under 5, so RED loses the card outright and BLUE's untouched 5 becomes
    # the sole qualifier.
    board.build_settlement(Color.ORANGE, 9, initial_build_phase=True)
    board.build_road(Color.ORANGE, (9, 8))
    board.build_road(Color.ORANGE, (8, 27))
    board.build_settlement(Color.ORANGE, 27)

    assert board.road_lengths[Color.RED] == 4
    assert board.road_color == Color.BLUE
    assert board.road_length == 5


def test_a27_longest_acyclic_path_closes_a_cycle_through_an_enemy_cut_node():
    """FIX A27 (live equivalence-scan regression, length undercount by exactly
    1: seeds 587, 954): reconstructs the exact real topology found via replay
    -- RED's roads form a CLOSED LOOP (0-5-16-21-19-20-0) where node 0 is a
    BLUE-owned city sitting at the one point where the loop is "cut". build_road
    deliberately never adds an enemy node to connected_components storage
    (see test_building_into_enemy_doesnt_merge_components -- adding it would
    wrongly let two genuinely separate pieces merge just because both touch
    the same enemy node from different sides), so the stored component here
    is only {5, 16, 19, 20, 21} -- node 0 itself is never a `start_node`
    candidate in longest_acyclic_path's `for start_node in node_set` loop.

    Reached from any INTERIOR node (the old behavior), the DFS arrives at
    node 0 from only one direction and correctly stops there (FIX A16) --
    but never gets to try node 0's OTHER exit fresh, which is exactly what
    would let it recover the full loop. Starting fresh AT node 0 (this fix)
    can expand in BOTH directions and walk the entire 6-edge loop, correctly
    ending back at node 0 (revisiting it as a leaf, not a pass-through) once
    every other edge has been used."""
    board = Board()
    board.build_settlement(Color.BLUE, 0, initial_build_phase=True)

    # Node 19 (not node 20 -- it's a direct neighbor of node 0 and would be
    # blocked by the distance-two rule) anchors RED's settlement.
    board.build_settlement(Color.RED, 19, initial_build_phase=True)
    board.build_road(Color.RED, (19, 20))
    board.build_road(Color.RED, (20, 0))
    board.build_road(Color.RED, (19, 21))
    board.build_road(Color.RED, (21, 16))
    board.build_road(Color.RED, (16, 5))
    board.build_road(Color.RED, (5, 0))  # closes the loop back through node 0

    assert board.connected_components[Color.RED] == [{5, 16, 19, 20, 21}]
    assert board.road_lengths[Color.RED] == 6
    assert board.road_color == Color.RED
    assert board.road_length == 6


def test_a28_settlement_cut_below_an_untouched_rival_transfers_the_card():
    """FIX A28 (live equivalence-scan regression, seeds 167, 491, 731 -- a gap
    in the A26 fix above): A26's `elif self.road_lengths[edge_color] >= 5:`
    branch only checked whether the severed incumbent still QUALIFIES
    (>=5) -- not whether they are still ON TOP. If an untouched rival's
    (already-existing, unchanged) length now STRICTLY EXCEEDS the
    incumbent's reduced length, that is a genuine strict beat (not a tie)
    and the card must transfer per official rules -- e.g. incumbent 9->6
    while another player sits untouched at 8: 8 > 6, card transfers.
    A26's fix wrongly let the incumbent keep it merely for still being
    >= 5, regardless of whether anyone else now exceeds them."""
    board = Board()
    # BLUE builds a length-6 network first and holds the card (6 >= 5).
    board.build_settlement(Color.BLUE, 3, initial_build_phase=True)
    for a, b in zip([3, 4, 15, 14, 37, 36], [4, 15, 14, 37, 36, 35]):
        board.build_road(Color.BLUE, (a, b))
    assert board.road_color == Color.BLUE and board.road_length == 6

    # RED builds a length-7 network and correctly takes the card (7 > 6).
    board.build_settlement(Color.RED, 31, initial_build_phase=True)
    chain_red = [31, 30, 29, 28, 27, 26, 25, 24]
    for a, b in zip(chain_red, chain_red[1:]):
        board.build_road(Color.RED, (a, b))
    assert board.road_color == Color.RED and board.road_length == 7

    # ORANGE cuts RED at node 29 (same approach as the A26 tests above),
    # splitting RED into a 2-piece and a 5-piece: RED still qualifies (>=5)
    # but BLUE's UNTOUCHED 6 now strictly exceeds RED's reduced 5.
    board.build_settlement(Color.ORANGE, 33, initial_build_phase=True)
    board.build_road(Color.ORANGE, (33, 32))
    board.build_road(Color.ORANGE, (32, 11))
    board.build_road(Color.ORANGE, (11, 10))
    board.build_road(Color.ORANGE, (10, 29))
    board.build_settlement(Color.ORANGE, 29)

    assert board.road_lengths[Color.RED] == 5
    assert board.road_lengths[Color.BLUE] == 6
    # BLUE's untouched, already-qualifying 6 strictly beats RED's reduced
    # 5 -- the card MUST transfer, even though RED still (individually)
    # qualifies at >=5.
    assert board.road_color == Color.BLUE
    assert board.road_length == 6


def test_a28_settlement_cut_tying_an_untouched_rival_keeps_the_incumbent():
    """Mirror-image sanity check requested alongside A28: a severed
    incumbent who still exactly TIES an untouched rival's length keeps the
    card -- ties never transfer, only a STRICT beat does. Same construction
    as the A26 tie test (test_a26_settlement_cut_tie_with_another_color_keeps_the_incumbent)
    reproduced here to guard against the A28 fix over-correcting into
    transferring on a mere tie."""
    board = Board()
    board.build_settlement(Color.BLUE, 3, initial_build_phase=True)
    for a, b in zip([3, 4, 15, 14, 37], [4, 15, 14, 37, 36]):
        board.build_road(Color.BLUE, (a, b))
    assert board.road_color == Color.BLUE and board.road_length == 5

    board.build_settlement(Color.RED, 31, initial_build_phase=True)
    chain_red = [31, 30, 29, 28, 27, 26, 25, 24]
    for a, b in zip(chain_red, chain_red[1:]):
        board.build_road(Color.RED, (a, b))
    assert board.road_color == Color.RED and board.road_length == 7

    board.build_settlement(Color.ORANGE, 33, initial_build_phase=True)
    board.build_road(Color.ORANGE, (33, 32))
    board.build_road(Color.ORANGE, (32, 11))
    board.build_road(Color.ORANGE, (11, 10))
    board.build_road(Color.ORANGE, (10, 29))
    board.build_settlement(Color.ORANGE, 29)

    assert board.road_lengths[Color.RED] == 5
    assert board.road_lengths[Color.BLUE] == 5
    assert board.road_color == Color.RED
    assert board.road_length == 5
