import pickle
import copy
from collections import defaultdict
from typing import Any, Set, Dict, Tuple, List
import functools

import networkx as nx  # type: ignore

from catanatron.models.player import Color
from catanatron.models.map import (
    BASE_MAP_TEMPLATE,
    MINI_MAP_TEMPLATE,
    NUM_NODES,
    CatanMap,
    NodeId,
)
from catanatron.models.enums import FastBuildingType, SETTLEMENT, CITY


# Used to find relationships between nodes and edges
base_map = CatanMap.from_template(BASE_MAP_TEMPLATE)
mini_map = CatanMap.from_template(MINI_MAP_TEMPLATE)
STATIC_GRAPH = nx.Graph()
for tile in base_map.tiles.values():
    STATIC_GRAPH.add_nodes_from(tile.nodes.values())
    STATIC_GRAPH.add_edges_from(tile.edges.values())


@functools.lru_cache(1)
def get_node_distances():
    return nx.floyd_warshall(STATIC_GRAPH)


@functools.lru_cache(3)  # None, range(54), range(24)
def get_edges(land_nodes=None):
    return list(STATIC_GRAPH.subgraph(land_nodes or range(NUM_NODES)).edges())


def _merge_components_sharing_nodes(components):
    """Union-find merge of `component sets that share at least one node.

    FIX A24: a settlement-triggered cut (see build_settlement's split, below)
    legitimately keeps the cut node in BOTH resulting pieces -- it is their
    shared boundary. Over the course of a game this can leave the SAME node
    duplicated across two (or, transitively, more) separate list-entries in
    connected_components (verified live: seed=97 produced
    [{4, 5}, {4, 37, 38, 39, 41, 14, 15, 17}], node 4 in both). Treating each
    stored entry as an independently anchored-or-not piece (the A17 filter's
    original approach) wrongly discards an entire piece -- and any node ONLY
    reachable within it -- whenever that piece alone has no friendly
    building, even though it is still reachable via the shared node from a
    genuinely anchored piece. Merging by shared membership first means the
    friendliness check operates on the real topology (is this piece
    connected to the network AT ALL, transitively), not on however many
    separate list-entries happen to be stored for it.
    """
    merged = []
    for component in components:
        overlapping = [i for i, existing in enumerate(merged) if existing & component]
        if not overlapping:
            merged.append(set(component))
            continue
        target = overlapping[0]
        merged[target] |= component
        for i in reversed(overlapping[1:]):
            merged[target] |= merged.pop(i)
    return merged


class Board:
    """Encapsulates all state information regarding the board.

    Attributes:
        buildings (Dict[NodeId, Tuple[Color, FastBuildingType]]): Mapping from
            node id to building (if there is a building there).
        roads (Dict[EdgeId, Color]): Mapping from edge
            to Color (if there is a road there). Contains inverted
            edges as well for ease of querying.
        connected_components (Dict[Color, List[Set[NodeId]]]): Cache
            datastructure to speed up maintaining longest road computation.
            To be queried by Color. Value is a list of node sets.
        board_buildable_ids (Set[NodeId]): Cache of buildable node ids in board.
        road_color (Color): Color of player with longest road.
        road_length (int): Number of roads of longest road
        robber_coordinate (Coordinate): Coordinate where robber is.
    """

    def __init__(self, catan_map=None, initialize=True):
        self.buildable_subgraph: Any = None
        self.buildable_edges_cache = {}
        self.player_port_resources_cache = {}
        if initialize:
            self.map: CatanMap = catan_map or CatanMap.from_template(
                BASE_MAP_TEMPLATE
            )  # Static State (no need to copy)

            self.buildings: Dict[NodeId, Tuple[Color, FastBuildingType]] = dict()
            self.roads = dict()  # (node_id, node_id) => color

            # color => int{}[] (list of node_id sets) one per component
            #   nodes in sets are incidental (might not be owned by player)
            self.connected_components: Any = defaultdict(list)
            self.board_buildable_ids = set(self.map.land_nodes)
            self.road_lengths = defaultdict(int)
            self.road_color = None
            self.road_length = 0

            # assumes there is at least one desert:
            self.robber_coordinate = filter(
                lambda coordinate: self.map.land_tiles[coordinate].resource is None,
                self.map.land_tiles.keys(),
            ).__next__()

            # Cache buildable subgraph
            self.buildable_subgraph = STATIC_GRAPH.subgraph(self.map.land_nodes)

    def build_settlement(self, color, node_id, initial_build_phase=False):
        """Adds a settlement, and ensures is a valid place to build.

        Args:
            color (Color): player's color
            node_id (int): where to build
            initial_build_phase (bool, optional):
                Whether this is part of initial building phase, so as to skip
                connectedness validation. Defaults to True.
        """
        buildable = self.buildable_node_ids(
            color, initial_build_phase=initial_build_phase
        )
        if node_id not in buildable:
            raise ValueError(
                "Invalid Settlement Placement: not connected and not initial-placement"
            )

        if node_id in self.buildings:
            raise ValueError("Invalid Settlement Placement: a building exists there")

        self.buildings[node_id] = (color, SETTLEMENT)

        previous_road_color = self.road_color
        if initial_build_phase:
            self.connected_components[color].append({node_id})
        else:
            # Maybe cut connected components.
            edges_by_color = defaultdict(list)
            for edge in STATIC_GRAPH.edges(node_id):
                edges_by_color[self.roads.get(edge, None)].append(edge)

            for edge_color, edges in edges_by_color.items():
                if edge_color == color or edge_color is None:
                    continue  # ignore
                if len(edges) == 2:  # rip, edge_color has been plowed
                    # consider cut was at b=node_id for edges (a, b) and (b, c)
                    a = [n for n in edges[0] if n != node_id].pop()
                    c = [n for n in edges[1] if n != node_id].pop()

                    # do dfs from a adding all encountered nodes
                    a_nodeset = self.dfs_walk(a, edge_color)
                    c_nodeset = self.dfs_walk(c, edge_color)

                    # split this components on here. NOTE: both resulting pieces are kept
                    # even if one no longer contains any of edge_color's own
                    # settlements/cities (a "stranded" piece severed from their real network)
                    # -- roads already built there still count toward longest-road LENGTH.
                    # buildable_edges() is where such a stranded piece must stop being used as
                    # an anchor for NEW construction (see FIX A17 refinement there).
                    b_index = self._get_connected_component_index(node_id, edge_color)
                    del self.connected_components[edge_color][b_index]
                    self.connected_components[edge_color].append(a_nodeset)
                    self.connected_components[edge_color].append(c_nodeset)

                    # Update longest road by plowed player. Compare again with all
                    self.road_lengths[edge_color] = max(
                        *[
                            len(longest_acyclic_path(self, component, edge_color))
                            for component in self.connected_components[edge_color]
                        ]
                    )
                    # FIX A26 (supersedes the original A15 recompute): a settlement-sever
                    # only ever DECREASES edge_color's own length -- it can never increase
                    # anyone's, and no other color's road_lengths entry is touched here. So
                    # the ONLY way this cut can change who holds Longest Road is if edge_color
                    # is the CURRENT incumbent, mirroring build_road's strictly-greater-to-
                    # take-it-away rule (its `candidate_length > self.road_length` guard, a
                    # few lines below): a mere TIE (or any other color's already-unchanged
                    # length) must never steal the card away mid-cut. The old code instead
                    # recomputed a GLOBAL max across ALL colors and unconditionally reassigned
                    # to it -- Python's max() resolves ties to whichever key was inserted
                    # first, not necessarily the incumbent, so a tie (or even an unrelated
                    # color) could silently flip who holds the card and its 2VP (live
                    # equivalence-scan regression, winner-flip in the worst case).
                    if edge_color != self.road_color:
                        pass  # not the incumbent -- this cut cannot affect who holds the card.
                    elif self.road_lengths[edge_color] >= 5 and self.road_lengths[
                        edge_color
                    ] >= max(self.road_lengths.values()):
                        # FIX A28 (closes a gap A26 left): still qualifies (>=5) is not
                        # enough on its own -- the incumbent must also still be ON TOP
                        # (>= every other color's length, ties included) to keep the
                        # card. A26 only checked qualification, so an untouched rival
                        # whose UNCHANGED length now strictly exceeds the incumbent's
                        # reduced length (e.g. incumbent 9->6 while another player sits
                        # untouched at 8) wrongly kept the card instead of transferring
                        # it -- a strict beat, not a tie, per official rules (live
                        # equivalence-scan regression, seeds 167/491/731). `>=` (not
                        # `>`) against the global max correctly keeps ties with the
                        # incumbent (ties never transfer) while still falling through
                        # below whenever someone else strictly exceeds.
                        self.road_length = self.road_lengths[edge_color]
                    else:
                        # FIX A15 (kept): dropped below 5, OR (FIX A28) an untouched
                        # rival's length now strictly exceeds the incumbent's reduced
                        # length -- either way the incumbent is dethroned and a fresh
                        # global-max pick (>=5) is unambiguous here.
                        candidate_color, candidate_length = max(
                            self.road_lengths.items(), key=lambda e: e[1]
                        )
                        if candidate_length >= 5:
                            self.road_color, self.road_length = candidate_color, candidate_length
                        else:
                            self.road_color, self.road_length = None, 0

        self.board_buildable_ids.discard(node_id)
        for n in STATIC_GRAPH.neighbors(node_id):
            self.board_buildable_ids.discard(n)

        self.buildable_edges_cache = {}  # Reset buildable_edges
        self.player_port_resources_cache = {}  # Reset port resources
        return previous_road_color, self.road_color, self.road_lengths

    def dfs_walk(self, node_id, color):
        """Generates set of nodes that are "connected" to given node.

        Args:
            node_id (int): Where to start search/walk.
            color (Color): Player color asking

        Returns:
            Set[int]: Nodes that are "connected" to this one
                by roads of the color player.
        """
        agenda = [node_id]  # assuming node_id is owned.
        visited = set()

        while len(agenda) != 0:
            n = agenda.pop()
            visited.add(n)

            if self.is_enemy_node(n, color):
                continue  # end of the road

            neighbors = [v for v in STATIC_GRAPH.neighbors(n) if v not in visited]
            expandable = [v for v in neighbors if self.roads.get((n, v), None) == color]
            agenda.extend(expandable)

        return visited

    def _get_connected_component_index(self, node_id, color):
        for i, component in enumerate(self.connected_components[color]):
            if node_id in component:
                return i

    def build_road(self, color, edge):
        buildable = self.buildable_edges(color)
        inverted_edge = (edge[1], edge[0])
        if edge not in buildable and inverted_edge not in buildable:
            raise ValueError("Invalid Road Placement")

        self.roads[edge] = color
        self.roads[inverted_edge] = color

        # Find connected components corresponding to edge nodes (buildings).
        a, b = edge
        a_index = self._get_connected_component_index(a, color)
        b_index = self._get_connected_component_index(b, color)

        # Extend or merge components
        if a_index is None and not self.is_enemy_node(a, color):
            component = self.connected_components[color][b_index]
            component.add(a)
        elif b_index is None and not self.is_enemy_node(b, color):
            component = self.connected_components[color][a_index]
            component.add(b)
        elif a_index is not None and b_index is not None and a_index != b_index:
            # Merge both components into one and delete the other.
            component = set.union(
                self.connected_components[color][a_index],
                self.connected_components[color][b_index],
            )
            self.connected_components[color][a_index] = component
            del self.connected_components[color][b_index]
        else:
            # In this case, a_index == b_index, which means that the edge
            # is already part of one component. No actions needed.
            chosen_index = a_index if a_index is not None else b_index
            component = self.connected_components[color][chosen_index]

        # find longest path on component under question
        previous_road_color = self.road_color
        candidate_length = len(longest_acyclic_path(self, component, color))
        self.road_lengths[color] = max(self.road_lengths[color], candidate_length)
        if candidate_length >= 5 and candidate_length > self.road_length:
            self.road_color = color
            self.road_length = candidate_length

        self.buildable_edges_cache = {}  # Reset buildable_edges
        return previous_road_color, self.road_color, self.road_lengths

    def build_city(self, color, node_id):
        building = self.buildings.get(node_id, None)
        if building is None or building[0] != color or building[1] != SETTLEMENT:
            raise ValueError("Invalid City Placement: no player settlement there")

        self.buildings[node_id] = (color, CITY)

    def buildable_node_ids(self, color: Color, initial_build_phase=False):
        if initial_build_phase:
            return sorted(list(self.board_buildable_ids))

        subgraphs = self.find_connected_components(color)
        nodes = set().union(*subgraphs)
        return sorted(list(nodes.intersection(self.board_buildable_ids)))

    def buildable_edges(self, color: Color):
        """List of (n1,n2) tuples. Edges are in n1 < n2 order."""
        if color in self.buildable_edges_cache:
            return self.buildable_edges_cache[color]

        expandable = set()

        # All nodes for this color.
        # TODO(tonypr): Explore caching for 'expandable_nodes'?
        # The 'expandable_nodes' set should only increase in size monotonically I think.
        # We can take advantage of that.
        expandable_nodes = set()
        # FIX A24: merge components sharing a node BEFORE the friendliness
        # check below -- see _merge_components_sharing_nodes' docstring. A
        # piece stored as a separate list-entry can still be part of the real
        # anchored network via a shared boundary node with another entry.
        for component in _merge_components_sharing_nodes(self.connected_components[color]):
            # FIX (A17 refinement): a settlement-triggered cut (build_settlement) keeps BOTH
            # resulting pieces tracked in connected_components -- correctly, since roads
            # already built in a now-severed piece still count toward longest-road LENGTH.
            # But a piece with NO settlement/city of `color` anywhere in it is a stranded
            # orphan: its only link back to color's real network was through the node an
            # enemy just settled on, so it must not anchor any NEW construction. Verified
            # live: RED's road reaches a through-node N, BLUE settles at N, and the
            # RED-owned piece beyond N (which has no RED building) kept offering its
            # frontier as buildable before this check.
            if any(self.is_friendly_node(node, color) for node in component):
                expandable_nodes.update(component)
        # FIX A17 (matches upstream issue #376, same root cause as A16): connected_components
        # may include enemy-owned BOUNDARY nodes (per the class docstring, "nodes in sets are
        # incidental... might not be owned by player"). A road may legally touch/end at such a
        # node but must not be anchored FURTHER OUT from it -- that would mean building through
        # the enemy settlement, which the rules forbid. `.edges(expandable_nodes)` returns every
        # edge incident to ANY node in the set regardless of ownership, so an enemy boundary
        # node was offering brand-new edges beyond itself as buildable. Anchor candidate edges
        # only at the non-enemy subset; an edge ENDING at an enemy node from a legitimate
        # (non-enemy) anchor is still included and still legal.
        anchor_nodes = {
            node for node in expandable_nodes if not self.is_enemy_node(node, color)
        }

        candidate_edges = self.buildable_subgraph.edges(anchor_nodes)
        for edge in candidate_edges:
            if self.get_edge_color(edge) is None:
                expandable.add(tuple(sorted(edge)))

        self.buildable_edges_cache[color] = list(expandable)
        return self.buildable_edges_cache[color]

    def get_player_port_resources(self, color):
        """Yields resources (None for 3:1) of ports owned by color"""
        if color in self.player_port_resources_cache:
            return self.player_port_resources_cache[color]

        resources = set()
        for resource, node_ids in self.map.port_nodes.items():
            if any(self.is_friendly_node(node_id, color) for node_id in node_ids):
                resources.add(resource)

        self.player_port_resources_cache[color] = resources
        return resources

    def find_connected_components(self, color: Color):
        """
        Returns:
            nx.Graph[]: connected subgraphs. subgraphs
                might include nodes that color doesnt own (on the way and on ends),
                just to make it is "closed" and easier for buildable_nodes to operate.
        """
        return self.connected_components[color]

    def continuous_roads_by_player(self, color: Color):
        paths = []
        components = self.find_connected_components(color)
        for component in components:
            paths.append(longest_acyclic_path(self, component, color))
        return paths

    def copy(self):
        board = Board(self.map, initialize=False)
        board.map = self.map  # reuse since its immutable
        board.buildings = self.buildings.copy()
        board.roads = self.roads.copy()
        board.connected_components = pickle.loads(
            pickle.dumps(self.connected_components)
        )
        board.board_buildable_ids = self.board_buildable_ids.copy()
        board.road_lengths = self.road_lengths.copy()
        board.road_color = self.road_color
        board.road_length = self.road_length

        board.robber_coordinate = self.robber_coordinate
        board.buildable_subgraph = self.buildable_subgraph
        board.buildable_edges_cache = copy.deepcopy(self.buildable_edges_cache)
        board.player_port_resources_cache = copy.deepcopy(
            self.player_port_resources_cache
        )
        return board

    # ===== Helper functions
    def get_node_color(self, node_id):
        # using try-except instead of .get for performance
        try:
            return self.buildings[node_id][0]
        except KeyError:
            return None

    def get_edge_color(self, edge):
        # using try-except instead of .get for performance
        try:
            return self.roads[edge]
        except KeyError:
            return None

    def is_enemy_node(self, node_id, color):
        node_color = self.get_node_color(node_id)
        return node_color is not None and node_color != color

    def is_enemy_road(self, edge, color):
        edge_color = self.get_edge_color(edge)
        return edge_color is not None and self.get_edge_color(edge) != color

    def is_friendly_node(self, node_id, color):
        return self.get_node_color(node_id) == color

    def is_friendly_road(self, edge, color):
        return self.get_edge_color(edge) == color


def longest_acyclic_path(board: Board, node_set: Set[int], color: Color):
    # FIX A27 (live equivalence-scan regression, length undercount by exactly
    # 1: seeds 587, 954): an enemy-owned cut-node adjacent to this component
    # via a friendly road must ALSO be tried as a DFS start, not just the
    # component's own stored members. build_road deliberately does NOT add
    # an enemy node to connected_components storage (adding it there would
    # make two genuinely SEPARATE pieces that merely both happen to touch
    # the same enemy node from different sides look like one merged
    # component to `_get_connected_component_index` -- see
    # test_building_into_enemy_doesnt_merge_components), so such a node is
    # invisible to the `for start_node in node_set` loop below. That's fine
    # when the enemy node only has ONE friendly exit -- starting from any
    # interior node_set member still reaches it and (per FIX A16) correctly
    # counts the final edge. It undercounts by 1 whenever the enemy node has
    # TWO OR MORE friendly exits (e.g. a cycle that closes back through it):
    # reached from the interior, expansion correctly stops AT the enemy node
    # (can't travel past it), but that single traversal never gets to try
    # the enemy node's OTHER exit(s) fresh, which a dedicated start there
    # would. Trying it as an extra start (without touching component
    # storage) recovers exactly that.
    start_candidates = set(node_set)
    for member in node_set:
        for neighbor in STATIC_GRAPH.neighbors(member):
            if not board.is_enemy_node(neighbor, color):
                continue
            if board.is_friendly_road(tuple(sorted((member, neighbor))), color):
                start_candidates.add(neighbor)

    paths = []
    for start_node in start_candidates:
        # do DFS when reach leaf node, stop and add to paths
        paths_from_this_node = []
        agenda: List[Tuple[int, Any]] = [(start_node, [])]
        while len(agenda) > 0:
            node, path_thus_far = agenda.pop()

            # FIX A16 (matches upstream issue #378 / PR #379): an enemy-owned node blocks
            # travel PAST it, but a road segment leading UP TO it still counts and it is a
            # valid DFS *start* (node_set includes boundary nodes precisely so a component can
            # be walked outward from either enemy-capped end). The bug was checking
            # is_enemy_node on the neighbor being expanded INTO, which drops the final edge
            # into every enemy-capped end entirely -- undercounting by 1 whenever the path is
            # approached from the interior. A single enemy cap was masked because starting the
            # DFS exactly at that boundary node still captured the true length; with BOTH ends
            # capped, only one end can ever be a "true" start per traversal, so the far end's
            # final edge was always dropped (undercount of 1, not 2). Checking is_enemy_node on
            # the POPPED node instead -- and only once we've actually traveled at least one
            # edge (`path_thus_far` non-empty) -- fixes both: it still permits an enemy node as
            # a fresh start, but forbids expanding further FROM it once reached.
            able_to_navigate = False
            if not (path_thus_far and board.is_enemy_node(node, color)):
                for neighbor_node in STATIC_GRAPH.neighbors(node):
                    edge = tuple(sorted((node, neighbor_node)))

                    # Must travel on a friendly road.
                    if not board.is_friendly_road(edge, color):
                        continue

                    if edge not in path_thus_far:
                        agenda.append((neighbor_node, path_thus_far + [edge]))
                        able_to_navigate = True

            if not able_to_navigate:  # then it is leaf node
                paths_from_this_node.append(path_thus_far)

        paths.extend(paths_from_this_node)

    return max(paths, key=len)
