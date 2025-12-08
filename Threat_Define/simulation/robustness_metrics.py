from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Dict, Iterable, List, Optional, Set, Tuple

import networkx as nx


@dataclass(frozen=True)
class PairStats:
    """Per ground-station pair communication statistics."""

    reachable: bool
    latency: float
    stretch: float
    hops: int


@dataclass
class RobustnessResults:
    srr: float
    dpr: float
    lccr: float
    cr: float
    tr: float
    d_t: float
    si: float
    hi: float
    rf: Optional[float] = None
    apv_leo: Optional[float] = None


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    if not vals:
        return 0.0
    return float(sum(vals) / len(vals))


def compute_srr(
    baseline_stats: Dict[Tuple[str, str], PairStats],
    attacked_stats: Dict[Tuple[str, str], PairStats],
) -> float:
    """Service Reachability Rate (SRR).

    SRR = reachable_pairs_attack / reachable_pairs_baseline
    """

    baseline_reachable = {
        pair for pair, stats in baseline_stats.items() if stats.reachable
    }
    if not baseline_reachable:
        return 0.0

    reachable_after_attack = sum(
        1
        for pair in baseline_reachable
        if attacked_stats.get(pair, PairStats(False, 0.0, 0.0, 0)).reachable
    )
    return float(reachable_after_attack / len(baseline_reachable))


def compute_dpr(attacked_stats: Dict[Tuple[str, str], PairStats]) -> float:
    """Disconnected Pair Ratio (DPR).

    DPR = unreachable_pairs_attack / total_pairs
    """

    total_pairs = len(attacked_stats)
    if total_pairs == 0:
        return 0.0
    unreachable = sum(1 for stats in attacked_stats.values() if not stats.reachable)
    return float(unreachable / total_pairs)


def compute_lccr(attacked_graph: nx.Graph, gs_nodes: Set[str]) -> float:
    """Largest Connected Component Ratio (LCCR) for ground stations.

    LCCR = |CC_max(G_atk^GS)| / |V_GS|
    where G_atk^GS is the subgraph induced by ground-station nodes.
    """

    if not gs_nodes:
        return 0.0

    present_gs = [node for node in gs_nodes if node in attacked_graph]
    if not present_gs:
        return 0.0

    subgraph = attacked_graph.subgraph(present_gs)
    # Use undirected view to evaluate connectivity among GS nodes.
    if nx.is_directed(subgraph):
        undirected = subgraph.to_undirected(as_view=True)
    else:
        undirected = subgraph

    components = list(nx.connected_components(undirected))
    if not components:
        return 0.0

    largest_cc = max(len(comp) for comp in components)
    return float(largest_cc / len(gs_nodes))


def compute_cr(
    baseline_coverage: Dict[int, Set[str]],
    attacked_coverage: Dict[int, Set[str]],
) -> float:
    """Coverage Retention (CR).

    CR = sum_t |U_atk(t)| / sum_t |U_0(t)|
    """

    baseline_sum = sum(len(stations) for stations in baseline_coverage.values())
    if baseline_sum == 0:
        return 0.0

    attacked_sum = sum(len(stations) for stations in attacked_coverage.values())
    return float(attacked_sum / baseline_sum)


def compute_tr(
    baseline_total_throughput: float,
    attacked_total_throughput: float,
) -> Tuple[float, float]:
    """Throughput Robustness (TR) and throughput loss ratio.

    TR = T_atk / T_0
    D_T = 1 - TR

    If the baseline throughput is non-positive, both metrics default to 0.0
    to avoid division by zero.
    """

    if baseline_total_throughput <= 0:
        return 0.0, 0.0

    tr = float(attacked_total_throughput / baseline_total_throughput)
    d_t = float(1.0 - tr)
    return tr, d_t


def _collect_stats_for_pairs(
    pairs: Iterable[Tuple[str, str]],
    stats_mapping: Dict[Tuple[str, str], PairStats],
    ignore_unreachable: bool,
    attribute: str,
) -> List[float]:
    collected: List[float] = []
    for pair in pairs:
        stats = stats_mapping.get(pair)
        if stats is None:
            continue
        if not stats.reachable and ignore_unreachable:
            continue
        value = getattr(stats, attribute)
        # For unreachable paths when not ignoring, treat as infinite penalty.
        if not stats.reachable and not ignore_unreachable:
            collected.append(float("inf"))
        else:
            collected.append(float(value))
    return collected


def compute_si(
    baseline_stats: Dict[Tuple[str, str], PairStats],
    attacked_stats: Dict[Tuple[str, str], PairStats],
    ignore_unreachable: bool = True,
) -> float:
    """Stretch Increase (SI).

    SI = (mean_stretch_atk - mean_stretch_baseline) / mean_stretch_baseline
    Baseline average considers the same pair set as the attack average when
    ``ignore_unreachable`` is True; otherwise unreachable pairs incur an
    infinite penalty.
    """

    baseline_reachable_pairs = {
        pair for pair, stats in baseline_stats.items() if stats.reachable
    }
    if not baseline_reachable_pairs:
        return 0.0

    if ignore_unreachable:
        pairs_to_use = {
            pair
            for pair in baseline_reachable_pairs
            if attacked_stats.get(pair, PairStats(False, 0.0, 0.0, 0)).reachable
        }
    else:
        pairs_to_use = baseline_reachable_pairs

    baseline_values = _collect_stats_for_pairs(
        pairs_to_use, baseline_stats, ignore_unreachable=False, attribute="stretch"
    )
    attacked_values = _collect_stats_for_pairs(
        pairs_to_use, attacked_stats, ignore_unreachable=ignore_unreachable, attribute="stretch"
    )

    baseline_mean = _mean(baseline_values)
    if baseline_mean == 0:
        return 0.0
    attack_mean = _mean(attacked_values)
    return float((attack_mean - baseline_mean) / baseline_mean)


def compute_hi(
    baseline_stats: Dict[Tuple[str, str], PairStats],
    attacked_stats: Dict[Tuple[str, str], PairStats],
    ignore_unreachable: bool = True,
) -> float:
    """Hop Inflation (HI).

    HI = (mean_hops_atk - mean_hops_baseline) / mean_hops_baseline
    Baseline and attack means follow the same pair selection logic as SI.
    """

    baseline_reachable_pairs = {
        pair for pair, stats in baseline_stats.items() if stats.reachable
    }
    if not baseline_reachable_pairs:
        return 0.0

    if ignore_unreachable:
        pairs_to_use = {
            pair
            for pair in baseline_reachable_pairs
            if attacked_stats.get(pair, PairStats(False, 0.0, 0.0, 0)).reachable
        }
    else:
        pairs_to_use = baseline_reachable_pairs

    baseline_values = _collect_stats_for_pairs(
        pairs_to_use, baseline_stats, ignore_unreachable=False, attribute="hops"
    )
    attacked_values = _collect_stats_for_pairs(
        pairs_to_use, attacked_stats, ignore_unreachable=ignore_unreachable, attribute="hops"
    )

    baseline_mean = _mean(baseline_values)
    if baseline_mean == 0:
        return 0.0
    attack_mean = _mean(attacked_values)
    return float((attack_mean - baseline_mean) / baseline_mean)


def compute_rf(
    baseline_throughput_by_gs: Dict[str, float],
    attacked_throughput_by_gs: Dict[str, float],
    region_of_gs: Dict[str, str],
) -> float:
    """Robustness Fairness (RF) across regions using throughput retention.

    For each region r:
        R_r = T_atk(r) / T_0(r)
    RF = sqrt( (1/K) * sum_{r}(R_r - mean(R))^2 )
    """

    per_region_baseline: Dict[str, float] = {}
    per_region_attack: Dict[str, float] = {}

    for gs_id, baseline_value in baseline_throughput_by_gs.items():
        region = region_of_gs.get(gs_id)
        if region is None:
            continue
        per_region_baseline[region] = per_region_baseline.get(region, 0.0) + float(
            baseline_value
        )
        attacked_value = float(attacked_throughput_by_gs.get(gs_id, 0.0))
        per_region_attack[region] = per_region_attack.get(region, 0.0) + attacked_value

    if not per_region_baseline:
        return 0.0

    retention: List[float] = []
    for region, baseline_value in per_region_baseline.items():
        if baseline_value <= 0:
            retention.append(0.0)
            continue
        attack_value = per_region_attack.get(region, 0.0)
        retention.append(float(attack_value / baseline_value))

    if not retention:
        return 0.0

    average_retention = _mean(retention)
    variance = _mean([(value - average_retention) ** 2 for value in retention])
    return float(sqrt(variance))


def compute_apv_leo(
    graph: nx.Graph,
    gs_pairs: List[Tuple[str, str]],
    attack_nodes: Set[str],
    weight: Optional[str] = None,
    max_pairs: Optional[int] = None,
) -> float:
    """Attack Path Vulnerability for LEO (APV_LEO).

    APV_LEO = affected_shortest_paths / total_shortest_paths
    Only a single shortest path per pair is considered to bound complexity.
    """

    if not gs_pairs:
        return 0.0

    selected_pairs = gs_pairs
    if max_pairs is not None and max_pairs > 0:
        selected_pairs = gs_pairs[:max_pairs]

    total_paths = 0
    affected_paths = 0

    for src, dst in selected_pairs:
        if src not in graph or dst not in graph:
            continue
        try:
            path = nx.shortest_path(graph, source=src, target=dst, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        total_paths += 1
        if any(node in attack_nodes for node in path):
            affected_paths += 1

    if total_paths == 0:
        return 0.0
    return float(affected_paths / total_paths)


class RobustnessEvaluator:
    """Convenience wrapper that aggregates all robustness metrics."""

    def __init__(
        self,
        baseline_stats: Dict[Tuple[str, str], PairStats],
        attacked_stats: Dict[Tuple[str, str], PairStats],
        attacked_graph: nx.Graph,
        gs_nodes: Set[str],
        baseline_coverage: Optional[Dict[int, Set[str]]] = None,
        attacked_coverage: Optional[Dict[int, Set[str]]] = None,
        baseline_total_throughput: float = 0.0,
        attacked_total_throughput: float = 0.0,
        baseline_throughput_by_gs: Optional[Dict[str, float]] = None,
        attacked_throughput_by_gs: Optional[Dict[str, float]] = None,
        region_of_gs: Optional[Dict[str, str]] = None,
        attack_nodes: Optional[Set[str]] = None,
        apv_weight: Optional[str] = None,
        apv_max_pairs: Optional[int] = None,
        ignore_unreachable: bool = True,
        graph_for_apv: Optional[nx.Graph] = None,
        baseline_result: Optional[Dict[str, object]] = None,
        attacked_result: Optional[Dict[str, object]] = None,
    ) -> None:
        self.baseline_stats = baseline_stats
        self.attacked_stats = attacked_stats
        self.attacked_graph = attacked_graph
        self.gs_nodes = gs_nodes
        self.baseline_coverage = baseline_coverage or {}
        self.attacked_coverage = attacked_coverage or {}
        self.baseline_total_throughput = baseline_total_throughput
        self.attacked_total_throughput = attacked_total_throughput
        self.baseline_throughput_by_gs = baseline_throughput_by_gs or {}
        self.attacked_throughput_by_gs = attacked_throughput_by_gs or {}
        self.region_of_gs = region_of_gs or {}
        self.attack_nodes = attack_nodes or set()
        self.apv_weight = apv_weight
        self.apv_max_pairs = apv_max_pairs
        self.ignore_unreachable = ignore_unreachable
        self.graph_for_apv = graph_for_apv or attacked_graph
        # Optional convenience: allow callers to pass pre-bundled results
        # (baseline_result/attacked_result) to hydrate the evaluator without
        # touching the rest of the pipeline. Missing fields are ignored to keep
        # backward compatibility.
        if baseline_result:
            self._hydrate_from_result(baseline_result, is_baseline=True)
        if attacked_result:
            self._hydrate_from_result(attacked_result, is_baseline=False)

    @staticmethod
    def _status(message: str) -> None:
        """Lightweight console status helper for evaluation progress."""

        print(f"[STATUS] {message}")

    def _hydrate_from_result(self, result: Dict[str, object], *, is_baseline: bool) -> None:
        """Ingest a loosely-typed result bundle into evaluator fields."""

        def _coerce_pair_stats(value: object) -> Dict[Tuple[str, str], PairStats]:
            if isinstance(value, dict):
                converted: Dict[Tuple[str, str], PairStats] = {}
                for key, item in value.items():
                    if isinstance(item, PairStats):
                        converted[tuple(key)] = item  # type: ignore[arg-type]
                        continue
                    if isinstance(item, dict):
                        converted[tuple(key)] = PairStats(
                            reachable=bool(item.get("reachable", False)),
                            latency=float(item.get("latency", 0.0)),
                            stretch=float(item.get("stretch", 0.0)),
                            hops=int(item.get("hops", 0)),
                        )  # type: ignore[arg-type]
                return converted
            return {}

        def _coerce_coverage(value: object) -> Dict[int, Set[str]]:
            if not isinstance(value, dict):
                return {}
            coverage: Dict[int, Set[str]] = {}
            for key, stations in value.items():
                try:
                    t = int(key)
                except Exception:
                    continue
                if isinstance(stations, (list, set, tuple)):
                    coverage[t] = {str(item) for item in stations}
            return coverage

        stats = _coerce_pair_stats(result.get("pair_stats"))
        coverage = _coerce_coverage(result.get("coverage"))
        throughput = result.get("total_throughput")
        throughput_by_gs = result.get("throughput_by_gs")
        graph = result.get("graph")
        gs_nodes = result.get("gs_nodes")
        attack_nodes = result.get("attack_nodes")

        if is_baseline:
            if stats:
                self.baseline_stats = stats
            if coverage:
                self.baseline_coverage = coverage
            if isinstance(throughput, (int, float)):
                self.baseline_total_throughput = float(throughput)
            if isinstance(throughput_by_gs, dict):
                self.baseline_throughput_by_gs = {
                    str(gs): float(val) for gs, val in throughput_by_gs.items()
                }
        else:
            if stats:
                self.attacked_stats = stats
            if coverage:
                self.attacked_coverage = coverage
            if isinstance(throughput, (int, float)):
                self.attacked_total_throughput = float(throughput)
            if isinstance(throughput_by_gs, dict):
                self.attacked_throughput_by_gs = {
                    str(gs): float(val) for gs, val in throughput_by_gs.items()
                }
            if isinstance(graph, nx.Graph):
                self.attacked_graph = graph
            if isinstance(gs_nodes, (set, list, tuple)):
                self.gs_nodes = {str(node) for node in gs_nodes}
            if isinstance(attack_nodes, (set, list, tuple)):
                self.attack_nodes = {str(node) for node in attack_nodes}
            if graph is not None and self.graph_for_apv is None:
                if isinstance(graph, nx.Graph):
                    self.graph_for_apv = graph

    def compute_all(self) -> RobustnessResults:
        """Compute every robustness indicator and bundle the results."""

        self._status("开始计算服务可达率 (SRR) 与失联对比例 (DPR)")
        srr = compute_srr(self.baseline_stats, self.attacked_stats)
        dpr = compute_dpr(self.attacked_stats)
        self._status("计算最大连通分量占比 (LCCR) 与覆盖保持率 (CR)")
        lccr = compute_lccr(self.attacked_graph, self.gs_nodes)
        cr = compute_cr(self.baseline_coverage, self.attacked_coverage)
        self._status("计算吞吐鲁棒性 (TR) 及其损失比")
        tr, d_t = compute_tr(
            self.baseline_total_throughput, self.attacked_total_throughput
        )
        self._status("计算时延拉伸增幅 (SI) 与平均跳数增幅 (HI)")
        si = compute_si(
            self.baseline_stats, self.attacked_stats, ignore_unreachable=self.ignore_unreachable
        )
        hi = compute_hi(
            self.baseline_stats, self.attacked_stats, ignore_unreachable=self.ignore_unreachable
        )

        rf: Optional[float] = None
        if self.baseline_throughput_by_gs and self.region_of_gs:
            self._status("计算区域鲁棒性不均衡度 (RF)")
            rf = compute_rf(
                self.baseline_throughput_by_gs,
                self.attacked_throughput_by_gs,
                self.region_of_gs,
            )

        apv_leo: Optional[float] = None
        if self.attack_nodes:
            self._status("计算攻击路径脆弱度 (APV_LEO)")
            gs_pairs = list(self.baseline_stats.keys())
            apv_leo = compute_apv_leo(
                self.graph_for_apv,
                gs_pairs,
                self.attack_nodes,
                weight=self.apv_weight,
                max_pairs=self.apv_max_pairs,
            )

        results = RobustnessResults(
            srr=srr,
            dpr=dpr,
            lccr=lccr,
            cr=cr,
            tr=tr,
            d_t=d_t,
            si=si,
            hi=hi,
            rf=rf,
            apv_leo=apv_leo,
        )
        self._status(
            "鲁棒性指标计算完成: "
            f"SRR={results.srr:.3f}, DPR={results.dpr:.3f}, LCCR={results.lccr:.3f}, "
            f"CR={results.cr:.3f}, TR={results.tr:.3f}, ΔT={results.d_t:.3f}, "
            f"SI={results.si:.3f}, HI={results.hi:.3f}, "
            f"RF={'N/A' if results.rf is None else f'{results.rf:.3f}'}, "
            f"APV_LEO={'N/A' if results.apv_leo is None else f'{results.apv_leo:.3f}'}"
        )
        return results