import argparse, os, json, time, random, csv
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

class LLMClient:
    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        raise NotImplementedError

class QwenLLM(LLMClient):
    def __init__(self, api_key: Optional[str]=None, model="qwen3-32b", temperature=0.2,
                 base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"):
        if OpenAI is None:
            raise RuntimeError("需要 openai SDK: pip install openai")
        self.client = OpenAI(api_key=api_key or os.getenv("DASH_SCOPE_KEY") or os.getenv("OPENAI_API_KEY"),
                             base_url=base_url)
        self.model = model
        self.temperature = temperature
    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            response_format={"type":"json_object"},
            messages=[{"role":"system","content":system_prompt},
                      {"role":"user","content":user_prompt}]
        )
        txt = resp.choices[0].message.content
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            # 去除可能的 Markdown 包裹
            return json.loads(txt.strip().split("```")[-1])

# 本地 stub（不联网也能跑通）
class StubLLM(LLMClient):
    def complete_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        # 生成“选择器场景”而非具体 link_id
        return {
          "scenario_name": "mixed_sunstorm_highload_batchring",
          "description": "Solar storm + high-lat load pulse + batch-ring wear + maintenance handoff.",
          "assumptions": ["Ka-band","+Grid","selectors-only (no link_id)"],
          "horizon_s": 1800,
          "events": [
            {"t": 300,  "selector":{"type":"random","fraction":0.10}, "action":"patch", "loss_pct":6.0, "duration":90,  "note":"solar_storm_spike"},
            {"t": 600,  "selector":{"type":"high_lat","fraction":0.08,"min_lat":40}, "action":"patch", "bw_mbps":800, "duration":240, "note":"sun_illumination_derate"},
            {"t": 900,  "selector":{"type":"batch_ring","fraction":0.10}, "action":"down", "duration":45, "note":"wear_tear_batch"},
            {"t": 1200, "selector":{"type":"maintenance","fraction":0.03}, "action":"down", "duration":30, "note":"scheduled_handoff"}
          ],
          "flows": [
            {"src":"gs_tokyo","dst":"gs_paris","rate_mbps":400.0},
            {"src":"gs_ny","dst":"gs_tokyo","rate_mbps":300.0}
          ]
        }

SYSTEM_PROMPT = (
    "You are a scenario designer for robust LEO network evaluation. "
    "Return STRICT JSON. Units: seconds/Mbps/ms/percent. IMPORTANT: "
    "For events, DO NOT use concrete link IDs. Instead, produce a 'selector' object indicating how to choose links "
    "(types: random, high_lat, batch_ring, maintenance). Include 'fraction' (0.01..0.20) and optional 'min_lat'."
)

USER_PROMPT_TEMPLATE = """
Generate a {mode} robustness scenario with selectors (NOT link IDs).
Goals: stress routing resilience, reveal recovery and SLA violations.
Constraints:
- horizon_s={horizon}
- 4..10 events; each has: t, selector{{type,fraction(0.01..0.2),min_lat?}}, action('patch'|'down'|'up'|'restore'),
  optional (bw_mbps, loss_pct, delay_ms, duration), note.
- Intensities guide: storm loss 3..12%; sun derate bw 30..70%; high_load target high latitude (>=40°N); maintenance ~3%.
- Return JSON keys: scenario_name, description, assumptions[], horizon_s, events[], flows[].
"""

# =========================
# 1) “模拟器适配器”与元数据抓取（duck-typing）
# =========================
@dataclass
class LinkMeta:
    link_id: str
    src: str
    dst: str
    src_lat: float
    dst_lat: float
    plane_id: str = "unknown"
    shell_id: str = "unknown"
    batch_id: str = "unknown"

class SimulatorAdapter:
    """
    从任意模拟器对象抽取链路元数据 & 扰动接口 & KPI 快照。
    适配顺序：
      - links 字典
      - graph.edges(data=True)
      - get_links()/enumerate_link_ids()
    """
    def __init__(self, sim):
        self.sim = sim

    # --- 枚举链路 ---
    def list_links(self) -> Dict[str, LinkMeta]:
        # 1) sim.links: Dict[str, link_obj]
        if hasattr(self.sim, "links") and isinstance(self.sim.links, dict):
            out = {}
            for lid, lo in self.sim.links.items():
                src = getattr(lo, "src", None)
                dst = getattr(lo, "dst", None)
                out[lid] = LinkMeta(
                    link_id=str(lid),
                    src=getattr(src, "name", str(src)),
                    dst=getattr(dst, "name", str(dst)),
                    src_lat=float(getattr(src, "lat", getattr(src, "lat_deg", 0.0)) or 0.0),
                    dst_lat=float(getattr(dst, "lat", getattr(dst, "lat_deg", 0.0)) or 0.0),
                    plane_id=str(getattr(src, "plane_id", "unknown")),
                    shell_id=str(getattr(src, "shell_id", "unknown")),
                    batch_id=str(getattr(src, "batch_id", "unknown")),
                )
            return out

        # 2) graph
        if hasattr(self.sim, "graph"):
            g = self.sim.graph
            out = {}
            if hasattr(g, "edges"):
                for u, v, d in g.edges(data=True):
                    lid = d.get("id") or f"{u}--{v}"
                    src_lat = float(d.get("src_lat", 0.0))
                    dst_lat = float(d.get("dst_lat", 0.0))
                    out[str(lid)] = LinkMeta(
                        link_id=str(lid), src=str(u), dst=str(v),
                        src_lat=src_lat, dst_lat=dst_lat,
                        plane_id=str(d.get("plane_id", "unknown")),
                        shell_id=str(d.get("shell_id", "unknown")),
                        batch_id=str(d.get("batch_id", "unknown")),
                    )
            if out: return out

        # 3) 其它 getter
        if hasattr(self.sim, "get_links"):
            out = {}
            for lo in self.sim.get_links():
                lid = getattr(lo, "name", getattr(lo, "id", None)) or f"{getattr(lo,'src','?')}--{getattr(lo,'dst','?')}"
                src = getattr(lo, "src", None); dst = getattr(lo, "dst", None)
                out[str(lid)] = LinkMeta(
                    link_id=str(lid),
                    src=getattr(src, "name", str(src)),
                    dst=getattr(dst, "name", str(dst)),
                    src_lat=float(getattr(src, "lat", 0.0) or 0.0),
                    dst_lat=float(getattr(dst, "lat", 0.0) or 0.0),
                )
            return out

        raise RuntimeError("无法从模拟器对象提取链路列表；请在 SimulatorAdapter.list_links 里添加一个你的适配分支。")

    # --- 扰动接口（多条路径兜底） ---
    def patch_link(self, link_id: str, bw_mbps=None, loss_pct=None, delay_ms=None, up=None):
        if hasattr(self.sim, "patch_link"):
            self.sim.patch_link(link_id,
                                capacity_mbps=bw_mbps,
                                loss_pct=loss_pct,
                                extra_delay_ms=delay_ms,
                                up=up)
            return
        # 直接改属性
        if hasattr(self.sim, "links") and link_id in self.sim.links:
            lo = self.sim.links[link_id]
            if bw_mbps is not None: setattr(lo, "capacity_mbps", bw_mbps)
            if loss_pct is not None: setattr(lo, "loss_pct", loss_pct)
            if delay_ms is not None: setattr(lo, "delay_ms", delay_ms)
            if up is not None: setattr(lo, "up", bool(up))
            return
        # graph 边属性
        if hasattr(self.sim, "graph") and hasattr(self.sim.graph, "edges"):
            u, v = link_id.split("--") if "--" in link_id else (None, None)
            if u and v and self.sim.graph.has_edge(u, v):
                if bw_mbps is not None: self.sim.graph[u][v]["capacity_mbps"] = bw_mbps
                if loss_pct is not None: self.sim.graph[u][v]["loss_pct"] = loss_pct
                if delay_ms is not None: self.sim.graph[u][v]["delay_ms"] = delay_ms
                if up is not None: self.sim.graph[u][v]["up"] = bool(up)
                return
        raise RuntimeError(f"无法对 link {link_id} 应用补丁；请在 SimulatorAdapter.patch_link 里添加你的适配。")

    # --- KPI 快照 ---
    def snapshot_metrics(self) -> Dict[str, Any]:
        for fn in ("snapshot_metrics","compute_metrics","measure_performance"):
            if hasattr(self.sim, fn):
                try:
                    return getattr(self.sim, fn)()
                except Exception:
                    pass
        return {"note": "no-metrics-interface"}

# =========================
# 2) 选择器解析（不需要 links_meta.csv）
# =========================
def resolve_selector(selector: Dict[str, Any], links: Dict[str, LinkMeta], seed=42) -> List[str]:
    """
    selector: {"type": "...", "fraction": 0.1, "min_lat": 40}
    支持:
      - random
      - high_lat
      - batch_ring
      - maintenance（小范围）
    """
    typ = (selector.get("type") or "random").lower()
    frac = max(0.01, min(0.20, float(selector.get("fraction", 0.05))))
    k = max(1, int(len(links) * frac))
    rnd = random.Random(seed)

    items = list(links.values())

    if typ == "high_lat":
        min_lat = float(selector.get("min_lat", 40.0))
        cand = [l for l in items if max(abs(l.src_lat), abs(l.dst_lat)) >= min_lat]
        if not cand: cand = items
        return [x.link_id for x in rnd.sample(cand, min(k, len(cand)))]

    if typ == "batch_ring":
        # 按 batch_id 最大的那一环优先
        by_batch: Dict[str, List[LinkMeta]] = {}
        for l in items:
            by_batch.setdefault(l.batch_id, []).append(l)
        batch = max(by_batch.keys(), key=lambda b: len(by_batch[b])) if by_batch else None
        cand = by_batch.get(batch, items)
        if len(cand) >= k:
            return [x.link_id for x in cand[:k]]
        else:
            # 不足则随机补足
            remainder = [x for x in items if x not in cand]
            cand2 = cand + rnd.sample(remainder, min(k-len(cand), len(remainder)))
            return [x.link_id for x in cand2]

    if typ == "maintenance":
        k = max(1, int(len(links) * min(frac, 0.05)))  # 维护量一般更小
        return [x.link_id for x in rnd.sample(items, min(k, len(items)))]

    # default: random
    return [x.link_id for x in rnd.sample(items, min(k, len(items)))]

# =========================
# 3) 场景执行与日志
# =========================
@dataclass
class EventTemplate:
    t: float
    selector: Dict[str, Any]
    action: str
    bw_mbps: Optional[float] = None
    loss_pct: Optional[float] = None
    delay_ms: Optional[float] = None
    duration: Optional[float] = None
    note: Optional[str] = None

def expand_events(templates: List[EventTemplate], links: Dict[str, LinkMeta], seed=42) -> List[Dict[str, Any]]:
    events = []
    tseed = seed
    for e in templates:
        ids = resolve_selector(e.selector, links, seed=tseed)
        tseed += 1337
        for lid in ids:
            events.append({
                "t": e.t, "link_id": lid, "action": e.action,
                "bw_mbps": e.bw_mbps, "loss_pct": e.loss_pct,
                "delay_ms": e.delay_ms, "duration": e.duration, "note": e.note
            })
    events.sort(key=lambda x: (x["t"], x["link_id"]))
    return events

def run_scenario(sim_adapter: SimulatorAdapter, scenario: Dict[str, Any], time_factor: float = 0.2,
                 out_log="results_log.csv"):
    os.makedirs(os.path.dirname(out_log), exist_ok=True)
    links = sim_adapter.list_links()

    # 1) 解析模板事件
    templates = []
    for e in scenario["events"]:
        templates.append(EventTemplate(
            t=float(e["t"]),
            selector=e["selector"],
            action=e.get("action","patch"),
            bw_mbps=e.get("bw_mbps"), loss_pct=e.get("loss_pct"),
            delay_ms=e.get("delay_ms"), duration=e.get("duration"),
            note=e.get("note")
        ))
    events = expand_events(templates, links)

    # 2) 回放
    start = time.time()
    records = []
    def wait_until(t_sim):
        target = start + t_sim * time_factor
        now = time.time()
        if target > now:
            time.sleep(target - now)

    for ev in events:
        wait_until(ev["t"])
        before = sim_adapter.snapshot_metrics()
        # 执行扰动
        up = None
        if ev["action"] == "down": up = False
        elif ev["action"] == "up": up = True
        sim_adapter.patch_link(ev["link_id"], bw_mbps=ev["bw_mbps"],
                               loss_pct=ev["loss_pct"], delay_ms=ev["delay_ms"], up=up)
        after = sim_adapter.snapshot_metrics()
        records.append({
            "t": ev["t"], "link_id": ev["link_id"], "action": ev["action"],
            "bw_mbps": ev["bw_mbps"] or "", "loss_pct": ev["loss_pct"] or "",
            "delay_ms": ev["delay_ms"] or "", "duration": ev["duration"] or "",
            "note": ev["note"] or "",
            "tp_before": before.get("throughput_Tbps",""),
            "tp_after": after.get("throughput_Tbps",""),
            "stretch_med_after": after.get("stretch_median",""),
            "coverage_after": after.get("coverage_pct",""),
        })
        # 简单的 duration 恢复（如果设置了）
        if ev.get("duration"):
            time.sleep(ev["duration"] * time_factor)
            # 恢复只做 up=True & 清空临时限制（简化；你可接入 sim.restore_link）
            sim_adapter.patch_link(ev["link_id"], bw_mbps=None, loss_pct=None, delay_ms=None, up=True)

    # 3) 写日志
    with open(out_log, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else
                           ["t","link_id","action","bw_mbps","loss_pct","delay_ms","duration","note",
                            "tp_before","tp_after","stretch_med_after","coverage_after"])
        w.writeheader()
        for r in records: w.writerow(r)
    print(f"[OK] Replayed {len(events)} expanded events. Log -> {out_log}")

# =========================
# 4) 将 LLM 场景（选择器）拿到
# =========================
def build_selector_scenario(use_llm: bool, mode: str, horizon: int) -> Dict[str, Any]:
    if use_llm:
        llm = QwenLLM()
    else:
        llm = StubLLM()
    user = USER_PROMPT_TEMPLATE.format(mode=mode, horizon=horizon)
    raw = llm.complete_json(SYSTEM_PROMPT, user)

    # 基本字段校验 & 清洗
    for k in ["scenario_name","description","assumptions","horizon_s","events","flows"]:
        if k not in raw: raise RuntimeError(f"LLM/stub 输出缺少字段: {k}")
    # 约束时间与强度
    out_events = []
    for e in raw["events"]:
        t = float(e.get("t", 0.0))
        if 0 <= t <= horizon:
            sel = e.get("selector") or {}
            frac = max(0.01, min(0.20, float(sel.get("fraction", 0.05))))
            sel["fraction"] = frac
            if sel.get("type","").lower() == "high_lat":
                sel["min_lat"] = float(sel.get("min_lat", 40.0))
            ev = {
                "t": t, "selector": sel, "action": e.get("action","patch"),
                "bw_mbps": e.get("bw_mbps"), "loss_pct": e.get("loss_pct"),
                "delay_ms": e.get("delay_ms"), "duration": e.get("duration"),
                "note": e.get("note","")
            }
            out_events.append(ev)
    raw["events"] = sorted(out_events, key=lambda x: x["t"])
    raw["horizon_s"] = min(horizon, float(raw["horizon_s"]))
    return raw


class _Node:
    def __init__(self, name, lat=0.0, plane_id="p0", shell_id="s1", batch_id="bA"):
        self.name=name; self.lat=lat; self.plane_id=plane_id; self.shell_id=shell_id; self.batch_id=batch_id

class _Link:
    def __init__(self, name, src, dst):
        self.name=name; self.src=src; self.dst=dst
        self.capacity_mbps=1000; self.loss_pct=0.0; self.delay_ms=0.0; self.up=True

class FakeSim:
    def __init__(self):
        self.links={}
        A=_Node("sat_L1_01", lat=48.0, plane_id="p1", shell_id="L1", batch_id="G1-01")
        B=_Node("sat_L1_02", lat=52.0, plane_id="p1", shell_id="L1", batch_id="G1-01")
        C=_Node("sat_L2_01", lat=12.0, plane_id="p2", shell_id="L2", batch_id="G2-03")
        D=_Node("sat_L2_02", lat=58.0, plane_id="p2", shell_id="L2", batch_id="G2-03")
        E=_Node("sat_L1_03", lat=33.0, plane_id="p1", shell_id="L1", batch_id="G1-02")
        self.links[f"{A.name}--{B.name}"]=_Link(f"{A.name}--{B.name}",A,B)
        self.links[f"{B.name}--{C.name}"]=_Link(f"{B.name}--{C.name}",B,C)
        self.links[f"{C.name}--{D.name}"]=_Link(f"{C.name}--{D.name}",C,D)
        self.links[f"{D.name}--{E.name}"]=_Link(f"{D.name}--{E.name}",D,E)
        self.links[f"{A.name}--{E.name}"]=_Link(f"{A.name}--{E.name}",A,E)
        self._tp=1.0

    def snapshot_metrics(self):
        down_cnt = sum(1 for l in self.links.values() if not l.up)
        avg_loss = sum(l.loss_pct for l in self.links.values())/max(1,len(self.links))
        self._tp = max(0.1, 1.2 - 0.15*down_cnt - 0.01*avg_loss)
        return {"throughput_Tbps": round(self._tp,3),
                "stretch_median": 1.2 + 0.02*down_cnt + 0.001*avg_loss,
                "coverage_pct": 95.0 - 2.0*down_cnt}

    def patch_link(self, link_id, capacity_mbps=None, loss_pct=None, extra_delay_ms=None, up=None):
        if link_id not in self.links: raise KeyError(link_id)
        l=self.links[link_id]
        if capacity_mbps is not None: l.capacity_mbps=capacity_mbps
        if loss_pct is not None: l.loss_pct=loss_pct
        if extra_delay_ms is not None: l.delay_ms=extra_delay_ms
        if up is not None: l.up=bool(up)

def get_simulator():
    return FakeSim()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["sunlight","solar_storm","high_load","wear_tear","mixed"], default="mixed")
    ap.add_argument("--horizon", type=int, default=1800)
    ap.add_argument("--use-llm", type=int, default=0, help="1=use Qwen via DashScope, 0=stub")
    ap.add_argument("--log", default="results/results_log.csv")
    ap.add_argument("--time-factor", type=float, default=0.2, help="wallclock = sim_time * time_factor")
    args=ap.parse_args()

    scenario = build_selector_scenario(use_llm=bool(args.use_llm), mode=args.mode, horizon=args.horizon)
    sim = get_simulator()
    adapter = SimulatorAdapter(sim)
    run_scenario(sim_adapter=adapter, scenario=scenario, time_factor=args.time_factor, out_log=args.log)

if __name__=="__main__":
    main()
