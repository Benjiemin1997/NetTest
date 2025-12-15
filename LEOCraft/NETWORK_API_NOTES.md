# LEOCraft network construction notes

- Satellite nodes are added during network graph creation via `Constellation.create_network_graph`, which calls the internal `_add_satellites_from_shell` helper to add each satellite node to `sat_net_graph`.
- Current codebase does not expose helper functions to remove satellite nodes after construction (no `remove_node`/`delete_satellite` found under `LEOCraft`).
- Inter-satellite and ground links are created in `_add_ISLs_from_shell` and ground station link builders; similar removal hooks are not present by default.

These notes summarize available APIs to help threat models align with the existing graph lifecycle.

## Runtime masking (NetTest integration layer)

- Because LEOCraft lacks public deletion APIs, `Threat_Define/simulation/environment.py` maintains a `runtime_state` mask (`offline_nodes`, `offline_edges`) plus a captured baseline graph.
- Threat models should call `LEONetworkModel.mark_satellite_status` / `disable_satellites` / `congest_links`, which update the mask and re-apply it onto the LEOCraft constellation graph before performance recomputation.
- This “static topology + runtime mask” design avoids rebuilding the constellation each step while still exposing outages/congestion to routing and metric calculations.
