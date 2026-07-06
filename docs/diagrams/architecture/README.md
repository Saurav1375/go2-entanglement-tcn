# Architecture Diagrams

Publication-quality architecture diagrams for the Leg-Entanglement Detection system (the **detector**
pipeline, ROS nodes, message/topic definitions, and the TCN model).

> ⚠️ **Recovery blocks are legacy.** The recovery portion of the *detailed* figure depicts the older
> FSM + strategy-manager design that has since been replaced. The current recovery is a **one-shot
> sequence** (stop → move back → stop → front jump → stop, once per alarm) — see
> [`../../recovery/RECOVERY.md`](../../recovery/RECOVERY.md). The detector blocks remain accurate.

## Figures

| figure | purpose | editable source | exports |
|---|---|---|---|
| **High-level** | one-glance overview of the whole pipeline (offline training → export → on-robot detector → recovery → closed loop) | `architecture_highlevel.svg` (hand-tuned SVG) · `architecture_highlevel.mmd` (Mermaid alt) | `.png` (2976×1680) · `.pdf` (1 page) |
| **Detailed** | appendix/documentation level: every processing block, ROS topic (with message type + QoS), the 10-state FSM, detector-aware strategy ordering, and verified Sport-API IDs | `architecture_detailed.dot` (Graphviz) · `architecture_detailed.mmd` (Mermaid alt) | `.svg` · `.png` (6241×3179) · `.pdf` (1 page) |

The **high-level** figure fits a single/double IEEE column; the **detailed** figure is a full-page
landscape appendix schematic.

## Styling (consistent across both figures)

- Font: Helvetica / Arial sans-serif.
- Print-safe palette, colour-coded by subsystem (see the legend in the high-level figure):
  - Data / train `#E8EEF7`·`#3B5B92` · Model / export `#E9E6F7`·`#5B4B9E` · Export `#E1F0EE`·`#2E7D74`
  - Detector `#E6F2E6`·`#3C7A3C` · Recovery `#FBEEDD`·`#B5701A` · Robot / operator `#ECEFF3`·`#37474F`
  - ROS 2 topic `#FFF6D6`·`#A38A1E`

## Regenerate the exports

```bash
cd docs/diagrams/architecture

# High-level (from the SVG source)
rsvg-convert -z 2.4 -f png -o architecture_highlevel.png architecture_highlevel.svg
rsvg-convert       -f pdf -o architecture_highlevel.pdf architecture_highlevel.svg

# Detailed (from the Graphviz source)
dot -Tsvg              architecture_detailed.dot -o architecture_detailed.svg
dot -Tpng -Gdpi=200    architecture_detailed.dot -o architecture_detailed.png
dot -Tpdf              architecture_detailed.dot -o architecture_detailed.pdf
```

The Mermaid `.mmd` sources are provided as editable alternates (render at
[mermaid.live](https://mermaid.live) or with `mmdc`); they encode the same structure. The `.svg`
files are also directly editable (Inkscape / draw.io / any text editor).

## Correspondence to code

- **Detector** blocks mirror `EntanglementEngine.push()` (ring buffer → `build_channel_matrix`
  [200,48]→[60,200] → `Normalizer.apply` → ONNX backend → sigmoids/temperature → physics-blended
  intensity → threshold+debounce+per-leg attribution → stationarity gate) and `node.py`.
- **Recovery** blocks mirror `recovery_node.py` + `recovery_fsm.py` (10 states) +
  `strategy_manager.py` (ordering policy) + `strategies.py`/`plan_runner.py` + `sport_client.py`
  (dry-run gate) with the verified `sport_api.py` IDs.
- **Topics / messages** match `EntanglementState.msg`, `/entanglement_state`, `/lowstate`,
  `/sportmodestate`, `/api/sport/{request,response}`, `/recovery_status`, `/recovery_estop|reset`.
- **TCN** mirrors `ml/model.py` (CausalConv1d stem 60→64, 5× residual TCN block d=1,2,4,8,16,
  RF=127, last-timestep embedding → bin/legs/intensity heads).
