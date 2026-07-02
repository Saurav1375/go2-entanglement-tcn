I have the fact sheet and the figure/table manifest from the outline. Let me compile the report.

Key manifest labels: fig:system, fig:data, tab:dataset, fig:dataset, fig:feat, tab:features, fig:tcn, tab:hparams, fig:runtime, fig:e2e, tab:oppoint, tab:detection, fig:confusion, fig:rocpr, fig:loro, fig:perleg, tab:perleg, fig:reliability, tab:ablation, fig:ablation, fig:field, fig:latency, tab:runtime.

Now let me cross-check every ref in the drafts against the manifest and check facts.

# Editorial Report — Leg-Entanglement Detection Paper

## (1) FACTUAL ERRORS

- **`abstract`**: "leave-one-recording-out cross-validation over $15$ folds gives an F1 of $0.807 \pm 0.142$." Correct per fact sheet (15 folds, 0.807±0.142). OK — but note the ablation section (`results`) reports the shipped 60-ch LORO as **0.832±0.166**, while the headline LORO is **0.807±0.142**. These are two different runs; ensure the abstract's 0.807 is the headline LORO (it is). No error, but see Inconsistencies.
- **`data`**: "ranges from roughly $410$ to $500$~Hz, with a median near $483$~Hz." Fact sheet says **~410–502 Hz**. Minor; "500" acceptable as rounded. Consistent with `feature` ("410 to 500 Hz").
- **`data`**: The held-out positives are listed as "`back_left_hand2`, `back_right_wire1`, `front_left_hand2`, and `front_both_wire2`". Correct per fact sheet (line 22). OK.
- **`results` (deployment robustness)**: "on the original four-file protocol, aggregate $F1$ moved from $0.851$ to $0.807$". The value **0.851** (v1 aggregate F1) is **not in the fact sheet**. The fact sheet provides no v1 aggregate F1. This is an unverifiable claim — flag it. Only per-phase firing rates (v1→v2) are authorized.
- **`discussion`**: "ROC-AUC from $0.830$ to $0.956$" — 0.830 is the baseline ROC-AUC (fact sheet line 67), 0.956 the TCN. Correct.
- **`ml`**: "receptive field is 127 samples, or 254\,ms" — correct. "136{,}393 parameters" — correct. "half-megabyte" / ~0.5 MB — correct.
- **`ml` / severity**: $I_{\text{leg}} = g\sqrt{d\,r}$ and "deployed intensity blends the network head and this physics index in equal parts" (0.5/0.5) — correct.
- No numeric contradiction found in `expsetup`, `future`, `limitations`, `conclusion` against the sheet (T=5.918, Brier 0.128→0.118, ECE 0.134→0.156, thresholds 0.9999/0.963, latency ~1.0/1.3 ms, Δ ≤ 3.8e-7/6.3e-7/2.1e-7 all match).

## (2) SCOPE LEAKS (recovery / actuation — must be removed)

- **`intro`**: "…is a prerequisite for any sensible response." — borderline; "response" gestures at recovery. Acceptable as motivation but tighten if strict.
- **`related`**: "when a leg becomes snagged, executes a **disentanglement maneuver** as that leg advances through its swing phase" — describes Yim et al.'s actuation. This is describing *prior work*, permissible, but the paper's own scope note ("we treat detection as a self-contained perception problem… decoupled from any actuation policy") is fine. No removal needed but confirm framing stays on "what is inferred," which it does.
- **`runtime`**: The content summary explicitly states "**No recovery controller/FSM mentioned**" and lists only `/entanglement_state` publication + stationarity gate (a perception-side buffer gate, not actuation). No leak. **However** the stationarity gate "arm 300 ms post-resume suppression" and "reset debounce" are detection-pipeline behaviors, acceptable. Verify the final `.tex` body (only a summary is provided here) contains no FSM/controller text.
- No FSM/controller/actuation leak in the visible section bodies. Primary action: **audit the actual `runtime_architecture.tex` file** (only a summary was supplied in the JSON).

## (3) INCONSISTENCIES ACROSS SECTIONS

- **Shipped-config LORO number**: `abstract`/`conclusion`/`results(LORO)`/`limitations`/`future` use headline LORO **0.807±0.142**; `results(ablation)` and `future` also cite the 60-ch ablation LORO **0.832±0.166**. Both are correct but adjacent; `future` uses **0.832±0.166** ("shipped 60-channel set") for the IMU comparison while the same section elsewhere cites 0.807±0.142. Add one sentence distinguishing "headline LORO (0.807)" from "ablation-protocol LORO (0.832)" to avoid a reader seeing two "shipped LORO F1" values.
- **LORO fold description**: `results` says "the weakest folds are the under-represented front-right **and rear-right stop and intensity cases**." "rear-right … intensity cases" is muddled/ungrounded — RR is the *best-represented* leg (7 recordings), so calling it under-represented contradicts `data`/`limitations`/`discussion` (which single out FR only). Fix: attribute weak folds to FR, not RR.
- **Sample-rate phrasing**: `data` "roughly 410 to 500 Hz, median near 483 Hz"; `feature` repeats "roughly 410 to 500 Hz with a median near 483 Hz" nearly verbatim — duplicated claim across III-A and III-B; the `feature` restatement can be trimmed to a back-reference.
- **Duplicated causal/train-deploy-equivalence claim**: stated in `abstract`, `intro` (fig:system para), `ml` ("no train-test discrepancy… verify empirically later"), `discussion`, `limitations`. The Δ≤3.8e-7 equivalence appears in `results`, `discussion`, `limitations` verbatim-ish. Merge/reduce repetition; keep the full statement once (Results) and back-reference.
- **Terminology "intensity" vs "severity"**: `ml`/`runtime`/loss use "intensity head"; abstract/intro/conclusion/limitations use "severity index." Both map to the same output. Standardize: "severity index" for the reported quantity, "intensity head" only for the network head. Currently mostly consistent but verify `results` LORO line "front-right and rear-right stop and **intensity** cases" misuses "intensity."
- **Baseline exact-match**: `results` "0.908 vs. 0.106" — matches sheet. `conclusion` omits it; fine.
- **`data` split counts** "16 training, 4 validation, 5 test" consistent with `expsetup`. OK.

## (4) WRITING FLAGS

- **Leaked drafting preamble / throat-clearing (must delete)** — several sections begin with meta-text that is not part of the paper:
  - `abstract`: "I have all the facts I need. Writing the abstract now." plus an entire **duplicate scratch abstract** (the fragmented "Held-out test split: … Offline caveat." block) precedes the real abstract. Delete the preamble and the scratch block.
  - `intro`: "The Introduction section is written to …section_introduction.tex. Raw LaTeX body below:" — delete.
  - `ml`: "I have everything I need. Here is the LaTeX body for the assigned subsection." — delete.
  - `runtime`: entire entry is a "Content summary," not LaTeX body — the actual `\subsection{Runtime Architecture}` body must be inserted.
  - `results`: "I have everything I need from the fact sheet. Here is the Results section." — delete.
  - `discussion`, `limitations`, `conclusion`: each opens with "I have … the facts …. Here is the … section." / "Now I'll write…" — delete all.
- **AI-tell / filler openers**: `discussion` "Two of the more instructive results concern…"; `results` "Our first question is whether…"; `expsetup` "Evaluation choices for this problem are dominated by one fact:" — acceptable but formulaic; vary.
- **Em-dash usage**: heavy in `related` and `ml` (e.g., `ml` "there is no train-test discrepancy…—" and multiple "which is how a small network reaches far back in time"). `runtime` summary claims "at most one em dash per paragraph (none used)" — verify in actual body. `data` uses em-dash-style "train--deploy" (that's in abstract). Reduce em-dash reliance in `related`/`ml`.
- **Uniform paragraph rhythm**: `related` and `expsetup` are four/six near-equal medium paragraphs each with the same "claim, then three-part elaboration" cadence. Vary length.
- **Banned/soft phrases**: "comfortably real-time," "comfortably inside/under" appears in `ml`, `expsetup`, `results`, `discussion`, `limitations`, `conclusion` (6×) — overused; vary. "honestly"/"honest" appears in `abstract`, `related`, `intro`, `results`, `discussion`, `limitations` (7×) — overused hedge; trim.

## (5) MISSING CROSS-REFS / DANGLING \ref

Manifest labels (from OUTLINE): fig:system, fig:data, tab:dataset, fig:dataset, fig:feat, tab:features, fig:tcn, tab:hparams, fig:runtime, fig:e2e, tab:oppoint, tab:detection, fig:confusion, fig:rocpr, fig:loro, fig:perleg, tab:perleg, fig:reliability, tab:ablation, fig:ablation, fig:field, fig:latency, tab:runtime.

- **Dangling \ref to non-existent labels:**
  - `conclusion`: "see Table~\ref{tab:latency}, Table~\ref{tab:runtime}" — **`tab:latency` does not exist** in the manifest. The latency table is `tab:runtime` (and fig:latency). Remove `tab:latency`.
- **Manifest labels never referenced anywhere in the drafts:**
  - **`fig:data`** — the data-collection pipeline figure. `data` references `fig:dataset` twice but never `fig:data` (it describes "Figure~\ref{fig:data} shows the collection pipeline" — actually it DOES: `data` says "Figure~\ref{fig:data} shows the collection pipeline end to end"). ✅ referenced.
  - **`tab:oppoint`** — operating-point table (III-D). Only named in the `runtime` *summary*; must be `\ref`'d in the actual runtime body. Verify.
  - **`fig:e2e`** — end-to-end figure. Only in `runtime` summary. Verify it is `\ref`'d in the real body; otherwise unreferenced.
  - **`fig:runtime`** — same; only in summary. Verify.
  - All others (fig:system, tab:dataset, fig:dataset, fig:feat, tab:features, fig:tcn, tab:hparams, tab:detection, fig:confusion, fig:rocpr, fig:loro, fig:perleg, tab:perleg, fig:reliability, tab:ablation, fig:ablation, fig:field, fig:latency, tab:runtime) are referenced in the visible bodies. ✅
- **`abstract`** references no figures (correct for IEEE abstract). OK.

**Action items:** (a) delete all leaked drafting preambles/meta-text and the duplicate scratch abstract; (b) replace `tab:latency` with `tab:runtime` in `conclusion`; (c) supply the real `runtime` LaTeX body and confirm fig:runtime, fig:e2e, tab:oppoint are actually `\ref`'d; (d) remove the unverifiable "0.851" v1 aggregate F1 in `results`; (e) fix the "rear-right … intensity" mischaracterization in `results` LORO paragraph; (f) reconcile the 0.807 vs 0.832 "shipped LORO" ambiguity.