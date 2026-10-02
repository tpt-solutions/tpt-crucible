//! Minimal 2D SVG swarm-topology map — the stepping stone before the wgpu
//! 3D view tracked in todo.md.
//!
//! Layout math lives in pure functions over [`TelemetryRow`]s so it stays
//! host-testable (`cargo test -p tpt-crucible-observer-web`) exactly like
//! [`crate::frames`]; only [`TopologyMap`] touches the DOM.

use leptos::prelude::*;

use crate::frames::TelemetryRow;

/// One node's slot on the grid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placed {
    /// Node id as reported by telemetry (`"esp32s3-07"`).
    pub node_id: String,
    /// Hardware family driving the color class.
    pub hw_type: String,
    /// Grid column, zero-based, left-to-right.
    pub col: usize,
    /// Grid row, zero-based, top-to-bottom.
    pub row: usize,
}

/// Cell width/height in SVG units.
pub const CELL_W: f64 = 110.0;
pub const CELL_H: f64 = 84.0;

/// Latest-position-per-node grid layout.
///
/// * dedupes to each node's **newest** sample (rows arrive newest-first,
///   so the first sighting wins),
/// * groups nodes by hardware family (stable alphabetical), then node id,
/// * flows the result into a near-square grid (`cols = ceil(sqrt n)`).
pub fn layout(rows: &[TelemetryRow]) -> Vec<Placed> {
    // Newest-first dedupe.
    let mut seen: Vec<TelemetryRow> = Vec::new();
    for row in rows {
        if !seen.iter().any(|r| r.node_id == row.node_id) {
            seen.push(row.clone());
        }
    }
    seen.sort_by(|a, b| a.hw_type.cmp(&b.hw_type).then(a.node_id.cmp(&b.node_id)));

    let cols = (seen.len() as f64).sqrt().ceil().max(1.0) as usize;
    seen.iter()
        .enumerate()
        .map(|(i, r)| Placed {
            node_id: r.node_id.clone(),
            hw_type: r.hw_type.clone(),
            col: i % cols,
            row: i / cols,
        })
        .collect()
}

/// Natural sort key stub removed — plain byte order keeps the pure fn
/// deterministic; numeric-aware ordering can land with the wgpu view.
/// SVG canvas size for a set of placements (viewBox dimensions).
pub fn canvas_size(placed: &[Placed]) -> (f64, f64) {
    let max_col = placed.iter().map(|p| p.col).max().unwrap_or(0);
    let max_row = placed.iter().map(|p| p.row).max().unwrap_or(0);
    ((max_col + 1) as f64 * CELL_W, (max_row + 1) as f64 * CELL_H)
}

/// CSS class suffix for a hardware family (kept in sync with style.css).
pub fn family_class(hw_type: &str) -> &'static str {
    match hw_type {
        "alloy" => "alloy",
        "fusion" => "fusion",
        "element" => "element",
        _ => "other",
    }
}

/// The reactive SVG map: one labeled hex-ish disc per live node.
#[component]
pub fn TopologyMap(rows: impl Fn() -> Vec<TelemetryRow> + Send + Sync + 'static) -> impl IntoView {
    // A `Memo` is `Copy`, so both reactive consumers can own it.
    let placed = Memo::new(move |_| layout(&rows()));
    let size = move || canvas_size(&placed.get());

    let nodes = move || {
        placed
            .get()
            .iter()
            .map(|p| {
                let cx = p.col as f64 * CELL_W + CELL_W / 2.0;
                let cy = p.row as f64 * CELL_H + CELL_H / 2.0;
                let label_y = cy + 3.0;
                let family_y = cy - 26.0;
                let cls = format!("topo-node {}", family_class(&p.hw_type));
                view! {
                    <g class={cls}>
                        <circle cx=cx cy=cy r=20 />
                        <text class="topo-label" x=cx y=label_y>{p.node_id.clone()}</text>
                        <text class="topo-family" x=cx y=family_y>{p.hw_type.to_uppercase()}</text>
                    </g>
                }
            })
            .collect::<Vec<_>>()
    };

    view! {
        <svg
            class="topo-svg"
            viewBox=move || format!("0 0 {} {}", size().0, size().1)
            preserveAspectRatio="xMidYMid meet"
        >
            {nodes}
        </svg>
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn row(node: &str, hw: &str, ts: u64) -> TelemetryRow {
        TelemetryRow {
            node_id: node.into(),
            hw_type: hw.into(),
            tokens_per_sec: None,
            mem_bandwidth_gbps: None,
            thermal_drift_c: None,
            latency_ms: None,
            timestamp_ms: ts,
        }
    }

    #[test]
    fn empty_rows_render_an_empty_canvas() {
        assert!(layout(&[]).is_empty());
        let (w, h) = canvas_size(&[]);
        assert!((w - CELL_W).abs() < 1e-9);
        assert!((h - CELL_H).abs() < 1e-9);
    }

    #[test]
    fn duplicates_keep_the_newest_sample() {
        let placed = layout(&[row("n1", "alloy", 42), row("n1", "alloy", 7)]);
        assert_eq!(placed.len(), 1);
        // Position exists either way; the dedupe count is the contract.
        assert_eq!(placed[0].col, 0);
        assert_eq!(placed[0].row, 0);
    }

    #[test]
    fn families_group_before_ids_and_grid_is_near_square() {
        let placed = layout(&[
            row("swarm-b", "alloy", 1),
            row("array-2", "element", 1),
            row("swarm-a", "alloy", 1),
            row("board-1", "fusion", 1),
            row("board-2", "fusion", 1),
        ]);
        // Five items → ceil(sqrt 5) = 3 columns, two rows.
        let key = |p: &Placed| (p.row, p.col);
        let mut sorted = placed.clone();
        sorted.sort_by_key(|p| key(p));
        let ids: Vec<&str> = sorted.iter().map(|p| p.node_id.as_str()).collect();
        // alloy < element < fusion (family-major), ids alphabetical inside.
        assert_eq!(
            ids,
            vec!["swarm-a", "swarm-b", "array-2", "board-1", "board-2"]
        );
        // Five items in three columns: the last one wraps to row 1, col 1.
        assert_eq!(sorted[4].row, 1);
        assert_eq!(sorted[4].col, 1);
        let (w, h) = canvas_size(&placed);
        assert!((w - 3.0 * CELL_W).abs() < 1e-9);
        assert!((h - 2.0 * CELL_H).abs() < 1e-9);
    }

    #[test]
    fn family_classes_cover_known_families() {
        assert_eq!(family_class("alloy"), "alloy");
        assert_eq!(family_class("fusion"), "fusion");
        assert_eq!(family_class("element"), "element");
        assert_eq!(family_class("mystery"), "other");
    }
}
