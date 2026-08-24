//! ComputationalGraph → TPT-UIR ingestion adapter for `tpt-crucible`.
//!
//! This implements the Phase 3 ingestion adapter described in
//! `tpt-uir/todo.md` (the `⛔ OUT OF REPO` Crucible tasks). It converts a
//! `tpt-catalyst` `ComputationalGraph` into the unified TPT-UIR
//! (`tpt_uir_core::Region`) using the Crucible dialect, emitting all tensor
//! shapes as `Dimension::Fixed` (never `Symbolic`/`Bounded`) so the result
//! satisfies the Crucible dialect invariant. A reverse converter reconstructs a
//! `ComputationalGraph` for lossless round-tripping.
//!
//! # Structure
//!
//! The whole graph is wrapped as a single [`Region`] containing exactly one
//! [`Block`]. A pure computational graph has no block-level inputs, so the
//! block has no arguments; every graph node becomes an `Operation` in that
//! block. Nodes are emitted in topological order so that each operation's
//! operands (the `from` node ids of incoming edges) are already defined by a
//! preceding operation, keeping the TPT-UIR SSA-valid.
//!
//! # Losslessness
//!
//! The minimal TPT-UIR core does not model every `ComputationalGraph` detail
//! (node `op_type`, `name`, free-form attributes, edge `tensor_name`). Those
//! are preserved as attributes so the reverse pass rebuilds the exact graph:
//!
//! * Original `op_type`/`name`/`id` are stored under [`ATTR_OP_TYPE`] /
//!   [`ATTR_NODE_NAME`] / [`ATTR_NODE_ID`].
//! * Free-form node attributes are JSON-serialized under [`ATTR_SOURCE_ATTRS`].
//! * Per-node incoming edges (including `tensor_name`, which the SSA operand
//!   list alone cannot capture) are JSON-serialized under [`ATTR_INCOMING`].
//!
//! Every emitted operation uses a `tpt_crucible.*` op name. Known Crucible
//! ops (`map_flash`, `route_fpga`, `analog_conv`) map to the dialect constants;
//! any other `op_type` is namespaced under `tpt_crucible` so the whole region
//! stays within the Crucible dialect.

use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use tpt_catalyst::ir::{ComputationalGraph, Edge, OpNode};
use tpt_uir_core::attr::{Attribute, AttributeValue};
use tpt_uir_core::op_name::OpName;
use tpt_uir_core::types::{Dimension, ShapeSpec};
use tpt_uir_core::{Block, Operation, OpId, Region, ValueId};
use tpt_uir_dialects::crucible::{
    CrucibleOp, TPT_CRUCIBLE_ANALOG_CONV, TPT_CRUCIBLE_MAP_FLASH, TPT_CRUCIBLE_ROUTE_FPGA,
};
use tpt_uir_dialects::{CrucibleDialect, ValidateDialect};

/// Attribute key holding the original graph node `op_type`.
pub const ATTR_OP_TYPE: &str = "crucible.op_type";
/// Attribute key holding the original graph node `name`.
pub const ATTR_NODE_NAME: &str = "crucible.node_name";
/// Attribute key holding the original graph node `id` (as `i64`).
pub const ATTR_NODE_ID: &str = "crucible.node_id";
/// Attribute key holding the original node attributes, JSON-serialized.
pub const ATTR_SOURCE_ATTRS: &str = "crucible.source_attrs";
/// Attribute key holding the original node's incoming edges, JSON-serialized.
pub const ATTR_INCOMING: &str = "crucible.incoming_edges";

/// One incoming edge of a node, as preserved for the reverse conversion.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct Incoming {
    from: usize,
    tensor_name: String,
}

#[derive(Debug, thiserror::Error)]
pub enum AdapterError {
    #[error("crucible dialect validation failed: {0:?}")]
    CrucibleValidation(Vec<String>),
    #[error("graph is not a DAG (cycle detected)")]
    Cycle,
    #[error("TPT-UIR serialization failed: {0}")]
    Serde(String),
}

// ---------------------------------------------------------------------------
// File I/O (.tptuir emission / consumption)
// ---------------------------------------------------------------------------

/// Convert a `ComputationalGraph` to TPT-UIR and write it to a `.tptuir` file.
pub fn write_tptuir(graph: &ComputationalGraph, path: &Path) -> Result<(), AdapterError> {
    let uir = from_crucible(graph)?;
    tpt_uir_serde::write_tptuir(path, &uir)
        .map_err(|e| AdapterError::Serde(format!("write_tptuir({}): {e}", path.display())))
}

/// Read a `.tptuir` file, deserialize it to TPT-UIR, and lower it back to a
/// `ComputationalGraph` (the reverse of [`write_tptuir`]).
pub fn read_tptuir(path: &Path) -> Result<ComputationalGraph, AdapterError> {
    let uir = tpt_uir_serde::read_tptuir(path)
        .map_err(|e| AdapterError::Serde(format!("read_tptuir({}): {e}", path.display())))?;
    Ok(to_crucible(&uir))
}

// ---------------------------------------------------------------------------
// ComputationalGraph -> TPT-UIR
// ---------------------------------------------------------------------------

/// Convert a `ComputationalGraph` into a TPT-UIR `Region` using the Crucible
/// dialect.
///
/// The returned region has exactly one block; every operation uses a
/// `tpt_crucible.*` op name and every emitted shape dimension is `Fixed`. The
/// region is guaranteed to pass both [`tpt_uir_core::validate_region`] and
/// [`CrucibleDialect::validate`].
pub fn from_crucible(graph: &ComputationalGraph) -> Result<Region, AdapterError> {
    let order = topo_order(graph).ok_or(AdapterError::Cycle)?;

    let mut operations = Vec::with_capacity(order.len());
    for &node_id in &order {
        let node = graph
            .nodes
            .iter()
            .find(|n| n.id == node_id)
            .expect("topo_order yields only known node ids");

        let op_name = op_name_for(&node.op_type);
        let (operands, incoming) = incoming_for(graph, node.id);
        let results = vec![node.id as ValueId];

        let mut attributes = vec![
            Attribute::string(ATTR_OP_TYPE, node.op_type.clone()),
            Attribute::string(ATTR_NODE_NAME, node.name.clone()),
            Attribute::i64(ATTR_NODE_ID, node.id as i64),
            Attribute::string(
                ATTR_SOURCE_ATTRS,
                serde_json::to_string(&node.attributes).unwrap_or_else(|_| "{}".to_string()),
            ),
            Attribute::string(
                ATTR_INCOMING,
                serde_json::to_string(&incoming).unwrap_or_else(|_| "[]".to_string()),
            ),
        ];

        // Emit any tensor shape as `Dimension::Fixed` only (Crucible invariant).
        if let Some(shape) = extract_fixed_shape(&node.attributes) {
            attributes.push(Attribute::shape("shape", shape));
        }

        let op = CrucibleOp::build(
            node.id as OpId,
            op_name,
            operands,
            results,
            attributes,
            vec![],
        );
        operations.push(op);
    }

    let region = Region {
        blocks: vec![Block {
            arguments: vec![],
            operations,
        }],
    };

    CrucibleDialect::validate(&region).map_err(AdapterError::CrucibleValidation)?;
    Ok(region)
}

/// Map a graph `op_type` to a `tpt_crucible.*` op name.
fn op_name_for(op_type: &str) -> OpName {
    match op_type.to_ascii_lowercase().as_str() {
        "map_flash" | "flash" => OpName::parse(TPT_CRUCIBLE_MAP_FLASH).unwrap(),
        "route_fpga" | "fpga" => OpName::parse(TPT_CRUCIBLE_ROUTE_FPGA).unwrap(),
        "analog_conv" | "analog" => OpName::parse(TPT_CRUCIBLE_ANALOG_CONV).unwrap(),
        other => OpName::new("tpt_crucible", other),
    }
}

/// Collect, in `graph.edges` order, the incoming edges of `node_id` and the
/// corresponding operand value ids (`from` node ids).
fn incoming_for(graph: &ComputationalGraph, node_id: usize) -> (Vec<ValueId>, Vec<Incoming>) {
    let mut incoming = Vec::new();
    for e in &graph.edges {
        if e.to == node_id {
            incoming.push(Incoming {
                from: e.from,
                tensor_name: e.tensor_name.clone(),
            });
        }
    }
    let operands = incoming.iter().map(|i| i.from as ValueId).collect();
    (operands, incoming)
}

/// Extract a tensor shape from common node-attribute keys, emitting every
/// dimension as `Dimension::Fixed`. Returns `None` when no usable shape is
/// present.
fn extract_fixed_shape(attrs: &HashMap<String, serde_json::Value>) -> Option<ShapeSpec> {
    const KEYS: [&str; 4] = ["shape", "output_shape", "tensor_shape", "dims"];
    for key in KEYS {
        if let Some(value) = attrs.get(key) {
            if let Some(arr) = value.as_array() {
                let mut dimensions = Vec::with_capacity(arr.len());
                let mut ok = true;
                for elem in arr {
                    match elem.as_u64() {
                        Some(n) => dimensions.push(Dimension::Fixed(n as usize)),
                        None => match elem.as_i64() {
                            Some(n) if n >= 0 => dimensions.push(Dimension::Fixed(n as usize)),
                            _ => {
                                ok = false;
                                break;
                            }
                        },
                    }
                }
                if ok && !dimensions.is_empty() {
                    return Some(ShapeSpec { dimensions });
                }
            }
        }
    }
    None
}

/// Topologically order node ids by edge direction (`from` precedes `to`).
/// Returns `None` if the graph contains a cycle.
fn topo_order(graph: &ComputationalGraph) -> Option<Vec<usize>> {
    let ids: Vec<usize> = graph.nodes.iter().map(|n| n.id).collect();
    let mut indegree: HashMap<usize, usize> = ids.iter().map(|&id| (id, 0)).collect();
    for e in &graph.edges {
        if indegree.contains_key(&e.to) {
            *indegree.get_mut(&e.to).unwrap() += 1;
        }
    }

    // FIFO queue seeded in declaration order. When the declaration order is
    // already a valid topological order (the common case for computational
    // graphs), this preserves it exactly, so the reverse conversion reproduces
    // the original node ordering.
    let mut queue: std::collections::VecDeque<usize> =
        ids.iter().copied().filter(|id| indegree[id] == 0).collect();
    let mut order = Vec::with_capacity(ids.len());
    while let Some(id) = queue.pop_front() {
        order.push(id);
        for e in &graph.edges {
            if e.from == id && indegree.contains_key(&e.to) {
                let deg = indegree.get_mut(&e.to).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    queue.push_back(e.to);
                }
            }
        }
    }

    if order.len() == ids.len() {
        Some(order)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// TPT-UIR -> ComputationalGraph
// ---------------------------------------------------------------------------

/// Reconstruct a `ComputationalGraph` from a TPT-UIR `Region` produced by
/// [`from_crucible`].
pub fn to_crucible(region: &Region) -> ComputationalGraph {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();

    let block = region
        .blocks
        .first()
        .expect("crucible region always has one block");

    for op in &block.operations {
        let op_type = attr_string(op, ATTR_OP_TYPE).unwrap_or("custom").to_string();
        let name = attr_string(op, ATTR_NODE_NAME).unwrap_or("").to_string();
        let id = attr_i64(op, ATTR_NODE_ID).unwrap_or(op.id as i64) as usize;

        let mut attributes: HashMap<String, serde_json::Value> = attr_string(op, ATTR_SOURCE_ATTRS)
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_default();

        // Re-emit any original shape under a node attribute so it survives the
        // round-trip even if the source graph tracked it outside `source_attrs`.
        if let Some(AttributeValue::Shape(shape)) = attr_value(op, "shape") {
            attributes
                .entry("shape".to_string())
                .or_insert_with(|| serde_json::to_value(shape_dimensions(shape)).unwrap());
        }

        let incoming: Vec<Incoming> = attr_string(op, ATTR_INCOMING)
            .and_then(|s| serde_json::from_str(s).ok())
            .unwrap_or_default();
        for in_edge in &incoming {
            edges.push(Edge {
                from: in_edge.from,
                to: id,
                tensor_name: in_edge.tensor_name.clone(),
            });
        }

        nodes.push(OpNode {
            id,
            op_type,
            name,
            attributes,
        });
    }

    ComputationalGraph { nodes, edges }
}

/// Flatten a `ShapeSpec` into a JSON array of fixed dimension sizes.
fn shape_dimensions(shape: &ShapeSpec) -> Vec<u64> {
    shape
        .dimensions
        .iter()
        .filter_map(|d| match d {
            Dimension::Fixed(n) => Some(*n as u64),
            _ => None,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Attribute helpers
// ---------------------------------------------------------------------------

fn attr_string<'a>(op: &'a Operation, key: &str) -> Option<&'a str> {
    op.attributes.iter().find(|a| a.key == key).and_then(|a| {
        if let AttributeValue::String(s) = &a.value {
            Some(s.as_str())
        } else {
            None
        }
    })
}

fn attr_i64(op: &Operation, key: &str) -> Option<i64> {
    op.attributes.iter().find(|a| a.key == key).and_then(|a| {
        if let AttributeValue::I64(v) = &a.value {
            Some(*v)
        } else {
            None
        }
    })
}

fn attr_value<'a>(op: &'a Operation, key: &str) -> Option<&'a AttributeValue> {
    op.attributes
        .iter()
        .find(|a| a.key == key)
        .map(|a| &a.value)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use tpt_catalyst::ir::ComputationalGraph;
    use tpt_uir_core::validate_region;

    fn node(id: usize, op_type: &str, shape: Option<&[u64]>) -> OpNode {
        let mut attrs = HashMap::new();
        if let Some(dims) = shape {
            attrs.insert(
                "shape".to_string(),
                serde_json::Value::Array(dims.iter().map(|d| serde_json::json!(*d)).collect()),
            );
        }
        OpNode {
            id,
            op_type: op_type.to_string(),
            name: format!("n{}", id),
            attributes: attrs,
        }
    }

    fn edge(from: usize, to: usize, name: &str) -> Edge {
        Edge {
            from,
            to,
            tensor_name: name.to_string(),
        }
    }

    fn sample_graph() -> ComputationalGraph {
        // in0 (1x4096) -> matmul -> add -> out (4096)
        //               in1(4096x4096) /
        ComputationalGraph {
            nodes: vec![
                node(0, "input", Some(&[1, 4096])),
                node(1, "weight", Some(&[4096, 4096])),
                node(2, "matmul", Some(&[1, 4096])),
                node(3, "add", Some(&[1, 4096])),
                node(4, "map_flash", None),
                node(5, "output", Some(&[1, 4096])),
            ],
            edges: vec![
                edge(0, 2, "a"),
                edge(1, 2, "b"),
                edge(2, 3, "x"),
                edge(0, 3, "y"),
                edge(3, 4, "logits"),
                edge(4, 5, "mapped"),
            ],
        }
    }

    #[test]
    fn test_from_crucible_wraps_single_region_block() {
        let region = from_crucible(&sample_graph()).expect("convert failed");
        assert_eq!(region.blocks.len(), 1, "must be a single region/block");
        let block = &region.blocks[0];
        assert!(block.arguments.is_empty(), "pure graph has no block args");
        assert_eq!(block.operations.len(), 6);
        assert!(
            validate_region(&region).is_ok(),
            "produced TPT-UIR must be SSA-valid"
        );
    }

    #[test]
    fn test_from_crucible_uses_crucible_dialect_only() {
        let region = from_crucible(&sample_graph()).unwrap();
        for op in &region.blocks[0].operations {
            assert_eq!(
                op.op_name.dialect, "tpt_crucible",
                "all ops must be crucible dialect, got {}",
                op.op_name
            );
        }
    }

    #[test]
    fn test_from_crucible_known_ops_map_to_constants() {
        let region = from_crucible(&sample_graph()).unwrap();
        let by_id: HashMap<u32, &Operation> = region
            .blocks[0]
            .operations
            .iter()
            .map(|op| (op.id, op))
            .collect();
        assert_eq!(by_id[&4].op_name.to_string(), TPT_CRUCIBLE_MAP_FLASH);
        assert_eq!(by_id[&2].op_name.to_string(), "tpt_crucible.matmul");
        assert_eq!(by_id[&3].op_name.to_string(), "tpt_crucible.add");
    }

    #[test]
    fn test_from_crucible_emits_fixed_shapes_only() {
        let region = from_crucible(&sample_graph()).unwrap();
        for op in &region.blocks[0].operations {
            for attr in &op.attributes {
                if let AttributeValue::Shape(shape) = &attr.value {
                    for d in &shape.dimensions {
                        assert!(
                            matches!(d, Dimension::Fixed(_)),
                            "crucible shapes must be Fixed, found {:?}",
                            d
                        );
                    }
                }
            }
        }
        // Crucible dialect validation must accept the result.
        assert!(CrucibleDialect::validate(&region).is_ok());
    }

    #[test]
    fn test_roundtrip_lossless() {
        let graph = sample_graph();
        let uir = from_crucible(&graph).unwrap();
        let back = to_crucible(&uir);
        assert_eq!(graph, back, "ComputationalGraph round-trip not lossless");
    }

    #[test]
    fn test_serialization_roundtrip() {
        let graph = sample_graph();
        let uir = from_crucible(&graph).unwrap();
        let bytes = tpt_uir_serde::serialize_region(&uir).expect("serialize failed");
        let back = tpt_uir_serde::deserialize_region(&bytes).expect("deserialize failed");
        assert_eq!(uir, back);
        assert_eq!(to_crucible(&back), graph);
    }

    #[test]
    fn test_topo_order_operands_precede() {
        let region = from_crucible(&sample_graph()).unwrap();
        let block = &region.blocks[0];
        // matmul (id 2) depends on input(0) and weight(1); both must appear
        // earlier in emission order.
        let pos = |id: u32| block.operations.iter().position(|o| o.id == id).unwrap();
        assert!(pos(2) > pos(0));
        assert!(pos(2) > pos(1));
        assert!(pos(3) > pos(2));
        assert!(pos(5) > pos(4));
    }

    #[test]
    fn test_cycle_rejected() {
        let graph = ComputationalGraph {
            nodes: vec![node(0, "a", None), node(1, "b", None)],
            edges: vec![edge(0, 1, "x"), edge(1, 0, "y")],
        };
        assert!(matches!(from_crucible(&graph), Err(AdapterError::Cycle)));
    }

    #[test]
    fn test_file_roundtrip() {
        let graph = sample_graph();
        let path = std::env::temp_dir().join(format!(
            "tptuir_crucible_{}.tptuir",
            std::process::id()
        ));
        write_tptuir(&graph, &path).expect("write_tptuir failed");
        let back = read_tptuir(&path).expect("read_tptuir failed");
        assert_eq!(graph, back);
        let _ = std::fs::remove_file(&path);
    }
}
