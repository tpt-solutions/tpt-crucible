//! Dead-node bypass: re-partitioning a swarm onto its survivors.
//!
//! The pipeline is: [`crate::heartbeat::FailureDetector`] flags nodes silent
//! past the timeout → [`bypass_dead_nodes`] prunes them from the topology
//! (re-slicing the latency/bandwidth matrices), re-runs the planner on the
//! survivors, and reports which IR nodes were displaced so the coordinator
//! knows what must be re-fetched. Inference degrades onto fewer nodes instead
//! of stalling on a shard whose host vanished (`spec2.txt`: "Fault-Tolerant
//! Inference").
//!
//! Survivors keep their original [`crate::topology::SwarmNode::id`];
//! [`Recovery::survivor_map`] maps new plan indices back to those ids so
//! firmware flashing and telemetry stay keyed by stable node identity.

use tpt_crucible_common::error::{Error, Result};
use tpt_crucible_common::{Graph, NodeId};

use crate::partition::{self, PartitionOptions, PartitionPlan};
use crate::topology::{SwarmNode, Topology};

/// Outcome of bypassing dead swarm nodes.
#[derive(Debug, Clone)]
pub struct Recovery {
    /// Original topology indices taken out of service (sorted, deduped).
    pub bypassed: Vec<usize>,
    /// New-plan index → original topology id, for every survivor.
    ///
    /// `survivor_map[k]` is the old [`crate::topology::SwarmNode::id`] that
    /// new shard index `k` now lives on.
    pub survivor_map: Vec<usize>,
    /// IR nodes that used to be hosted on a bypassed node.
    pub displaced: Vec<NodeId>,
    /// Fresh assignment over the surviving fleet.
    pub plan: PartitionPlan,
}

impl Recovery {
    /// True when no IR node was hosted on a bypassed node (the failure cost
    /// the fleet nothing but spare capacity).
    pub fn lossless(&self) -> bool {
        self.displaced.is_empty()
    }
}

/// Re-plan `graph` over `topology` minus `dead`.
///
/// `dead` holds original topology ids; duplicates and out-of-range entries
/// are tolerated (deduped / ignored) so callers can pass raw detector output
/// mixed with manual overrides.
///
/// # Errors
/// * [`Error::InvalidArgument`] when every node is dead,
/// * whatever [`crate::partition::partition`] raises on the pruned fleet
///   (e.g. [`Error::OutOfMemory`] when the survivors cannot hold the model).
pub fn bypass_dead_nodes(
    graph: &Graph,
    topology: &Topology,
    options: &PartitionOptions,
    dead: &[usize],
) -> Result<Recovery> {
    let n = topology.nodes.len();
    let mut bypassed: Vec<usize> = dead.iter().copied().filter(|&d| d < n).collect();
    bypassed.sort_unstable();
    bypassed.dedup();

    if bypassed.len() >= n {
        return Err(Error::InvalidArgument(
            "cannot bypass every node; no survivors left to plan onto".into(),
        ));
    }

    // Prune the node list, keeping each survivor's original id.
    let mut survivor_map = Vec::with_capacity(n - bypassed.len());
    let mut live: Vec<&SwarmNode> = Vec::with_capacity(n - bypassed.len());
    for node in &topology.nodes {
        if !bypassed.contains(&node.id) {
            survivor_map.push(node.id);
            live.push(node);
        }
    }

    // Re-slice the latency/bandwidth matrices over the survivors. The
    // sub-matrices keep row/col order aligned with `live`.
    let k = live.len();
    let mut latency = vec![vec![0.0; k]; k];
    let mut bandwidth = vec![vec![0.0; k]; k];
    for (i, &oi) in survivor_map.iter().enumerate() {
        for (j, &oj) in survivor_map.iter().enumerate() {
            latency[i][j] = topology.latency_ms[oi][oj];
            bandwidth[i][j] = topology.bandwidth[oi][oj];
        }
    }
    let live_topo = Topology::from_parts(live.into_iter().cloned().collect(), latency, bandwidth)?;

    // What did the dead nodes host on the *original* placement?
    let original = partition::partition(graph, topology, options)?;
    let mut displaced: Vec<NodeId> = Vec::new();
    for shard in &original.shards {
        if bypassed.contains(&shard.swarm_node) {
            displaced.extend_from_slice(&shard.graph_nodes);
        }
    }

    // Fresh assignment over the surviving fleet.
    let plan = partition::partition(graph, &live_topo, options)?;

    Ok(Recovery {
        bypassed,
        survivor_map,
        displaced,
        plan,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tpt_crucible_common::ops::SoftmaxAttrs;
    use tpt_crucible_common::{DType, Op, Tensor, TensorDesc};

    const MIB: u64 = 1 << 20;

    /// x → matmul(w) → softmax → out; f16 weights of `rows x rows`.
    fn mlp_graph(rows: usize) -> Graph {
        let mut g = Graph::new("mlp");
        let x = g.push(
            "",
            Op::Input(TensorDesc::new(vec![1, rows], DType::F32)),
            Vec::<NodeId>::new(),
        );
        let w = g.push(
            "",
            Op::Constant {
                tensor: Tensor::zeros(TensorDesc::new(vec![rows, rows], DType::F16)),
            },
            Vec::<NodeId>::new(),
        );
        let mm = g.push("", Op::MatMul, vec![x, w]);
        let sm = g.push(
            "",
            Op::Softmax {
                attrs: SoftmaxAttrs { axis: -1 },
            },
            vec![mm],
        );
        let out = g.push("", Op::Output { name: "out".into() }, vec![sm]);
        g.mark_output(out);
        g
    }

    fn fleet(n: usize, mem: u64) -> Topology {
        Topology::homogeneous(n, "esp32s3", mem)
    }

    /// x → matmul(w1) → softmax → matmul(w2) → out; two ~0.78 MiB f16
    /// weights so a 1 MiB/node cap forces a two-shard placement.
    fn two_mm_graph() -> Graph {
        let mut g = Graph::new("two-mm");
        let x = g.push(
            "",
            Op::Input(TensorDesc::new(vec![1, 640], DType::F32)),
            Vec::<NodeId>::new(),
        );
        let w1 = g.push(
            "",
            Op::Constant {
                tensor: Tensor::zeros(TensorDesc::new(vec![640, 640], DType::F16)),
            },
            Vec::<NodeId>::new(),
        );
        let mm1 = g.push("", Op::MatMul, vec![x, w1]);
        let sm = g.push(
            "",
            Op::Softmax {
                attrs: SoftmaxAttrs { axis: -1 },
            },
            vec![mm1],
        );
        let w2 = g.push(
            "",
            Op::Constant {
                tensor: Tensor::zeros(TensorDesc::new(vec![640, 640], DType::F16)),
            },
            Vec::<NodeId>::new(),
        );
        let mm2 = g.push("", Op::MatMul, vec![sm, w2]);
        let out = g.push("", Op::Output { name: "out".into() }, vec![mm2]);
        g.mark_output(out);
        g
    }

    #[test]
    fn bypass_replans_onto_survivors() {
        // Two ~0.78 MiB weight blocks under a 1 MiB/node cap spill across
        // two nodes; killing either host forces a fresh plan over survivors.
        let g = two_mm_graph();
        let topo = fleet(4, 4 * MIB);
        let opts = PartitionOptions {
            max_weight_bytes_per_node: Some(MIB),
            ..Default::default()
        };

        let original = partition::partition(&g, &topo, &opts).unwrap();
        assert_eq!(original.shards.len(), 2);
        let dead_shard = original.shards[0].swarm_node;

        let rec = bypass_dead_nodes(&g, &topo, &opts, &[dead_shard]).unwrap();
        assert_eq!(rec.bypassed, vec![dead_shard]);
        assert_eq!(rec.survivor_map.len(), 3);
        assert!(!rec.displaced.is_empty(), "shard host died: nodes moved");

        // The recovered plan only references survivors (new shard indices map
        // back to live original ids through survivor_map).
        for new_idx in rec.plan.assignments.values() {
            let old_id = rec.survivor_map[*new_idx];
            assert!(
                !rec.bypassed.contains(&old_id),
                "plan routed to bypassed node {old_id}"
            );
        }
    }

    #[test]
    fn lossless_bypass_when_dead_nodes_were_idle() {
        // A 4-node fleet with a model that fits on one node: killing any node
        // that hosts nothing displaces nothing.
        let g = mlp_graph(64); // 8 KiB weights
        let topo = fleet(4, MIB);
        let opts = PartitionOptions::default();

        let original = partition::partition(&g, &topo, &opts).unwrap();
        let busy = original.shards[0].swarm_node;
        let idle: usize = (0..4).find(|&n| n != busy).unwrap();

        let rec = bypass_dead_nodes(&g, &topo, &opts, &[idle]).unwrap();
        assert!(rec.lossless());
        assert!(rec.displaced.is_empty());
        // Same single-shard placement, now indexed over three survivors.
        assert_eq!(rec.plan.shards.len(), 1);
    }

    #[test]
    fn detector_output_feeds_bypass_directly() {
        use crate::heartbeat::{FailureDetector, HeartbeatMsg, NodeStatus};
        use std::time::{Duration, Instant};

        let g = mlp_graph(64);
        let topo = fleet(4, MIB);
        let opts = PartitionOptions::default();

        let mut det = FailureDetector::new(Duration::from_millis(100));
        let t0 = Instant::now();
        let beat = |id: u32| HeartbeatMsg {
            node_id: id,
            seq: 1,
            timestamp_ms: 0,
            status: NodeStatus::Healthy,
        };
        // Nodes 0-2 check in at t0; 0 and 1 refresh at t0+150ms.
        for id in 0u32..3 {
            det.record(&beat(id), t0);
        }
        let mid = t0 + Duration::from_millis(150);
        det.record(&beat(0), mid);
        det.record(&beat(1), mid);

        // At t0+200ms: 0 and 1 are fresh; 2 timed out; 3 was never seen.
        let now = t0 + Duration::from_millis(200);
        let dead: Vec<usize> = (0usize..4)
            .filter(|&n| !det.is_alive(n as u32, now))
            .collect();
        assert_eq!(dead, vec![2, 3]);

        let rec = bypass_dead_nodes(&g, &topo, &opts, &dead).unwrap();
        assert_eq!(rec.bypassed, vec![2, 3]);
        assert_eq!(rec.survivor_map, vec![0, 1]);
        for new_idx in rec.plan.assignments.values() {
            assert!(*new_idx < 2, "plan must stay on the live pair");
        }
    }

    #[test]
    fn out_of_range_and_duplicate_dead_ids_tolerated() {
        let g = mlp_graph(64);
        let topo = fleet(2, MIB);
        let rec = bypass_dead_nodes(&g, &topo, &PartitionOptions::default(), &[1, 1, 99]).unwrap();
        assert_eq!(rec.bypassed, vec![1]);
        assert_eq!(rec.survivor_map.len(), 1);
    }

    #[test]
    fn killing_every_node_is_rejected() {
        let g = mlp_graph(64);
        let topo = fleet(2, MIB);
        let err = bypass_dead_nodes(&g, &topo, &PartitionOptions::default(), &[0, 1]).unwrap_err();
        assert!(err.to_string().contains("no survivors"), "{err}");
    }
}
