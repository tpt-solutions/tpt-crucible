//! TPT-IR to swarm partitioning.
//!
//! v1 planner (documented heuristics):
//!
//! * walks the IR in SSA order (a valid topological order by construction);
//! * keeps consecutive ops on the same node unless a link is cheaper than
//!   re-fetching weights;
//! * charges every constant to the first op that consumes it;
//! * respects a per-node byte budget (explicit cap, or node memory minus the
//!   reserved KV-cache share);
//! * under [`Strategy::HeadParallel`] / [`Strategy::Hybrid`], every
//!   [`Op::Attention`] gets a [`HeadParallelGroup`] that spreads query heads
//!   across the fleet while FFN sublayers stay layer-serial — the
//!   transformer-native split from spec2.txt differentiator #5;
//! * with [`PartitionOptions::fpga_offload`], hybrid nodes' fabrics are read
//!   by the planner: GEMM-class segments ([`Op::MatMul`] / [`Op::Attention`])
//!   prefer landing on a board whose fabric can stage the shard's weights in
//!   block RAM, and such shards record which ops route to the overlay.

use std::collections::BTreeMap;

use petgraph::prelude::*;
use serde::{Deserialize, Serialize};
use tpt_crucible_common::error::{Error, Result};
use tpt_crucible_common::{Graph, NodeId, Op};

use crate::kv_cache::{self, KvCachePlan, KvCacheRequest};
use crate::topology::Topology;

/// Partitioning policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum Strategy {
    /// Pipeline stages in IR order; attention runs whole on one node.
    LayerSerial,
    /// Every attention splits its query heads across all nodes.
    HeadParallel,
    /// Attention-head parallel + FFN layer-serial (the default).
    #[default]
    Hybrid,
}

/// Knobs for [`partition`].
#[derive(Debug, Clone, Default)]
pub struct PartitionOptions {
    /// Policy to apply; defaults to [`Strategy::Hybrid`].
    pub strategy: Option<Strategy>,
    /// Hard cap on weight bytes per node; defaults to node memory.
    pub max_weight_bytes_per_node: Option<u64>,
    /// When set, reserve this much cache per node before partitioning.
    pub kv_request: Option<KvCacheRequest>,
    /// Route GEMM-class ops on hybrid boards through their FPGA fabric.
    ///
    /// When enabled, a segment containing [`Op::MatMul`] or [`Op::Attention`]
    /// prefers opening on a node whose [`crate::topology::FpgaProfile`] has
    /// enough block RAM to stage the segment's weights; such shards report
    /// [`Shard::fpga_offload`] and name the fabric-routed ops. Pure-silicon
    /// execution remains the fallback whenever the fabric cannot fit.
    pub fpga_offload: bool,
}

/// A group of swarm nodes jointly executing one attention op,
/// each holding `head_count` of the op's query/kv heads.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HeadParallelGroup {
    /// The attention op being spread.
    pub graph_node: NodeId,
    /// Head ranges per participating node.
    pub slices: Vec<crate::kv_cache::HeadSlice>,
}

/// Weights and ops landed on one swarm node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Shard {
    /// Swarm node index.
    pub swarm_node: usize,
    /// IR nodes executing here (constants included, SSA order kept).
    pub graph_nodes: Vec<NodeId>,
    /// Sum of constant payload bytes charged to this shard.
    pub weight_bytes: u64,
    /// KV-cache bytes reserved here (0 when no kv request was given).
    pub kv_cache_bytes: u64,
    /// GEMM-class ops of this shard routed through the node's FPGA fabric
    /// (empty unless [`PartitionOptions::fpga_offload`] was set and the
    /// fabric fits the shard's weights in block RAM).
    pub fpga_ops: Vec<NodeId>,
}

impl Shard {
    /// True when part of this shard executes on the node's FPGA fabric.
    pub fn fpga_offload(&self) -> bool {
        !self.fpga_ops.is_empty()
    }
}

/// Coarse quality metrics of a plan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanStats {
    /// Edges whose endpoints sit on different nodes.
    pub cut_edges: u64,
    /// Total weight bytes placed anywhere (replicas double-counted).
    pub weight_bytes_total: u64,
    /// Largest shard weight footprint (capacity-planning headline).
    pub max_shard_weight_bytes: u64,
    /// GEMM-class ops routed through an FPGA fabric (0 unless
    /// [`PartitionOptions::fpga_offload`] was set).
    pub fpga_offload_ops: u64,
}

/// The complete assignment produced by [`partition`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PartitionPlan {
    /// Policy that produced this plan.
    pub strategy: Strategy,
    /// IR node id to swarm node index.
    pub assignments: BTreeMap<u32, usize>,
    /// Per-node shards (only nodes that received work appear).
    pub shards: Vec<Shard>,
    /// Attention ops spread across the fleet (empty for `LayerSerial`).
    pub head_parallel_groups: Vec<HeadParallelGroup>,
    /// KV-cache distribution, when requested.
    pub kv_plan: Option<KvCachePlan>,
    /// Quality metrics.
    pub stats: PlanStats,
}

impl PartitionPlan {
    /// Swarm node executing `id`, if assigned.
    pub fn node_of(&self, id: NodeId) -> Option<usize> {
        self.assignments.get(&id.0).copied()
    }
}

/// Partition `graph` across `topology`.
///
/// # Errors
///
/// * [`Error::InvalidArgument`] on structurally invalid IR,
/// * [`Error::OutOfMemory`] when no node can hold a shard under the budget.
pub fn partition(
    graph: &Graph,
    topology: &Topology,
    options: &PartitionOptions,
) -> Result<PartitionPlan> {
    graph.validate()?;

    // Mirror the IR into petgraph for structural checks and cut stats.
    let mut dg = DiGraph::new();
    let handles: Vec<_> = graph
        .nodes
        .iter()
        .map(|n| dg.add_node(n.op.name()))
        .collect();
    for (i, n) in graph.nodes.iter().enumerate() {
        for &inp in &n.inputs {
            dg.add_edge(handles[inp.0 as usize], handles[i], ());
        }
    }
    if petgraph::algo::toposort(&dg, None).is_err() {
        return Err(Error::InvalidArgument(
            "ir graph contains a cycle; tpt-ir must be a dag".into(),
        ));
    }

    // ---- weight accounting -------------------------------------------------
    // Constants cost their payload; ops inherit the constants they alone feed.
    // Shared constants are charged to their first consumer only.
    let mut consumers = vec![0u32; graph.len()];
    for n in &graph.nodes {
        for &inp in &n.inputs {
            consumers[inp.0 as usize] += 1;
        }
    }
    let const_bytes = |id: NodeId| -> u64 {
        match graph.get_node(id).map(|n| &n.op) {
            Some(Op::Constant { tensor }) => tensor.desc.byte_size() as u64,
            _ => 0,
        }
    };
    let mut op_bytes = vec![0u64; graph.len()];
    for (i, n) in graph.nodes.iter().enumerate() {
        if matches!(n.op, Op::Input(_) | Op::Output { .. }) {
            continue;
        }
        for &inp in &n.inputs {
            if matches!(graph.get_node(inp).unwrap().op, Op::Constant { .. })
                && consumers[inp.0 as usize] == 1
            {
                op_bytes[i] += const_bytes(inp);
            }
        }
    }

    // ---- kv reservation ----------------------------------------------------
    let memories: Vec<u64> = topology.nodes.iter().map(|n| n.memory_bytes).collect();
    let kv_plan = match options.kv_request {
        Some(ref req) => Some(kv_cache::plan(req, &memories)?),
        None => None,
    };
    let kv_reserved = |node: usize| -> u64 { kv_plan.as_ref().map_or(0, |p| p.bytes_on(node)) };

    let strategy = options.strategy.unwrap_or_default();
    let budget_for = |node: usize| -> u64 {
        options
            .max_weight_bytes_per_node
            .unwrap_or_else(|| topology.nodes[node].memory_bytes)
            .saturating_sub(kv_reserved(node))
    };

    // ---- greedy layer-serial walk ------------------------------------------
    #[derive(Default)]
    struct ShardAcc {
        /// Topology index this shard lives on.
        swarm_node: usize,
        bytes: u64,
        nodes: Vec<NodeId>,
        /// Ops of this shard routed through the node's FPGA fabric.
        fpga_ops: Vec<NodeId>,
    }
    let mut shards: Vec<ShardAcc> = Vec::new();
    let mut assignment: Vec<Option<usize>> = vec![None; graph.len()];
    let mut current: Option<usize> = None;

    // GEMM-class ops are the ones an FPGA overlay accelerates.
    let is_gemm_class = |op: &Op| matches!(op, Op::MatMul | Op::Attention { .. });
    let mut current_fabric = false;

    for (i, n) in graph.nodes.iter().enumerate() {
        if matches!(n.op, Op::Constant { .. }) {
            continue; // constants ride along with their consumer
        }
        let b = op_bytes[i];
        let stay_feasible = match current {
            Some(pos) => {
                let sn = shards[pos].swarm_node;
                shards[pos].bytes + b <= budget_for(sn)
            }
            None => false,
        };
        let pos = if stay_feasible {
            current.unwrap()
        } else {
            // Decide FPGA routing for the *segment* about to open. Its first
            // op is often an Input (zero bytes, non-GEMM), so scan forward
            // through the ops that will share this shard (up to the largest
            // node budget) looking for GEMM-class work; the fabric must be
            // able to stage the weights accumulated through that op.
            let global_max_budget = (0..topology.nodes.len())
                .map(&budget_for)
                .max()
                .unwrap_or(0);
            let mut gemm_segment = false;
            let mut segment_bram = 0u64;
            if options.fpga_offload {
                let mut acc = 0u64;
                for (idx, n2) in graph.nodes.iter().enumerate().skip(i) {
                    if matches!(
                        n2.op,
                        Op::Constant { .. } | Op::Input(_) | Op::Output { .. }
                    ) {
                        continue;
                    }
                    acc += op_bytes[idx];
                    if is_gemm_class(&n2.op) {
                        segment_bram = acc;
                        gemm_segment = true;
                        break;
                    }
                    if acc > global_max_budget {
                        break;
                    }
                }
            }

            // Open a shard on the best feasible node. With FPGA offload
            // enabled and a GEMM-class segment, hybrid boards whose fabric
            // can stage the segment's weights in block RAM sort ahead of
            // plain silicon; ties break by most remaining headroom.
            let mut best: Option<(usize, u64, bool)> = None; // (node, room, fabric)
            for s in 0..topology.nodes.len() {
                let used = shards
                    .iter()
                    .filter(|acc| acc.swarm_node == s)
                    .map(|acc| acc.bytes)
                    .sum::<u64>();
                let room = budget_for(s).saturating_sub(used);
                if room < b {
                    continue;
                }
                let fabric = gemm_segment
                    && segment_bram > 0
                    && topology.nodes[s]
                        .fpga
                        .is_some_and(|f| f.block_ram_bytes >= segment_bram);
                let better = match best {
                    None => true,
                    Some((_, r, f)) => (fabric && !f) || (fabric == f && room > r),
                };
                if better {
                    best = Some((s, room, fabric));
                }
            }
            let (s, _, fabric) = best.ok_or_else(|| Error::OutOfMemory {
                node: "any swarm node".into(),
                needed: b,
                available: memories.iter().copied().max().unwrap_or(0),
            })?;
            current_fabric = fabric;
            let pos = shards.len();
            shards.push(ShardAcc {
                swarm_node: s,
                bytes: 0,
                nodes: Vec::new(),
                fpga_ops: Vec::new(),
            });
            current = Some(pos);
            pos
        };
        if current_fabric && is_gemm_class(&n.op) {
            shards[pos].fpga_ops.push(NodeId(i as u32));
        }
        shards[pos].bytes += b;
        shards[pos].nodes.push(NodeId(i as u32));
        assignment[i] = Some(pos);
    }

    // Each constant rides with its first assigned consumer; strays join the
    // last open shard.
    for (i, n) in graph.nodes.iter().enumerate() {
        if !matches!(n.op, Op::Constant { .. }) {
            continue;
        }
        let consumers = graph.consumers_of(NodeId(i as u32));
        let target = consumers
            .iter()
            .find_map(|c| assignment[c.0 as usize])
            .or(current);
        let pos = match target {
            Some(p) => p,
            None => {
                shards.push(ShardAcc {
                    swarm_node: 0,
                    bytes: 0,
                    nodes: Vec::new(),
                    fpga_ops: Vec::new(),
                });
                0
            }
        };
        assignment[i] = Some(pos);
        shards[pos].nodes.push(NodeId(i as u32));
    }

    // ---- transformer-native attention split --------------------------------
    // HeadParallel / Hybrid: spread each attention's heads across the fleet.
    // LayerSerial keeps attention whole on its shard.
    let mut head_parallel_groups = Vec::new();
    if matches!(strategy, Strategy::HeadParallel | Strategy::Hybrid) {
        for (i, n) in graph.nodes.iter().enumerate() {
            let (num_heads, num_kv_heads) = match &n.op {
                Op::Attention { attrs } => (attrs.num_heads, attrs.num_kv_heads),
                _ => continue,
            };
            let fleet = topology.nodes.len() as u32;
            let per_node = num_heads.div_ceil(fleet);
            let mut slices = Vec::with_capacity(fleet as usize);
            let mut start = 0u32;
            let mut remaining = num_heads.min(num_kv_heads.max(1));
            for s in 0..fleet {
                if remaining == 0 {
                    break;
                }
                let count = remaining.min(per_node);
                slices.push(crate::kv_cache::HeadSlice {
                    node: s as usize,
                    head_start: start,
                    head_count: count,
                });
                remaining -= count;
                start += count;
            }
            head_parallel_groups.push(HeadParallelGroup {
                graph_node: NodeId(i as u32),
                slices,
            });
        }
    }

    // ---- stats + plan -------------------------------------------------------
    let cut_edges = graph
        .nodes
        .iter()
        .enumerate()
        .flat_map(|(i, n)| n.inputs.iter().map(move |&inp| (i, inp)))
        .filter(|&(i, inp)| assignment[i] != assignment[inp.0 as usize])
        .count() as u64;

    let weight_bytes_total = shards.iter().map(|acc| acc.bytes).sum();
    let max_shard_weight_bytes = shards.iter().map(|acc| acc.bytes).max().unwrap_or(0);
    let fpga_offload_ops = shards.iter().map(|acc| acc.fpga_ops.len() as u64).sum();

    // Map every IR node to its *swarm node id* via its shard.
    let mut assignments = BTreeMap::new();
    for (i, slot) in assignment.iter().enumerate() {
        if let Some(pos) = slot {
            assignments.insert(i as u32, shards[*pos].swarm_node);
        }
    }

    let out_shards = shards
        .into_iter()
        .map(|acc| Shard {
            swarm_node: acc.swarm_node,
            graph_nodes: acc.nodes,
            weight_bytes: acc.bytes,
            kv_cache_bytes: kv_reserved(acc.swarm_node),
            fpga_ops: acc.fpga_ops,
        })
        .collect();

    Ok(PartitionPlan {
        strategy,
        assignments,
        shards: out_shards,
        head_parallel_groups,
        kv_plan,
        stats: PlanStats {
            cut_edges,
            weight_bytes_total,
            max_shard_weight_bytes,
            fpga_offload_ops,
        },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kv_cache::KvCacheRequest;
    use crate::topology::{FpgaProfile, NodeArch, SwarmNode, Topology};
    use tpt_crucible_common::ops::{AttentionAttrs, SoftmaxAttrs};
    use tpt_crucible_common::{DType, Tensor, TensorDesc};

    const MIB: u64 = 1 << 20;

    /// x → matmul(w) → softmax → out; f16 weights of `rows x rows`.
    fn mlp_graph(rows: usize) -> Graph {
        let mut g = Graph::new("mlp");
        let x = g.push(
            "x",
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

    #[test]
    fn small_model_lands_on_one_node() {
        let g = mlp_graph(64); // 8 KiB of f16 weights
        let plan = partition(&g, &fleet(4, 2 * MIB), &PartitionOptions::default()).unwrap();
        assert_eq!(plan.shards.len(), 1);
        assert_eq!(plan.stats.cut_edges, 0);
        assert_eq!(plan.assignments.len(), g.len());
        assert_eq!(plan.strategy, Strategy::Hybrid);
    }

    #[test]
    fn big_model_spills_across_nodes() {
        // Two chained matmuls of ~0.78 MiB weights each: under a 1 MiB/node
        // cap the second must spill onto another node.
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

        let opts = PartitionOptions {
            max_weight_bytes_per_node: Some(MIB),
            ..Default::default()
        };
        let plan = partition(&g, &fleet(16, 8 * MIB), &opts).unwrap();
        assert_eq!(plan.shards.len(), 2);
        assert_eq!(plan.stats.cut_edges, 1);
        assert_eq!(plan.stats.weight_bytes_total, 2 * 640 * 640 * 2);
    }

    #[test]
    fn impossible_fit_is_a_loud_oom() {
        let g = mlp_graph(1024); // 2 MiB weights
        let opts = PartitionOptions {
            max_weight_bytes_per_node: Some(64 * 1024),
            ..Default::default()
        };
        let err = partition(&g, &fleet(4, MIB), &opts).unwrap_err();
        assert!(matches!(err, Error::OutOfMemory { .. }));
    }

    #[test]
    fn kv_reservation_shrinks_budgets_and_fails_cleanly() {
        let g = mlp_graph(256);
        let kv = KvCacheRequest {
            num_layers: 22,
            num_kv_heads: 8,
            head_dim: 128,
            dtype: DType::F16,
            max_seq_len: 512,
            batch: 1,
        };
        let opts = PartitionOptions {
            kv_request: Some(kv),
            ..Default::default()
        };
        // KV needs ~44 MiB across the fleet here; 8 MiB nodes must refuse.
        assert!(matches!(
            partition(&g, &fleet(4, 8 * MIB), &opts),
            Err(Error::OutOfMemory { .. })
        ));

        let plan = partition(&g, &fleet(4, 128 * MIB), &opts).unwrap();
        let kv_plan = plan.kv_plan.as_ref().unwrap();
        assert_eq!(kv_plan.assignments.len(), 4);
        // Every shard carries exactly its reserved slice.
        for s in &plan.shards {
            assert_eq!(s.kv_cache_bytes, kv_plan.bytes_on(s.swarm_node));
        }
    }

    #[test]
    fn hybrid_splits_attention_heads_layer_serial_rest() {
        let mut g = Graph::new("attn-net");
        let input = |g: &mut Graph| {
            g.push(
                "",
                Op::Input(TensorDesc::new(vec![], DType::F32)),
                Vec::<NodeId>::new(),
            )
        };
        let q = input(&mut g);
        let k = input(&mut g);
        let v = input(&mut g);
        let attrs = AttentionAttrs {
            num_heads: 8,
            num_kv_heads: 8,
            head_dim: 64,
            causal: true,
            scale: None,
        };
        let att = g.push("", Op::Attention { attrs }, vec![q, k, v]);
        let att_out = g.push("", Op::Output { name: "o".into() }, vec![att]);
        g.mark_output(att_out);
        g.validate().unwrap();

        let plan = partition(&g, &fleet(4, MIB), &PartitionOptions::default()).unwrap();
        assert_eq!(plan.head_parallel_groups.len(), 1);
        let slices = &plan.head_parallel_groups[0].slices;
        assert_eq!(slices.len(), 4);
        assert_eq!(slices.iter().map(|s| s.head_count).sum::<u32>(), 8);

        let serial = partition(
            &g,
            &fleet(4, MIB),
            &PartitionOptions {
                strategy: Some(Strategy::LayerSerial),
                ..Default::default()
            },
        )
        .unwrap();
        assert!(serial.head_parallel_groups.is_empty());
    }

    #[test]
    fn heterogeneous_memory_steers_placement() {
        let g = mlp_graph(1024); // 2 MiB weights
        let nodes = vec![
            SwarmNode::new(0, "big", NodeArch::Esp32, 4 * MIB),
            SwarmNode::new(1, "small", NodeArch::Esp32, MIB / 2),
        ];
        let topo = Topology::from_parts(
            nodes,
            vec![vec![0.0, 1.0], vec![1.0, 0.0]],
            vec![vec![0.0, 1.0e6], vec![1.0e6, 0.0]],
        )
        .unwrap();
        let plan = partition(&g, &topo, &PartitionOptions::default()).unwrap();
        // The whole model must sit on the big node.
        assert!(plan.assignments.values().all(|&n| n == 0));
    }

    #[test]
    fn hybrid_node_partitions_like_a_plain_one() {
        // With FPGA offload disabled (the default), a silicon+FPGA node must
        // not change placement or panic.
        let g = mlp_graph(64);
        let nodes = vec![SwarmNode::with_fpga(
            0,
            "hybrid",
            NodeArch::RiscV,
            2 * MIB,
            FpgaProfile {
                luts: 20_000,
                dsp_slices: 64,
                block_ram_bytes: 512 * 1024,
            },
        )];
        let topo = Topology::from_parts(nodes, vec![vec![0.0]], vec![vec![0.0]]).unwrap();
        let plan = partition(&g, &topo, &PartitionOptions::default()).unwrap();
        assert_eq!(plan.shards.len(), 1);
        assert_eq!(topo.nodes[0].fpga.unwrap().luts, 20_000);
        assert!(!plan.shards[0].fpga_offload());
        assert_eq!(plan.stats.fpga_offload_ops, 0);
    }

    /// [plain 4 MiB node, hybrid node with the given BRAM].
    fn hybrid_fleet(bram: u64) -> Topology {
        let nodes = vec![
            SwarmNode::new(0, "plain", NodeArch::Esp32, 4 * MIB),
            SwarmNode::with_fpga(
                1,
                "hybrid",
                NodeArch::RiscV,
                4 * MIB,
                FpgaProfile {
                    luts: 60_000,
                    dsp_slices: 128,
                    block_ram_bytes: bram,
                },
            ),
        ];
        Topology::from_parts(
            nodes,
            vec![vec![0.0, 1.0], vec![1.0, 0.0]],
            vec![vec![0.0, 1.0e6], vec![1.0e6, 0.0]],
        )
        .unwrap()
    }

    #[test]
    fn fpga_offload_prefers_hybrid_for_gemm_segments() {
        // 2 MiB of weights fit either node's memory; with offload enabled the
        // GEMM segment should land on the fabric-equipped board.
        let g = mlp_graph(1024); // 2 MiB weights
        let topo = hybrid_fleet(4 * MIB);
        let opts = PartitionOptions {
            fpga_offload: true,
            ..Default::default()
        };
        let plan = partition(&g, &topo, &opts).unwrap();
        assert_eq!(plan.shards.len(), 1);
        assert_eq!(plan.shards[0].swarm_node, 1, "GEMM segment prefers hybrid");
        assert!(plan.shards[0].fpga_offload());
        assert_eq!(plan.stats.fpga_offload_ops, 1); // the matmul
        assert!(plan.shards[0]
            .fpga_ops
            .iter()
            .all(|&id| matches!(g.get_node(id).unwrap().op, Op::MatMul)));
    }

    #[test]
    fn fpga_offload_falls_back_when_bram_too_small() {
        // The fabric cannot stage 2 MiB of weights: keep the shard on plain
        // silicon rather than failing.
        let g = mlp_graph(1024);
        let topo = hybrid_fleet(64 * 1024);
        let opts = PartitionOptions {
            fpga_offload: true,
            ..Default::default()
        };
        let plan = partition(&g, &topo, &opts).unwrap();
        assert_eq!(plan.shards[0].swarm_node, 0, "falls back to plain node");
        assert!(!plan.shards[0].fpga_offload());
        assert_eq!(plan.stats.fpga_offload_ops, 0);
    }

    #[test]
    fn fpga_offload_ignores_non_gemm_segments() {
        // Elementwise-only graphs gain nothing from a fabric; placement must
        // stay on the most-headroom (plain) node and mark nothing offloaded.
        let mut g = Graph::new("eltwise");
        let x = g.push(
            "",
            Op::Input(TensorDesc::new(vec![8], DType::F32)),
            Vec::<NodeId>::new(),
        );
        let y = g.push("", Op::Gelu, vec![x]);
        let o = g.push("", Op::Output { name: "o".into() }, vec![y]);
        g.mark_output(o);

        let topo = hybrid_fleet(4 * MIB);
        let opts = PartitionOptions {
            fpga_offload: true,
            ..Default::default()
        };
        let plan = partition(&g, &topo, &opts).unwrap();
        assert!(!plan.shards[0].fpga_offload());
        assert_eq!(plan.stats.fpga_offload_ops, 0);
    }

    #[test]
    fn runtime_report_updates_capability_between_runs() {
        // A hybrid board whose fabric starts small: the GEMM segment falls
        // back to silicon. After an overlay load the node re-reports a larger
        // fabric; re-partitioning against the same topology object now
        // offloads onto it — and tearing the overlay down reverts.
        let g = mlp_graph(1024); // 2 MiB weights
        let mut topo = hybrid_fleet(64 * 1024);
        let opts = PartitionOptions {
            fpga_offload: true,
            ..Default::default()
        };

        let before = partition(&g, &topo, &opts).unwrap();
        assert_eq!(before.shards[0].swarm_node, 0);
        assert!(!before.shards[0].fpga_offload());

        topo.report_fpga(
            1,
            Some(FpgaProfile {
                luts: 60_000,
                dsp_slices: 128,
                block_ram_bytes: 4 * MIB,
            }),
        )
        .unwrap();
        let after = partition(&g, &topo, &opts).unwrap();
        assert_eq!(after.shards[0].swarm_node, 1);
        assert!(after.shards[0].fpga_offload());

        // Overlay torn down again: capability reverts.
        topo.report_fpga(1, None).unwrap();
        let reverted = partition(&g, &topo, &opts).unwrap();
        assert!(!reverted.shards[0].fpga_offload());
    }

    #[test]
    fn report_fpga_rejects_unknown_node() {
        let mut topo = hybrid_fleet(4 * MIB);
        let err = topo.report_fpga(9, None).unwrap_err();
        assert!(matches!(
            err,
            tpt_crucible_common::Error::InvalidArgument(_)
        ));
    }
}
