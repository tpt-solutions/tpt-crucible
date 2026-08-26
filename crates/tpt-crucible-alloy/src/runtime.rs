//! The swarm runtime: shard distribution, heartbeat-driven liveness, and
//! automatic recovery (`spec2.txt` §3.4 "Fault-Tolerant Execution").
//!
//! [`ExecutionEngine`] is the coordinator-side control plane:
//!
//! 1. `launch` partitions the graph and produces one bincode
//!    [`ShardDeployment`] per busy node — the payload a flashed node loads at
//!    boot (the firmware generator in [`crate::firmware`] emits the code that
//!    consumes it);
//! 2. nodes stream [`HeartbeatMsg`]s; the engine feeds them into a
//!    [`FailureDetector`];
//! 3. every `tick`, nodes silent past the timeout are handed to
//!    [`crate::recovery::bypass_dead_nodes`]: the plan is re-cut over the
//!    survivors, displaced work is reported, and a fresh generation of
//!    deployments is produced — inference degrades instead of stalling;
//! 4. when nothing survives, the engine reports [`EngineEvent::FleetLost`]
//!    once and goes quiet (deliberately terminal: re-flashing the fleet is an
//!    operator action, not an automatic one).
//!
//! Networking is intentionally absent: time is injected so behavior is
//! deterministic in tests and portable to wasm, matching the rest of Alloy.
//! A real coordinator wires `tick`/`on_heartbeat` to its transport of choice.

use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tpt_crucible_common::error::Result;
use tpt_crucible_common::Graph;

use crate::heartbeat::{FailureDetector, HeartbeatMsg};
use crate::partition::{self, PartitionOptions, PartitionPlan};
use crate::recovery::{self, Recovery};
use crate::topology::Topology;

/// One node's boot payload: which IR nodes, weights, and KV-slice it owns.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShardDeployment {
    /// Original topology id of the node this deployment targets.
    pub node_id: usize,
    /// Node label (echoed for logging/telemetry).
    pub node_name: String,
    /// Plan generation that produced this deployment.
    pub generation: u64,
    /// Bincode-encoded [`crate::partition::Shard`] (compact on-device decode,
    /// same codec family as binary TPT-IR).
    pub payload: Vec<u8>,
}

impl ShardDeployment {
    /// Decode the embedded shard back.
    ///
    /// # Errors
    /// A deserialization error when the bytes are corrupt.
    pub fn decode(&self) -> Result<crate::partition::Shard> {
        Ok(bincode::deserialize(&self.payload)?)
    }
}

/// Coordinator lifecycle events surfaced by [`ExecutionEngine::tick`].
#[derive(Debug, Clone, PartialEq)]
pub enum EngineEvent {
    /// Nodes went silent past the timeout (original topology ids).
    NodeLost {
        /// Ids newly detected as dead.
        nodes: Vec<usize>,
    },
    /// The fleet was re-planned without the lost nodes.
    Recovered {
        /// New plan generation.
        generation: u64,
        /// Bypassed ids (cumulative for this generation).
        bypassed: Vec<usize>,
        /// Count of IR nodes that must be re-fetched elsewhere.
        displaced: usize,
    },
    /// Every node died; the engine stops scheduling.
    FleetLost,
}

/// The coordinator-side swarm runtime.
///
/// * `graph`/`topology` are captured at launch; recovery always plans against
///   the original topology (bypassing dead ids) rather than mutating it, so
///   id semantics stay stable across generations.
/// * Shard index → original node id translation lives in `survivor_map`,
///   refreshed by every recovery.
pub struct ExecutionEngine<'a> {
    graph: &'a Graph,
    topology: Topology,
    options: PartitionOptions,
    detector: FailureDetector,
    plan: PartitionPlan,
    /// Current-plan shard index → original topology id.
    survivor_map: Vec<usize>,
    generation: u64,
    known_dead: Vec<usize>,
    failed: bool,
}

impl<'a> ExecutionEngine<'a> {
    /// Partition the graph and arm the failure detector.
    ///
    /// # Errors
    /// Whatever [`crate::partition::partition`] raises (invalid IR, shards
    /// that fit nowhere).
    pub fn launch(
        graph: &'a Graph,
        topology: &Topology,
        options: PartitionOptions,
        heartbeat_timeout: Duration,
    ) -> Result<Self> {
        let plan = partition::partition(graph, topology, &options)?;
        let survivor_map = topology.nodes.iter().map(|n| n.id).collect();
        Ok(Self {
            graph,
            topology: topology.clone(),
            options,
            detector: FailureDetector::new(heartbeat_timeout),
            plan,
            survivor_map,
            generation: 0,
            known_dead: Vec::new(),
            failed: false,
        })
    }

    /// Deployments for the current generation, one per busy node.
    ///
    /// Deterministic order: by original node id.
    pub fn distribute(&self) -> Vec<ShardDeployment> {
        let mut out: Vec<ShardDeployment> = self
            .plan
            .shards
            .iter()
            .enumerate()
            .map(|(shard_idx, shard)| {
                let node_id = self.survivor_map.get(shard_idx).copied().unwrap_or(0);
                let name = self
                    .topology
                    .nodes
                    .get(node_id)
                    .map(|n| n.name.clone())
                    .unwrap_or_default();
                ShardDeployment {
                    node_id,
                    node_name: name,
                    generation: self.generation,
                    // Bincode of an in-memory shard is infallible.
                    payload: bincode::serialize(shard).expect("shard serialization is infallible"),
                }
            })
            .collect();
        out.sort_by_key(|d| d.node_id);
        out
    }

    /// Record an incoming heartbeat observed at `now`.
    pub fn on_heartbeat(&mut self, msg: &HeartbeatMsg, now: Instant) {
        self.detector.record(msg, now);
    }

    /// Current plan generation (0 = the launch plan).
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// The active plan (read-only view for telemetry/manifests).
    pub fn plan(&self) -> &PartitionPlan {
        &self.plan
    }

    /// Original topology id hosting current-plan shard `shard_idx`.
    pub fn original_id_of(&self, shard_idx: usize) -> Option<usize> {
        self.survivor_map.get(shard_idx).copied()
    }

    /// True once [`EngineEvent::FleetLost`] fired; the engine is inert after.
    pub fn is_failed(&self) -> bool {
        self.failed
    }
}

impl<'a> ExecutionEngine<'a> {
    /// Advance time to `now`: detect silent nodes, re-plan over survivors.
    ///
    /// Emits events in causal order (`NodeLost` before its `Recovered`). A
    /// quiet fleet emits nothing; a terminal fleet emits nothing after its
    /// single [`EngineEvent::FleetLost`].
    pub fn tick(&mut self, now: Instant) -> Vec<EngineEvent> {
        if self.failed {
            return Vec::new();
        }
        let silent: Vec<usize> = self
            .detector
            .dead_nodes(now)
            .into_iter()
            // Heartbeat ids are the original topology ids (u32 on the wire).
            .map(|id| id as usize)
            .collect();
        let fresh: Vec<usize> = silent
            .iter()
            .copied()
            .filter(|id| !self.known_dead.contains(id))
            .collect();
        if fresh.is_empty() {
            return Vec::new();
        }

        let mut events = vec![EngineEvent::NodeLost {
            nodes: fresh.clone(),
        }];
        self.known_dead.extend(fresh);
        self.known_dead.sort_unstable();

        match recovery::bypass_dead_nodes(self.graph, &self.topology, &self.options, &silent) {
            Ok(rec) => {
                let Recovery {
                    bypassed,
                    survivor_map,
                    displaced,
                    plan,
                } = rec;
                self.survivor_map = survivor_map;
                self.plan = plan;
                self.generation += 1;
                events.push(EngineEvent::Recovered {
                    generation: self.generation,
                    bypassed,
                    displaced: displaced.len(),
                });
            }
            Err(_) => {
                // No survivors left (or nothing fits on them): terminal.
                self.failed = true;
                events.push(EngineEvent::FleetLost);
            }
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::heartbeat::NodeStatus;
    use crate::partition::Strategy;
    use std::time::Duration;
    use tpt_crucible_common::{DType, Op, Tensor, TensorDesc};

    const TIMEOUT: Duration = Duration::from_millis(100);
    const MIB: u64 = 1 << 20;

    /// A chain of `n` MatMul layers (weights sized so each layer is
    /// `dim*dim*4` bytes).
    fn mlp_graph(dim: usize, layers: usize) -> Graph {
        let mut g = Graph::new("mlp");
        let mut cur = g.push(
            "x",
            Op::Input(TensorDesc::new(vec![1, dim], DType::F32)),
            Vec::<_>::new(),
        );
        for l in 0..layers {
            let w = g.push(
                format!("w{l}"),
                Op::Constant {
                    tensor: Tensor::from_f32(vec![dim, dim], &vec![0.01; dim * dim]),
                },
                Vec::<_>::new(),
            );
            cur = g.push(format!("mm{l}"), Op::MatMul, vec![cur, w]);
        }
        let out = g.push("out", Op::Output { name: "y".into() }, vec![cur]);
        g.mark_output(out);
        g
    }

    fn fleet(count: usize) -> Topology {
        Topology::homogeneous(count, "esp32s3", 8 * MIB)
    }

    fn beat(id: u32, seq: u64) -> HeartbeatMsg {
        HeartbeatMsg {
            node_id: id,
            seq,
            timestamp_ms: 0,
            status: NodeStatus::Healthy,
        }
    }

    #[test]
    fn launch_distributes_decodable_deployments() {
        let g = mlp_graph(64, 2); // ~32 KiB weights → single shard
        let topo = fleet(4);
        let engine =
            ExecutionEngine::launch(&g, &topo, PartitionOptions::default(), TIMEOUT).unwrap();
        let deployments = engine.distribute();
        assert_eq!(deployments.len(), engine.plan().shards.len());
        assert!(deployments.len() <= 4);
        for d in &deployments {
            assert_eq!(d.generation, 0);
            assert_eq!(
                d.node_name, topo.nodes[d.node_id].name,
                "names echo the topology"
            );
            let shard = d.decode().unwrap();
            assert!(!shard.graph_nodes.is_empty());
        }
        // Sorted by original node id.
        assert!(deployments.windows(2).all(|w| w[0].node_id < w[1].node_id));
    }

    #[test]
    fn quiet_fleet_ticks_silently() {
        let g = mlp_graph(64, 2);
        let topo = fleet(4);
        let mut engine =
            ExecutionEngine::launch(&g, &topo, PartitionOptions::default(), TIMEOUT).unwrap();
        let t0 = Instant::now();
        for id in 0u32..4 {
            engine.on_heartbeat(&beat(id, 1), t0);
        }
        assert!(engine.tick(t0).is_empty());
        assert!(engine.tick(t0 + Duration::from_millis(50)).is_empty());
        // Everyone refreshes well inside the timeout window; still quiet
        // past it.
        for id in 0u32..4 {
            engine.on_heartbeat(&beat(id, 2), t0 + Duration::from_millis(150));
        }
        assert!(engine.tick(t0 + Duration::from_millis(200)).is_empty());
        assert_eq!(engine.generation(), 0);
    }

    #[test]
    fn silent_nodes_trigger_recovery_onto_survivors() {
        // ~4 MiB of weights split under a 1.5 MiB per-node cap: shards land
        // on nodes 0, 1, 2 ("most remaining headroom" placement).
        let g = mlp_graph(512, 4);
        let topo = fleet(6);
        let options = PartitionOptions {
            max_weight_bytes_per_node: Some(1_500_000),
            ..Default::default()
        };
        let mut engine = ExecutionEngine::launch(&g, &topo, options, TIMEOUT).unwrap();
        assert!(
            engine.plan().shards.len() > 1,
            "cap must force a multi-shard plan"
        );
        assert!(
            engine.plan().shards.iter().any(|s| s.swarm_node == 1),
            "node 1 hosts a shard by construction"
        );
        let t0 = Instant::now();
        for id in 0u32..6 {
            engine.on_heartbeat(&beat(id, 1), t0);
        }
        // Everyone but node 1 refreshes; node 1 goes silent.
        for id in [0u32, 2, 3, 4, 5] {
            engine.on_heartbeat(&beat(id, 2), t0 + Duration::from_millis(150));
        }

        let events = engine.tick(t0 + Duration::from_millis(200));
        assert_eq!(events.len(), 2);
        match &events[0] {
            EngineEvent::NodeLost { nodes } => assert_eq!(nodes, &vec![1]),
            other => panic!("expected NodeLost, got {other:?}"),
        }
        let EngineEvent::Recovered {
            generation,
            bypassed,
            displaced,
        } = &events[1]
        else {
            panic!("expected Recovered, got {:?}", events[1]);
        };
        assert_eq!(*generation, 1);
        assert_eq!(bypassed, &vec![1]);
        assert!(
            *displaced > 0,
            "node 1 hosted a shard: its IR nodes must move"
        );

        // The recovered plan avoids the dead node everywhere.
        assert_ne!(engine.original_id_of(0), Some(1));
        for d in engine.distribute() {
            assert_eq!(d.generation, 1);
            assert_ne!(d.node_id, 1, "no deployment may target a dead node");
            d.decode().unwrap();
        }
        for shard in &engine.plan().shards {
            assert_ne!(engine.original_id_of(shard.swarm_node), Some(1));
        }
    }

    #[test]
    fn idle_node_death_is_lossless() {
        // Model fits one node: killing any *other* node displaces nothing.
        let g = mlp_graph(64, 2); // ~32 KiB
        let topo = fleet(4);
        let mut engine =
            ExecutionEngine::launch(&g, &topo, PartitionOptions::default(), TIMEOUT).unwrap();
        let busy = engine.plan().shards[0].swarm_node as u32;

        let t0 = Instant::now();
        for id in 0u32..4 {
            engine.on_heartbeat(&beat(id, 1), t0);
        }
        engine.on_heartbeat(&beat(busy, 2), t0 + Duration::from_millis(150));

        let events = engine.tick(t0 + Duration::from_millis(200));
        assert!(matches!(
            events.as_slice(),
            [
                EngineEvent::NodeLost { .. },
                EngineEvent::Recovered { displaced: 0, .. }
            ]
        ));
        assert!(engine.generation() >= 1);
    }

    #[test]
    fn total_silence_loses_the_fleet_once() {
        let g = mlp_graph(64, 2);
        let topo = fleet(2);
        let mut engine =
            ExecutionEngine::launch(&g, &topo, PartitionOptions::default(), TIMEOUT).unwrap();
        let t0 = Instant::now();
        for id in 0u32..2 {
            engine.on_heartbeat(&beat(id, 1), t0);
        }
        let events = engine.tick(t0 + TIMEOUT + Duration::from_millis(1));
        assert_eq!(
            events,
            vec![
                EngineEvent::NodeLost { nodes: vec![0, 1] },
                EngineEvent::FleetLost
            ]
        );
        assert!(engine.is_failed());
        // Terminal: further ticks are quiet.
        assert!(engine
            .tick(t0 + TIMEOUT + Duration::from_secs(1))
            .is_empty());
    }

    #[test]
    fn unrecovered_plan_respects_default_strategy() {
        let g = mlp_graph(64, 2);
        let topo = fleet(3);
        let engine = ExecutionEngine::launch(
            &g,
            &topo,
            PartitionOptions {
                strategy: Some(Strategy::LayerSerial),
                ..Default::default()
            },
            TIMEOUT,
        )
        .unwrap();
        assert_eq!(engine.plan().strategy, Strategy::LayerSerial);
    }
}
