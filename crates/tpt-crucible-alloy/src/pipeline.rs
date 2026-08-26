//! Rolling pipeline parallelism (`spec2.txt` §3.4 "Fault-Tolerant Execution
//! & Pipeline Parallelism").
//!
//! A layer-serial swarm wastes the fleet: while stage *k* computes, stages
//! behind it idle. Rolling pipelining keeps every stage busy by feeding
//! micro-batches back-to-back — stage *k* starts micro-batch *m+1* the moment
//! it frees, overlapping with its neighbors.
//!
//! [`schedule`] simulates that schedule exactly (greedy in-flight, fixed
//! order — optimal for pipeline flow shops) and reports the makespan against
//! the strict serial baseline, so callers can prove the stall elimination
//! instead of asserting it. [`PipelineModel::from_plan`] estimates per-stage
//! compute from
//! shard weight bytes and inter-stage transfer from the measured topology
//! matrices.
//!
//! ```text
//! stage:      S0   S1   S2
//! mb 0        [──] [──] [──]
//! mb 1             [──] [──] [──]     ← starts before mb 0 finishes
//! time ─────────────────────────────►
//! ```

use serde::{Deserialize, Serialize};
use tpt_crucible_common::error::{Error, Result};

use crate::partition::PartitionPlan;
use crate::topology::Topology;

/// One scheduled execution slot: a micro-batch on a pipeline stage.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Slot {
    /// Micro-batch index (0-based).
    pub microbatch: u32,
    /// Pipeline stage index (= shard index of the plan).
    pub stage: usize,
    /// Start time in ms.
    pub start_ms: f64,
    /// End time in ms (`start_ms + compute`, never overlapping a previous
    /// slot on the same stage).
    pub end_ms: f64,
}

/// Per-stage timing model for [`schedule`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineModel {
    /// Compute cost per stage in ms (stage = plan shard order).
    pub stage_compute_ms: Vec<f64>,
    /// `transfer_ms[i][j]`: activation-transfer cost from stage `i`'s node to
    /// stage `j`'s node (0 on the diagonal).
    pub transfer_ms: Vec<Vec<f64>>,
}

impl PipelineModel {
    /// Build a model from a partition plan: compute estimated as
    /// `weight_bytes / bytes_per_ms`, transfer taken from the measured
    /// latency/bandwidth matrices for `activation_bytes` activations.
    ///
    /// `bytes_per_ms` is the fleet's observed compute throughput; 50 KiB/ms
    /// (~50 MB/s through one ESP32-S3 MAC array) is a sane first guess and
    /// should be replaced by reported telemetry once nodes run.
    ///
    /// # Errors
    /// [`Error::InvalidArgument`] when `bytes_per_ms` is not finite positive,
    /// the plan has no shards, or shard owners fall outside the topology.
    pub fn from_plan(
        plan: &PartitionPlan,
        topo: &Topology,
        bytes_per_ms: f64,
        activation_bytes: u64,
    ) -> Result<Self> {
        if !bytes_per_ms.is_finite() || bytes_per_ms <= 0.0 {
            return Err(Error::InvalidArgument(format!(
                "bytes_per_ms must be finite positive, got {bytes_per_ms}"
            )));
        }
        let n = plan.shards.len();
        if n == 0 {
            return Err(Error::InvalidArgument(
                "plan has no shards; nothing to pipeline".into(),
            ));
        }
        let mut stage_compute_ms = Vec::with_capacity(n);
        for shard in &plan.shards {
            let owner = shard.swarm_node;
            let name = topo
                .nodes
                .get(owner)
                .map(|n| n.name.as_str())
                .unwrap_or("?");
            if owner >= topo.nodes.len() {
                return Err(Error::InvalidArgument(format!(
                    "shard owner {owner} ({name}) is outside the topology"
                )));
            }
            stage_compute_ms.push(shard.weight_bytes as f64 / bytes_per_ms);
        }
        let mut transfer_ms = vec![vec![0.0f64; n]; n];
        for (i, row) in transfer_ms.iter_mut().enumerate() {
            for (j, cell) in row.iter_mut().enumerate() {
                if i == j {
                    continue;
                }
                let (a, b) = (plan.shards[i].swarm_node, plan.shards[j].swarm_node);
                *cell = topo.transfer_seconds(a, b, activation_bytes) * 1000.0;
            }
        }
        Ok(Self {
            stage_compute_ms,
            transfer_ms,
        })
    }
}

/// The rolling-pipeline schedule over [`PipelineModel`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineSchedule {
    /// Model this schedule was computed from.
    #[serde(flatten)]
    pub model: PipelineModel,
    /// Micro-batches pushed through the pipeline.
    pub microbatches: u32,
    /// Every (microbatch, stage) slot, ordered by start time.
    pub slots: Vec<Slot>,
    /// When the last slot finishes (ms).
    pub makespan_ms: f64,
    /// What strict layer-serial execution would have taken (ms): each
    /// micro-batch traverses all stages before the next starts.
    pub serial_makespan_ms: f64,
}

impl PipelineSchedule {
    /// `serial / rolling` speedup (>1 means pipelining won).
    pub fn speedup(&self) -> f64 {
        if self.makespan_ms <= 0.0 {
            return 1.0;
        }
        self.serial_makespan_ms / self.makespan_ms
    }

    /// True when rolling actually eliminated stalls (strictly earlier finish
    /// than serial, with more than one stage and more than one micro-batch).
    pub fn eliminates_stalls(&self) -> bool {
        self.model.stage_compute_ms.len() > 1
            && self.microbatches > 1
            && self.makespan_ms < self.serial_makespan_ms - 1e-9
    }

    /// Steady-state period in ms: once filled, one result pops out every
    /// bottleneck-stage compute time. Single-stage pipelines return their own
    /// compute time.
    pub fn steady_state_period_ms(&self) -> f64 {
        self.model
            .stage_compute_ms
            .iter()
            .copied()
            .fold(0.0f64, f64::max)
    }
}

/// Simulate rolling pipelining of `microbatches` through the model's stages.
///
/// Greedy in-order flow shop: each slot starts as soon as both its stage is
/// free and the previous stage has finished that micro-batch (plus transfer).
/// This is the classic optimal pipeline schedule for fixed order.
///
/// # Errors
/// [`Error::InvalidArgument`] when the model is empty, a cost is negative or
/// non-finite, or `microbatches` is zero.
pub fn schedule(model: &PipelineModel, microbatches: u32) -> Result<PipelineSchedule> {
    let stages = model.stage_compute_ms.len();
    if stages == 0 {
        return Err(Error::InvalidArgument(
            "pipeline model has no stages".into(),
        ));
    }
    if microbatches == 0 {
        return Err(Error::InvalidArgument(
            "microbatches must be at least 1".into(),
        ));
    }
    if model
        .stage_compute_ms
        .iter()
        .chain(model.transfer_ms.iter().flatten())
        .any(|v| !v.is_finite() || *v < 0.0)
    {
        return Err(Error::InvalidArgument(
            "stage/transfer costs must be finite and non-negative".into(),
        ));
    }

    let mut slots = Vec::with_capacity(stages * microbatches as usize);
    let mut stage_free = vec![0.0f64; stages];
    // Completion time of each (micro-batch, stage); flat [mb * stages + s].
    let mut done = vec![0.0f64; stages * microbatches as usize];

    for mb in 0..microbatches as usize {
        for s in 0..stages {
            let compute = model.stage_compute_ms[s];
            let upstream = if s == 0 {
                0.0
            } else {
                done[mb * stages + s - 1] + model.transfer_ms[s - 1][s]
            };
            let start = stage_free[s].max(upstream);
            let end = start + compute;
            stage_free[s] = end;
            done[mb * stages + s] = end;
            slots.push(Slot {
                microbatch: mb as u32,
                stage: s,
                start_ms: start,
                end_ms: end,
            });
        }
    }

    // Serial baseline: one micro-batch traverses every stage before the next
    // enters — compute plus the full inter-stage transfer chain, × M.
    let serial_per_mb: f64 = model.stage_compute_ms.iter().sum::<f64>()
        + (0..stages - 1)
            .map(|s| model.transfer_ms[s][s + 1])
            .sum::<f64>();
    let serial_makespan_ms = serial_per_mb * microbatches as f64;

    Ok(PipelineSchedule {
        model: PipelineModel {
            stage_compute_ms: model.stage_compute_ms.clone(),
            transfer_ms: model.transfer_ms.clone(),
        },
        microbatches,
        makespan_ms: slots.last().map(|s| s.end_ms).unwrap_or(0.0),
        serial_makespan_ms,
        slots,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model(compute: &[f64]) -> PipelineModel {
        let n = compute.len();
        PipelineModel {
            stage_compute_ms: compute.to_vec(),
            transfer_ms: vec![vec![0.0; n]; n],
        }
    }

    #[test]
    fn hand_checked_two_stage_overlap() {
        // S=2, costs [10, 20] ms, M=2, no transfer:
        // mb0: S0 0-10, S1 10-30 | mb1: S0 10-20, S1 30-50.
        let sched = schedule(&model(&[10.0, 20.0]), 2).unwrap();
        assert!((sched.makespan_ms - 50.0).abs() < 1e-9);
        assert!((sched.serial_makespan_ms - 60.0).abs() < 1e-9);
        assert!(sched.eliminates_stalls());
        assert!((sched.speedup() - 1.2).abs() < 1e-9);
        let s1 = &sched.slots[3];
        assert_eq!((s1.microbatch, s1.stage), (1, 1));
        assert!((s1.start_ms - 30.0).abs() < 1e-9, "waits for its data");
    }

    #[test]
    fn steady_state_matches_bottleneck() {
        let sched = schedule(&model(&[5.0, 20.0, 7.0]), 8).unwrap();
        assert!((sched.steady_state_period_ms() - 20.0).abs() < 1e-9);
        // Fill (5+20+7) then (M-1)*bottleneck.
        let expect = 32.0 + 7.0 * 20.0;
        assert!((sched.makespan_ms - expect).abs() < 1e-9);
    }

    #[test]
    fn stages_never_overlap_and_data_flows_in_order() {
        let sched = schedule(&model(&[3.0, 9.0, 4.0]), 12).unwrap();
        for s in 0..3usize {
            let mut per_stage: Vec<&Slot> = sched.slots.iter().filter(|sl| sl.stage == s).collect();
            per_stage.sort_by(|a, b| a.start_ms.total_cmp(&b.start_ms));
            for w in per_stage.windows(2) {
                assert!(w[1].start_ms >= w[0].end_ms - 1e-9, "stage {s} overlapped");
            }
        }
        // Slots are emitted mb-major/stage-minor: the previous slot of one
        // with stage > 0 is exactly its upstream producer.
        for (i, slot) in sched.slots.iter().enumerate() {
            if slot.stage > 0 {
                let prev = &sched.slots[i - 1];
                assert_eq!(prev.microbatch, slot.microbatch);
                assert_eq!(prev.stage, slot.stage - 1);
                assert!(slot.start_ms >= prev.end_ms - 1e-9);
            }
        }
    }

    #[test]
    fn single_stage_and_single_microbatch_degenerate() {
        let one = schedule(&model(&[7.0]), 4).unwrap();
        assert!((one.makespan_ms - 28.0).abs() < 1e-9);
        assert_eq!(one.serial_makespan_ms, one.makespan_ms);
        assert!(!one.eliminates_stalls());

        let once = schedule(&model(&[1.0, 2.0, 3.0]), 1).unwrap();
        assert!(!once.eliminates_stalls());
        assert!((once.makespan_ms - 6.0).abs() < 1e-9);
    }

    #[test]
    fn transfers_shift_the_schedule() {
        let m = PipelineModel {
            stage_compute_ms: vec![10.0, 10.0],
            transfer_ms: vec![vec![0.0, 5.0], vec![5.0, 0.0]],
        };
        let sched = schedule(&m, 2).unwrap();
        // Timeline: mb0 S0 0-10, S1 15-25 (5 ms wire); mb1 S0 10-20, S1 25-35.
        let s1 = &sched.slots[3];
        assert!((s1.start_ms - 25.0).abs() < 1e-9);
        assert!((sched.makespan_ms - 35.0).abs() < 1e-9);
    }

    #[test]
    fn invalid_inputs_rejected() {
        assert!(schedule(&model(&[]), 1).is_err());
        assert!(schedule(&model(&[1.0]), 0).is_err());
        assert!(schedule(&model(&[-1.0, 2.0]), 1).is_err());
        assert!(schedule(&model(&[1.0, f64::NAN]), 1).is_err());
    }

    #[test]
    fn from_plan_estimates_from_topology() {
        use crate::{partition, topology::Topology};
        use tpt_crucible_common::{DType, Op, Tensor, TensorDesc};

        // Two 256 KiB MatMul layers under a 300 KiB cap → shards on two nodes
        // (each layer must fit the cap; their sum may not).
        let mut g = tpt_crucible_common::Graph::new("mlp");
        let x = g.push(
            "x",
            Op::Input(TensorDesc::new(vec![1, 256], DType::F32)),
            Vec::<_>::new(),
        );
        let w = g.push(
            "w",
            Op::Constant {
                tensor: Tensor::from_f32(vec![256, 256], &vec![0.01; 256 * 256]),
            },
            Vec::<_>::new(),
        );
        let h = g.push("mm0", Op::MatMul, vec![x, w]);
        let w2 = g.push(
            "w2",
            Op::Constant {
                tensor: Tensor::from_f32(vec![256, 256], &vec![0.01; 256 * 256]),
            },
            Vec::<_>::new(),
        );
        let h2 = g.push("mm1", Op::MatMul, vec![h, w2]);
        let o = g.push("out", Op::Output { name: "y".into() }, vec![h2]);
        g.mark_output(o);

        let topo = Topology::homogeneous(4, "esp32s3", 8 << 20);
        let opts = partition::PartitionOptions {
            max_weight_bytes_per_node: Some(300_000),
            ..Default::default()
        };
        let plan = partition::partition(&g, &topo, &opts).unwrap();
        assert_eq!(plan.shards.len(), 2, "cap forces exactly two stages");

        let m = PipelineModel::from_plan(&plan, &topo, 100_000.0, 4096).unwrap();
        // Each stage carries one 256 KiB weight tensor → ~2.62 ms at 100 KiB/ms.
        for c in &m.stage_compute_ms {
            assert!((c - 262_144.0 / 100_000.0).abs() < 1e-9, "{c}");
        }
        // Stage owners sit on different nodes: 1 ms hop + 4096 B @ 1 MiB/s.
        assert!((m.transfer_ms[0][1] - (1.0 + 4096.0 / 1_048_576.0 * 1000.0)).abs() < 1e-9);
        assert!(m.transfer_ms[1][0] > 0.0);
        let sched = schedule(&m, 6).unwrap();
        assert!(sched.eliminates_stalls(), "multi-stage rolling must win");

        // Bad throughput rejected; empty plans rejected.
        assert!(PipelineModel::from_plan(&plan, &topo, 0.0, 1).is_err());
        let empty = partition::PartitionPlan {
            strategy: plan.strategy,
            assignments: Default::default(),
            head_parallel_groups: Vec::new(),
            kv_plan: None,
            stats: plan.stats,
            shards: Vec::new(),
        };
        assert!(PipelineModel::from_plan(&empty, &topo, 1.0, 1).is_err());
    }
}
