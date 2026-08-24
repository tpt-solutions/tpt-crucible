//! # tpt-crucible-alloy
//!
//! **Module 3 (Swarm)** (`spec2.txt` §3.4): partitions TPT-IR across a swarm
//! of microcontrollers and generates per-node firmware.
//!
//! * [`topology`] — swarm description and latency-matrix auto-discovery.
//! * [`partition`] — topology-aware partitioning with transformer-native
//!   strategies (attention-head parallel + layer-serial hybrid).
//! * [`kv_cache`] — KV-cache distribution that prevents OOMs on
//!   memory-constrained nodes.
//! * [`heartbeat`] — node liveness protocol codec + failure detector.
//! * [`firmware`] — per-node firmware projects and master flashing scripts.
//!
//! ## Example
//!
//! ```
//! use tpt_crucible_alloy::{kv_cache::KvCacheRequest, partition, topology};
//! use tpt_crucible_common::{DType, Graph, Op, Tensor, TensorDesc};
//!
//! let mut g = Graph::new("tiny");
//! let x = g.push("", Op::Input(TensorDesc::new(vec![1, 8], DType::F32)), Vec::<_>::new());
//! let w = g.push("", Op::Constant { tensor: Tensor::zeros(TensorDesc::new(vec![8, 8], DType::F16)) }, Vec::<_>::new());
//! let y = g.push("", Op::MatMul, vec![x, w]);
//! let out = g.push("", Op::Output { name: "y".into() }, vec![y]);
//! g.mark_output(out);
//!
//! let topo = topology::Topology::homogeneous(4, "esp32s3", 8 << 20);
//! let plan = partition::partition(&g, &topo, &partition::PartitionOptions::default()).unwrap();
//! assert_eq!(plan.shards.len(), 1);
//! ```
//!
//! Pure Rust throughout: this crate is the one that compiles to
//! `wasm32-unknown-unknown` for the browser SiL demo.

pub mod firmware;
pub mod heartbeat;
pub mod kv_cache;
pub mod partition;
pub mod topology;

/// Re-export of the shared IR crate for downstream convenience.
pub use tpt_crucible_common as common;
