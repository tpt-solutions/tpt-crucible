//! KV-cache memory planning.
//!
//! Attention KV caches dominate RAM on long-context LLM inference; on an
//! ESP32 with a few MiB of PSRAM, an unplanned cache means a silent OOM
//! hours into a flash session (`spec2.txt`: "KV Cache-Aware Memory
//! Planning"). [`plan`] distributes heads across nodes up-front and refuses
//! plans that would not fit.

use serde::{Deserialize, Serialize};
use tpt_crucible_common::dtype::DType;
use tpt_crucible_common::error::{Error, Result};

/// The cache workload to plan for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KvCacheRequest {
    /// Transformer blocks contributing a K and V cache each.
    pub num_layers: u32,
    /// Key/value heads per layer (GQA-aware).
    pub num_kv_heads: u32,
    /// Elements per head dimension.
    pub head_dim: u32,
    /// Element type of cached values (usually f16).
    pub dtype: DType,
    /// Maximum sequence length to reserve for.
    pub max_seq_len: u32,
    /// Concurrent sequences per node.
    pub batch: u32,
}

impl KvCacheRequest {
    /// Total bytes required if the entire cache lived on one node.
    pub fn total_bytes(&self) -> u64 {
        let elem = self.dtype.item_size().unwrap_or(2) as u64;
        // K and V, per head, per element, per token, per sequence:
        2u64 * self.num_kv_heads as u64
            * self.head_dim as u64
            * elem
            * self.max_seq_len as u64
            * self.batch as u64
            * self.num_layers as u64
    }
}

/// One contiguous slice of kv heads assigned to a node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HeadSlice {
    /// Owning swarm node index.
    pub node: usize,
    /// First kv head handled by this node.
    pub head_start: u32,
    /// Number of kv heads handled by this node.
    pub head_count: u32,
}

/// A feasible distribution of the cache across the swarm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCachePlan {
    /// Head slices, one entry per node that holds cache state.
    pub assignments: Vec<HeadSlice>,
    /// Bytes reserved on each node (indexed by node id).
    pub per_node_bytes: Vec<u64>,
    /// Whole-cache footprint across all nodes.
    pub total_bytes: u64,
}

impl KvCachePlan {
    /// Reserved bytes on a given node id (0 when out of range or unassigned).
    pub fn bytes_on(&self, node: usize) -> u64 {
        self.per_node_bytes.get(node).copied().unwrap_or(0)
    }
}

/// Distribute `req` across `node_memory` (bytes usable per node).
///
/// Heads are handed out round-robin so every node gets an equal share; the
/// plan fails loudly with [`Error::OutOfMemory`] naming the tightest node
/// instead of letting firmware discover the OOM in the field.
pub fn plan(req: &KvCacheRequest, node_memory: &[u64]) -> Result<KvCachePlan> {
    if node_memory.is_empty() {
        return Err(Error::InvalidArgument(
            "cannot plan kv cache over zero nodes".into(),
        ));
    }
    if req.num_kv_heads == 0 || req.max_seq_len == 0 || req.num_layers == 0 {
        return Err(Error::InvalidArgument(
            "kv cache request needs at least one layer, head, and token".into(),
        ));
    }

    let nodes = node_memory.len() as u32;
    let heads_per_node = req.num_kv_heads.div_ceil(nodes);

    let mut assignments = Vec::with_capacity(nodes as usize);
    let mut per_node_bytes = vec![0u64; node_memory.len()];
    let mut remaining_head = req.num_kv_heads;
    let mut head_start = 0u32;
    for (idx, &mem) in node_memory.iter().enumerate() {
        if remaining_head == 0 {
            break;
        }
        let count = remaining_head.min(heads_per_node);
        let slice_req = KvCacheRequest {
            num_kv_heads: count,
            ..*req
        };
        let bytes = slice_req.total_bytes();
        if bytes > mem {
            return Err(Error::OutOfMemory {
                node: format!("node-{idx:02}"),
                needed: bytes,
                available: mem,
            });
        }
        assignments.push(HeadSlice {
            node: idx,
            head_start,
            head_count: count,
        });
        per_node_bytes[idx] = bytes;
        remaining_head -= count;
        head_start += count;
    }

    debug_assert_eq!(remaining_head, 0, "round-robin must cover every head");
    Ok(KvCachePlan {
        total_bytes: req.total_bytes(),
        assignments,
        per_node_bytes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req() -> KvCacheRequest {
        KvCacheRequest {
            num_layers: 22,
            num_kv_heads: 4,
            head_dim: 64,
            dtype: DType::F16,
            max_seq_len: 512,
            batch: 1,
        }
    }

    const MIB: u64 = 1 << 20;

    #[test]
    fn totals_match_hand_math() {
        // 2(KV) * 4 heads * 64 dim * 2B(f16) * 512 tok * 1 batch * 22 layers
        assert_eq!(req().total_bytes(), 2 * 4 * 64 * 2 * 512 * 22);
    }

    #[test]
    fn even_split_across_nodes() {
        let p = plan(&req(), &[8 * MIB; 4]).unwrap();
        assert_eq!(p.assignments.len(), 4);
        assert_eq!(
            p.assignments[0],
            HeadSlice {
                node: 0,
                head_start: 0,
                head_count: 1
            }
        );
        assert_eq!(p.assignments[3].head_start, 3);
        for n in 0..4 {
            assert_eq!(p.bytes_on(n), req().total_bytes() / 4);
        }
    }

    #[test]
    fn fewer_heads_than_nodes_leaves_some_idle() {
        let p = plan(&req(), &[8 * MIB; 8]).unwrap();
        // 4 kv heads spread over 8 nodes -> first 4 carry one each.
        assert_eq!(p.assignments.len(), 4);
        assert_eq!(p.bytes_on(5), 0);
    }

    #[test]
    fn oom_names_the_tight_node() {
        let err = plan(&req(), &[8 * MIB, 8 * MIB, MIB, 8 * MIB]).unwrap_err();
        match err {
            Error::OutOfMemory {
                node,
                needed,
                available,
            } => {
                assert_eq!(node, "node-02");
                assert_eq!(available, MIB);
                assert!(needed > available);
            }
            other => panic!("expected OOM, got {other:?}"),
        }
    }

    #[test]
    fn empty_swarm_rejected() {
        assert!(plan(&req(), &[]).is_err());
    }
}
