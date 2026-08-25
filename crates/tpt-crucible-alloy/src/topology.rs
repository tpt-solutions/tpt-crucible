//! Swarm topology: nodes, links, and latency-matrix auto-discovery.
//!
//! The physical layout is *reported by the nodes themselves*: every node
//! pings every other node and publishes a latency row;
//! [`Topology::from_parts`] turns those reports into the cost model used for
//! partitioning. No manual topology files, no hardcoded switch trees.

use serde::{Deserialize, Serialize};

/// Instruction set of a swarm node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NodeArch {
    /// ESP32 family (Xtensa LX6/LX7).
    Esp32,
    /// RP2040 / RP2350 (dual Cortex-M0+/M33).
    Rp2040,
    /// Generic 32/64-bit RISC-V microcontroller.
    RiscV,
    /// Browser tab (the wasm SiL demo).
    Wasm,
    /// Desktop-class x86-64 node.
    X86_64,
    /// Anything else.
    Other,
}

impl NodeArch {
    /// Canonical lowercase name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Esp32 => "esp32",
            Self::Rp2040 => "rp2040",
            Self::RiscV => "riscv",
            Self::Wasm => "wasm",
            Self::X86_64 => "x86_64",
            Self::Other => "other",
        }
    }
}

/// Reconfigurable fabric attached to a node, alongside its normal silicon.
///
/// A node with `fpga: Some(_)` is a hybrid board: `arch` still describes its
/// fixed ISA, and this describes the FPGA fabric sitting next to it. The two
/// are orthogonal, so this is a field on [`SwarmNode`] rather than a new
/// [`NodeArch`] variant.
///
/// This is descriptive metadata only today — no partitioning or firmware
/// logic reads it yet. It exists so `tpt-crucible-fusion` has a named shape
/// to target once it compiles real overlays (see its module docs).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct FpgaProfile {
    /// Available look-up tables.
    pub luts: u32,
    /// Available DSP slices.
    pub dsp_slices: u32,
    /// Block RAM capacity in bytes.
    pub block_ram_bytes: u64,
}

/// One physical node of the swarm.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmNode {
    /// Stable index into the topology matrices.
    pub id: usize,
    /// Human label, e.g. `"esp32-07"`.
    pub name: String,
    /// Instruction set.
    pub arch: NodeArch,
    /// Usable memory in bytes (PSRAM + SRAM minus firmware overhead).
    pub memory_bytes: u64,
    /// Reconfigurable fabric attached to this node, if any (hybrid board).
    pub fpga: Option<FpgaProfile>,
}

impl SwarmNode {
    /// Convenience constructor for a plain (non-hybrid) node.
    pub fn new(id: usize, name: impl Into<String>, arch: NodeArch, memory_bytes: u64) -> Self {
        Self {
            id,
            name: name.into(),
            arch,
            memory_bytes,
            fpga: None,
        }
    }

    /// Convenience constructor for a hybrid silicon+FPGA node.
    pub fn with_fpga(
        id: usize,
        name: impl Into<String>,
        arch: NodeArch,
        memory_bytes: u64,
        fpga: FpgaProfile,
    ) -> Self {
        Self {
            id,
            name: name.into(),
            arch,
            memory_bytes,
            fpga: Some(fpga),
        }
    }
}

/// A swarm: nodes plus pairwise communication costs.
///
/// Matrices are `n*n`, indexed by [`SwarmNode::id`]; diagonal entries are
/// zero. Latency is milliseconds per message, bandwidth bytes-per-second -
/// both measured, not assumed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Topology {
    /// Participating nodes.
    pub nodes: Vec<SwarmNode>,
    /// Pairwise one-way latency in ms.
    pub latency_ms: Vec<Vec<f64>>,
    /// Pairwise bandwidth in bytes/second.
    pub bandwidth: Vec<Vec<f64>>,
}

impl Topology {
    /// Assemble a topology from node-reported measurements
    /// ("auto-discovery"): `latency_ms[i][j]` and `bandwidth[i][j]` come
    /// from the swarm's own ping/bandwidth probes.
    ///
    /// # Errors
    ///
    /// Fails when a matrix does not match the node count or is not square.
    pub fn from_parts(
        nodes: Vec<SwarmNode>,
        latency_ms: Vec<Vec<f64>>,
        bandwidth: Vec<Vec<f64>>,
    ) -> tpt_crucible_common::Result<Self> {
        let n = nodes.len();
        let shape_ok = |m: &Vec<Vec<f64>>| m.len() == n && m.iter().all(|row| row.len() == n);
        if !shape_ok(&latency_ms) || !shape_ok(&bandwidth) {
            return Err(tpt_crucible_common::Error::InvalidArgument(format!(
                "latency/bandwidth matrices must be {n}x{n} to match the node list"
            )));
        }
        Ok(Self {
            nodes,
            latency_ms,
            bandwidth,
        })
    }

    /// A full mesh of `count` identical nodes with uniform link costs.
    ///
    /// Handy for first-flash fleets and tests before real measurements
    /// exist; replace with [`Topology::from_parts`] once nodes report
    /// numbers.
    pub fn homogeneous(count: usize, arch_name: &str, memory_bytes_per_node: u64) -> Self {
        let arch = match arch_name {
            "esp32" | "esp32s2" | "esp32s3" | "esp32c3" => NodeArch::Esp32,
            "rp2040" | "rp2350" => NodeArch::Rp2040,
            "wasm" => NodeArch::Wasm,
            "x86_64" => NodeArch::X86_64,
            _ => NodeArch::Other,
        };
        let nodes: Vec<SwarmNode> = (0..count)
            .map(|i| {
                SwarmNode::new(
                    i,
                    format!("{arch_name}-{i:02}"),
                    arch,
                    memory_bytes_per_node,
                )
            })
            .collect();
        // Conservative defaults: ~1 ms hop, ~1 MiB/s practical WiFi5.
        let mut latency = vec![vec![0.0; count]; count];
        let mut bandwidth = vec![vec![0.0; count]; count];
        for i in 0..count {
            for j in 0..count {
                if i != j {
                    latency[i][j] = 1.0;
                    bandwidth[i][j] = 1024.0 * 1024.0;
                }
            }
        }
        Self::from_parts(nodes, latency, bandwidth).expect("homogeneous matrices are well-formed")
    }

    /// Estimated transfer+latency cost in seconds for `bytes` between nodes.
    pub fn transfer_seconds(&self, from: usize, to: usize, bytes: u64) -> f64 {
        if from == to {
            return 0.0;
        }
        let lat = self.latency_ms[from][to] / 1000.0;
        let bw = self.bandwidth[from][to].max(1.0);
        lat + bytes as f64 / bw
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn homogeneous_mesh_costs() {
        let topo = Topology::homogeneous(4, "esp32s3", 8 << 20);
        assert_eq!(topo.nodes.len(), 4);
        assert_eq!(topo.nodes[2].name, "esp32s3-02");
        assert_eq!(topo.nodes[2].arch, NodeArch::Esp32);
        assert_eq!(topo.transfer_seconds(1, 1, 1 << 20), 0.0);
        // 1 MiB over 1 MiB/s + 1 ms = 1.001 s.
        let t = topo.transfer_seconds(0, 3, 1 << 20);
        assert!((t - 1.001).abs() < 1e-9);
    }

    #[test]
    fn matrix_shape_mismatch_rejected() {
        let nodes = vec![SwarmNode::new(0, "a", NodeArch::Esp32, 100)];
        let bad = vec![vec![0.0; 2]; 2];
        let good = vec![vec![0.0]];
        assert!(Topology::from_parts(nodes.clone(), bad.clone(), good.clone()).is_err());
        assert!(Topology::from_parts(nodes, good, bad).is_err());
    }

    #[test]
    fn arch_names_roundtrip() {
        assert_eq!(NodeArch::Rp2040.name(), "rp2040");
        assert_eq!(serde_json::to_string(&NodeArch::Wasm).unwrap(), "\"wasm\"");
    }
}
