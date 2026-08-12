//! Shared configuration types used across backends.

/// The memory-vs-throughput tradeoff for the ONNX Runtime embedder (and a
/// hint for how you might want to size other resources around it).
///
/// This only affects the ORT backend's session configuration today —
/// LiteRT's memory behavior is governed by which accelerator you pick
/// (`Accelerators`) rather than a comparable arena/threading knob in the
/// 0.2.x API.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MemoryProfile {
    /// Prioritize throughput: memory arena and the memory-pattern optimizer
    /// stay on, and intra-op parallelism scales with available cores. Good
    /// default for a server doing continuous, concurrent recognition.
    #[default]
    Balanced,
    /// Prioritize a small, predictable resident footprint over raw
    /// throughput: arena allocator and memory-pattern optimization both off,
    /// single-threaded. Good for running alongside other memory-hungry app
    /// code on a phone or a small edge box, at the cost of some latency and
    /// more allocator churn per call.
    LowMemory,
}

impl MemoryProfile {
    pub fn use_arena(&self) -> bool {
        matches!(self, MemoryProfile::Balanced)
    }

    pub fn use_memory_pattern(&self) -> bool {
        matches!(self, MemoryProfile::Balanced)
    }

    pub fn intra_threads(&self) -> usize {
        match self {
            MemoryProfile::Balanced => std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4),
            MemoryProfile::LowMemory => 1,
        }
    }
}
