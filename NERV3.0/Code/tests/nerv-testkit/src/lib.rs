//! # nerv-testkit — the in-process multinode harness (WP §13.4, T5–T7)
//!
//! * `clock`      — the deterministic logical clock (erratum 193).
//! * `loopback`   — the in-memory transport (bounded queues).
//! * `scheduler`  — the adversarial scheduler and its strategies.
//! * `harness`    — the multinode harness: nodes, wiring, step execution.
//! * `schedules`  — the TLA+ schedule corpus (T5–T7 as data).
//!
//! Everything runs in one process, one thread, zero I/O — the
//! determinism the CI jobs pin (DSR-11).


#![forbid(unsafe_code)]
#![deny(clippy::float_arithmetic, clippy::float_cmp)]


pub mod clock;
pub mod harness;
pub mod loopback;
pub mod scheduler;
pub mod schedules;


pub use clock::DetClock;
pub use harness::{Harness, HarnessError, NodeId, TestNode};
pub use loopback::{Loopback, Message, Queue};
pub use scheduler::{Schedule, ScheduleStrategy, Scheduler};
pub use schedules::{run_corpus, CorpusResult, CORPUS};
