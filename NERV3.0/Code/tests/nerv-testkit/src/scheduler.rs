//! The adversarial scheduler (erratum 194): owns the clock and the
//! transport, selects messages according to a strategy, and delivers
//! them to the target node's inbox.


use crate::clock::DetClock;
use crate::loopback::{Loopback, Message};


/// The scheduling strategy: how the scheduler picks the next message
/// from the ready set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ScheduleStrategy {
    /// FIFO — the honest baseline.
    InOrder,
    /// Reverse the ready set's order at each step.
    Reverse,
    /// Pick the message from the source with the fewest deliveries so far.
    RoundRobin,
    /// Drop messages from the named source (censorship — T7).
    DropSource { source: &'static str },
    /// Hold messages from the named source for `delay_ns` before delivery (T5/T6).
    DelaySource { source: &'static str, delay_ns: u64 },
    /// Drop a specific fraction (permille) deterministically by index.
    DropPermille { permille: u64, seed: u64 },
}


/// The scheduler: clock + transport + strategy.
pub struct Scheduler {
    pub clock: DetClock,
    pub transport: Loopback,
    strategy: ScheduleStrategy,
    step_count: u64,
    delivery_counts: std::collections::BTreeMap<String, u64>,
    prng_state: u64,
}


/// One step's outcome: what the scheduler did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StepResult {
    /// A message was delivered to `to`.
    Delivered { from: String, to: String },
    /// No messages were ready.
    Idle,
    /// A message was dropped by the strategy.
    Dropped { from: String, to: String },
}


impl Scheduler {
    pub fn new(strategy: ScheduleStrategy) -> Scheduler {
        Scheduler {
            clock: DetClock::new(),
            transport: Loopback::new(),
            strategy,
            step_count: 0,
            delivery_counts: std::collections::BTreeMap::new(),
            prng_state: 0x9E37_79B9_7F4A_7C15,
        }
    }


    pub fn strategy(&self) -> &ScheduleStrategy {
        &self.strategy
    }


    pub fn step_count(&self) -> u64 {
        self.step_count
    }


    fn prng(&mut self) -> u64 {
        self.prng_state = self.prng_state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.prng_state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }


    /// Execute one scheduler step: pick a ready message, apply the
    /// strategy, deliver or drop. Returns the result and the message
    /// (if delivered).
    pub fn step(&mut self) -> (StepResult, Option<Message>) {
        self.step_count += 1;
        self.clock.tick();


        // Apply the censorship strategy: drop from the censored source.
        if let ScheduleStrategy::DropSource { source } = &self.strategy {
            self.transport.drop_from(source);
        }


        // Apply the delay strategy: push delayed messages back with
        // a later ready time.
        if let ScheduleStrategy::DelaySource { source, delay_ns } = &self.strategy {
            let source: &'static str = *source;
            let delay = *delay_ns;
            let now = self.clock.now_ns();
            let ready = self.transport.ready(now);
            for msg in ready {
                if msg.from == source {
                    // Re-queue with the delay applied.
                    let _ = self.transport.send(
                        &msg.from, &msg.to, msg.payload, now, delay,
                    );
                } else {
                    // Re-queue immediately.
                    let _ = self.transport.send(
                        &msg.from, &msg.to, msg.payload, now, 0,
                    );
                }
            }
        }


        // Get the ready set.
        let now = self.clock.now_ns();
        let mut ready = self.transport.ready(now);


        if ready.is_empty() {
            return (StepResult::Idle, None);
        }


        // Apply the drop-permille strategy.
        if let ScheduleStrategy::DropPermille { permille, .. } = &self.strategy {
            let permille = *permille;
            let idx = self.prng() % 1000;
            if idx < permille {
                let msg = ready.remove(0);
                return (StepResult::Dropped { from: msg.from, to: msg.to }, None);
            }
        }


        // Select according to the strategy.
        let selected = match &self.strategy {
            ScheduleStrategy::InOrder => ready.remove(0),
            ScheduleStrategy::Reverse => ready.pop().unwrap(),
            ScheduleStrategy::RoundRobin => {
                // Pick from the source with the fewest deliveries.
                let mut best = 0;
                let mut best_count = u64::MAX;
                for (i, m) in ready.iter().enumerate() {
                    let count = self.delivery_counts.get(&m.from).copied().unwrap_or(0);
                    if count < best_count {
                        best_count = count;
                        best = i;
                    }
                }
                ready.remove(best)
            }
            ScheduleStrategy::DropSource { .. } => ready.remove(0),
            ScheduleStrategy::DelaySource { .. } => ready.remove(0),
            ScheduleStrategy::DropPermille { .. } => ready.remove(0),
        };


        // Re-queue unselected messages.
        for m in ready {
            let _ = self.transport.send(&m.from, &m.to, m.payload, now, 0);
        }


        // Record the delivery.
        *self.delivery_counts.entry(selected.from.clone()).or_insert(0) += 1;


        let result = StepResult::Delivered { from: selected.from.clone(), to: selected.to.clone() };
        (result, Some(selected))
    }


    /// Run `n` steps, returning the results.
    pub fn run(&mut self, n: u64) -> Vec<StepResult> {
        let mut out = Vec::with_capacity(n as usize);
        for _ in 0..n {
            let (result, _) = self.step();
            out.push(result);
        }
        out
    }
}
