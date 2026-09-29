//! The in-memory transport (erratum 194): bounded queues between named
//! endpoints, owned by the scheduler.


use std::collections::BTreeMap;
use std::collections::VecDeque;


pub const DEFAULT_CAPACITY: usize = 4096;


/// One message in the transport.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub from: String,
    pub to: String,
    pub payload: Vec<u8>,
    /// The logical time the message was sent.
    pub sent_ns: u64,
    /// The earliest logical time the message may be delivered.
    pub ready_ns: u64,
}


/// A bounded queue from one source to one destination.
#[derive(Debug)]
pub struct Queue {
    pub capacity: usize,
    messages: VecDeque<Message>,
}


impl Queue {
    pub fn new(capacity: usize) -> Queue {
        Queue { capacity, messages: VecDeque::new() }
    }


    pub fn push(&mut self, msg: Message) -> bool {
        if self.messages.len() >= self.capacity {
            return false;
        }
        self.messages.push_back(msg);
        true
    }


    pub fn pop(&mut self) -> Option<Message> {
        self.messages.pop_front()
    }


    pub fn peek(&self) -> Option<&Message> {
        self.messages.front()
    }


    pub fn len(&self) -> usize {
        self.messages.len()
    }


    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
    }


    pub fn retain_fresh(&mut self, now_ns: u64) -> Vec<Message> {
        let mut ready = Vec::new();
        let mut retained = VecDeque::new();
        while let Some(m) = self.messages.pop_front() {
            if m.ready_ns <= now_ns {
                ready.push(m);
            } else {
                retained.push_back(m);
            }
        }
        self.messages = retained;
        ready
    }
}


/// The loopback transport: all queues, keyed by (from, to).
#[derive(Debug)]
pub struct Loopback {
    queues: BTreeMap<(String, String), Queue>,
    capacity: usize,
}


impl Loopback {
    pub fn new() -> Loopback {
        Loopback::with_capacity(DEFAULT_CAPACITY)
    }


    pub fn with_capacity(capacity: usize) -> Loopback {
        Loopback { queues: BTreeMap::new(), capacity }
    }


    fn queue_for(&mut self, from: &str, to: &str) -> &mut Queue {
        self.queues
            .entry((from.to_string(), to.to_string()))
            .or_insert_with(|| Queue::new(self.capacity))
    }


    /// Send a message from `from` to `to` at logical time `now_ns`,
    /// with an optional delay before it becomes deliverable.
    pub fn send(&mut self, from: &str, to: &str, payload: Vec<u8>, now_ns: u64, delay_ns: u64) -> bool {
        let msg = Message {
            from: from.to_string(),
            to: to.to_string(),
            payload,
            sent_ns: now_ns,
            ready_ns: now_ns.saturating_add(delay_ns),
        };
        self.queue_for(from, to).push(msg)
    }


    /// All messages that are ready for delivery at `now_ns`, across all
    /// queues. The scheduler picks from this set according to its strategy.
    pub fn ready(&mut self, now_ns: u64) -> Vec<Message> {
        let mut out = Vec::new();
        for queue in self.queues.values_mut() {
            out.extend(queue.retain_fresh(now_ns));
        }
        out
    }


    /// Drain a specific queue (deliver all regardless of readiness).
    pub fn drain(&mut self, from: &str, to: &str) -> Vec<Message> {
        match self.queues.get_mut(&(from.to_string(), to.to_string())) {
            Some(q) => {
                let mut out = Vec::new();
                while let Some(m) = q.pop() {
                    out.push(m);
                }
                out
            }
            None => Vec::new(),
        }
    }


    /// Drop all messages from a specific source (censorship).
    pub fn drop_from(&mut self, from: &str) -> usize {
        let mut dropped = 0;
        for ((f, _), q) in self.queues.iter_mut() {
            if f == from {
                dropped += q.len();
                *q = Queue::new(self.capacity);
            }
        }
        dropped
    }


    /// Drop messages between a specific pair.
    pub fn drop_pair(&mut self, from: &str, to: &str) -> usize {
        match self.queues.get_mut(&(from.to_string(), to.to_string())) {
            Some(q) => {
                let n = q.len();
                *q = Queue::new(self.capacity);
                n
            }
            None => 0,
        }
    }


    /// Total messages in flight.
    pub fn total_len(&self) -> usize {
        self.queues.values().map(|q| q.len()).sum()
    }


    pub fn is_empty(&self) -> bool {
        self.total_len() == 0
    }


    /// Register a pair (pre-allocates the queue).
    pub fn connect(&mut self, from: &str, to: &str) {
        self.queue_for(from, to);
    }
}


impl Default for Loopback {
    fn default() -> Self {
        Loopback::new()
    }
}


#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn send_and_ready() {
        let mut lb = Loopback::new();
        lb.connect("a", "b");
        assert!(lb.send("a", "b", vec![1], 0, 0));
        let ready = lb.ready(0);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].from, "a");
        assert_eq!(ready[0].to, "b");
        assert!(lb.is_empty());
    }


    #[test]
    fn delayed_messages() {
        let mut lb = Loopback::new();
        lb.send("a", "b", vec![1], 0, 1_000);
        assert!(lb.ready(0).is_empty(), "not ready yet");
        assert_eq!(lb.total_len(), 1, "still in flight");
        let ready = lb.ready(1_000);
        assert_eq!(ready.len(), 1);
        assert!(lb.is_empty());
    }


    #[test]
    fn capacity_bounded() {
        let mut lb = Loopback::with_capacity(2);
        assert!(lb.send("a", "b", vec![1], 0, 0));
        assert!(lb.send("a", "b", vec![2], 0, 0));
        assert!(!lb.send("a", "b", vec![3], 0, 0), "at capacity");
        assert_eq!(lb.total_len(), 2);
    }


    #[test]
    fn drop_from_censors() {
        let mut lb = Loopback::new();
        lb.send("a", "b", vec![1], 0, 0);
        lb.send("a", "c", vec![2], 0, 0);
        lb.send("b", "c", vec![3], 0, 0);
        let dropped = lb.drop_from("a");
        assert_eq!(dropped, 2);
        assert_eq!(lb.total_len(), 1);
        let ready = lb.ready(0);
        assert_eq!(ready.len(), 1);
        assert_eq!(ready[0].from, "b");
    }


    #[test]
    fn drain_pair() {
        let mut lb = Loopback::new();
        lb.send("a", "b", vec![1], 0, 0);
        lb.send("a", "b", vec![2], 0, 0);
        lb.send("b", "a", vec![3], 0, 0);
        let drained = lb.drain("a", "b");
        assert_eq!(drained.len(), 2);
        assert_eq!(lb.total_len(), 1);
    }
}
File: tests/nerv-testkit/src/scheduler.rs 
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
    DropSource { source: String },
    /// Hold messages from the named source for `delay_ns` before delivery (T5/T6).
    DelaySource { source: String, delay_ns: u64 },
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
            let source = source.clone();
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


#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn in_order_delivery() {
        let mut s = Scheduler::new(ScheduleStrategy::InOrder);
        s.transport.connect("a", "c");
        s.transport.connect("b", "c");
        s.transport.send("a", "c", vec![1], 0, 0);
        s.transport.send("b", "c", vec![2], 0, 0);


        let (r1, m1) = s.step();
        assert!(matches!(r1, StepResult::Delivered { ref from, .. } if from == "a"));
        assert!(m1.is_some());


        let (r2, m2) = s.step();
        assert!(matches!(r2, StepResult::Delivered { ref from, .. } if from == "b"));
        assert!(m2.is_some());


        let (r3, _) = s.step();
        assert!(matches!(r3, StepResult::Idle));
    }


    #[test]
    fn reverse_delivery() {
        let mut s = Scheduler::new(ScheduleStrategy::Reverse);
        s.transport.send("a", "c", vec![1], 0, 0);
        s.transport.send("b", "c", vec![2], 0, 0);


        let (r, m) = s.step();
        assert!(matches!(r, StepResult::Delivered { ref from, .. } if from == "b"));
        assert_eq!(m.unwrap().payload, vec![2]);
    }


    #[test]
    fn round_robin_fairness() {
        let mut s = Scheduler::new(ScheduleStrategy::RoundRobin);
        for i in 0..10 {
            s.transport.send("a", "c", vec![i], 0, 0);
            s.transport.send("b", "c", vec![i], 0, 0);
        }
        let mut counts = std::collections::BTreeMap::new();
        for _ in 0..20 {
            let (r, _) = s.step();
            if let StepResult::Delivered { from, .. } = r {
                *counts.entry(from).or_insert(0) += 1;
            }
        }
        assert_eq!(counts.get("a"), counts.get("b"), "round-robin is fair");
        assert_eq!(counts.get("a"), Some(&10));
    }


    #[test]
    fn drop_source_censors() {
        let mut s = Scheduler::new(ScheduleStrategy::DropSource { source: "a".into() });
        s.transport.send("a", "c", vec![1], 0, 0);
        s.transport.send("b", "c", vec![2], 0, 0);


        let (r, _) = s.step();
        // "a" is dropped; "b" is delivered.
        assert!(matches!(r, StepResult::Delivered { ref from, .. } if from == "b"));
        assert_eq!(s.transport.total_len(), 0, "everything else delivered or dropped");
    }


    #[test]
    fn delay_source_holds_then_releases() {
        let mut s = Scheduler::new(ScheduleStrategy::DelaySource {
            source: "slow".into(),
            delay_ns: 1_000_000, // 1 ms in logical time.
        });
        s.transport.send("fast", "c", vec![1], 0, 0);
        s.transport.send("slow", "c", vec![2], 0, 0);


        // Step 1: "fast" is delivered; "slow" is re-queued with delay.
        let (r1, _) = s.step();
        assert!(matches!(r1, StepResult::Delivered { ref from, .. } if from == "fast"));


        // Step 2: "slow" was delayed by 1ms = 1_000_000 ns; after ~1M
        // ticks it becomes ready. Let's verify it's not delivered at tick 2.
        let (r2, _) = s.step();
        // The slow message is either idle (not yet ready) or still delayed.
        // The delay is 1_000_000 ns, and we've only ticked twice.
        assert!(
            matches!(r2, StepResult::Idle) || matches!(r2, StepResult::Delivered { ref from, .. } if from != "slow"),
            "slow must not be delivered yet: {r2:?}"
        );


        // Fast-forward the clock.
        s.clock.tick_ms(2);
        let (r3, _) = s.step();
        assert!(matches!(r3, StepResult::Delivered { ref from, .. } if from == "slow"));
    }


    #[test]
    fn drop_permille() {
        let mut s = Scheduler::new(ScheduleStrategy::DropPermille { permille: 500, seed: 42 });
        let mut delivered = 0;
        let mut dropped = 0;
        for i in 0..100 {
            s.transport.send("a", "c", vec![i], 0, 0);
            let (r, _) = s.step();
            match r {
                StepResult::Delivered { .. } => delivered += 1,
                StepResult::Dropped { .. } => dropped += 1,
                StepResult::Idle => {}
            }
        }
        assert!(delivered + dropped == 100);
        assert!(delivered > 30 && delivered < 70, "roughly 50%: {delivered}");
    }
}
