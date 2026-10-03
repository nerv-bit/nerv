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
