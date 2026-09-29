//! The deterministic clock (erratum 193): a logical nanosecond counter
//! advanced only by the scheduler. No wall-clock reads, ever.


/// The deterministic clock. `tick()` advances by one logical step;
/// `advance_to()` jumps forward (for timeout scenarios).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DetClock {
    now_ns: u64,
}


impl DetClock {
    pub fn new() -> DetClock {
        DetClock::default()
    }


    pub fn now_ns(&self) -> u64 {
        self.now_ns
    }


    pub fn now_ms(&self) -> u64 {
        self.now_ns / 1_000_000
    }


    pub fn now_secs(&self) -> u64 {
        self.now_ns / 1_000_000_000
    }


    pub fn tick(&mut self) {
        self.now_ns = self.now_ns.saturating_add(1);
    }


    pub fn tick_ms(&mut self, ms: u64) {
        self.now_ns = self.now_ns.saturating_add(ms * 1_000_000);
    }


    pub fn tick_secs(&mut self, secs: u64) {
        self.now_ns = self.now_ns.saturating_add(secs * 1_000_000_000);
    }


    pub fn advance_to_ns(&mut self, target: u64) {
        if target > self.now_ns {
            self.now_ns = target;
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn clock_advances_only_explicitly() {
        let mut c = DetClock::new();
        assert_eq!(c.now_ns(), 0);
        c.tick();
        assert_eq!(c.now_ns(), 1);
        c.tick_ms(5);
        assert_eq!(c.now_ns(), 5_000_001);
        c.tick_secs(1);
        assert_eq!(c.now_ns(), 1_000_000_001);
        assert_eq!(c.now_ms(), 1000);
        assert_eq!(c.now_secs(), 1);
    }


    #[test]
    fn advance_to_is_monotone() {
        let mut c = DetClock::new();
        c.tick_ms(100);
        c.advance_to_ns(50_000_000);
        assert_eq!(c.now_ns(), 100_000_000, "going back is a no-op");
        c.advance_to_ns(200_000_000);
        assert_eq!(c.now_ns(), 200_000_000);
    }


    #[test]
    fn saturates() {
        let mut c = DetClock { now_ns: u64::MAX - 1 };
        c.tick();
        assert_eq!(c.now_ns(), u64::MAX);
        c.tick();
        assert_eq!(c.now_ns(), u64::MAX, "saturating");
    }
}
