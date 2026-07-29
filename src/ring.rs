//! The far-end reference ring: a fixed-capacity buffer of mono `f32` samples
//! addressed by an absolute sample index that never wraps.
//!
//! The engine feeds every far-end sample here and reads aligned spans back out
//! of it. Addressing is by absolute index (total samples ever pushed), so the
//! alignment bookkeeping in the engine is plain integer arithmetic that is
//! immune to the ring's internal wrap. When the buffer is full a push overwrites
//! the oldest retained sample; the number of samples dropped that way is derived
//! from the absolute counter, so it needs no separate bookkeeping.

/// A fixed-capacity ring of far-end reference samples, addressed by absolute
/// index.
///
/// Absolute index `a` (with `0 <= a < next_abs`) is the `a`-th sample ever
/// pushed. The buffer retains only the most recent `capacity` samples: sample
/// `a` is held while `a >= next_abs - capacity`. A read of an index that has
/// been dropped (too old) or not yet been pushed (in the future) returns
/// [`None`], which the engine renders as a starved (silent) sample.
pub(crate) struct ReferenceRing {
    buf: Vec<f32>,
    capacity: usize,
    next_abs: u64,
}

impl ReferenceRing {
    /// Creates a ring that retains the most recent `capacity` samples.
    ///
    /// `capacity` must be greater than zero; the engine derives it from the
    /// validated configuration, so it is always well above zero in practice.
    pub(crate) fn new(capacity: usize) -> Self {
        assert!(
            capacity > 0,
            "reference ring capacity must be greater than zero"
        );
        Self {
            buf: vec![0.0; capacity],
            capacity,
            next_abs: 0,
        }
    }

    /// Appends `samples` to the ring, overwriting the oldest retained samples
    /// once the ring is full. Advances the absolute counter by `samples.len()`.
    pub(crate) fn push(&mut self, samples: &[f32]) {
        for &sample in samples {
            let slot = (self.next_abs % self.capacity as u64) as usize;
            self.buf[slot] = sample;
            self.next_abs += 1;
        }
    }

    /// Returns the sample at absolute index `abs`, or [`None`] if it has been
    /// dropped (too old) or has not been pushed yet (in the future).
    pub(crate) fn get(&self, abs: u64) -> Option<f32> {
        let oldest = self.oldest_retained();
        if abs >= oldest && abs < self.next_abs {
            Some(self.buf[(abs % self.capacity as u64) as usize])
        } else {
            None
        }
    }

    /// The absolute index of the next sample a push will occupy, i.e. the total
    /// number of samples ever pushed.
    pub(crate) fn next_abs(&self) -> u64 {
        self.next_abs
    }

    /// The ring's retained depth in samples: how far back from the frontier a
    /// read can still be served. An alignment whose next read sits further
    /// behind the frontier than this cannot be served by any retained sample,
    /// whatever the rest of the engine believes about it.
    pub(crate) fn capacity(&self) -> usize {
        self.capacity
    }

    /// The number of samples dropped to overflow since construction (or the last
    /// [`clear`](Self::clear)): the count of pushes beyond the ring's capacity.
    pub(crate) fn dropped(&self) -> u64 {
        self.next_abs.saturating_sub(self.capacity as u64)
    }

    /// Clears all retained samples and resets the absolute counter, returning the
    /// ring to its just-constructed state without reallocating.
    pub(crate) fn clear(&mut self) {
        self.next_abs = 0;
    }

    /// The absolute index of the oldest retained sample (equal to the dropped
    /// count). When the ring is empty this equals `next_abs`, so the retained
    /// window is empty.
    fn oldest_retained(&self) -> u64 {
        self.next_abs.saturating_sub(self.capacity as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pushed_samples_read_back_by_absolute_index() {
        let mut ring = ReferenceRing::new(8);
        ring.push(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(ring.get(0), Some(1.0));
        assert_eq!(ring.get(3), Some(4.0));
        assert_eq!(ring.get(4), None);
        assert_eq!(ring.next_abs(), 4);
        assert_eq!(ring.dropped(), 0);
    }

    #[test]
    fn reads_before_the_stream_and_past_the_frontier_are_none() {
        let mut ring = ReferenceRing::new(8);
        ring.push(&[1.0, 2.0]);
        // Not yet pushed (future).
        assert_eq!(ring.get(2), None);
        assert_eq!(ring.get(100), None);
        // Empty ring: every read is None.
        let empty = ReferenceRing::new(8);
        assert_eq!(empty.get(0), None);
    }

    #[test]
    fn overflow_drops_oldest_and_counts_them() {
        let mut ring = ReferenceRing::new(4);
        ring.push(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(ring.next_abs(), 6);
        assert_eq!(ring.dropped(), 2);
        // Absolute indices 0 and 1 were overwritten.
        assert_eq!(ring.get(0), None);
        assert_eq!(ring.get(1), None);
        // The most recent four remain, addressed by their original absolute index.
        assert_eq!(ring.get(2), Some(3.0));
        assert_eq!(ring.get(5), Some(6.0));
    }

    #[test]
    fn absolute_addressing_survives_the_internal_wrap() {
        let mut ring = ReferenceRing::new(3);
        ring.push(&[1.0, 2.0, 3.0]); // abs 0,1,2
        ring.push(&[4.0, 5.0]); // abs 3,4; drops abs 0,1
        assert_eq!(ring.dropped(), 2);
        assert_eq!(ring.get(2), Some(3.0));
        assert_eq!(ring.get(3), Some(4.0));
        assert_eq!(ring.get(4), Some(5.0));
        assert_eq!(ring.get(1), None);
    }

    #[test]
    fn clear_returns_the_ring_to_empty() {
        let mut ring = ReferenceRing::new(4);
        ring.push(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        ring.clear();
        assert_eq!(ring.next_abs(), 0);
        assert_eq!(ring.dropped(), 0);
        assert_eq!(ring.get(0), None);
        // Pushing after a clear addresses from absolute zero again.
        ring.push(&[9.0]);
        assert_eq!(ring.get(0), Some(9.0));
    }
}
