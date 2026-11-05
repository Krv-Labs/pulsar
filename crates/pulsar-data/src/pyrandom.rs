//! CPython's `random.Random` (MT19937), reproduced bit-for-bit.
//!
//! Pulsar v0.5 imputed string columns with `sample_categorical` through
//! `random.Random(seed).choices(...)`. The golden fixtures (#60) freeze those
//! draws, so the port keeps the same generator rather than switching RNGs and
//! regenerating fixtures. Only the pieces `choices` needs are implemented:
//! integer seeding (`init_by_array`), `random()`, and the cumulative-weight
//! bisection `choices` does.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;

/// A Mersenne Twister seeded the way `random.Random(int)` seeds it.
pub struct PyRandom {
    mt: [u32; N],
    index: usize,
}

impl PyRandom {
    /// Equivalent to `random.Random(seed)` for a non-negative integer seed.
    pub fn new(seed: u64) -> Self {
        // CPython splits abs(seed) into 32-bit little-endian words and calls
        // init_by_array; zero becomes the single-word key [0].
        let mut key = Vec::new();
        let mut n = seed;
        loop {
            key.push((n & 0xffff_ffff) as u32);
            n >>= 32;
            if n == 0 {
                break;
            }
        }
        let mut rng = Self {
            mt: [0; N],
            index: N,
        };
        rng.init_by_array(&key);
        rng
    }

    /// Equivalent to numpy's legacy `np.random.RandomState(seed)`: the
    /// same generator seeded with `init_genrand` instead of `init_by_array`.
    pub fn numpy_legacy(seed: u32) -> Self {
        let mut rng = Self {
            mt: [0; N],
            index: N,
        };
        rng.init_genrand(seed);
        rng
    }

    fn init_genrand(&mut self, s: u32) {
        self.mt[0] = s;
        for i in 1..N {
            let prev = self.mt[i - 1];
            self.mt[i] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        self.index = N;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19_650_218);
        let mut i = 1usize;
        let mut j = 0usize;
        let len = key.len();
        for _ in 0..N.max(len) {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
            if j >= len {
                j = 0;
            }
        }
        for _ in 0..N - 1 {
            let prev = self.mt[i - 1];
            self.mt[i] = (self.mt[i] ^ (prev ^ (prev >> 30)).wrapping_mul(1_566_083_941))
                .wrapping_sub(i as u32);
            i += 1;
            if i >= N {
                self.mt[0] = self.mt[N - 1];
                i = 1;
            }
        }
        self.mt[0] = 0x8000_0000;
    }

    fn generate(&mut self) {
        for kk in 0..N {
            let y = (self.mt[kk] & UPPER_MASK) | (self.mt[(kk + 1) % N] & LOWER_MASK);
            let mut v = self.mt[(kk + M) % N] ^ (y >> 1);
            if y & 1 == 1 {
                v ^= MATRIX_A;
            }
            self.mt[kk] = v;
        }
        self.index = 0;
    }

    /// `genrand_uint32`: the next raw 32-bit output.
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            self.generate();
        }
        let mut y = self.mt[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// `random()`: a float in `[0, 1)` with 53 random bits.
    pub fn random(&mut self) -> f64 {
        let a = (self.next_u32() >> 5) as f64;
        let b = (self.next_u32() >> 6) as f64;
        (a * 67_108_864.0 + b) * (1.0 / 9_007_199_254_740_992.0)
    }

    /// One draw of `random.choices(population, weights=weights)`: the index
    /// into `population`. `cum_weights` is the running sum of the weights.
    pub fn choice_index(&mut self, cum_weights: &[f64]) -> usize {
        let n = cum_weights.len();
        debug_assert!(n > 0, "choices needs at least one weight");
        let total = cum_weights[n - 1];
        let x = self.random() * total;
        // bisect.bisect_right(cum_weights, x, 0, n - 1)
        let (mut lo, mut hi) = (0usize, n - 1);
        while lo < hi {
            let mid = (lo + hi) / 2;
            if x < cum_weights[mid] {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo
    }
}

/// Running sums of `weights`, as `random.choices` computes them.
pub fn cumulative(weights: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(weights.len());
    let mut acc = 0.0;
    for &w in weights {
        acc += w;
        out.push(acc);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference values from CPython 3.12 `random.Random(seed).random()`.
    #[test]
    fn random_matches_cpython() {
        let mut r = PyRandom::new(7);
        let got: Vec<f64> = (0..4).map(|_| r.random()).collect();
        assert_eq!(
            got,
            [
                0.32383276483316237,
                0.15084917392450192,
                0.6509344730398537,
                0.07243628666754276
            ]
        );
        let mut r = PyRandom::new(42);
        assert_eq!(r.random(), 0.6394267984578837);
        assert_eq!(r.random(), 0.025010755222666936);
        assert_eq!(r.random(), 0.27502931836911926);
        let mut r = PyRandom::new(0);
        assert_eq!(r.random(), 0.8444218515250481);
        assert_eq!(r.random(), 0.7579544029403025);
        // Multi-word key (seed > 2^32).
        let mut r = PyRandom::new((1u64 << 40) + 5);
        assert_eq!(r.random(), 0.5043802970418443);
        assert_eq!(r.random(), 0.2686044399723282);
        assert_eq!(r.random(), 0.9257865475671585);
    }

    #[test]
    fn choices_match_cpython() {
        // random.Random(7).choices(["male", "female"], weights=[168, 165], k=11)
        let mut r = PyRandom::new(7);
        let cum = cumulative(&[168.0, 165.0]);
        let got: Vec<usize> = (0..11).map(|_| r.choice_index(&cum)).collect();
        assert_eq!(got, [0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0]);
        // random.Random(3).choices(["a", "b", "c"], weights=[1, 5, 2], k=8)
        let mut r = PyRandom::new(3);
        let cum = cumulative(&[1.0, 5.0, 2.0]);
        let got: Vec<usize> = (0..8).map(|_| r.choice_index(&cum)).collect();
        assert_eq!(got, [1, 1, 1, 1, 1, 0, 0, 2]);
    }
}
