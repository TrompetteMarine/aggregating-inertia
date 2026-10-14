//! xoshiro256** seeded by splitmix64, uniforms on (0, 1) and Box-Muller normals with caching:
//! the generator of the reference C Monte Carlo (mc.c), reproduced bit for bit.

#[inline]
fn rotl(x: u64, k: u32) -> u64 {
    x.rotate_left(k)
}

#[inline]
pub fn splitmix64(x: &mut u64) -> u64 {
    *x = x.wrapping_add(0x9e3779b97f4a7c15);
    let mut z = *x;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

#[derive(Clone, Debug)]
pub struct Xoshiro {
    s: [u64; 4],
    has: bool,
    cache: f64,
}

impl Xoshiro {
    /// The per-path stream of mc.c: splitmix64 seeded with seed ^ (golden * (n + 1)).
    pub fn for_path(seed: u64, n: u64) -> Self {
        let mut sm = seed ^ 0x9E3779B97F4A7C15u64.wrapping_mul(n.wrapping_add(1));
        let s = [splitmix64(&mut sm), splitmix64(&mut sm), splitmix64(&mut sm), splitmix64(&mut sm)];
        Xoshiro { s, has: false, cache: 0.0 }
    }

    pub fn from_seed(seed: u64) -> Self {
        let mut sm = seed;
        let s = [splitmix64(&mut sm), splitmix64(&mut sm), splitmix64(&mut sm), splitmix64(&mut sm)];
        Xoshiro { s, has: false, cache: 0.0 }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = rotl(s[1].wrapping_mul(5), 7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = rotl(s[3], 45);
        result
    }

    /// Uniform on (0, 1): ((x >> 11) + 0.5) 2^-53.
    #[inline]
    pub fn unif(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 0.5) * (1.0 / 9007199254740992.0)
    }

    /// Standard normal by Box-Muller, the second variate cached.
    #[inline]
    pub fn gauss(&mut self) -> f64 {
        if self.has {
            self.has = false;
            return self.cache;
        }
        let u1 = self.unif();
        let u2 = self.unif();
        let rad = (-2.0 * u1.ln()).sqrt();
        let ang = 6.283185307179586476925 * u2;
        self.cache = rad * ang.sin();
        self.has = true;
        rad * ang.cos()
    }

    /// Exponential with unit mean.
    #[inline]
    pub fn exp1(&mut self) -> f64 {
        -self.unif().ln()
    }

    /// Jump the stream ahead by 2^128 calls (xoshiro256 jump), for independent substreams.
    pub fn jump(&mut self) {
        const JUMP: [u64; 4] = [0x180ec6d33cfd0aba, 0xd5a61266f0c9392c, 0xa9582618e03fc9aa, 0x39abdc4529b1661c];
        let mut s0 = 0u64;
        let mut s1 = 0u64;
        let mut s2 = 0u64;
        let mut s3 = 0u64;
        for j in JUMP {
            for b in 0..64 {
                if j & (1u64 << b) != 0 {
                    s0 ^= self.s[0];
                    s1 ^= self.s[1];
                    s2 ^= self.s[2];
                    s3 ^= self.s[3];
                }
                self.next_u64();
            }
        }
        self.s = [s0, s1, s2, s3];
        self.has = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn moments() {
        let mut r = Xoshiro::from_seed(1);
        let n = 200_000;
        let (mut s1, mut s2) = (0.0, 0.0);
        for _ in 0..n {
            let z = r.gauss();
            s1 += z;
            s2 += z * z;
        }
        assert!((s1 / n as f64).abs() < 0.01);
        assert!((s2 / n as f64 - 1.0).abs() < 0.01);
    }
}

/// numpy's `default_rng(seed)` for a seed below 2^32 (SeedSequence followed by PCG64 with the XSL-RR output) and
/// its `random()`. Used only for the horizontal jitter of Figure 6, so that the figure data match the reference
/// pipeline's digit for digit.
#[derive(Clone, Debug)]
pub struct NumpyPcg64 {
    state: u128,
    inc: u128,
}

impl NumpyPcg64 {
    const MULT: u128 = 0x2360_ED05_1FC6_5DA4_4385_DF64_9FCC_F645;

    pub fn default_rng(seed: u32) -> Self {
        // SeedSequence: hash the entropy into a pool of four words, then draw the state words from the pool
        const INIT_A: u32 = 0x43b0_d7e5;
        const MULT_A: u32 = 0x931e_8875;
        const INIT_B: u32 = 0x8b51_f9dd;
        const MULT_B: u32 = 0x58f3_8ded;
        const MIX_MULT_L: u32 = 0xca01_f9dd;
        const MIX_MULT_R: u32 = 0x4973_f715;
        fn hashmix(v: u32, hc: &mut u32) -> u32 {
            let mut v = v ^ *hc;
            *hc = hc.wrapping_mul(MULT_A);
            v = v.wrapping_mul(*hc);
            v ^ (v >> 16)
        }
        fn mix(x: u32, y: u32) -> u32 {
            let r = MIX_MULT_L.wrapping_mul(x).wrapping_sub(MIX_MULT_R.wrapping_mul(y));
            r ^ (r >> 16)
        }
        let mut hc = INIT_A;
        let mut pool = [0u32; 4];
        for (i, w) in pool.iter_mut().enumerate() {
            *w = hashmix(if i == 0 { seed } else { 0 }, &mut hc);
        }
        for s in 0..4 {
            for d in 0..4 {
                if s != d {
                    let h = hashmix(pool[s], &mut hc);
                    pool[d] = mix(pool[d], h);
                }
            }
        }
        let mut hb = INIT_B;
        let mut w = [0u32; 8];
        for (i, x) in w.iter_mut().enumerate() {
            let mut v = pool[i % 4] ^ hb;
            hb = hb.wrapping_mul(MULT_B);
            v = v.wrapping_mul(hb);
            *x = v ^ (v >> 16);
        }
        let word = |k: usize| (w[2 * k] as u64 | ((w[2 * k + 1] as u64) << 32)) as u128;
        let (initstate, initseq) = ((word(0) << 64) | word(1), (word(2) << 64) | word(3));
        let mut r = NumpyPcg64 { state: 0, inc: (initseq << 1) | 1 };
        r.step();
        r.state = r.state.wrapping_add(initstate);
        r.step();
        r
    }

    #[inline]
    fn step(&mut self) {
        self.state = self.state.wrapping_mul(Self::MULT).wrapping_add(self.inc);
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.step();
        let s = self.state;
        (((s >> 64) as u64) ^ (s as u64)).rotate_right((s >> 122) as u32)
    }

    /// `Generator.random()`: (x >> 11) 2^-53, on [0, 1).
    #[inline]
    pub fn random(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / 9007199254740992.0)
    }
}

#[cfg(test)]
mod numpy_tests {
    use super::NumpyPcg64;

    /// numpy.random.default_rng(seed).random(5), from numpy 2.4.4.
    #[test]
    fn numpy_default_rng() {
        for (seed, want) in [
            (3u32, [0.08564916714362436, 0.2368105065960997, 0.8012744652063969, 0.5821620360643678, 0.09412864224039919]),
            (5u32, [0.8050029237453802, 0.8079407897364937, 0.515325561042142, 0.2858013800881416, 0.053930702381656426]),
        ] {
            let mut r = NumpyPcg64::default_rng(seed);
            for w in want {
                assert_eq!(r.random(), w);
            }
        }
    }
}
