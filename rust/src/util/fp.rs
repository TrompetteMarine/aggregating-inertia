//! Flush-to-zero / denormals-are-zero for the duration of a sweep, as the reference C kernel does
//! (masses in the far tails underflow into the subnormal range, where x86 arithmetic is slow).

#[cfg(target_arch = "x86_64")]
pub struct FtzGuard {
    saved: u32,
}

#[cfg(target_arch = "x86_64")]
impl FtzGuard {
    #[allow(deprecated)]
    pub fn new() -> Self {
        use std::arch::x86_64::{_mm_getcsr, _mm_setcsr};
        // SAFETY: reading and writing MXCSR only changes the rounding of subnormal results on this
        // thread; the previous value is restored on drop.
        let saved = unsafe { _mm_getcsr() };
        unsafe { _mm_setcsr(saved | 0x8040) };
        FtzGuard { saved }
    }
}

#[cfg(target_arch = "x86_64")]
impl Drop for FtzGuard {
    #[allow(deprecated)]
    fn drop(&mut self) {
        use std::arch::x86_64::_mm_setcsr;
        unsafe { _mm_setcsr(self.saved) };
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub struct FtzGuard;

#[cfg(not(target_arch = "x86_64"))]
impl FtzGuard {
    pub fn new() -> Self {
        FtzGuard
    }
}

impl Default for FtzGuard {
    fn default() -> Self {
        Self::new()
    }
}
