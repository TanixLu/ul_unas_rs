//! Streaming UL-UNAS DNS3 speech denoising with no third-party dependencies.
//!
//! Input is normalized, finite mono PCM at 16 kHz, in frames of 256 samples.
//! Weights are embedded and shared by all instances. Construction allocates;
//! processing and reset perform no heap allocations, locks, or file I/O.
//!
//! ```
//! use ul_unas_rs::{Denoiser, FRAME_SIZE};
//!
//! let mut denoiser = Denoiser::new();
//! let mut frame = [0.0; FRAME_SIZE];
//! denoiser.process_frame(&mut frame);
//! denoiser.reset();
//! ```
#![deny(unsafe_op_in_unsafe_fn)]

mod fft;
mod network;
mod simd;
mod spectral;
mod weights;

use fft::Complex;
use network::Network;
use spectral::Stft;

/// Required input and output sample rate, in Hz.
pub const SAMPLE_RATE: u32 = 16_000;
/// Number of mono samples consumed and produced per call.
pub const FRAME_SIZE: usize = 256;
/// STFT delay in samples (16 ms), excluding caller buffering and resampling.
pub const ALGORITHM_DELAY: usize = 256;

/// One independent audio stream, including recurrent and overlap-add state.
///
/// Audio I/O, resampling, gain, and limiting are the caller's responsibility.
/// Use a separate instance for each channel. No CPU features are required;
/// AVX2 kernels are selected automatically when available on x86/x86-64.
pub struct Denoiser {
    network: Network,
    stft: Stft,
    spectrum: [Complex; 257],
}

impl Denoiser {
    /// Creates a stream with zeroed history and embedded DNS3 weights.
    pub fn new() -> Self {
        Self {
            network: Network::new(),
            stft: Stft::new(),
            spectrum: [Complex::new(0.0, 0.0); 257],
        }
    }

    /// Clears all neural and STFT history without allocating.
    /// Call before processing an unrelated stream.
    pub fn reset(&mut self) {
        self.network.reset();
        self.stft.reset();
        self.spectrum.fill(Complex::new(0.0, 0.0));
    }

    /// Replaces a frame of finite, normalized PCM with denoised audio.
    ///
    /// Each call consumes and produces exactly [`FRAME_SIZE`] samples.
    /// To emit the delayed tail at end of stream, process one extra zero frame.
    /// Output is not clipped or limited.
    pub fn process_frame(&mut self, frame: &mut [f32; FRAME_SIZE]) {
        self.stft.analyze(frame, &mut self.spectrum);
        self.network.denoise(&mut self.spectrum);
        self.stft.synthesize(&mut self.spectrum, frame);
    }

    /// Processes a frame into a separate output buffer, preserving the input.
    pub fn process(&mut self, input: &[f32; FRAME_SIZE], output: &mut [f32; FRAME_SIZE]) {
        self.stft.analyze(input, &mut self.spectrum);
        self.network.denoise(&mut self.spectrum);
        self.stft.synthesize(&mut self.spectrum, output);
    }
}

impl Default for Denoiser {
    fn default() -> Self {
        Self::new()
    }
}
