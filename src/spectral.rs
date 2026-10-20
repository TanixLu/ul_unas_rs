//! UL-UNAS's periodic Hann STFT with normalized 50% overlap-add.

pub(super) use crate::fft::Complex;
use crate::{FRAME_SIZE, fft::Fft};

pub(super) struct Stft {
    window: [f32; 512],
    norm: [f32; FRAME_SIZE],
    fft: Fft,
    previous: [f32; FRAME_SIZE],
    overlap: [f32; FRAME_SIZE],
    time: [f32; 512],
}

impl Stft {
    pub(super) fn new() -> Self {
        let window: [f32; 512] = std::array::from_fn(|i| {
            (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / 512.0).cos()) as f32
        });
        Self {
            norm: std::array::from_fn(|i| {
                1.0 / (window[i] * window[i] + window[i + 256] * window[i + 256])
            }),
            window,
            fft: Fft::new(),
            previous: [0.0; FRAME_SIZE],
            overlap: [0.0; FRAME_SIZE],
            time: [0.0; 512],
        }
    }

    pub(super) fn reset(&mut self) {
        self.fft.reset();
        self.previous.fill(0.0);
        self.overlap.fill(0.0);
        self.time.fill(0.0);
    }

    pub(super) fn analyze(&mut self, input: &[f32; FRAME_SIZE], spectrum: &mut [Complex; 257]) {
        for (i, &sample) in input.iter().enumerate() {
            self.time[i] = self.previous[i] * self.window[i];
            self.time[i + 256] = sample * self.window[i + 256];
        }
        self.previous.copy_from_slice(input);
        self.fft.forward(&self.time, spectrum);
    }

    pub(super) fn synthesize(
        &mut self,
        spectrum: &mut [Complex; 257],
        output: &mut [f32; FRAME_SIZE],
    ) {
        spectrum[0].im = 0.0;
        spectrum[256].im = 0.0;
        self.fft.inverse(spectrum, &mut self.time);
        for (i, sample) in output.iter_mut().enumerate() {
            *sample = (self.overlap[i] + self.time[i] * (self.window[i] / 512.0)) * self.norm[i];
            self.overlap[i] = self.time[i + 256] * (self.window[i + 256] / 512.0);
        }
    }
}
