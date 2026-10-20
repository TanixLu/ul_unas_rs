//! Fixed 512-real FFT using a packed 256-complex radix-2 transform.
//! Both directions are unnormalized, matching realfft's convention.

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Complex {
    pub re: f32,
    pub im: f32,
}

impl Complex {
    pub const fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    pub fn norm(self) -> f32 {
        self.re.hypot(self.im)
    }
}

impl std::ops::MulAssign<f32> for Complex {
    fn mul_assign(&mut self, rhs: f32) {
        self.re *= rhs;
        self.im *= rhs;
    }
}

pub(crate) struct Fft {
    twiddle: [Complex; 256],
    work: [Complex; 256],
}

impl Fft {
    pub fn new() -> Self {
        Self {
            twiddle: std::array::from_fn(|k| {
                let angle = -std::f64::consts::TAU * k as f64 / 512.0;
                Complex::new(angle.cos() as f32, angle.sin() as f32)
            }),
            work: [Complex::new(0.0, 0.0); 256],
        }
    }

    pub fn reset(&mut self) {
        self.work.fill(Complex::new(0.0, 0.0));
    }

    fn complex_forward(&mut self) {
        for i in 0..256_usize {
            let j = i.reverse_bits() >> (usize::BITS - 8);
            if i < j {
                self.work.swap(i, j);
            }
        }
        for stage in 1..=8 {
            let size = 1 << stage;
            let half = size / 2;
            for start in (0..256).step_by(size) {
                for k in 0..half {
                    let a = self.work[start + k];
                    let b = self.work[start + k + half];
                    let w = self.twiddle[k * (512 / size)];
                    let re = b.re * w.re - b.im * w.im;
                    let im = b.re * w.im + b.im * w.re;
                    self.work[start + k] = Complex::new(a.re + re, a.im + im);
                    self.work[start + k + half] = Complex::new(a.re - re, a.im - im);
                }
            }
        }
    }

    pub fn forward(&mut self, time: &[f32; 512], spectrum: &mut [Complex; 257]) {
        for (z, pair) in self.work.iter_mut().zip(time.as_chunks::<2>().0) {
            *z = Complex::new(pair[0], pair[1]);
        }
        self.complex_forward();
        let z = self.work[0];
        spectrum[0] = Complex::new(z.re + z.im, 0.0);
        spectrum[256] = Complex::new(z.re - z.im, 0.0);
        for (k, bin) in spectrum.iter_mut().enumerate().take(256).skip(1) {
            let a = self.work[k];
            let b = self.work[256 - k];
            let w = self.twiddle[k];
            let dr = a.re - b.re;
            let di = a.im + b.im;
            *bin = Complex::new(
                0.5 * (a.re + b.re + w.re * di + w.im * dr),
                0.5 * (a.im - b.im - w.re * dr + w.im * di),
            );
        }
    }

    pub fn inverse(&mut self, spectrum: &[Complex; 257], time: &mut [f32; 512]) {
        // Reconstruct the packed transform and conjugate it so the same
        // complex forward kernel computes the inverse. DC/Nyquist are real.
        self.work[0] = Complex::new(
            0.5 * (spectrum[0].re + spectrum[256].re),
            -0.5 * (spectrum[0].re - spectrum[256].re),
        );
        for k in 1..256 {
            let a = spectrum[k];
            let b = spectrum[256 - k];
            let w = self.twiddle[k];
            let dr = a.re - b.re;
            let di = a.im + b.im;
            self.work[k] = Complex::new(
                0.5 * (a.re + b.re + w.im * dr - w.re * di),
                -0.5 * (a.im - b.im + w.re * dr + w.im * di),
            );
        }
        self.complex_forward();
        // A 256-complex inverse has gain 256. Double it to keep the
        // original unnormalized 512-real inverse gain.
        for (pair, z) in time.as_chunks_mut::<2>().0.iter_mut().zip(&self.work) {
            pair[0] = 2.0 * z.re;
            pair[1] = -2.0 * z.im;
        }
    }
}
