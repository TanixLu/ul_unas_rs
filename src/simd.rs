//! Single-threaded FP32 kernels, selected once during stream construction.
//! Deliberately uses separate multiply/add operations, preserving the model's
//! original non-fused arithmetic. Other architectures use portable scalar code.

type Linear = unsafe fn(&[f32], &[f32], &[f32], &mut [f32]);
type Norm = unsafe fn(&mut [f32], f32, f32);

#[derive(Clone, Copy)]
pub(super) struct Kernels {
    linear: Linear,
    norm: Norm,
}

impl Kernels {
    pub(super) fn new() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if std::is_x86_feature_detected!("avx2") {
            return Self {
                linear: x86::linear,
                norm: x86::norm,
            };
        }
        Self {
            linear: scalar_linear,
            norm: scalar_norm,
        }
    }

    pub(super) fn linear(self, weight: &[f32], input: &[f32], bias: &[f32], output: &mut [f32]) {
        assert!(!input.is_empty());
        assert_eq!(weight.len(), output.len() * input.len());
        assert_eq!(bias.len(), output.len());
        // SAFETY: the constructor checks CPU support, and dimensions above
        // ensure that each matrix row and output element are in bounds.
        unsafe { (self.linear)(weight, input, bias, output) }
    }

    pub(super) fn norm(self, data: &mut [f32], scale: f32, bias: f32) {
        // SAFETY: the constructor checks CPU support; kernels handle tails.
        unsafe { (self.norm)(data, scale, bias) }
    }
}

fn scalar_linear(weight: &[f32], input: &[f32], bias: &[f32], output: &mut [f32]) {
    for ((row, &bias), output) in weight.chunks_exact(input.len()).zip(bias).zip(output) {
        let mut sum = 0.0;
        for (&a, &b) in row.iter().zip(input) {
            sum += a * b;
        }
        *output = bias + sum;
    }
}

fn scalar_norm(data: &mut [f32], scale: f32, bias: f32) {
    for value in data {
        *value = *value * scale + bias;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn linear(weight: &[f32], input: &[f32], bias: &[f32], output: &mut [f32]) {
        let vector_len = input.len() / 8 * 8;
        for ((row, &bias), output) in weight.chunks_exact(input.len()).zip(bias).zip(output) {
            let mut sum = _mm256_setzero_ps();
            for i in (0..vector_len).step_by(8) {
                // SAFETY: i + 8 <= input.len() == row.len(); unaligned loads.
                let (a, b) = unsafe {
                    (
                        _mm256_loadu_ps(row.as_ptr().add(i)),
                        _mm256_loadu_ps(input.as_ptr().add(i)),
                    )
                };
                sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
            }
            let sum = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps::<1>(sum));
            let sum = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
            let sum = _mm_add_ss(sum, _mm_shuffle_ps::<0x55>(sum, sum));
            let mut value = bias + _mm_cvtss_f32(sum);
            for i in vector_len..input.len() {
                value += row[i] * input[i];
            }
            *output = value;
        }
    }

    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn norm(data: &mut [f32], scale: f32, bias: f32) {
        let scale_v = _mm256_set1_ps(scale);
        let bias_v = _mm256_set1_ps(bias);
        let (chunks, tail) = data.as_chunks_mut::<8>();
        for chunk in chunks {
            // SAFETY: chunks contain eight writable f32s; unaligned accesses.
            unsafe {
                let value = _mm256_loadu_ps(chunk.as_ptr());
                _mm256_storeu_ps(
                    chunk.as_mut_ptr(),
                    _mm256_add_ps(_mm256_mul_ps(value, scale_v), bias_v),
                );
            }
        }
        super::scalar_norm(tail, scale, bias);
    }
}
