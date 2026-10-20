# ul_unas_rs

A pure Rust streaming speech denoising library for
[UL-UNAS DNS3](https://github.com/Xiaobin-Rong/ul-unas/tree/00f7c700da43d38347f30a6ccebd86fcbc798e07),
ported from VoiceBox. Uses `std` with no third-party dependencies.

- Processes 16 kHz mono audio in frames of 256 `f32` samples.
- Algorithmic delay: 256 samples (16 ms).
- Selects AVX2 kernels on supported x86/x86-64 CPUs, with a scalar fallback.
- Embedded FP32 weights shared by all instances.
- No allocations, locks, or file I/O during frame processing or reset.

```rust
use ul_unas_rs::{Denoiser, FRAME_SIZE};

let mut denoiser = Denoiser::new();
let mut frame = [0.0_f32; FRAME_SIZE]; // Fill with finite, normalized PCM samples.
denoiser.process_frame(&mut frame); // Denoised audio replaces the input.
denoiser.reset(); // Clear state before starting a new stream.
```

Audio I/O and resampling are handled by the caller. Use a separate denoiser for
each channel. Pad partial frames with zeros and process one extra zero frame at
end of stream to emit the delayed tail. Output is not clipped.
Use `denoiser.process(&input, &mut output)` to preserve the input.

Licensed under the [MIT License](LICENSE). See [NOTICE](NOTICE) for upstream code
and model attribution.
