use hound;
use std::env;
use std::process;
// use rayon::prelude::*;
const SAMPLE_RATE: u32 = 44100;
const DURATION_SECS: usize = 5;
const N_SAMPLES: usize = SAMPLE_RATE as usize * DURATION_SECS;
const DT: f32 = 1.0 / SAMPLE_RATE as f32;

const HAMMER_MASS: f32 = 0.95;
const HAMMER_K: f32 = 250.0;
const HAMMER_B: f32 = 0.8;
const TENSION: f32 = 0.64;
const DECAY: f32 = 0.9995;
const CLAMP: f32 = 10.0;

struct Mode {
    omega: f32,  // 角周波数 ω = 2π f
    zeta: f32,   // 減衰比 ζ
    k: f32,      // 入力結合係数
    y_prev: f32, // y[n-1]
    y_curr: f32, // y[n]
    a1: f32,     // 差分係数1
    a2: f32,     // 差分係数2
}

impl Mode {
    fn new(f: f32, zeta: f32, k: f32, fs: f32) -> Self {
        let omega = 2.0 * std::f32::consts::PI * f;
        let r = (-zeta * omega * (1.0 / fs)).exp();
        let theta = omega * (1.0 / fs) * (1.0 - zeta * zeta).sqrt();
        let a1 = 2.0 * r * theta.cos();
        let a2 = -(r * r);
        Mode {
            omega,
            zeta,
            k,
            y_prev: 0.0,
            y_curr: 0.0,
            a1,
            a2,
        }
    }

    fn process(&mut self, input: f32) -> f32 {
        //  y[n+1] = a1*y[n] + a2*y[n-1] + k*input
        let y_next = self.a1 * self.y_curr + self.a2 * self.y_prev + self.k * input;
        self.y_prev = self.y_curr;
        self.y_curr = y_next;
        y_next
    }
}

fn update_step(
    u_prev: &mut [f32],
    u: &mut [f32],
    u_next: &mut [f32],
    hammer_pos: &mut f32,
    hammer_vel: &mut f32,
    hit_index: usize,
    samples: &mut Vec<f32>,
    modes: &mut [Mode],
) {
    let len = u.len();
    assert!(len >= 3);
    let penetration = *hammer_pos - u[hit_index];
    let contact_force = if penetration < 0.0 {
        (-HAMMER_K * penetration.abs().powf(1.5)).clamp(-CLAMP, CLAMP)
    } else {
        0.0
    };

    let accel = (contact_force - HAMMER_B * *hammer_vel) / HAMMER_MASS;
    *hammer_vel += DT * accel;
    *hammer_pos += if penetration < 0.0 {
        -penetration
    } else {
        DT * *hammer_vel
    };

    u_next[1..len - 1]
        .iter_mut() // par_iter_mut()
        .enumerate()
        .for_each(|(i, u_next_elem)| {
            let idx = i + 1;
            let f = if idx == hit_index { contact_force } else { 0.0 };
            let window = &u[idx - 1..=idx + 1];
            *u_next_elem = 2.0 * window[1] - u_prev[idx]
                + TENSION * (window[2] - 2.0 * window[1] + window[0])
                + DT * DT * f;
            *u_next_elem *= DECAY;
        });

    let mut out = if penetration > -0.0 { u[len / 2] } else { 0.0 }; //+ modal_sum / 10.0;

    let mut modal_sum = 0.0;
    for mode in modes.iter_mut() {
        modal_sum += mode.process(out);
    }
    out = modal_sum;
    samples.push(out);

    u_prev.swap_with_slice(u);
    u.swap_with_slice(u_next);
}
fn main() {
    let len: usize = env::args()
        .nth(1)
        .map(|s| {
            s.parse().unwrap_or_else(|_| {
                eprintln!("Expected a number, got {}", s);
                process::exit(1)
            })
        })
        .unwrap_or(81 as usize);
    let hz = SAMPLE_RATE as f32 * (TENSION).sqrt() / (len - 1) as f32;
    println!("{}Hz", hz);
    let mut u_prev = vec![0.0f32; len];
    let mut u = vec![0.0f32; len];
    let mut u_next = vec![0.0f32; len];
    let mut modes = vec![
        Mode::new(hz * 1.0, 1.0, 1.0, SAMPLE_RATE as f32), // 基音
        Mode::new(hz * 2.0, 1.0, 0.093, SAMPLE_RATE as f32), // 第2倍音
        // Mode::new(hz * 3.0, 0.15, 0.8, SAMPLE_RATE as f32), // 第3倍音
        // Mode::new(hz * 4.0, 0.08, 0.1, SAMPLE_RATE as f32), // 第4倍音
        // Mode::new(hz * 5.0, 0.04, 0.999, SAMPLE_RATE as f32), // 第5倍音
        Mode::new(hz * 6.0, 0.002, 0.0001, SAMPLE_RATE as f32), // 第6倍音
        // Mode::new(hz * 7.0, 0.01, 0.3, SAMPLE_RATE as f32), // 第7倍音
        // Mode::new(hz * 8.0, 0.005, 0.1, SAMPLE_RATE as f32), // 第8倍音
        // Mode::new(hz * 9.0, 0.0015, 0.09, SAMPLE_RATE as f32), // 第9倍音c
        // Mode::new(hz * 10.0, 0.00015, 0.005, SAMPLE_RATE as f32), // 第10倍音
        Mode::new(hz / 2.0, 0.1, 0.6, SAMPLE_RATE as f32), // 第2倍音
    ];

    let mut hammer_pos = 1.1;
    let mut hammer_vel = -500.0;
    let hit_index = len / 2;

    let mut samples = Vec::with_capacity(N_SAMPLES);

    for _ in 0..N_SAMPLES {
        update_step(
            &mut u_prev[..],
            &mut u[..],
            &mut u_next[..],
            &mut hammer_pos,
            &mut hammer_vel,
            hit_index,
            &mut samples,
            &mut modes,
        );
    }
    let max_amp = samples.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
    let gain = if max_amp > 0.0 { 0.9 / max_amp } else { 1.0 };
    println!("Max amplitude: {}", max_amp);

    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create("strike.wav", spec).unwrap();
    let mean = samples.iter().copied().sum::<f32>() / samples.len() as f32;
    for s in samples.iter_mut() {
        *s -= mean;
    }

    let fade_len = 200.min(samples.len());
    for i in 0..fade_len {
        let idx = samples.len() - fade_len + i;
        let fade_gain = 1.0 - i as f32 / fade_len as f32;
        samples[idx] *= fade_gain;
    }
    for s in samples {
        let amp = (s * gain * i16::MAX as f32).clamp(-i16::MAX as f32, i16::MAX as f32);
        writer.write_sample(amp as i16).unwrap();
    }
    writer.finalize().unwrap();
}
