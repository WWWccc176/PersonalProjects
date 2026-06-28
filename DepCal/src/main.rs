#![allow(non_upper_case_globals)]

use std::io;

const g: f64 = 9.80665;
const v_s: f64 = 343.0;

fn main() {
    println!("Please enter the total time in seconds:");

    let mut input = String::new();

    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read input");

    let time: f64 = input.trim().parse().expect("Invalid number");

    let depth = cal_depth(time);
    let fall_time = t1(depth);
    let sound_time = t2(depth);

    println!("depth: {:.10} m", depth);
    println!("fall time: {:.10} s", fall_time);
    println!("sound time: {:.10} s", sound_time);
}

fn cal_depth(time: f64) -> f64 {
    let t_limit = v_s / g;
    let h_limit = v_s * v_s / (2.0 * g);
    let total_limit = t_limit + h_limit / v_s;

    if time <= total_limit {
        let a = (2.0 / g).sqrt();
        let b = (2.0 / g + 4.0 * time / v_s).sqrt();
        let x = 2.0 * time / (a + b);

        x * x
    } else {
        (time - t_limit + h_limit / v_s) / (1.0 / v_s + 1.0 / v_s)
    }
}

fn t1(h: f64) -> f64 {
    let h_limit = v_s * v_s / (2.0 * g);

    if h <= h_limit {
        (2.0 * h / g).sqrt()
    } else {
        v_s / g + (h - h_limit) / v_s
    }
}

fn t2(h: f64) -> f64 {
    h / v_s
}
