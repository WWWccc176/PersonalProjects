#![allow(non_upper_case_globals)]

use std::io::{self, Write};

const G: f64 = 6.674_30e-11;
const M_EARTH: f64 = 5.972_2e24;
const R_EARTH: f64 = 6_371_000.0;
const v_r: f64 = 7.292_115_0e-5;

const GAMMA_AIR: f64 = 1.4;
const R_UNIVERSAL: f64 = 8.314_462_618;
const MOLAR_MASS_DRY_AIR: f64 = 0.028_964_4;
const R_SPECIFIC_DRY_AIR: f64 = R_UNIVERSAL / MOLAR_MASS_DRY_AIR;
const SEA_LEVEL_PRESSURE: f64 = 101_325.0;
const CELSIUS_TO_KELVIN: f64 = 273.15;

fn main() {
    let total_time = read_number(
        "1. Please enter the total time duration in seconds: ",
        |value| value > 0.0,
        "The total time must be a positive number.",
    );

    let latitude_deg = read_latitude("2. Please enter the latitude as degrees.minutes.seconds: ");

    let altitude_m = read_number(
        "3. Please enter the altitude relative to sea level in meters: ",
        |value| value > -R_EARTH,
        "The altitude must be greater than minus the Earth's radius.",
    );

    let temperature_c = read_number(
        "4. Please enter the current air temperature in degrees Celsius: ",
        |value| value > -CELSIUS_TO_KELVIN,
        "The temperature must be above absolute zero (-273.15 C).",
    );

    let g_local = effective_gravity(latitude_deg, altitude_m);

    if !g_local.is_finite() || g_local <= 0.0 {
        eprintln!("The calculated local gravity is invalid for the supplied data.");
        return;
    }

    let sound_speed = speed_of_sound(altitude_m, temperature_c, g_local);

    if !sound_speed.is_finite() || sound_speed <= 0.0 {
        eprintln!("The calculated speed of sound is invalid for the supplied data.");
        return;
    }

    let depth = cal_depth(total_time, g_local, sound_speed);
    let fall_time = t1(depth, g_local, sound_speed);
    let sound_time = t2(depth, sound_speed);

    println!("\nCalculated result:");
    println!("depth: {:.2} m", depth);
    println!(
        "fall time: {:.2} s    sound travel time: {:.2} s",
        fall_time, sound_time
    );

    println!("\nCalculated local conditions:");
    println!("latitude: {:.8} degrees", latitude_deg);
    println!("altitude: {:.2} m", altitude_m);
    println!("temperature: {:.2} C", temperature_c);
    println!("effective gravity: {:.2} m/s^2", g_local);
    println!("speed of sound: {:.2} m/s", sound_speed);
}

fn effective_gravity(latitude_deg: f64, altitude_m: f64) -> f64 {
    let latitude_rad = latitude_deg.to_radians();
    let radius = R_EARTH + altitude_m;

    // g = GM / r² - v_r²r cos²(latitude)
    G * M_EARTH / radius.powi(2) - v_r.powi(2) * radius * latitude_rad.cos().powi(2)
}

fn air_pressure(altitude_m: f64, temperature_k: f64, g_local: f64) -> f64 {
    let exponent = -g_local * altitude_m / (R_SPECIFIC_DRY_AIR * temperature_k);

    SEA_LEVEL_PRESSURE * exponent.exp()
}

fn air_density(pressure_pa: f64, temperature_k: f64) -> f64 {
    pressure_pa / (R_SPECIFIC_DRY_AIR * temperature_k)
}

fn speed_of_sound(altitude_m: f64, temperature_c: f64, g_local: f64) -> f64 {
    let temperature_k = temperature_c + CELSIUS_TO_KELVIN;

    let pressure = air_pressure(altitude_m, temperature_k, g_local);

    let density = air_density(pressure, temperature_k);

    if pressure > 0.0 && density > 0.0 {
        // c = sqrt(gamma × pressure / density)
        (GAMMA_AIR * pressure / density).sqrt()
    } else {
        // c = sqrt(gamma × R × temperature)
        (GAMMA_AIR * R_SPECIFIC_DRY_AIR * temperature_k).sqrt()
    }
}

fn cal_depth(total_time: f64, g_local: f64, sound_speed: f64) -> f64 {
    let t_limit = sound_speed / g_local;
    let h_limit = sound_speed.powi(2) / (2.0 * g_local);
    let total_limit = t_limit + h_limit / sound_speed;

    if total_time <= total_limit {
        // total_time = sqrt(2h / g) + h / sound_speed
        let a = (2.0 / g_local).sqrt();

        let b = (2.0 / g_local + 4.0 * total_time / sound_speed).sqrt();

        let sqrt_depth = 2.0 * total_time / (a + b);

        sqrt_depth.powi(2)
    } else {
        // total_time = t_limit + (h - h_limit) / sound_speed + h / sound_speed
        let numerator = total_time - t_limit + h_limit / sound_speed;

        let denominator = 2.0 / sound_speed;

        numerator / denominator
    }
}

fn t1(depth_m: f64, g_local: f64, sound_speed: f64) -> f64 {
    let h_limit = sound_speed.powi(2) / (2.0 * g_local);

    if depth_m <= h_limit {
        (2.0 * depth_m / g_local).sqrt()
    } else {
        sound_speed / g_local + (depth_m - h_limit) / sound_speed
    }
}

fn t2(depth_m: f64, sound_speed: f64) -> f64 {
    depth_m / sound_speed
}

fn read_number<F>(prompt: &str, validator: F, error_message: &str) -> f64
where
    F: Fn(f64) -> bool,
{
    loop {
        let input = read_line(prompt);

        match input.parse::<f64>() {
            Ok(value) if value.is_finite() && validator(value) => {
                return value;
            }
            _ => {
                eprintln!("{error_message}");
            }
        }
    }
}

fn read_latitude(prompt: &str) -> f64 {
    loop {
        let input = read_line(prompt);

        match parse_latitude(&input) {
            Ok(latitude) => return latitude,
            Err(message) => eprintln!("{message}"),
        }
    }
}

fn parse_latitude(input: &str) -> Result<f64, String> {
    let upper = input.trim().to_ascii_uppercase();

    if upper.is_empty() {
        return Err("Latitude cannot be empty.".to_string());
    }

    if upper.contains('N') && upper.contains('S') {
        return Err("Latitude cannot contain both N and S.".to_string());
    }

    let cleaned: String = upper
        .chars()
        .map(|character| {
            if character.is_ascii_digit() || matches!(character, '+' | '-') {
                character
            } else {
                ' '
            }
        })
        .collect();

    let values: Result<Vec<f64>, _> = cleaned.split_whitespace().map(str::parse::<f64>).collect();

    let values = values.map_err(|_| "Latitude contains an invalid number.".to_string())?;

    if values.is_empty() || values.len() > 3 {
        return Err(
            "Enter latitude as degrees.minutes.seconds, using one to three values.".to_string(),
        );
    }

    let degrees = values[0];
    let minutes = values.get(1).copied().unwrap_or(0.0);
    let seconds = values.get(2).copied().unwrap_or(0.0);

    if !degrees.is_finite() || !minutes.is_finite() || !seconds.is_finite() {
        return Err("Latitude values must be finite numbers.".to_string());
    }

    if !(0.0..60.0).contains(&minutes) || !(0.0..60.0).contains(&seconds) {
        return Err("Latitude minutes and seconds must be in the range [0, 60).".to_string());
    }

    // latitude = degrees + minutes / 60 + seconds / 3600
    let magnitude = degrees.abs() + minutes / 60.0 + seconds / 3600.0;

    if magnitude > 90.0 {
        return Err("Latitude must be between 90 degrees south and 90 degrees north.".to_string());
    }

    let sign = if upper.contains('S') {
        -1.0
    } else if upper.contains('N') {
        1.0
    } else if degrees.is_sign_negative() {
        -1.0
    } else {
        1.0
    };

    Ok(sign * magnitude)
}

fn read_line(prompt: &str) -> String {
    loop {
        print!("{prompt}");

        io::stdout().flush().expect("Failed to flush stdout");

        let mut input = String::new();

        match io::stdin().read_line(&mut input) {
            Ok(0) => {
                eprintln!("No input was received. Please try again.");
            }
            Ok(_) => {
                return input.trim().to_string();
            }
            Err(error) => {
                eprintln!("Failed to read input: {error}");
            }
        }
    }
}
