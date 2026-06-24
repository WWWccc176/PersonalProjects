use crate::math::matrix::Matrix;
use rug::Rational;
use std::fs;
use std::path::Path;

pub enum DynMatrix {
    F64(Matrix<f64>),
    Big(Matrix<Rational>),
}

pub fn read_matrix_txt<P: AsRef<Path>>(path: P) -> Result<DynMatrix, String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    parse_matrix_text(&text)
}

pub fn write_text<P: AsRef<Path>>(path: P, text: &str) -> Result<(), String> {
    fs::write(path, text).map_err(|e| e.to_string())
}

pub fn parse_matrix_text(text: &str) -> Result<DynMatrix, String> {
    let s: String = text.chars().filter(|c| !c.is_control()).collect();
    let s = s.trim();
    if !s.starts_with("[[") || !s.ends_with("]]") {
        return Err("Matrix must look like [[1 2][3 4]]".to_string());
    }

    let inner = &s[1..s.len() - 1];
    let rows: Vec<Vec<String>> = inner
        .split("][")
        .map(|r| {
            r.trim_start_matches('[')
                .trim_end_matches(']')
                .split_whitespace()
                .map(|s| s.to_string())
                .collect()
        })
        .collect();

    if rows.is_empty() || rows[0].is_empty() {
        return Err("Empty matrix".to_string());
    }

    let cols = rows[0].len();
    if rows.iter().any(|r| r.len() != cols) {
        return Err("All rows must have the same length".to_string());
    }

    let use_big = rows.iter().any(|r| r.iter().any(|s| needs_big_number(s)));

    if use_big {
        let mut data = Vec::new();
        for row in &rows {
            for s in row {
                let rat = Rational::parse(s.as_str()).map_err(|_| format!("Parse error: {}", s))?;
                data.push(Rational::from(rat));
            }
        }
        Ok(DynMatrix::Big(
            Matrix::new(rows.len(), cols, data).map_err(|e| e.to_string())?,
        ))
    } else {
        let mut data = Vec::new();
        for row in &rows {
            for s in row {
                let val = s
                    .parse::<f64>()
                    .map_err(|_| format!("Parse error: {}", s))?;
                data.push(val);
            }
        }
        Ok(DynMatrix::F64(
            Matrix::new(rows.len(), cols, data).map_err(|e| e.to_string())?,
        ))
    }
}

fn needs_big_number(s: &str) -> bool {
    let digit_count = s.chars().filter(|c| c.is_ascii_digit()).count();
    if digit_count > 15 {
        return true;
    }
    if s.contains('/') {
        return true;
    }
    false
}

