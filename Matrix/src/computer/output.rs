use crate::math::compressions::diagnal_low_rank::DiagonalLowRankCompression;
use crate::math::compressions::low_rank::LowRankCompression;
use crate::math::decompositions::evd::EvdResult;
use crate::math::decompositions::svd::SvdResult;
use crate::math::matrix::Matrix;

pub fn format_evd(result: &EvdResult) -> String {
    let mut out = String::new();
    out.push_str("eigen-v: ");
    for v in &result.eigenvalues {
        out.push_str(&format!("{}, ", format_number(*v)));
    }
    out.push_str("\nD     : ");
    out.push_str(&format_matrix_pretty(&result.d, 8));
    out.push_str("\nP     : ");
    out.push_str(&format_matrix_pretty(&result.p, 8));
    out.push_str("\nP^-1  : ");
    out.push_str(&format_matrix_pretty(&result.p_inv, 8));
    out.push('\n');
    out
}

pub fn format_svd(result: &SvdResult) -> String {
    let mut out = String::new();
    out.push_str("singular-v: ");
    for v in &result.singular_values {
        out.push_str(&format!("{}, ", format_number(*v)));
    }
    out.push_str("\nU     : ");
    out.push_str(&format_matrix_pretty(&result.u, 8));
    out.push_str("\nSigma : ");
    out.push_str(&format_matrix_pretty(&result.sigma, 8));
    out.push_str("\nV^T   : ");
    out.push_str(&format_matrix_pretty(&result.vt, 8));
    out.push('\n');
    out
}

pub fn format_low_rank(result: &LowRankCompression) -> String {
    let mut out = String::new();
    out.push_str(&format!("k     : {}\n", result.k));
    out.push_str("U     : ");
    out.push_str(&format_matrix_pretty(&result.u, 8));
    out.push_str("\nV     : ");
    out.push_str(&format_matrix_pretty(&result.v, 8));
    out.push('\n');
    out
}

pub fn format_diagonal_low_rank(result: &DiagonalLowRankCompression) -> String {
    let mut out = String::new();
    out.push_str(&format!("k     : {}\n", result.k));
    out.push_str("diag  : ");
    for v in &result.diagonal {
        out.push_str(&format!("{}, ", format_number(*v)));
    }
    out.push_str("\nU     : ");
    out.push_str(&format_matrix_pretty(&result.u, 8));
    out.push_str("\nV     : ");
    out.push_str(&format_matrix_pretty(&result.v, 8));
    out.push('\n');
    out
}

pub fn format_scalar(name: &str, value: f64) -> String {
    format!("{}: {}\n", name, format_number(value))
}

pub fn format_usize(name: &str, value: usize) -> String {
    format!("{}: {}\n", name, value)
}

pub fn format_vector(name: &str, values: &[f64]) -> String {
    let mut out = format!("{}: ", name);
    for v in values {
        out.push_str(&format!("{}, ", format_number(*v)));
    }
    out.push('\n');
    out
}

pub fn format_matrix_labeled(label: &str, m: &Matrix<f64>) -> String {
    let mut s = String::from(label);
    s.push_str(&format_matrix_pretty(m, label.len()));
    s.push('\n');
    s
}

pub fn format_matrix_pretty(a: &Matrix<f64>, indent: usize) -> String {
    let mut out = String::new();
    for i in 0..a.rows() {
        if i == 0 {
            out.push('[');
        } else {
            out.push('\n');
            out.push_str(&" ".repeat(indent));
        }
        out.push('[');
        for j in 0..a.cols() {
            if j > 0 {
                out.push(' ');
            }
            out.push_str(&format_number(a[(i, j)]));
        }
        out.push(']');
        if i + 1 == a.rows() {
            out.push(']');
        }
    }
    out
}

pub fn format_number(x: f64) -> String {
    if x.abs() < 1e-12 {
        return "0".to_string();
    }
    if (x - x.round()).abs() < 1e-12 {
        return format!("{}", x.round() as i64);
    }
    let s = format!("{:.12}", x);
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}
