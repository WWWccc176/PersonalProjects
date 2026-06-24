use crate::math::matrix::Matrix;
use crate::math::scalar::Scalar;

pub struct EvdResult {
    pub eigenvalues: Vec<f64>,
    pub d: Matrix<f64>,
    pub p: Matrix<f64>,
    pub p_inv: Matrix<f64>,
}

pub fn evd_symmetric<T: Scalar>(a: &Matrix<T>) -> Result<EvdResult, String> {
    if !a.is_square() {
        return Err("Matrix must be square".to_string());
    }
    let n = a.rows();
    let mut d = Matrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            d[(i, j)] = a[(i, j)].to_f64();
        }
    }

    let mut p = Matrix::<f64>::identity(n);
    let max_iter = 100;

    for _ in 0..max_iter {
        let mut off_diag = 0.0;
        for i in 0..n {
            for j in i + 1..n {
                if d[(i, j)].abs() > off_diag {
                    off_diag = d[(i, j)].abs();
                }
            }
        }
        if off_diag < 1e-10 {
            break;
        }

        for p_idx in 0..n {
            for q_idx in p_idx + 1..n {
                if d[(p_idx, q_idx)].abs() < 1e-15 {
                    continue;
                }

                let theta = (d[(q_idx, q_idx)] - d[(p_idx, p_idx)]) / (2.0 * d[(p_idx, q_idx)]);
                let t = theta.signum() / (theta.abs() + (1.0 + theta * theta).sqrt());
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;

                let mut new_d = d.clone();
                for i in 0..n {
                    new_d[(i, p_idx)] = c * d[(i, p_idx)] + s * d[(i, q_idx)];
                    new_d[(i, q_idx)] = -s * d[(i, p_idx)] + c * d[(i, q_idx)];
                }
                d = new_d.clone();
                let mut new_d2 = d.clone();
                for j in 0..n {
                    new_d2[(p_idx, j)] = c * d[(p_idx, j)] + s * d[(q_idx, j)];
                    new_d2[(q_idx, j)] = -s * d[(p_idx, j)] + c * d[(q_idx, j)];
                }
                d = new_d2;

                let mut new_p = p.clone();
                for i in 0..n {
                    new_p[(i, p_idx)] = c * p[(i, p_idx)] + s * p[(i, q_idx)];
                    new_p[(i, q_idx)] = -s * p[(i, p_idx)] + c * p[(i, q_idx)];
                }
                p = new_p;
            }
        }
    }

    let mut eigenvalues = vec![0.0; n];
    for i in 0..n {
        eigenvalues[i] = d[(i, i)];
    }

    let p_inv = p.transpose();
    Ok(EvdResult {
        eigenvalues,
        d,
        p,
        p_inv,
    })
}

