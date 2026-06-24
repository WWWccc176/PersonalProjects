use crate::math::matrix::Matrix;
use crate::math::scalar::Scalar;

pub struct AsuvResult {
    pub matrix: Matrix<f64>,
}

pub fn compress_asuv_placeholder<T: Scalar>(a: &Matrix<T>, k: usize) -> Result<AsuvResult, String> {
    let m = a.rows();
    let n = a.cols();
    let cols = std::cmp::max(1 + k, std::cmp::max(m, n));
    let mut mat = Matrix::zeros(1 + 2 * k, cols);

    mat[(0, 0)] = k as f64;
    for i in 0..k {
        mat[(0, 1 + i)] = 1.0;
    }

    for i in 0..k {
        for j in 0..m {
            mat[(1 + i, j)] = if i == j { 1.0 } else { 0.0 };
        }
    }

    for i in 0..k {
        for j in 0..n {
            mat[(1 + k + i, j)] = if i == j { 1.0 } else { 0.0 };
        }
    }

    Ok(AsuvResult { matrix: mat })
}

pub fn decompress_asuv(mat: &Matrix<f64>) -> Result<Matrix<f64>, String> {
    let k = mat[(0, 0)] as usize;
    let cols = mat.cols();

    let mut s = Matrix::zeros(k, k);
    for i in 0..k {
        s[(i, i)] = mat[(0, 1 + i)];
    }

    let mut ut = Matrix::zeros(k, cols);
    for i in 0..k {
        for j in 0..cols {
            ut[(i, j)] = mat[(1 + i, j)];
        }
    }
    let u = ut.transpose();

    let mut vt = Matrix::zeros(k, cols);
    for i in 0..k {
        for j in 0..cols {
            vt[(i, j)] = mat[(1 + k + i, j)];
        }
    }

    let us = u.matmul(&s).map_err(|e| e.to_string())?;
    us.matmul(&vt).map_err(|e| e.to_string())
}
