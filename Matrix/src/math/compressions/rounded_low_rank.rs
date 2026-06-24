use crate::math::decompositions::svd::svd;
use crate::math::matrix::Matrix;
use crate::math::scalar::Scalar;

pub struct RoundedLowRankCompression {
    pub k: usize,
    pub u: Matrix<f64>,
    pub v: Matrix<f64>,
}

pub fn decompose_rounded_low_rank<T: Scalar>(
    a: &Matrix<T>,
    k: usize,
    eps: f64,
) -> Result<RoundedLowRankCompression, String> {
    let svd_res = svd(a)?;
    let m = a.rows();
    let n = a.cols();

    let mut u = Matrix::zeros(m, k);
    let mut v = Matrix::zeros(k, n);

    for i in 0..m {
        for j in 0..k {
            let s = svd_res.singular_values[j];
            let s_rounded = if s.abs() < eps { 0.0 } else { s };
            u[(i, j)] = svd_res.u[(i, j)] * s_rounded.sqrt();
        }
    }
    for i in 0..k {
        for j in 0..n {
            let s = svd_res.singular_values[i];
            let s_rounded = if s.abs() < eps { 0.0 } else { s };
            v[(i, j)] = svd_res.vt[(i, j)] * s_rounded.sqrt();
        }
    }
    Ok(RoundedLowRankCompression { k, u, v })
}

pub fn decompress_rounded_low_rank(rlr: &RoundedLowRankCompression) -> Matrix<f64> {
    rlr.u.matmul(&rlr.v).unwrap()
}
