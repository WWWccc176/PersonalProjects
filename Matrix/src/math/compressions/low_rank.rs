use crate::math::decompositions::svd::svd;
use crate::math::matrix::Matrix;
use crate::math::scalar::Scalar;

pub struct LowRankCompression {
    pub k: usize,
    pub u: Matrix<f64>,
    pub v: Matrix<f64>,
}

pub fn decompose_low_rank<T: Scalar>(
    a: &Matrix<T>,
    k: usize,
) -> Result<LowRankCompression, String> {
    let svd_res = svd(a)?;
    let m = a.rows();
    let n = a.cols();

    let mut u = Matrix::zeros(m, k);
    let mut v = Matrix::zeros(k, n);

    for i in 0..m {
        for j in 0..k {
            u[(i, j)] = svd_res.u[(i, j)] * svd_res.singular_values[j].sqrt();
        }
    }
    for i in 0..k {
        for j in 0..n {
            v[(i, j)] = svd_res.vt[(i, j)] * svd_res.singular_values[i].sqrt();
        }
    }
    Ok(LowRankCompression { k, u, v })
}

pub fn decompress_low_rank(lr: &LowRankCompression) -> Matrix<f64> {
    lr.u.matmul(&lr.v).unwrap()
}
