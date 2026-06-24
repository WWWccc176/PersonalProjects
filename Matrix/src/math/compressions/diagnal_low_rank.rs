use crate::math::decompositions::svd::svd;
use crate::math::matrix::Matrix;
use crate::math::scalar::Scalar;

pub struct DiagonalLowRankCompression {
    pub k: usize,
    pub diagonal: Vec<f64>,
    pub u: Matrix<f64>,
    pub v: Matrix<f64>,
}

pub fn decompose_diagonal_low_rank<T: Scalar>(
    a: &Matrix<T>,
    k: usize,
) -> Result<DiagonalLowRankCompression, String> {
    let svd_res = svd(a)?;
    let m = a.rows();
    let n = a.cols();

    let mut u = Matrix::zeros(m, k);
    let mut v = Matrix::zeros(k, n);
    let diagonal = svd_res.singular_values.iter().take(k).cloned().collect();

    for i in 0..m {
        for j in 0..k {
            u[(i, j)] = svd_res.u[(i, j)];
        }
    }
    for i in 0..k {
        for j in 0..n {
            v[(i, j)] = svd_res.vt[(i, j)];
        }
    }
    Ok(DiagonalLowRankCompression { k, diagonal, u, v })
}

pub fn decompress_diagonal_low_rank(dlr: &DiagonalLowRankCompression) -> Matrix<f64> {
    let mut s = Matrix::zeros(dlr.k, dlr.k);
    for i in 0..dlr.k {
        s[(i, i)] = dlr.diagonal[i];
    }
    let us = dlr.u.matmul(&s).unwrap();
    us.matmul(&dlr.v).unwrap()
}
