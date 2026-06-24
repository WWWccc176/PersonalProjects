use crate::math::decompositions::svd::svd;
use crate::math::matrix::Matrix;
use crate::math::scalar::Scalar;

pub fn condition_number_2<T: Scalar>(a: &Matrix<T>) -> Result<f64, String> {
    let svd_res = svd(a)?;
    let mut max = 0.0;
    let mut min = f64::MAX;
    for &s in &svd_res.singular_values {
        if s > max {
            max = s;
        }
        if s < min && s > 1e-15 {
            min = s;
        }
    }
    if min == f64::MAX {
        min = 0.0;
    }
    Ok(max / min)
}
