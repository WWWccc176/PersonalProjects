use crate::math::decompositions::svd::svd;
use crate::math::matrix::Matrix;
use crate::math::scalar::Scalar;

pub fn singular_values<T: Scalar>(a: &Matrix<T>) -> Result<Vec<f64>, String> {
    let res = svd(a)?;
    Ok(res.singular_values)
}
