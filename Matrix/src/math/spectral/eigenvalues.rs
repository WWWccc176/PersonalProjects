use crate::math::decompositions::evd::evd_symmetric;
use crate::math::matrix::Matrix;
use crate::math::scalar::Scalar;

pub fn eigenvalues_symmetric<T: Scalar>(a: &Matrix<T>) -> Result<Vec<f64>, String> {
    let res = evd_symmetric(a)?;
    Ok(res.eigenvalues)
}
