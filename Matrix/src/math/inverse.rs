use crate::math::matrix::Matrix;
use crate::math::scalar::Scalar;
use crate::math::solve::rref_solve;

pub fn inverse<T: Scalar>(a: &Matrix<T>) -> Result<Matrix<T>, String> {
    if !a.is_square() {
        return Err("Matrix must be square".to_string());
    }
    let n = a.rows();
    let identity = Matrix::identity(n);
    rref_solve(a, &identity)
}

