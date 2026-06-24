use crate::math::matrix::Matrix;
use crate::math::scalar::Scalar;

pub fn trace<T: Scalar>(m: &Matrix<T>) -> T {
    let mut sum = T::zero();
    for i in 0..m.rows().min(m.cols()) {
        sum = sum + m[(i, i)].clone();
    }
    sum
}
