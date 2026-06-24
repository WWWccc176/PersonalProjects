use crate::math::matrix::Matrix;
use crate::math::scalar::Scalar;

pub fn frobenius_norm<T: Scalar>(m: &Matrix<T>) -> f64 {
    let mut sum = 0.0;
    for val in m.data() {
        let f = val.to_f64();
        sum += f * f;
    }
    sum.sqrt()
}
