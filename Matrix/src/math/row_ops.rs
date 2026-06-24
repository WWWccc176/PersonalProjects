use crate::math::matrix::Matrix;
use crate::math::scalar::Scalar;

pub fn swap_rows<T: Scalar>(m: &mut Matrix<T>, r1: usize, r2: usize) {
    if r1 == r2 {
        return;
    }
    let cols = m.cols();
    for j in 0..cols {
        let temp = m[(r1, j)].clone();
        m[(r1, j)] = m[(r2, j)].clone();
        m[(r2, j)] = temp;
    }
}
