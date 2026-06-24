use crate::math::matrix::Matrix;
use crate::math::row_ops::swap_rows;
use crate::math::scalar::Scalar;

pub fn determinant<T: Scalar>(a: &Matrix<T>) -> Result<T, String> {
    if !a.is_square() {
        return Err("Matrix must be square".to_string());
    }
    let n = a.rows();
    let mut m = a.clone();
    let mut det = T::one();

    for col in 0..n {
        let mut pivot = col;
        for r in (col + 1)..n {
            if m[(r, col)].abs() > m[(pivot, col)].abs() {
                pivot = r;
            }
        }

        if m[(pivot, col)].is_zero() {
            return Ok(T::zero());
        }

        if pivot != col {
            swap_rows(&mut m, col, pivot);
            det = -det;
        }

        let pivot_val = m[(col, col)].clone();
        det = det * pivot_val.clone();

        for r in (col + 1)..n {
            let factor = m[(r, col)].clone() / pivot_val.clone();
            for j in 0..n {
                m[(r, j)] = m[(r, j)].clone() - factor.clone() * m[(col, j)].clone();
            }
        }
    }
    Ok(det)
}
