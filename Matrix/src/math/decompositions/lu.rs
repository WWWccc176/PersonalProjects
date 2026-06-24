use crate::math::matrix::Matrix;
use crate::math::scalar::Scalar;

pub fn lu<T: Scalar>(a: &Matrix<T>) -> Result<(Matrix<T>, Matrix<T>), String> {
    if !a.is_square() {
        return Err("Matrix must be square".to_string());
    }
    let n = a.rows();
    let mut l = Matrix::<T>::identity(n);
    let mut u = Matrix::<T>::zeros(n, n);

    for i in 0..n {
        for j in 0..n {
            let sum = (0..i)
                .map(|k| l[(i, k)].clone() * u[(k, j)].clone())
                .fold(T::zero(), |acc, x| acc + x);
            if i <= j {
                u[(i, j)] = a[(i, j)].clone() - sum;
            } else {
                if u[(i - 1, i - 1)].is_zero() {
                    return Err("LU decomposition failed".to_string());
                }
                l[(i, j)] = (a[(i, j)].clone() - sum) / u[(j, j)].clone();
            }
        }
    }
    Ok((l, u))
}

