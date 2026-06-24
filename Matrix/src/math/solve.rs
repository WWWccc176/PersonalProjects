use crate::math::matrix::Matrix;
use crate::math::row_ops::swap_rows;
use crate::math::scalar::Scalar;

pub fn rref_solve<T: Scalar>(left: &Matrix<T>, right: &Matrix<T>) -> Result<Matrix<T>, String> {
    if left.rows() != left.cols() {
        return Err("Left matrix must be square".to_string());
    }
    if left.rows() != right.rows() {
        return Err("Dimension mismatch".to_string());
    }

    let n = left.rows();
    let rhs_cols = right.cols();
    let mut left = left.clone();
    let mut right = right.clone();

    for col in 0..n {
        let mut pivot_row = col;
        let mut pivot_abs = left[(col, col)].abs();
        for r in (col + 1)..n {
            let val = left[(r, col)].abs();
            if val > pivot_abs {
                pivot_abs = val;
                pivot_row = r;
            }
        }

        if pivot_abs.is_zero() {
            return Err("Singular matrix".to_string());
        }

        swap_rows(&mut left, col, pivot_row);
        swap_rows(&mut right, col, pivot_row);

        let pivot = left[(col, col)].clone();
        for j in 0..n {
            left[(col, j)] = left[(col, j)].clone() / pivot.clone();
        }
        for j in 0..rhs_cols {
            right[(col, j)] = right[(col, j)].clone() / pivot.clone();
        }

        for r in 0..n {
            if r == col {
                continue;
            }
            let factor = left[(r, col)].clone();
            for j in 0..n {
                left[(r, j)] = left[(r, j)].clone() - factor.clone() * left[(col, j)].clone();
            }
            for j in 0..rhs_cols {
                right[(r, j)] = right[(r, j)].clone() - factor.clone() * right[(col, j)].clone();
            }
        }
    }
    Ok(right)
}

pub fn solve_linear_system<T: Scalar>(a: &Matrix<T>, b: &Matrix<T>) -> Result<Matrix<T>, String> {
    rref_solve(a, b)
}

