use crate::math::matrix::Matrix;
use crate::math::row_ops::swap_rows;
use crate::math::scalar::Scalar;

pub fn rank_rref<T: Scalar>(a: &Matrix<T>) -> usize {
    let mut m = a.clone();
    let rows = m.rows();
    let cols = m.cols();
    let mut rank = 0;

    for col in 0..cols {
        if rank >= rows {
            break;
        }

        let mut pivot = rank;
        while pivot < rows && m[(pivot, col)].is_zero() {
            pivot += 1;
        }
        if pivot == rows {
            continue;
        }

        swap_rows(&mut m, rank, pivot);

        let pivot_val = m[(rank, col)].clone();
        for j in 0..cols {
            m[(rank, j)] = m[(rank, j)].clone() / pivot_val.clone();
        }

        for r in 0..rows {
            if r == rank || m[(r, col)].is_zero() {
                continue;
            }
            let factor = m[(r, col)].clone();
            for j in 0..cols {
                m[(r, j)] = m[(r, j)].clone() - factor.clone() * m[(rank, j)].clone();
            }
        }
        rank += 1;
    }
    rank
}
