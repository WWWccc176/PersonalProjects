use crate::math::matrix::Matrix;
use crate::math::scalar::Scalar;

#[derive(Debug, Clone)]
pub struct LuResult<T: Scalar> {
    pub p: Matrix<T>,
    pub l: Matrix<T>,
    pub u: Matrix<T>,
    pub permutation: Vec<usize>,
    pub swap_count: usize,
}

pub fn lu<T: Scalar>(a: &Matrix<T>) -> Result<LuResult<T>, String> {
    if !a.is_square() {
        return Err("Matrix must be square".to_string());
    }

    let n = a.rows();

    let mut p = Matrix::<T>::identity(n);
    let mut l = Matrix::<T>::identity(n);
    let mut u = a.clone();

    let mut permutation: Vec<usize> = (0..n).collect();
    let mut swap_count = 0usize;

    for k in 0..n {
        let pivot_row = find_nonzero_pivot(&u, k);

        let Some(pivot_row) = pivot_row else {
            return Err("LU decomposition failed: matrix is singular".to_string());
        };

        if pivot_row != k {
            swap_rows(&mut u, k, pivot_row);
            swap_rows(&mut p, k, pivot_row);
            swap_permutation(&mut permutation, k, pivot_row);

            // Only swap the already computed part of L.
            // Columns k..n of L have not been computed yet.
            swap_l_previous_columns(&mut l, k, pivot_row, k);

            swap_count += 1;
        }

        let pivot = u[(k, k)].clone();

        if pivot.is_zero() {
            return Err("LU decomposition failed: zero pivot after row exchange".to_string());
        }

        for i in (k + 1)..n {
            let factor = u[(i, k)].clone() / pivot.clone();

            l[(i, k)] = factor.clone();

            for j in k..n {
                let correction = factor.clone() * u[(k, j)].clone();
                u[(i, j)] = u[(i, j)].clone() - correction;
            }

            // In exact arithmetic this is already zero.
            // Setting it explicitly keeps the triangular structure clean.
            u[(i, k)] = T::zero();
        }
    }

    Ok(LuResult {
        p,
        l,
        u,
        permutation,
        swap_count,
    })
}

fn find_nonzero_pivot<T: Scalar>(u: &Matrix<T>, col: usize) -> Option<usize> {
    (col..u.rows()).find(|&row| !u[(row, col)].is_zero())
}

fn swap_rows<T: Scalar>(a: &mut Matrix<T>, r1: usize, r2: usize) {
    if r1 == r2 {
        return;
    }

    for j in 0..a.cols() {
        let tmp = a[(r1, j)].clone();
        a[(r1, j)] = a[(r2, j)].clone();
        a[(r2, j)] = tmp;
    }
}

fn swap_l_previous_columns<T: Scalar>(
    l: &mut Matrix<T>,
    r1: usize,
    r2: usize,
    computed_cols: usize,
) {
    if r1 == r2 {
        return;
    }

    for j in 0..computed_cols {
        let tmp = l[(r1, j)].clone();
        l[(r1, j)] = l[(r2, j)].clone();
        l[(r2, j)] = tmp;
    }
}

fn swap_permutation(permutation: &mut [usize], i: usize, j: usize) {
    permutation.swap(i, j);
}

