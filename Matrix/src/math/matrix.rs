use crate::math::scalar::Scalar;

#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T: Scalar> {
    rows: usize,
    cols: usize,
    data: Vec<T>,
}

impl<T: Scalar> Matrix<T> {
    pub fn new(rows: usize, cols: usize, data: Vec<T>) -> Result<Self, String> {
        if rows * cols != data.len() {
            return Err(format!(
                "Invalid shape: {}x{} != {}",
                rows,
                cols,
                data.len()
            ));
        }
        Ok(Self { rows, cols, data })
    }

    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![T::zero(); rows * cols],
        }
    }

    pub fn identity(n: usize) -> Self {
        let mut out = Self::zeros(n, n);
        for i in 0..n {
            out[(i, i)] = T::one();
        }
        out
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
    pub fn data(&self) -> &[T] {
        &self.data
    }
    pub fn data_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    pub fn transpose(&self) -> Self {
        let mut out = Self::zeros(self.cols, self.rows);
        for (i, row) in self.data.chunks(self.cols).enumerate() {
            for (j, val) in row.iter().enumerate() {
                out[(j, i)] = val.clone();
            }
        }
        out
    }

    pub fn matmul(&self, rhs: &Self) -> Result<Self, String> {
        if self.cols != rhs.rows {
            return Err("Dimension mismatch for matmul".to_string());
        }
        let mut out = Self::zeros(self.rows, rhs.cols);
        let ncols = rhs.cols;
        for i in 0..self.rows {
            let out_row = &mut out.data[i * ncols..(i + 1) * ncols];
            for k in 0..self.cols {
                let a = self[(i, k)].clone();
                let rhs_row = &rhs.data[k * ncols..(k + 1) * ncols];
                for j in 0..ncols {
                    out_row[j] = out_row[j].clone() + a.clone() * rhs_row[j].clone();
                }
            }
        }
        Ok(out)
    }

    pub fn add(&self, rhs: &Self) -> Result<Self, String> {
        if self.shape() != rhs.shape() {
            return Err("Dimension mismatch".to_string());
        }
        Ok(Self {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(&rhs.data)
                .map(|(a, b)| a.clone() + b.clone())
                .collect(),
        })
    }

    pub fn sub(&self, rhs: &Self) -> Result<Self, String> {
        if self.shape() != rhs.shape() {
            return Err("Dimension mismatch".to_string());
        }
        Ok(Self {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .zip(&rhs.data)
                .map(|(a, b)| a.clone() - b.clone())
                .collect(),
        })
    }

    pub fn scale(&self, alpha: T) -> Self {
        Self {
            rows: self.rows,
            cols: self.cols,
            data: self
                .data
                .iter()
                .map(|x| alpha.clone() * x.clone())
                .collect(),
        }
    }
}

impl<T: Scalar> std::ops::Index<(usize, usize)> for Matrix<T> {
    type Output = T;
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        &self.data[i * self.cols + j]
    }
}

impl<T: Scalar> std::ops::IndexMut<(usize, usize)> for Matrix<T> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        &mut self.data[i * self.cols + j]
    }
}

