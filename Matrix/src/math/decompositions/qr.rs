use crate::math::matrix::Matrix;
use crate::math::scalar::Scalar;

pub fn qr<T: Scalar>(a: &Matrix<T>) -> Result<(Matrix<f64>, Matrix<f64>), String> {
    let m = a.rows();
    let n = a.cols();
    let mut q = Matrix::<f64>::identity(m);
    let mut r = Matrix::zeros(m, n);

    for i in 0..m {
        for j in 0..n {
            r[(i, j)] = a[(i, j)].to_f64();
        }
    }

    for k in 0..n.min(m) {
        let mut norm = 0.0;
        for i in k..m {
            norm += r[(i, k)].powi(2);
        }
        let norm = norm.sqrt();

        if norm < 1e-12 {
            continue;
        }

        let mut v = vec![0.0; m];
        v[k] = if r[(k, k)] >= 0.0 { -norm } else { norm };
        for i in k + 1..m {
            v[i] = r[(i, k)];
        }

        let v_norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if v_norm < 1e-12 {
            continue;
        }

        for j in k..n {
            let dot: f64 = (k..m).map(|i| v[i] * r[(i, j)]).sum();
            for i in k..m {
                r[(i, j)] -= 2.0 * (dot / v_norm) * v[i];
            }
        }

        for j in 0..m {
            let dot: f64 = (k..m).map(|i| v[i] * q[(j, i)]).sum();
            for i in k..m {
                q[(j, i)] -= 2.0 * (dot / v_norm) * v[i];
            }
        }
    }
    Ok((q, r))
}
