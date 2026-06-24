use crate::math::decompositions::evd::evd_symmetric;
use crate::math::matrix::Matrix;
use crate::math::scalar::Scalar;

pub struct SvdResult {
    pub singular_values: Vec<f64>,
    pub u: Matrix<f64>,
    pub sigma: Matrix<f64>,
    pub vt: Matrix<f64>,
}

pub fn svd<T: Scalar>(a: &Matrix<T>) -> Result<SvdResult, String> {
    let (m, n) = a.shape();
    if m >= n {
        svd_tall(a)
    } else {
        let at = a.transpose();
        let r = svd_tall(&at)?;
        Ok(SvdResult {
            singular_values: r.singular_values,
            u: r.vt.transpose(),
            sigma: r.sigma,
            vt: r.u.transpose(),
        })
    }
}

fn svd_tall<T: Scalar>(a: &Matrix<T>) -> Result<SvdResult, String> {
    let m = a.rows();
    let n = a.cols();
    let at = a.transpose();
    let ata = at.matmul(a).map_err(|e| e.to_string())?;

    let evd = evd_symmetric(&ata)?;
    let v = evd.p;

    let singular_values: Vec<f64> = evd.eigenvalues.iter().map(|&x| x.abs().sqrt()).collect();
    let mut u = Matrix::zeros(m, n);

    for j in 0..n {
        let sigma = singular_values[j];
        if sigma.abs() < 1e-12 {
            continue;
        }
        for i in 0..m {
            let sum = (0..n).map(|k| a[(i, k)].to_f64() * v[(k, j)]).sum::<f64>();
            u[(i, j)] = sum / sigma;
        }
    }

    let mut sigma_mat = Matrix::zeros(n, n);
    for i in 0..n {
        sigma_mat[(i, i)] = singular_values[i];
    }
    let vt = v.transpose();

    Ok(SvdResult {
        singular_values,
        u,
        sigma: sigma_mat,
        vt,
    })
}

