use crate::computer::io::{DynMatrix, read_matrix_txt, write_text};
use crate::computer::output::*;
use crate::math::decompositions::evd::evd_symmetric;
use crate::math::decompositions::svd::svd;
use crate::math::determinant::determinant;
use crate::math::ranks::rank_rref;
use crate::math::scalar::Scalar;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int};

fn ffi_compute(
    input: *const c_char,
    output: *const c_char,
    op: fn(&DynMatrix) -> Result<String, String>,
) -> c_int {
    let input = match unsafe { cstr_to_str(input) } {
        Ok(s) => s,
        Err(c) => return c,
    };
    let output = match unsafe { cstr_to_str(output) } {
        Ok(s) => s,
        Err(c) => return c,
    };

    let dyn_m = match read_matrix_txt(input) {
        Ok(m) => m,
        Err(_) => return -2,
    };
    let text = match op(&dyn_m) {
        Ok(s) => s,
        Err(_) => return -3,
    };
    match write_text(output, &text) {
        Ok(()) => 0,
        Err(_) => -4,
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn matrix_evd_file(input: *const c_char, output: *const c_char) -> c_int {
    ffi_compute(input, output, |m| match m {
        DynMatrix::F64(mat) => Ok(format_evd(&evd_symmetric(mat)?)),
        DynMatrix::Big(mat) => Ok(format_evd(&evd_symmetric(mat)?)),
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn matrix_svd_file(input: *const c_char, output: *const c_char) -> c_int {
    ffi_compute(input, output, |m| match m {
        DynMatrix::F64(mat) => Ok(format_svd(&svd(mat)?)),
        DynMatrix::Big(mat) => Ok(format_svd(&svd(mat)?)),
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn matrix_rank_file(input: *const c_char, output: *const c_char) -> c_int {
    ffi_compute(input, output, |m| match m {
        DynMatrix::F64(mat) => Ok(format_usize("rank", rank_rref(mat))),
        DynMatrix::Big(mat) => Ok(format_usize("rank", rank_rref(mat))),
    })
}

#[unsafe(no_mangle)]
pub extern "C" fn matrix_det_file(input: *const c_char, output: *const c_char) -> c_int {
    ffi_compute(input, output, |m| match m {
        DynMatrix::F64(mat) => Ok(format_scalar("det", determinant(mat)?.to_f64())),
        DynMatrix::Big(mat) => Ok(format_scalar("det", determinant(mat)?.to_f64())),
    })
}

unsafe fn cstr_to_str<'a>(ptr: *const c_char) -> std::result::Result<&'a str, c_int> {
    if ptr.is_null() {
        return Err(-1);
    }
    unsafe { CStr::from_ptr(ptr) }.to_str().map_err(|_| -1)
}
