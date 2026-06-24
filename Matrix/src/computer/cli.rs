use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use crate::computer::io::{DynMatrix, read_matrix_txt, write_text};
use crate::computer::output::*;
use crate::math::compressions::diagnal_low_rank::decompose_diagonal_low_rank;
use crate::math::compressions::low_rank::decompose_low_rank;
use crate::math::decompositions::evd::evd_symmetric;
use crate::math::decompositions::svd::svd;
use crate::math::determinant::determinant;
use crate::math::inverse::inverse;
use crate::math::properties::trace;
use crate::math::ranks::rank_rref;
use crate::math::scalar::Scalar;
use crate::math::solve::solve_linear_system;
use crate::math::spectral::condition::condition_number_2;
use crate::math::spectral::eigenvalues::eigenvalues_symmetric;
use crate::math::spectral::singular_values::singular_values;

/// 把 `Result<T, String>` 转成 `anyhow::Result<T>`，方便用 `?`
trait StrErrExt<T> {
    fn anyhow(self) -> anyhow::Result<T>;
}
impl<T> StrErrExt<T> for std::result::Result<T, String> {
    fn anyhow(self) -> anyhow::Result<T> {
        self.map_err(|e| anyhow::anyhow!(e))
    }
}

#[derive(Parser)]
#[command(name = "mtx")]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Subcommand)]
pub enum Command {
    Rank {
        input: PathBuf,
        output: PathBuf,
    },
    Det {
        input: PathBuf,
        output: PathBuf,
    },
    Inv {
        input: PathBuf,
        output: PathBuf,
    },
    Trace {
        input: PathBuf,
        output: PathBuf,
    },
    Eigvals {
        input: PathBuf,
        output: PathBuf,
    },
    Singvals {
        input: PathBuf,
        output: PathBuf,
    },
    Condition {
        input: PathBuf,
        output: PathBuf,
    },
    Solve {
        a: PathBuf,
        b: PathBuf,
        output: PathBuf,
    },
    Evd {
        input: PathBuf,
        output: PathBuf,
    },
    Svd {
        input: PathBuf,
        output: PathBuf,
    },
    Compress {
        #[command(subcommand)]
        method: CompressCommand,
    },
}

#[derive(Subcommand)]
pub enum CompressCommand {
    LowRank {
        input: PathBuf,
        output: PathBuf,
        #[arg(long)]
        k: usize,
    },
    DiagonalLowRank {
        input: PathBuf,
        output: PathBuf,
        #[arg(long)]
        k: usize,
    },
}

pub fn run_cli(cli: Cli) -> Result<()> {
    match cli.command {
        Command::Rank { input, output } => {
            let dyn_m = read_matrix_txt(input).anyhow()?;
            let text = match dyn_m {
                DynMatrix::F64(m) => format_usize("rank", rank_rref(&m)),
                DynMatrix::Big(m) => format_usize("rank", rank_rref(&m)),
            };
            write_text(output, &text).anyhow()?;
        }
        Command::Det { input, output } => {
            let dyn_m = read_matrix_txt(input).anyhow()?;
            let text = match dyn_m {
                DynMatrix::F64(m) => format_scalar("det", determinant(&m).anyhow()?.to_f64()),
                DynMatrix::Big(m) => format_scalar("det", determinant(&m).anyhow()?.to_f64()),
            };
            write_text(output, &text).anyhow()?;
        }
        Command::Inv { input, output } => {
            let dyn_m = read_matrix_txt(input).anyhow()?;
            let text = match dyn_m {
                DynMatrix::F64(m) => {
                    let inv = inverse(&m).anyhow()?;
                    let mut f_inv = crate::math::matrix::Matrix::zeros(inv.rows(), inv.cols());
                    for i in 0..inv.rows() {
                        for j in 0..inv.cols() {
                            f_inv[(i, j)] = inv[(i, j)].to_f64();
                        }
                    }
                    format_matrix_labeled("A^-1  : ", &f_inv)
                }
                DynMatrix::Big(m) => {
                    let inv = inverse(&m).anyhow()?;
                    let mut f_inv = crate::math::matrix::Matrix::zeros(inv.rows(), inv.cols());
                    for i in 0..inv.rows() {
                        for j in 0..inv.cols() {
                            f_inv[(i, j)] = inv[(i, j)].to_f64();
                        }
                    }
                    format_matrix_labeled("A^-1  : ", &f_inv)
                }
            };
            write_text(output, &text).anyhow()?;
        }
        Command::Trace { input, output } => {
            let dyn_m = read_matrix_txt(input).anyhow()?;
            let text = match dyn_m {
                DynMatrix::F64(m) => format_scalar("trace", trace(&m).to_f64()),
                DynMatrix::Big(m) => format_scalar("trace", trace(&m).to_f64()),
            };
            write_text(output, &text).anyhow()?;
        }
        Command::Eigvals { input, output } => {
            let dyn_m = read_matrix_txt(input).anyhow()?;
            let text = match dyn_m {
                DynMatrix::F64(m) => format_vector("eigen-v", &eigenvalues_symmetric(&m).anyhow()?),
                DynMatrix::Big(m) => format_vector("eigen-v", &eigenvalues_symmetric(&m).anyhow()?),
            };
            write_text(output, &text).anyhow()?;
        }
        Command::Singvals { input, output } => {
            let dyn_m = read_matrix_txt(input).anyhow()?;
            let text = match dyn_m {
                DynMatrix::F64(m) => format_vector("singular-v", &singular_values(&m).anyhow()?),
                DynMatrix::Big(m) => format_vector("singular-v", &singular_values(&m).anyhow()?),
            };
            write_text(output, &text).anyhow()?;
        }
        Command::Condition { input, output } => {
            let dyn_m = read_matrix_txt(input).anyhow()?;
            let text = match dyn_m {
                DynMatrix::F64(m) => format_scalar("condition-2", condition_number_2(&m).anyhow()?),
                DynMatrix::Big(m) => format_scalar("condition-2", condition_number_2(&m).anyhow()?),
            };
            write_text(output, &text).anyhow()?;
        }
        Command::Solve { a, b, output } => {
            let dyn_a = read_matrix_txt(a).anyhow()?;
            let dyn_b = read_matrix_txt(b).anyhow()?;
            let text = match (dyn_a, dyn_b) {
                (DynMatrix::F64(a), DynMatrix::F64(b)) => {
                    let x = solve_linear_system(&a, &b).anyhow()?;
                    let mut f_x = crate::math::matrix::Matrix::zeros(x.rows(), x.cols());
                    for i in 0..x.rows() {
                        for j in 0..x.cols() {
                            f_x[(i, j)] = x[(i, j)].to_f64();
                        }
                    }
                    format_matrix_labeled("x     : ", &f_x)
                }
                (DynMatrix::Big(a), DynMatrix::Big(b)) => {
                    let x = solve_linear_system(&a, &b).anyhow()?;
                    let mut f_x = crate::math::matrix::Matrix::zeros(x.rows(), x.cols());
                    for i in 0..x.rows() {
                        for j in 0..x.cols() {
                            f_x[(i, j)] = x[(i, j)].to_f64();
                        }
                    }
                    format_matrix_labeled("x     : ", &f_x)
                }
                _ => anyhow::bail!("Matrix A and B must have the same precision type"),
            };
            write_text(output, &text).anyhow()?;
        }
        Command::Evd { input, output } => {
            let dyn_m = read_matrix_txt(input).anyhow()?;
            let text = match dyn_m {
                DynMatrix::F64(m) => format_evd(&evd_symmetric(&m).anyhow()?),
                DynMatrix::Big(m) => format_evd(&evd_symmetric(&m).anyhow()?),
            };
            write_text(output, &text).anyhow()?;
        }
        Command::Svd { input, output } => {
            let dyn_m = read_matrix_txt(input).anyhow()?;
            let text = match dyn_m {
                DynMatrix::F64(m) => format_svd(&svd(&m).anyhow()?),
                DynMatrix::Big(m) => format_svd(&svd(&m).anyhow()?),
            };
            write_text(output, &text).anyhow()?;
        }
        Command::Compress { method } => match method {
            CompressCommand::LowRank { input, output, k } => {
                let dyn_m = read_matrix_txt(input).anyhow()?;
                let text = match dyn_m {
                    DynMatrix::F64(m) => format_low_rank(&decompose_low_rank(&m, k).anyhow()?),
                    DynMatrix::Big(m) => format_low_rank(&decompose_low_rank(&m, k).anyhow()?),
                };
                write_text(output, &text).anyhow()?;
            }
            CompressCommand::DiagonalLowRank { input, output, k } => {
                let dyn_m = read_matrix_txt(input).anyhow()?;
                let text = match dyn_m {
                    DynMatrix::F64(m) => {
                        format_diagonal_low_rank(&decompose_diagonal_low_rank(&m, k).anyhow()?)
                    }
                    DynMatrix::Big(m) => {
                        format_diagonal_low_rank(&decompose_diagonal_low_rank(&m, k).anyhow()?)
                    }
                };
                write_text(output, &text).anyhow()?;
            }
        },
    }
    Ok(())
}

