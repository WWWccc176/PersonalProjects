use std::fmt;

#[derive(Debug)]
pub enum MatrixError {
    InvalidShape,
    DimensionMismatch,
    NotSquare,
    SingularMatrix,
    ParseError(String),
    InvalidTextFormat(String),
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            MatrixError::InvalidShape => write!(f, "Invalid shape"),
            MatrixError::DimensionMismatch => write!(f, "Dimension mismatch"),
            MatrixError::NotSquare => write!(f, "Matrix must be square"),
            MatrixError::SingularMatrix => write!(f, "Singular matrix"),
            MatrixError::ParseError(s) => write!(f, "Parse error: {}", s),
            MatrixError::InvalidTextFormat(s) => write!(f, "Invalid text format: {}", s),
        }
    }
}

impl std::error::Error for MatrixError {}

