use rug::Rational;
use std::ops::{Add, Div, Mul, Neg, Sub};

pub trait Scalar:
    Clone
    + Sized
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + PartialEq
    + PartialOrd
    + std::fmt::Debug
{
    fn zero() -> Self;
    fn one() -> Self;
    fn is_zero(&self) -> bool;
    fn abs(&self) -> Self;
    fn from_i64(v: i64) -> Self;
    fn to_f64(&self) -> f64;
}

impl Scalar for f64 {
    fn zero() -> Self {
        0.0
    }
    fn one() -> Self {
        1.0
    }
    fn is_zero(&self) -> bool {
        self.abs() < 1e-12
    }
    fn abs(&self) -> Self {
        f64::abs(*self)
    }
    fn from_i64(v: i64) -> Self {
        v as f64
    }
    fn to_f64(&self) -> f64 {
        *self
    }
}

impl Scalar for Rational {
    fn zero() -> Self {
        Rational::new()
    }
    fn one() -> Self {
        Rational::from(1)
    }
    fn is_zero(&self) -> bool {
        self == &Self::zero()
    }
    // 调用 rug::Rational 的固有 abs(self)，不再递归调用 trait 方法
    fn abs(&self) -> Self {
        self.clone().abs()
    }
    fn from_i64(v: i64) -> Self {
        Rational::from(v)
    }
    fn to_f64(&self) -> f64 {
        rug::Float::with_val(53, self).to_f64()
    }
}
