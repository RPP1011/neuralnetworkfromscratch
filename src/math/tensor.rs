use std::{
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    ops::{Add, Div, Mul, Sub},
    rc::Rc,
    sync::Mutex,
};

use super::tensor_context::{TensorContext, TensorRef};

#[derive(Debug, Clone)]
pub struct Tensor {
    pub tensor_context: Rc<RefCell<TensorContext>>,
    pub tensor_ref: TensorRef,
    pub data: Vec<f64>,
    pub grad: Option<Vec<f64>>,
    pub operation: Option<Operation>,
}

impl Tensor {
    pub fn add(&self, other: TensorRef) -> TensorRef {
        let tensor_context : &mut TensorContext = &mut self.tensor_context.as_ref().borrow_mut();
        tensor_context.add(self.tensor_ref, other)
    }
    
    pub fn mul(&self, other: TensorRef) -> TensorRef {
        let tensor_context : &mut TensorContext = &mut self.tensor_context.as_ref().borrow_mut();
        tensor_context.mul(self.tensor_ref, other)
    }

    pub fn backwards(&self) {
        let tensor_context : &mut TensorContext = &mut self.tensor_context.as_ref().borrow_mut();
        tensor_context.backwards(self.tensor_ref);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    Add(TensorRef, TensorRef),
    Sub,
    Mul(TensorRef, TensorRef),
    Div,
    Exp,
    Log,
    Sum,
    Mean,
    Dot,
    Transpose,
    Reshape,
    Slice,
}
