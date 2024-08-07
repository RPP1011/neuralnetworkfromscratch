use std::{
    borrow::{BorrowMut},
    cell::RefCell,
    ops::{Add, Mul},
    rc::Rc,
};

use super::tensor_context::{TensorContext, TensorRef};

#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub tensor_context: Rc<RefCell<TensorContext>>,
    pub tensor_ref: TensorRef,
    pub data: Vec<f64>,
    pub grad: Option<Vec<f64>>,
    pub operation: Option<Operation>,
}



impl Tensor {   
    // Returns a new tensor owned by its own context
    pub fn new(sizes: Vec<usize>, data : Vec<f64>) -> Tensor {
        Tensor {
            shape: sizes,
            tensor_context: Rc::new(RefCell::new(TensorContext::new(1))),
            tensor_ref: 0,
            data,
            grad: None,
            operation: None,
        }
    }

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



#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    Add(Vec<TensorRef>),
    Sub(TensorRef, TensorRef),
    Mul(TensorRef, TensorRef),
    Div,
    Exp,
    Pow(TensorRef, f64),
    Log,
    Sum(TensorRef),
    Mean,
    Dot,
    Tanh(TensorRef),
    Transpose,
    Reshape,
    Slice,
    ReLU(TensorRef),
    Concat(Vec<TensorRef>),
    Composite(Vec<(Operation, TensorRef)>),
}
