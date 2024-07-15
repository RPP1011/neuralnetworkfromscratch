use std::{borrow::BorrowMut, cell::RefCell, rc::Rc};

use super::{
    tensor::{self, Operation, Tensor},
    tensor_context::{TensorContext, TensorRef},
};

#[derive(Debug, Clone)]
pub struct CompositeOperation {
    pub tensor_context: Rc<RefCell<TensorContext>>,
    pub operations: Vec<(Operation, TensorRef)>,
    pub output_tensor: TensorRef,
}

impl CompositeOperation {
    pub fn dot_product(context: Rc<RefCell<TensorContext>>, left:TensorRef, right:TensorRef) -> CompositeOperation {
        let mul_tensor = context.as_ref().borrow_mut().mul(left, right);
        let sum_tensor = context.as_ref().borrow_mut().sum(mul_tensor);

        CompositeOperation {
            tensor_context: context,
            operations: vec![(Operation::Mul(left, right),mul_tensor), (Operation::Sum(mul_tensor), sum_tensor)],
            output_tensor: sum_tensor,
        }
    }

    pub fn perform(&self) {
        self.operations.iter().for_each(|op| {
            match &(op.0) {
                Operation::Mul(left, right) => {
                    self.tensor_context.as_ref().borrow_mut().mul_inplace(left.clone(), right.clone(), op.1.clone());
                }
                Operation::Sum(tensor) => {
                    self.tensor_context.as_ref().borrow_mut().sum_inplace(tensor.clone(), op.1.clone());
                }
                _ => {}
            }
        });
    }

    pub fn backprop(&self) {
        self.tensor_context.as_ref().borrow_mut().backwards(self.output_tensor);
    }
}
