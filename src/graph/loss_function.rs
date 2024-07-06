use std::{cell::RefCell, rc::Rc};

use crate::math::{tensor::Tensor, tensor_context::{self, TensorContext, TensorRef}};

pub enum LossFunction {
    MeanSquaredError,
    CrossEntropy,
    SparseCrossEntropy,
}

impl LossFunction {
    pub fn loss(&self,tensor_context: Rc<RefCell<TensorContext>>,  input : TensorRef, desired: TensorRef) -> TensorRef {
        match self {
            LossFunction::MeanSquaredError => LossFunction::mean_squared_error(tensor_context, input, desired),
            LossFunction::CrossEntropy => LossFunction::cross_entropy(tensor_context, input, desired),
            LossFunction::SparseCrossEntropy => LossFunction::sparse_cross_entropy(tensor_context, input, desired),
        }
    }

    fn mean_squared_error(tensor_context: Rc<RefCell<TensorContext>>, input : TensorRef, desired: TensorRef) -> TensorRef {
        let loss_tensor = tensor_context.borrow_mut().sub(input, desired);
        let loss_tensor = tensor_context.borrow_mut().pow(loss_tensor, 2.0);
        let loss = tensor_context.borrow_mut().sum(loss_tensor);

        loss
    }

    fn cross_entropy(tensor_context: Rc<RefCell<TensorContext>>,_input : TensorRef, _desired: TensorRef) -> TensorRef {
        todo!("Implement Cross Entropy Loss Function")
    }

    fn sparse_cross_entropy(tensor_context: Rc<RefCell<TensorContext>>,_input : TensorRef, _desired: TensorRef) -> TensorRef {
        todo!("Implement Sparse Cross Entropy Loss Function")
    }
}