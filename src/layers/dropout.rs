use crate::math::{tensor::Tensor, tensor_context::TensorRef};

use super::layers::layers::Layer;


pub struct Dropout {
    pub rate: f64,
    pub training: bool,
}

impl Layer for Dropout {
    fn forward(&self, input: TensorRef) -> TensorRef {
        todo!("Implement dropout Forward pass")
    }

    fn backward(&self, input: TensorRef) -> TensorRef {
        todo!("Implement dropout backward pass")
    }
}

impl Dropout {
    pub fn new(rate: f64, training: bool) -> Dropout {
        Dropout {
            rate,
            training
        }
    }
}