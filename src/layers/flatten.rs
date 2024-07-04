use crate::math::{tensor::Tensor, tensor_context::TensorRef};

use super::layers::layers::Layer;

pub struct Flatten {
    pub input_shape: Vec<usize>
}

impl Layer for Flatten {
    fn forward(&self, input: TensorRef) -> TensorRef {
        todo!()
    }

    fn backward(&self, input: TensorRef) -> TensorRef {
        todo!()
    }
}

impl Flatten {
    pub fn new(input_vector_shape : Vec<usize>) -> Flatten {
        Flatten {
            input_shape : input_vector_shape
        }
    }
}