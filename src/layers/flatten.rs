use crate::math::tensor::Tensor;

use super::layers::layers::Layer;

pub struct Flatten {
    pub input_shape: Vec<usize>
}

impl Layer for Flatten {
    fn forward(&self, input: Tensor) -> Tensor {
        todo!()
    }

    fn backward(&self, input: Tensor) -> Tensor {
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