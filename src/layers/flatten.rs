use std::{cell::RefCell, rc::Rc};

use crate::math::{tensor_context::{TensorContext, TensorRef}};

use super::layers::layers::Layer;

pub struct Flatten {
    pub input_shape: Vec<usize>,
    tensor_context : Rc<RefCell<TensorContext>>,
    output_tensor: Option<TensorRef>,
}

impl Layer for Flatten {
    fn forward(&self, input: TensorRef) -> TensorRef { 
        self.tensor_context.borrow_mut().concat_inplace(vec![input], self.output_tensor.unwrap());
        self.output_tensor.unwrap()
    }

    fn compile(&mut self, input: TensorRef) -> TensorRef {
        self.output_tensor = Some(self.tensor_context.borrow_mut().concat(vec![input]));
        self.output_tensor.unwrap() 
    }
}

impl Flatten {
    pub fn new(tensor_context: Rc<RefCell<TensorContext>>, input_vector_shape : Vec<usize>) -> Flatten {
        Flatten {
            input_shape : input_vector_shape,
            tensor_context,
            output_tensor: None,
        }
    }
}