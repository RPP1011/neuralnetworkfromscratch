use std::{cell::RefCell, rc::Rc};

use crate::math::tensor_context::{TensorContext, TensorRef};

use super::layers::layers::Layer;

pub struct Input {
    pub tensor_context: Rc<RefCell<TensorContext>>,
    pub size: Vec<usize>,
    fixed_input_tensor: Option<TensorRef>,
}

impl Layer for Input {
    fn forward(&self, input: TensorRef) -> TensorRef {
        self.tensor_context.borrow_mut().copy_data(input, self.fixed_input_tensor.unwrap());
        self.fixed_input_tensor.unwrap()
    }

    fn compile(&mut self, input: TensorRef) -> TensorRef {
        let input_size = self.tensor_context.borrow().get_tensor(input);

        if input_size.shape != self.size {
            panic!("Input size does not match the expected size of the input layer");
        }

        let new_fixed_tensor = self.tensor_context.borrow_mut().new_tensor(self.size.clone(), vec![0.0; self.size.iter().product::<usize>()]);
        self.fixed_input_tensor = Some(new_fixed_tensor);
        new_fixed_tensor
    }
    
    fn get_parameters(&self) -> Vec<TensorRef> {
        vec![]
    }
}

impl Input {
    pub fn new(tensor_context: Rc<RefCell<TensorContext>>, size: Vec<usize>) -> Input {
        Input {
            tensor_context,
            size,
            fixed_input_tensor: None,
        }
    }
}