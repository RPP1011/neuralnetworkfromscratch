use std::{cell::RefCell, rc::Rc};

use crate::math::{tensor_context::{TensorContext, TensorRef}};

use super::layers::layers::Layer;
use crate::math::tensor::Tensor;

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

    fn get_parameters(&self) -> Vec<TensorRef> {
        vec![]
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
#[cfg(test)]
mod tests {
    use crate::create_tensor_context;

    use super::*;

    #[test]
    fn test_flatten_compile() {
        let tensor_context = create_tensor_context!(1024);
        let input_shape = vec![2, 3, 4];
        let input = tensor_context.borrow_mut().new_tensor(vec![2, 3, 4], vec![1.0; 24]);
        let mut flatten = Flatten::new(tensor_context.clone(), input_shape);

        let output = flatten.compile(input);

        assert_eq!(tensor_context.borrow_mut().get_tensor(output).shape, vec![24]);
    }

    #[test]
    fn test_flatten_forward() {
        let tensor_context = create_tensor_context!(1024);
        let input_shape = vec![2, 3, 4];
        let input = tensor_context.borrow_mut().new_tensor(vec![2, 3, 4], vec![1.0; 24]);
        let mut flatten = Flatten::new(tensor_context.clone(), input_shape);

        flatten.compile(input);
        let output1 = flatten.forward(input);

        assert_eq!(tensor_context.borrow_mut().get_tensor(output1).shape, vec![24]);

        // Flatten another tensor
        let input = tensor_context.borrow_mut().new_tensor(vec![2, 3, 4], vec![2.0; 24]);
        let output2 = flatten.forward(input);
        assert_eq!(tensor_context.borrow_mut().get_tensor(output2).shape, vec![24]);

        assert_eq!(output1, output2);
    }
}
