use std::{cell::RefCell, iter::zip, ops::Mul, rc::Rc};

use crate::{math::tensor_context::{TensorContext, TensorRef}, nuerons::activation_function::ActivationFunction};

use super::activation_function;



#[derive(Debug, Clone)]
pub struct Nueron {
    pub tensor_context: Rc<RefCell<TensorContext>>,
    pub weights : TensorRef,
    pub bias : TensorRef,
    pub activation_function: ActivationFunction,
    relevant_tensors : Vec<TensorRef>
}

impl Nueron {
    pub fn new(tensor_context: Rc<RefCell<TensorContext>>, weight_count: usize, activation_function: ActivationFunction) -> Nueron{
        let random_weights = tensor_context.borrow_mut().new_tensor(vec![weight_count], vec![rand::random::<f64>()]);
        let random_bias = tensor_context.borrow_mut().new_tensor(vec![1], vec![rand::random::<f64>()]);
        Nueron {
            tensor_context,
            weights: random_weights,
            bias: random_bias,
            activation_function,
            relevant_tensors: vec![random_weights, random_bias]
        }
    }

    pub fn initialize(&mut self, input_tensor: TensorRef) {
        // Generate perceptron output tensor
        let context = &mut self.tensor_context.borrow_mut();

        let dot_product_tensor = context.dot_product(input_tensor, self.weights);
        let sum_tensor = context.add(dot_product_tensor, self.bias);

        let activation_function_tensor = context.apply(self.activation_function, sum_tensor);
        self.relevant_tensors.push(dot_product_tensor);
        self.relevant_tensors.push(sum_tensor);
        self.relevant_tensors.push(activation_function_tensor);
    }

    pub fn feed_forward(&self, inputs: TensorRef) -> TensorRef {
        // let tensor_context = &mut self.tensor_context.borrow_mut();
        // let weights = tensor_context.get_tensor(self.weights);
        // let bias = tensor_context.get_tensor(self.bias);
        // let weighted_sum = tensor_context.add(tensor_context.mul(self.weights, inputs.tensor_ref), self.bias);
        // activation_function::apply(self.activation_function, weighted_sum)
        0
    }
}