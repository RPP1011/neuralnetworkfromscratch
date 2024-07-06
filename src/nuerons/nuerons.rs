use std::{cell::RefCell, iter::zip, ops::Mul, rc::Rc};

use graphviz_rust::attributes::weight;

use crate::{math::tensor_context::{TensorContext, TensorRef}, nuerons::activation_function::ActivationFunction};

use super::activation_function;



#[derive(Debug, Clone)]
pub struct Nueron {
    pub activation_function: ActivationFunction,
    tensor_context: Rc<RefCell<TensorContext>>,
    weights : Option<TensorRef>,
    bias : Option<TensorRef>,
    dot_product: Option<TensorRef>,
    activation_input: Option<TensorRef>,
    activation_output: Option<TensorRef>,
}

impl Nueron {
    pub fn new(tensor_context: Rc<RefCell<TensorContext>>, weight_count: usize, activation_function: ActivationFunction) -> Nueron{
        let random_weights = tensor_context.borrow_mut().new_tensor(vec![weight_count], vec![rand::random::<f64>()]);
        let random_bias = tensor_context.borrow_mut().new_tensor(vec![1], vec![rand::random::<f64>()]);
        Nueron {
            tensor_context,
            activation_function,
            bias: Some(random_bias),
            weights: Some(random_weights),
            dot_product: None,
            activation_input: None,
            activation_output: None,
        }
    }

    pub fn initialize(&mut self, input_tensor: TensorRef) -> TensorRef {
        let dot_product_tensor = self.tensor_context.borrow_mut().dot_product(input_tensor, self.weights.unwrap());
        // let sum_tensor = self.tensor_context.borrow_mut().add(dot_product_tensor, self.bias.unwrap());
        // let activation_function_tensor = self.tensor_context.borrow_mut().apply(self.activation_function, sum_tensor);
        
        // self.dot_product = Some(dot_product_tensor);
        // self.activation_input = Some(sum_tensor);
        // self.activation_output = Some(activation_function_tensor);
        // activation_function_tensor
        dot_product_tensor
    }

    pub fn feed_forward(&self, input_tensor: TensorRef) -> TensorRef {
        let tensor_context = &mut self.tensor_context.borrow_mut();
        let weights = self.weights.unwrap();
        let bias = self.bias.unwrap();
        tensor_context.dot_product_inplace(weights, input_tensor, self.dot_product.unwrap());
        tensor_context.add_inplace(self.dot_product.unwrap(), bias, self.activation_input.unwrap());
        tensor_context.apply_inplace(self.activation_function, self.activation_input.unwrap(), self.activation_output.unwrap());
        self.activation_output.unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::tensor::Tensor;

    #[test]
    fn test_nueron() {
        let tensor_context = Rc::new(RefCell::new(TensorContext::new(1024)));
        tensor_context.borrow_mut().set_self_reference(tensor_context.clone());

        let mut nueron = Nueron::new(tensor_context.clone(), 2, ActivationFunction::ReLU);
        let input_tensor = tensor_context.borrow_mut().new_tensor(vec![2], vec![1.0, 1.0]);
        nueron.initialize(input_tensor);
        let output_tensor = nueron.feed_forward(input_tensor);
        let output = tensor_context.borrow().get_tensor(output_tensor);
        println!("{:?}", output.data);
    }

    #[test]
    fn test_neuron_output_backpropagation() {
        let tensor_context = Rc::new(RefCell::new(TensorContext::new(1024)));
        tensor_context.borrow_mut().set_self_reference(tensor_context.clone());

        let mut nueron = Nueron::new(tensor_context.clone(), 2, ActivationFunction::ReLU);
        let input_tensor = tensor_context.borrow_mut().new_tensor(vec![2], vec![1.0, 1.0]);
        nueron.initialize(input_tensor);
        let output_tensor = nueron.feed_forward(input_tensor);
        
        tensor_context.borrow_mut().set_grad(output_tensor, vec![3.0]);
        
        let output = tensor_context.borrow().get_tensor(output_tensor);
        println!("{:?}", output.data);
        let output_grad = tensor_context.borrow_mut().new_tensor(vec![1], vec![1.0]);
        

        tensor_context.borrow_mut().backwards(output_tensor);
        let input_grad = tensor_context.borrow().get_tensor(input_tensor);
        println!("{:?}", input_grad.data);
    }
}