use std::{cell::RefCell, rc::Rc};

use crate::{
    math::{composite_operations::CompositeOperation, tensor_context::{TensorContext, TensorRef}},
    nuerons::activation_function::ActivationFunction,
};

#[derive(Debug, Clone)]
pub struct Neuron {
    pub activation_function: ActivationFunction,
    tensor_context: Rc<RefCell<TensorContext>>,
    weights : Option<TensorRef>,
    bias : Option<TensorRef>,
    dot_product: Option<CompositeOperation>,
    sum_output: Option<TensorRef>,
    activation_output: Option<TensorRef>,
}

impl Neuron {
    pub fn new(
        tensor_context: Rc<RefCell<TensorContext>>,
        weight_count: usize,
        activation_function: ActivationFunction,
    ) -> Neuron {
        let random_weights = tensor_context
            .borrow_mut()
            .new_tensor(vec![weight_count], vec![rand::random::<f64>()]);
        let random_bias = tensor_context
            .borrow_mut()
            .new_tensor(vec![1], vec![rand::random::<f64>()]);
        Neuron {
            tensor_context,
            activation_function,
            bias: Some(random_bias),
            weights: Some(random_weights),
            dot_product: None,
            sum_output: None,
            activation_output: None
        }
    }

    pub fn new_ones(
        tensor_context: Rc<RefCell<TensorContext>>,
        weight_count: usize,
        activation_function: ActivationFunction,
    ) -> Neuron {
        let ones_weights = tensor_context
            .borrow_mut()
            .new_tensor(vec![weight_count], vec![1.0]);
        let ones_bias = tensor_context.borrow_mut().new_tensor(vec![1], vec![0.0]);
        Neuron {
            tensor_context,
            activation_function,
            bias: Some(ones_bias),
            weights: Some(ones_weights),
            dot_product: None,
            sum_output: None,
            activation_output: None,
        }
    }

    pub fn initialize(&mut self, input_tensor: TensorRef) -> TensorRef {
        self.dot_product = Some(CompositeOperation::dot_product(self.tensor_context.clone(), input_tensor, self.weights.unwrap()));
        let dot_output = self.dot_product.as_ref().unwrap().output_tensor;
        self.sum_output = Some(self.tensor_context.borrow_mut().add(dot_output, self.bias.unwrap()));
        self.activation_output = Some(self.tensor_context.borrow_mut().apply( self.activation_function, self.sum_output.unwrap()));
        self.activation_output.unwrap()
    }

    pub fn feed_forward(&self, input_tensor: TensorRef) -> TensorRef {
        let bias = self.bias.unwrap();
        self.dot_product.as_ref().unwrap().perform();
        self.tensor_context.borrow_mut().add_inplace(
            self.dot_product.as_ref().unwrap().output_tensor,
            bias,
            self.sum_output.unwrap(),
        );

        self.tensor_context.borrow_mut().apply_inplace(
            self.activation_function,
            self.sum_output.unwrap(),
            self.activation_output.unwrap(),
        );

        self.activation_output.unwrap()
    }

    pub fn get_parameters(&self) -> Vec<TensorRef> {
        vec![self.weights.unwrap(), self.bias.unwrap()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_nueron() {
        let tensor_context = Rc::new(RefCell::new(TensorContext::new(1024)));
        tensor_context
            .borrow_mut()
            .set_self_reference(tensor_context.clone());

        let mut nueron = Neuron::new(tensor_context.clone(), 2, ActivationFunction::ReLU);
        let input_tensor = tensor_context
            .borrow_mut()
            .new_tensor(vec![2], vec![1.0, 1.0]);
        nueron.initialize(input_tensor);
        let output_tensor = nueron.feed_forward(input_tensor);
        let output = tensor_context.borrow().get_tensor(output_tensor);
        println!("{:?}", output.data);
    }

    #[test]
    fn test_neuron_output_backpropagation() {
        let tensor_context = Rc::new(RefCell::new(TensorContext::new(1024)));
        tensor_context
            .borrow_mut()
            .set_self_reference(tensor_context.clone());

        let mut nueron = Neuron::new(tensor_context.clone(), 2, ActivationFunction::ReLU);
        let input_tensor = tensor_context
            .borrow_mut()
            .new_tensor(vec![2], vec![1.0, 1.0]);
        nueron.initialize(input_tensor);
        let output_tensor = nueron.feed_forward(input_tensor);

        tensor_context
            .borrow_mut()
            .set_grad(output_tensor, vec![3.0]);

        let output = tensor_context.borrow().get_tensor(output_tensor);
        println!("{:?}", output.data);
        let _output_grad = tensor_context.borrow_mut().new_tensor(vec![1], vec![1.0]);

        tensor_context.borrow_mut().backwards(output_tensor);
        let input_grad = tensor_context.borrow().get_tensor(input_tensor);
        println!("{:?}", input_grad.data);
    }

    #[test]
    fn test_neuron_feed_forward_initializing() {
        let tensor_context = Rc::new(RefCell::new(TensorContext::new(1024)));
        tensor_context
            .borrow_mut()
            .set_self_reference(tensor_context.clone());

        let mut neuron = Neuron::new(tensor_context.clone(), 2, ActivationFunction::ReLU);

        // Test with input tensor [1.0, 1.0]
        let input_tensor1 = tensor_context
            .borrow_mut()
            .new_tensor(vec![2], vec![1.0, 1.0]);
        neuron.initialize(input_tensor1);
        let output_tensor1 = neuron.feed_forward(input_tensor1);
        let output1 = tensor_context.borrow().get_tensor(output_tensor1);
        println!("{:?}", output1.data);

        // Test with input tensor [0.5, 0.5]
        let input_tensor2 = tensor_context
            .borrow_mut()
            .new_tensor(vec![2], vec![0.5, 0.5]);
        neuron.initialize(input_tensor2);
        let output_tensor2 = neuron.feed_forward(input_tensor2);
        let output2 = tensor_context.borrow().get_tensor(output_tensor2);
        println!("{:?}", output2.data);

        // Test with input tensor [2.0, 2.0]
        let input_tensor3 = tensor_context
            .borrow_mut()
            .new_tensor(vec![2], vec![2.0, 2.0]);
        neuron.initialize(input_tensor3);
        let output_tensor3 = neuron.feed_forward(input_tensor3);
        let output3 = tensor_context.borrow().get_tensor(output_tensor3);
        println!("{:?}", output3.data);
    }

    #[test]
    fn test_neuron_feed_forward_without_initializing() {
        let tensor_context = Rc::new(RefCell::new(TensorContext::new(1024)));
        tensor_context
            .borrow_mut()
            .set_self_reference(tensor_context.clone());

        let mut neuron = Neuron::new(tensor_context.clone(), 2, ActivationFunction::ReLU);

        // Test with input tensor [1.0, 1.0]
        let input_tensor1 = tensor_context
            .borrow_mut()
            .new_tensor(vec![2], vec![1.0, 1.0]);
        neuron.initialize(input_tensor1); // only initialization
        let output_tensor1 = neuron.feed_forward(input_tensor1);
        let output1 = tensor_context.borrow().get_tensor(output_tensor1);
        
        println!("{:?}", output1.data);

        // Test with input tensor [0.5, 0.5]
        let input_tensor2 = tensor_context
            .borrow_mut()
            .new_tensor(vec![2], vec![0.5, 0.5]);
        let output_tensor2 = neuron.feed_forward(input_tensor2);
        let output2 = tensor_context.borrow().get_tensor(output_tensor2);
        println!("{:?}", output2.data);

        // Test with input tensor [2.0, 2.0]
        let input_tensor3 = tensor_context
            .borrow_mut()
            .new_tensor(vec![2], vec![2.0, 2.0]);
        let output_tensor3 = neuron.feed_forward(input_tensor3);
        let output3 = tensor_context.borrow().get_tensor(output_tensor3);
        println!("{:?}", output3.data);
    }
}
