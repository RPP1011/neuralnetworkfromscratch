use core::str;
use std::{cell::RefCell, rc::Rc, vec};

use crate::nuerons::activation_function;

use super::tensor::{self, Operation, Tensor};

pub type TensorRef = usize;

#[derive(Debug)]
pub struct TensorContext {
    tensors : Vec<Tensor>,
    self_reference: Option<Rc<RefCell<TensorContext>>>
}

impl TensorContext {
    pub fn new(capacity: usize) -> TensorContext {
        let context = TensorContext {
            tensors: Vec::with_capacity(capacity),
            self_reference: None
        };
        context
    }

    pub fn set_self_reference(&mut self, self_reference: Rc<RefCell<TensorContext>>) {
        self.self_reference = Some(self_reference);
    }

    pub fn new_tensor(&mut self, shape: Vec<usize>, data: Vec<f64>) -> TensorRef {
        let grad: Option<Vec<f64>> = None;
        let operation: Option<Operation> = None;
        let tensor = Tensor {
            shape,
            tensor_context: self.self_reference.as_mut().unwrap().clone(),
            tensor_ref: self.tensors.len(),
            data,
            grad,
            operation,
        };
        self.tensors.push(tensor);
        self.tensors.len() - 1
    }

    pub fn get_tensor(&self, tensor_ref : TensorRef) -> Tensor {
        self.tensors[tensor_ref].clone()
    }

    pub fn add(&mut self, tensor_ref1 : TensorRef, tensor_ref2 : TensorRef) -> TensorRef {
        let tensors = &mut self.tensors;
        let tensor1 = &tensors[tensor_ref1];
        let tensor2 = &tensors[tensor_ref2];
        let data = tensor1.data.iter().zip(tensor2.data.iter()).map(|(a, b)| a + b).collect();
        let grad: Option<Vec<f64>> = None;
        let operation = Some(Operation::Add(vec![tensor_ref1, tensor_ref2]));
        let tensor = Tensor {
            shape: tensor1.shape.clone(),
            tensor_context: self.self_reference.as_mut().unwrap().clone(),
            tensor_ref: tensors.len(),
            data,
            grad,
            operation,
        };
        tensors.push(tensor);
        tensors.len() - 1
    }

    pub fn dot_product(&mut self, tensor_ref1 : TensorRef, tensor_ref2 : TensorRef) -> TensorRef {
        1
    }


    pub fn mul(&mut self, tensor_ref1 : TensorRef, tensor_ref2 : TensorRef) -> TensorRef {
        let tensors = &mut self.tensors;
        let tensor1 = &tensors[tensor_ref1];
        let tensor2 = &tensors[tensor_ref2];
        let data = tensor1.data.iter().zip(tensor2.data.iter()).map(|(a, b)| a * b).collect();
        let grad: Option<Vec<f64>> = None;
        let operation = Some(Operation::Mul(tensor_ref1, tensor_ref2));
        let tensor = Tensor {
            shape: tensor1.shape.clone(),
            tensor_context: self.self_reference.as_mut().unwrap().clone(),
            tensor_ref: tensors.len(),
            data,
            grad,
            operation,
        };
        tensors.push(tensor);
        tensors.len() - 1
    }

    pub fn backwards(&mut self, tensor_ref : TensorRef) {
        let mut tensor_size = 0;
        let mut output_grad = vec![];
        {
            let tensor = &self.tensors[tensor_ref];
            tensor_size = tensor.data.len();
            output_grad = tensor.grad.clone().unwrap_or(vec![0.0; tensor_size]);
        }

        let mut operation = None;
        {
            let tensor = &self.tensors[tensor_ref];
            operation = tensor.operation.clone();
        }

        match operation {
            Some(Operation::Add(predecessors)) => {
                for predecessor in predecessors.iter() {
                    let predecessor_grad = self.tensors[*predecessor].grad.as_ref().unwrap();
                    let predecessor_grad = predecessor_grad.iter().zip(output_grad.iter()).map(|(a, b)| a + b).collect();
                    self.tensors[*predecessor].grad = Some(predecessor_grad);
                }
            }
            _ => {}
        }
    }

    pub fn apply(&mut self, activation_function: activation_function::ActivationFunction, tensor_ref : TensorRef) -> TensorRef {
        match activation_function {
            activation_function::ActivationFunction::Sigmoid => todo!(),
            activation_function::ActivationFunction::ReLU => todo!(),
            activation_function::ActivationFunction::Tanh => todo!(),
            activation_function::ActivationFunction::Softmax => todo!(),
            activation_function::ActivationFunction::LeakyReLU => todo!(),
        };
        0
    }
}