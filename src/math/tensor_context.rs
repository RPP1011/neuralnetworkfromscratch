use core::str;
use std::{cell::RefCell, rc::Rc, vec};

use super::tensor::{self, Operation, Tensor};

pub type TensorRef = usize;

#[derive(Debug)]
pub struct TensorContext {
    tensors : Vec<Tensor>,
    self_reference: Rc<RefCell<TensorContext>>,
    capacity : usize
}

impl TensorContext {
    pub fn get_tensor(self, tensor_ref : TensorRef) -> Tensor {
        self.tensors[tensor_ref].clone()
    }

    pub fn add(&mut self, tensor_ref1 : TensorRef, tensor_ref2 : TensorRef) -> TensorRef {
        let tensors = &mut self.tensors;
        let tensor1 = &tensors[tensor_ref1];
        let tensor2 = &tensors[tensor_ref2];
        let data = tensor1.data.iter().zip(tensor2.data.iter()).map(|(a, b)| a + b).collect();
        let grad: Option<Vec<f64>> = None;
        let operation = Some(Operation::Add(tensor_ref1, tensor_ref2));
        let tensor = Tensor {
            tensor_context: self.self_reference.clone(),
            tensor_ref: tensors.len(),
            data,
            grad,
            operation,
        };
        tensors.push(tensor);
        tensors.len() - 1
    }

    pub fn mul(&mut self, tensor_ref1 : TensorRef, tensor_ref2 : TensorRef) -> TensorRef {
        let tensors = &mut self.tensors;
        let tensor1 = &tensors[tensor_ref1];
        let tensor2 = &tensors[tensor_ref2];
        let data = tensor1.data.iter().zip(tensor2.data.iter()).map(|(a, b)| a * b).collect();
        let grad: Option<Vec<f64>> = None;
        let operation = Some(Operation::Mul(tensor_ref1, tensor_ref2));
        let tensor = Tensor {
            tensor_context: self.self_reference.clone(),
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

        let operation = &self.tensors[tensor_ref].operation.unwrap();
        match operation {
            Operation::Add(tensor_ref1, tensor_ref2) => {
                {
                    let tensor1 = &self.tensors[*tensor_ref1];
                    let tensor1_grad = tensor1.grad.as_ref().unwrap();
                    let tensor1_grad = tensor1_grad.iter().zip(output_grad.iter()).map(|(a, b)| a + b).collect();
                    self.tensors[*tensor_ref1].grad = Some(tensor1_grad);
                }
                {
                    let tensor2 = &self.tensors[*tensor_ref2];
                    let tensor2_grad = tensor2.grad.as_ref().unwrap();
                    let tensor2_grad = tensor2_grad.iter().zip(output_grad.iter()).map(|(a, b)| a + b).collect();
                    self.tensors[*tensor_ref2].grad = Some(tensor2_grad);
                }
            }
            _ => {}
        }
    }
}