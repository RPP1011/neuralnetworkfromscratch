use core::str;
use std::{cell::RefCell, rc::Rc, vec};

use crate::nuerons::activation_function;

use super::tensor::{self, Operation, Tensor};

pub type TensorRef = usize;

#[derive(Debug)]
pub struct TensorContext {
    tensors: Vec<Tensor>,
    self_reference: Option<Rc<RefCell<TensorContext>>>,
}
#[macro_export]
macro_rules! create_tensor_context {
    ($size:expr) => {{
        let tensor_context = tensor_context::TensorContext::new($size);
        let tensor_context_ref = Rc::new(RefCell::new(tensor_context));
        tensor_context_ref
            .as_ref()
            .borrow_mut()
            .set_self_reference(tensor_context_ref.clone());
        tensor_context_ref
    }};
}

impl TensorContext {
    pub fn new(capacity: usize) -> TensorContext {
        let context = TensorContext {
            tensors: Vec::with_capacity(capacity),
            self_reference: None,
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

    pub fn get_tensor(&self, tensor_ref: TensorRef) -> Tensor {
        self.tensors[tensor_ref].clone()
    }

    pub fn add(&mut self, tensor_ref1: TensorRef, tensor_ref2: TensorRef) -> TensorRef {
        let tensors = &mut self.tensors;
        let tensor1 = &tensors[tensor_ref1];
        let tensor2 = &tensors[tensor_ref2];
        let data = tensor1
            .data
            .iter()
            .zip(tensor2.data.iter())
            .map(|(a, b)| a + b)
            .collect();
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

    pub fn sum(&mut self, tensor_ref: TensorRef) -> TensorRef {
        let tensors = &mut self.tensors;
        let tensor = &tensors[tensor_ref];
        let sum = tensor.data.iter().sum();
        let data = vec![sum];
        let grad: Option<Vec<f64>> = None;
        let operation = Some(Operation::Sum);
        let tensor = Tensor {
            shape: vec![1],
            tensor_context: self.self_reference.as_mut().unwrap().clone(),
            tensor_ref: tensors.len(),
            data,
            grad,
            operation,
        };
        tensors.push(tensor);
        tensors.len() - 1
    }

    pub fn dot_product(&mut self, tensor_ref1: TensorRef, tensor_ref2: TensorRef) -> TensorRef {
        let mul_result = self.mul(tensor_ref1, tensor_ref2);
        self.sum(mul_result)
    }

    pub fn mul(&mut self, tensor_ref1: TensorRef, tensor_ref2: TensorRef) -> TensorRef {
        let tensors = &mut self.tensors;
        let tensor1 = &tensors[tensor_ref1];
        let tensor2 = &tensors[tensor_ref2];
        let data = tensor1
            .data
            .iter()
            .zip(tensor2.data.iter())
            .map(|(a, b)| a * b)
            .collect();
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

    pub fn backwards(&mut self, tensor_ref: TensorRef) {
        let tensor = &self.tensors[tensor_ref];
        let tensor_size = tensor.data.len();
        let output_grad = tensor.grad.clone().unwrap_or_else(|| vec![0.0; tensor_size]);
        let operation = tensor.operation.clone();
    
        match operation {
            Some(Operation::Add(predecessors)) => {
                for predecessor_ref in predecessors {
                    let grad = &mut self.tensors[predecessor_ref].grad;
                    match grad {
                        Some(grad) => grad.iter_mut().zip(&output_grad).for_each(|(a, b)| *a = *b),
                        None => self.tensors[predecessor_ref].grad = Some(output_grad.clone()),
                    }
                }
            }
            Some(Operation::Mul(left, right)) => {
                let right_tensor_data = self.tensors[right].data.clone();
                let left_tensor_data = self.tensors[left].data.clone();
    
                if let Some(left_grad) = &mut self.tensors[left].grad {
                    left_grad.iter_mut().zip(&output_grad).zip(&right_tensor_data).for_each(|((a, b), c)| *a = b * c);
                } else {
                    self.tensors[left].grad = Some(output_grad.iter().zip(&right_tensor_data).map(|(a, b)| a * b).collect());
                }
    
                if let Some(right_grad) = &mut self.tensors[right].grad {
                    right_grad.iter_mut().zip(&output_grad).zip(&left_tensor_data).for_each(|((a, b), c)| *a = b * c);
                } else {
                    self.tensors[right].grad = Some(output_grad.iter().zip(&left_tensor_data).map(|(a, b)| a * b).collect());
                }
            }
            _ => {}
        }
    }
    

    pub fn apply(
        &mut self,
        activation_function: activation_function::ActivationFunction,
        tensor_ref: TensorRef,
    ) -> TensorRef {
        match activation_function {
            activation_function::ActivationFunction::Sigmoid => todo!(),
            activation_function::ActivationFunction::ReLU => {
                let tensor = &self.tensors[tensor_ref];
                let data = tensor.data.iter().map(|a| a.max(0.0)).collect();
                let grad: Option<Vec<f64>> = None;
                let operation = Some(Operation::ReLU(tensor_ref));
                let tensor = Tensor {
                    shape: tensor.shape.clone(),
                    tensor_context: self.self_reference.as_mut().unwrap().clone(),
                    tensor_ref: self.tensors.len(),
                    data,
                    grad,
                    operation,
                };
                self.tensors.push(tensor);
                self.tensors.len() - 1
            }
            activation_function::ActivationFunction::Tanh => todo!(),
            activation_function::ActivationFunction::Softmax => todo!(),
            activation_function::ActivationFunction::LeakyReLU => todo!(),
        }
    }

    pub fn add_inplace(
        &mut self,
        tensor_ref1: TensorRef,
        tensor_ref2: TensorRef,
        output_tensor_ref: TensorRef,
    ) {
        let tensors = &mut self.tensors;
        let tensor1 = &tensors[tensor_ref1];
        let tensor2 = &tensors[tensor_ref2];
        let data = tensor1
            .data
            .iter()
            .zip(tensor2.data.iter())
            .map(|(a, b)| a + b)
            .collect();
        tensors[output_tensor_ref].data = data;
    }

    pub fn sum_inplace(&mut self, tensor_ref: TensorRef, output_tensor_ref: TensorRef) {
        let tensor = &self.tensors[tensor_ref];
        let sum = tensor.data.iter().sum();
        self.tensors[output_tensor_ref].data = vec![sum];
    }

    pub fn dot_product_inplace(
        &mut self,
        tensor_ref1: TensorRef,
        tensor_ref2: TensorRef,
        output_tensor_ref: TensorRef,
    ) {
        let mul_result = self.mul(tensor_ref1, tensor_ref2);
        self.sum_inplace(mul_result, output_tensor_ref);
    }

    pub fn mul_inplace(
        &mut self,
        tensor_ref1: TensorRef,
        tensor_ref2: TensorRef,
        output_tensor_ref: TensorRef,
    ) {
        let tensors = &mut self.tensors;
        let tensor1 = &tensors[tensor_ref1];
        let tensor2 = &tensors[tensor_ref2];
        let data = tensor1
            .data
            .iter()
            .zip(tensor2.data.iter())
            .map(|(a, b)| a * b)
            .collect();
        tensors[output_tensor_ref].data = data;
    }

    pub fn apply_inplace(
        &mut self,
        activation_function: activation_function::ActivationFunction,
        tensor_ref: TensorRef,
        output_tensor_ref: TensorRef,
    ) {
        match activation_function {
            activation_function::ActivationFunction::Sigmoid => todo!(),
            activation_function::ActivationFunction::ReLU => {
                let tensor = &self.tensors[tensor_ref];
                let data = tensor.data.iter().map(|a| a.max(0.0)).collect();
                self.tensors[output_tensor_ref].data = data;
            }
            activation_function::ActivationFunction::Tanh => todo!(),
            activation_function::ActivationFunction::Softmax => todo!(),
            activation_function::ActivationFunction::LeakyReLU => todo!(),
        };
    }

    pub fn set_grad(&mut self, tensor_ref: TensorRef, grad: Vec<f64>) {
        self.tensors[tensor_ref].grad = Some(grad);
    }

    pub fn set_data(&mut self, tensor_ref: TensorRef, data: Vec<f64>) {
        self.tensors[tensor_ref].data = data;
    }

    pub fn update_data_from_grad(&mut self, tensor_ref: TensorRef, step: f64) {
        let tensor = &mut self.tensors[tensor_ref];
        let grad = tensor.grad.clone().unwrap();
        tensor
            .data
            .iter_mut()
            .zip(grad.iter())
            .for_each(|(data, grad)| *data += step * grad);
    }
}

#[cfg(test)]
mod tests {
    use crate::math::tensor_context;

    use super::*;

    //#region
    #[test]
    fn test_add() {
        let tensor_context = create_tensor_context!(20);
        let tensor_ref1 = tensor_context.borrow_mut().new_tensor(vec![1], vec![1.0]);
        let tensor_ref2 = tensor_context.borrow_mut().new_tensor(vec![1], vec![1.0]);
        let tensor_ref3 = tensor_context.borrow_mut().add(tensor_ref1, tensor_ref2);
        let tensor = tensor_context.borrow_mut().get_tensor(tensor_ref3);
        assert_eq!(tensor.data, vec![2.0]);
    }

    #[test]
    fn test_add_inplace() {
        let tensor_context = create_tensor_context!(20);
        let tensor_ref1 = tensor_context.borrow_mut().new_tensor(vec![1], vec![1.0]);
        let tensor_ref2 = tensor_context.borrow_mut().new_tensor(vec![1], vec![1.0]);
        let tensor_ref3 = tensor_context.borrow_mut().add(tensor_ref1, tensor_ref2);
        let tensor = tensor_context.borrow_mut().get_tensor(tensor_ref3);
        assert_eq!(tensor.data, vec![2.0]);

        tensor_context
            .borrow_mut()
            .add_inplace(tensor_ref1, tensor_ref2, tensor_ref3);
        let tensor = tensor_context.borrow_mut().get_tensor(tensor_ref3);
        assert_eq!(tensor.data, vec![2.0]);

        tensor_context
            .borrow_mut()
            .add_inplace(tensor_ref1, tensor_ref2, tensor_ref3);
        let tensor = tensor_context.borrow_mut().get_tensor(tensor_ref3);
        assert_eq!(tensor.data, vec![2.0]);
    }

    #[test]
    pub fn test_add_inplace_after_modify() {
        let tensor_context = create_tensor_context!(20);
        let tensor_ref1 = tensor_context.borrow_mut().new_tensor(vec![1], vec![1.0]);
        let tensor_ref2 = tensor_context.borrow_mut().new_tensor(vec![1], vec![1.0]);
        let tensor_ref3 = tensor_context.borrow_mut().add(tensor_ref1, tensor_ref2);
        let tensor = tensor_context.borrow_mut().get_tensor(tensor_ref3);
        assert_eq!(tensor.data, vec![2.0]);

        tensor_context.borrow_mut().set_data(tensor_ref1, vec![2.0]);
        tensor_context.borrow_mut().set_data(tensor_ref2, vec![2.0]);
        tensor_context
            .borrow_mut()
            .add_inplace(tensor_ref1, tensor_ref2, tensor_ref3);
        let tensor = tensor_context.borrow_mut().get_tensor(tensor_ref3);
        assert_eq!(tensor.data, vec![4.0]);
    }

    #[test]
    pub fn test_sum() {
        let tensor_context = create_tensor_context!(20);
        let tensor_ref1 = tensor_context
            .borrow_mut()
            .new_tensor(vec![2], vec![1.0, 1.0]);
        let tensor_ref2 = tensor_context.borrow_mut().sum(tensor_ref1);
        let tensor = tensor_context.borrow_mut().get_tensor(tensor_ref2);
        assert_eq!(tensor.data, vec![2.0]);
    }

    #[test]
    pub fn test_sum_inplace() {
        let tensor_context = create_tensor_context!(20);
        let tensor_ref1 = tensor_context
            .borrow_mut()
            .new_tensor(vec![2], vec![1.0, 1.0]);
        let tensor_ref2 = tensor_context.borrow_mut().sum(tensor_ref1);
        let tensor = tensor_context.borrow_mut().get_tensor(tensor_ref2);
        assert_eq!(tensor.data, vec![2.0]);

        tensor_context
            .borrow_mut()
            .set_data(tensor_ref1, vec![2.0, 2.0]);

        tensor_context
            .borrow_mut()
            .sum_inplace(tensor_ref1, tensor_ref2);
        let tensor = tensor_context.borrow_mut().get_tensor(tensor_ref2);
        assert_eq!(tensor.data, vec![4.0]);
    }

    //#endregion
    #[test]
    pub fn test_mul() {
        let tensor_context = create_tensor_context!(20);
        let tensor_ref1 = tensor_context.borrow_mut().new_tensor(vec![1], vec![2.0]);
        let tensor_ref2 = tensor_context.borrow_mut().new_tensor(vec![1], vec![2.0]);
        let tensor_ref3 = tensor_context.borrow_mut().mul(tensor_ref1, tensor_ref2);
        let tensor = tensor_context.borrow_mut().get_tensor(tensor_ref3);
        assert_eq!(tensor.data, vec![4.0]);
    }

    #[test]
    pub fn test_mul_inplace() {
        let tensor_context = create_tensor_context!(20);
        let tensor_ref1 = tensor_context.borrow_mut().new_tensor(vec![1], vec![2.0]);
        let tensor_ref2 = tensor_context.borrow_mut().new_tensor(vec![1], vec![2.0]);
        let tensor_ref3 = tensor_context.borrow_mut().mul(tensor_ref1, tensor_ref2);
        let tensor = tensor_context.borrow_mut().get_tensor(tensor_ref3);
        assert_eq!(tensor.data, vec![4.0]);

        tensor_context.borrow_mut().set_data(tensor_ref1, vec![3.0]);
        tensor_context
            .borrow_mut()
            .mul_inplace(tensor_ref1, tensor_ref2, tensor_ref3);
        let tensor = tensor_context.borrow_mut().get_tensor(tensor_ref3);
        assert_eq!(tensor.data, vec![6.0]);
    }

    #[test]
    pub fn test_backwards_add() {
        let tensor_context = create_tensor_context!(20);
        let tensor_ref1 = tensor_context.borrow_mut().new_tensor(vec![1], vec![1.0]);
        let tensor_ref2 = tensor_context.borrow_mut().new_tensor(vec![1], vec![1.0]);
        let tensor_ref3 = tensor_context.borrow_mut().add(tensor_ref1, tensor_ref2);
        let tensor = tensor_context.borrow_mut().get_tensor(tensor_ref3);
        assert_eq!(tensor.data, vec![2.0]);

        let tensor1 = tensor_context.borrow_mut().get_tensor(tensor_ref1);
        let tensor2 = tensor_context.borrow_mut().get_tensor(tensor_ref2);
        assert_eq!(tensor1.grad, None);
        assert_eq!(tensor2.grad, None);

        tensor_context.borrow_mut().set_grad(tensor_ref3, vec![1.0]);
        tensor_context.borrow_mut().backwards(tensor_ref3);

        let tensor1 = tensor_context.borrow_mut().get_tensor(tensor_ref1);
        let tensor2 = tensor_context.borrow_mut().get_tensor(tensor_ref2);
        assert_eq!(tensor1.grad, Some(vec![1.0]));
        assert_eq!(tensor2.grad, Some(vec![1.0]));
    }

    #[test]
    pub fn test_backwards_mul() {
        let tensor_context = create_tensor_context!(20);
        let tensor_ref1 = tensor_context.borrow_mut().new_tensor(vec![1], vec![1.0]);
        let tensor_ref2 = tensor_context.borrow_mut().new_tensor(vec![1], vec![2.0]);

        assert_eq!(tensor_context.borrow_mut().get_tensor(tensor_ref1).grad, None);
        assert_eq!(tensor_context.borrow_mut().get_tensor(tensor_ref2).grad, None);

        let tensor_ref3 = tensor_context.borrow_mut().mul(tensor_ref1, tensor_ref2);
        let tensor = tensor_context.borrow_mut().get_tensor(tensor_ref3);
        assert_eq!(tensor.data, vec![2.0]);

        tensor_context.borrow_mut().set_grad(tensor_ref3, vec![1.0]);

        tensor_context.borrow_mut().backwards(tensor_ref3);

        let tensor1 = tensor_context.borrow_mut().get_tensor(tensor_ref1);
        let tensor2 = tensor_context.borrow_mut().get_tensor(tensor_ref2);

        assert_eq!(tensor1.grad, Some(vec![2.0]));
        assert_eq!(tensor2.grad, Some(vec![1.0]));

    }
}
