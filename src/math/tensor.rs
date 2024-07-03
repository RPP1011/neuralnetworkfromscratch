use std::{
    borrow::{Borrow, BorrowMut},
    cell::RefCell,
    ops::{Add, Div, Mul, Sub},
    rc::Rc,
    sync::Mutex,
};

#[derive(Debug, Clone)]
pub struct Tensor {
    shape: Vec<usize>,
    data: Vec<f64>,
    grad: Option<Vec<f64>>,
    operation: Option<Operation>,
    precursor: Option<Vec<Tensor>>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f64>) -> Tensor {
        Tensor {
            shape,
            data,
            grad: None,
            operation: None,
            precursor: None,
        }
    }

    pub fn backwards(self) -> Vec<Tensor> {
        let self_clone: Tensor = self.clone();

        match self.operation {
            Some(mut operation) => operation.backwards(self_clone),
            None => vec![self_clone],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Operation {
    Add,
    Mul,
}

impl Operation {
    pub fn backwards(&mut self, tensor: Tensor) -> Vec<Tensor> {
        let tensor_clone = tensor.clone();

        let out_grad = tensor.grad.unwrap_or(vec![0.0; tensor.data.len()]);

        // Precursors are the tensors that were used to create the current tensor, always either singleton or binary depending on the operation
        let precursors = match &tensor.precursor {
            Some(precursors) => precursors.clone(),
            None => vec![],
        };

        match self {
            Operation::Add => {
                
                let mut tensors = Vec::new();
                for precursor in precursors {
                    let mut precursor_clone = precursor.clone();
                    let mut grad = precursor.grad.unwrap_or(vec![0.0; precursor.data.len()]);
                    for i in 0..grad.len() {
                        grad[i] += out_grad[i];
                    }
                    precursor_clone.grad = Some(grad);
                    tensors.push(precursor_clone);
                }
                tensors
            }
            Operation::Mul => {
                let mut left_grad = vec![0.0; precursors[0].data.len()];
                let mut right_grad = vec![0.0; precursors[1].data.len()];
                for i in 0..out_grad.len() {
                    left_grad[i] += out_grad[i] * precursors[1].data[i];
                    right_grad[i] += out_grad[i] * precursors[0].data[i];
                }

                let mut left_precursor = precursors[0].clone();
                let mut right_precursor = precursors[1].clone();
                left_precursor.grad = Some(left_grad);
                right_precursor.grad = Some(right_grad);
                
                vec![left_precursor, right_precursor]
            }
        }
    }
}

impl Add<Tensor> for Tensor {
    type Output = Tensor;

    fn add(self, rhs: Tensor) -> Tensor {
        let mut data = vec![0.0; self.data.len()];
        for i in 0..self.data.len() {
            data[i] = self.data[i] + rhs.data[i];
        }

        let self_clone: Tensor = self.clone();

        Tensor {
            shape: self.shape,
            data,
            grad: None,
            operation: Some(Operation::Add),
            precursor: Some(vec![self_clone, rhs.clone()]),
        }
    }
}

impl Mul<Tensor> for Tensor {
    type Output = Tensor;

    fn mul(self, rhs: Tensor) -> Self::Output {
        let mut data = vec![0.0; self.data.len()];
        for i in 0..self.data.len() {
            data[i] = self.data[i] * rhs.data[i];
        }

        let self_clone: Tensor = self.clone();

        Tensor {
            shape: self.shape,
            data,
            grad: None,
            operation: Some(Operation::Mul),
            precursor: Some(vec![self_clone, rhs.clone()]),
        }
    }
}

impl Div<Tensor> for Tensor {
    type Output = Tensor;

    fn div(self, rhs: Tensor) -> Self::Output {
        let mut data = vec![0.0; self.data.len()];
        for i in 0..self.data.len() {
            data[i] = self.data[i] / rhs.data[i];
        }

        let self_clone: Tensor = self.clone();
        let mut rhs_clone: Tensor = rhs.clone();
        for i in 0..rhs_clone.data.len() {
            rhs_clone.data[i] = 1.0 / rhs_clone.data[i];
        }


        Tensor {
            shape: self.shape,
            data,
            grad: None,
            operation: Some(Operation::Mul),
            precursor: Some(vec![self_clone, rhs_clone]),
        }
    }
}

impl Sub<Tensor> for Tensor {
    type Output = Tensor;

    fn sub(self, rhs: Tensor) -> Self::Output {
        let mut data = vec![0.0; self.data.len()];
        for i in 0..self.data.len() {
            data[i] = self.data[i] - rhs.data[i];
        }

        let mut negative_rhs = rhs.clone();
        for i in 0..negative_rhs.data.len() {
            negative_rhs.data[i] = -negative_rhs.data[i];
        }

        let self_clone: Tensor = self.clone();

        Tensor {
            shape: self.shape,
            data,
            grad: None,
            operation: Some(Operation::Add),
            precursor: Some(vec![self_clone, negative_rhs]),
        }
    }   
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }

    fn ne(&self, other: &Self) -> bool {
        self.shape != other.shape || self.data != other.data
    }
}

// Set up basic tests for 1 + 1 and gradient calculation
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        let a = Tensor::new(vec![1], vec![1.0]);
        let b = Tensor::new(vec![1], vec![1.0]);

        let c = Tensor {
            shape: vec![1],
            data: vec![2.0],
            grad: None,
            operation: None,
            precursor: None,
        };

        assert_eq!(c, a + b);
    }

    #[test]
    fn test_add_wrong() {
        let a = Tensor::new(vec![1], vec![1.0]);
        let b = Tensor::new(vec![1], vec![1.0]);

        let c = Tensor {
            shape: vec![1],
            data: vec![3.0],
            grad: None,
            operation: None,
            precursor: None,
        };

        assert_ne!(c, a + b);
    }

    #[test]
    fn test_backwards() {
        let a = Tensor::new(vec![1], vec![1.0]);
        let b = Tensor::new(vec![1], vec![2.0]);

        let c = a + b;
        println!("{:?}", c.shape);

        let mut d = c.clone();
        d.grad = Some(vec![3.0]);
        let e = d.backwards();
        e.iter().for_each(|x| println!("{:?}", x));
    }

    #[test]
    fn test_mul() {
        let a = Tensor::new(vec![1], vec![1.0]);
        let b = Tensor::new(vec![1], vec![2.0]);

        let c = Tensor {
            shape: vec![1],
            data: vec![2.0],
            grad: None,
            operation: None,
            precursor: None,
        };

        assert_eq!(c, a * b);
    }

    #[test]
    fn test_mul_wrong() {
        let a = Tensor::new(vec![1], vec![1.0]);
        let b = Tensor::new(vec![1], vec![2.0]);

        let c = Tensor {
            shape: vec![1],
            data: vec![3.0],
            grad: None,
            operation: None,
            precursor: None,
        };

        assert_ne!(c, a * b);
    }

    #[test]
    fn test_backwards_mul() {
        let a = Tensor::new(vec![1], vec![1.0]);
        let b = Tensor::new(vec![1], vec![2.0]);

        let c = a * b;
        println!("{:?}", c.shape);

        assert_eq!(vec![2.0], c.data);

        let mut d = c.clone();
        d.grad = Some(vec![3.0]);
        let e = d.backwards();
        assert_eq!(vec![6.0], e[0].clone().grad.unwrap());
        assert_eq!(vec![3.0], e[1].clone().grad.unwrap());
    }

    #[test]
    fn test_sub() {
        let a = Tensor::new(vec![1], vec![1.0]);
        let b = Tensor::new(vec![1], vec![1.0]);

        let c = Tensor {
            shape: vec![1],
            data: vec![0.0],
            grad: None,
            operation: None,
            precursor: None,
        };

        assert_eq!(c, a - b);
    }

    #[test]
    fn test_sub_wrong() {
        let a = Tensor::new(vec![1], vec![1.0]);
        let b = Tensor::new(vec![1], vec![1.0]);

        let c = Tensor {
            shape: vec![1],
            data: vec![3.0],
            grad: None,
            operation: None,
            precursor: None,
        };

        assert_ne!(c, a - b);
    }

    #[test]
    fn test_backwards_sub() {
        let a = Tensor::new(vec![1], vec![1.0]);
        let b = Tensor::new(vec![1], vec![2.0]);

        let c = a - b;

        assert_eq!(vec![-1.0], c.data);

        let mut d = c.clone();
        d.grad = Some(vec![3.0]);
        let e = d.backwards();
        assert_eq!(vec![3.0], e[0].clone().grad.unwrap());
        assert_eq!(vec![3.0], e[1].clone().grad.unwrap());
    }

    #[test]
    fn test_div() {
        let a = Tensor::new(vec![1], vec![1.0]);
        let b = Tensor::new(vec![1], vec![1.0]);

        let c = Tensor {
            shape: vec![1],
            data: vec![1.0],
            grad: None,
            operation: None,
            precursor: None,
        };

        assert_eq!(c, a / b);
    }

    #[test]
    fn test_div_wrong() {
        let a = Tensor::new(vec![1], vec![1.0]);
        let b = Tensor::new(vec![1], vec![1.0]);

        let c = Tensor {
            shape: vec![1],
            data: vec![3.0],
            grad: None,
            operation: None,
            precursor: None,
        };

        assert_ne!(c, a / b);
    }

    #[test]
    fn test_backwards_div() {
        let a = Tensor::new(vec![1], vec![1.0]);
        let b = Tensor::new(vec![1], vec![2.0]);

        let c = a / b;

        assert_eq!(vec![0.5], c.data);

        let mut d = c.clone();
        d.grad = Some(vec![3.0]);
        let e = d.backwards();
        assert_eq!(vec![1.5], e[0].clone().grad.unwrap());
        assert_eq!(vec![3.0], e[1].clone().grad.unwrap());
    }
}
