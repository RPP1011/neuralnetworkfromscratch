use crate::math::tensor::Tensor;

pub enum LossFunction {
    MeanSquaredError,
    CrossEntropy,
    SparseCrossEntropy,
}

impl LossFunction {
    pub fn loss(&self, input : Vec<Tensor>, desired: Vec<Tensor>) -> Tensor {
        match self {
            LossFunction::MeanSquaredError => LossFunction::mean_squared_error(input, desired),
            LossFunction::CrossEntropy => LossFunction::cross_entropy(input, desired),
            LossFunction::SparseCrossEntropy => LossFunction::sparse_cross_entropy(input, desired),
        }
    }

    fn mean_squared_error(input : Vec<Tensor>, desired: Vec<Tensor>) -> Tensor {
        let loss = input.iter().zip(desired.iter()).map(|(input, desired)| {
            let mut sum = 0.0;
            for i in 0..input.data.len() {
                sum += (input.data[i] - desired.data[i]).powi(2);
            }
            sum
        }).sum();
        Tensor::new(vec![1], vec![loss])
    }

    fn cross_entropy(_input : Vec<Tensor>, _desired: Vec<Tensor>) -> Tensor {
        todo!("Implement Cross Entropy Loss Function")
    }

    fn sparse_cross_entropy(_input : Vec<Tensor>, _desired: Vec<Tensor>) -> Tensor {
        todo!("Implement Sparse Cross Entropy Loss Function")
    }
}