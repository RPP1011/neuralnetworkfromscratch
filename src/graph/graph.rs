use std::rc::Rc;

use crate::{layers::layers::layers::Layer, math::tensor::Tensor};

use super::{loss_function::LossFunction, network_metric::Metric, optimizer::Optimizer};


pub struct Sequential {
    pub layers: std::vec::Vec<Rc<dyn Layer>>
}

impl Model for Sequential {
    // For now this just fixes the layer weights so the model can be used for testing
    fn compile(&self, optimizer: Optimizer, loss: LossFunction, metrics: Vec<Metric>) {
        
    }

    fn fit(&self, data:Tensor, labels:Tensor, epochs: usize) {
        // Split tensor into input tensors, assume that first dimension is always the input count
        let data_shape = data.shape;
        let input_count = data_shape[0];
        let input_shape = data_shape[1..].to_vec();
        let input_shape_product = input_shape.iter().fold(1, |acc, x| acc * x);

        let input_tensors: Vec<Tensor> = (0..input_count).map(|i| {
            let start = i * input_shape_product;
            let end = start + input_shape_product;
            Tensor::new(input_shape.clone(), data.data[start..end].to_vec())
        }).collect();
        

        for _ in 0..epochs {
            for input in input_tensors.iter() {
                let mut previous_layer_output = input.clone();
                for layer in self.layers.iter() {
                    previous_layer_output = layer.forward(previous_layer_output);
                }

                let final_output = previous_layer_output;

                for layer in self.layers.iter().rev() {
                    layer.backward();
                }
            }
        }
    }

    fn predict(&self, data:Vec<f64>) -> f64 {
        todo!()
    }

    fn evaluate(&self, data:Vec<f64>, labels:Vec<f64>) -> (f64, f64) {
        todo!()
    }

    fn save(&self) {
        todo!()
    }
}

pub trait Model {
    fn compile(&self, optimizer: Optimizer, loss: LossFunction, metrics: Vec<Metric>);
    fn fit(&self, data:Tensor, labels:Tensor, epochs: usize);
    fn predict(&self, data:Vec<f64>) -> f64;
    fn evaluate(&self, data:Vec<f64>, labels:Vec<f64>) -> (f64, f64);
    fn save(&self);
}

