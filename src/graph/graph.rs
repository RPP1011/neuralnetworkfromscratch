use std::{borrow::Borrow, cell::RefCell, rc::Rc};

use crate::{layers::layers::layers::Layer, math::{tensor::{self, Tensor}, tensor_context::{TensorContext, TensorRef}}};

use super::{loss_function::LossFunction, network_metric::Metric, optimizer::Optimizer};


pub struct Sequential {
    pub layers: std::vec::Vec<Box<dyn Layer>>,
    pub context: Rc<RefCell<TensorContext>>,
    output_value: Option<TensorRef>
}

impl Sequential {
    pub fn new(context: Rc<RefCell<TensorContext>>, layers: Vec<Box<dyn Layer>>) -> Sequential {
        Sequential {
            layers,
            context,
            output_value: None
        }
    }
}

impl Model for Sequential {
    // For now this just fixes the layer weights so the model can be used for testing
    fn compile(&mut self, input_shape: Vec<usize>, output_shape: Vec<usize>, optimizer: Optimizer, loss: LossFunction, metrics: Vec<Metric>) {
        // Iterate thropugh all layers and compile them
        let dummy = self.context.borrow_mut().new_tensor(input_shape.clone(), vec![0.0; input_shape.iter().fold(1, |acc, x| acc * x)]);
        let mut last_layer = Some(dummy);
        for layer in self.layers.iter_mut() {
            last_layer = Some(layer.compile(last_layer.unwrap()));
        }
        self.output_value = last_layer; 
    }

    fn fit(&mut self, data_tensor:TensorRef, labels:TensorRef, epochs: usize) {
        // Split tensor into input tensors, assume that first dimension is always the input count
        let data = self.context.borrow_mut().get_tensor(data_tensor).clone();
        let data_shape = data.shape;
        let input_count = data_shape[0];
        let input_shape = data_shape[1..].to_vec();
        let input_shape_product = input_shape.iter().fold(1, |acc, x| acc * x);

        let input_tensors: Vec<TensorRef> = (0..input_count).map(|i| {
            let start = i * input_shape_product;
            let end = start + input_shape_product;
            self.context.borrow_mut().new_tensor(input_shape.clone(), data.data[start..end].to_vec())
        }).collect();
        

        for _ in 0..epochs {
            for input in input_tensors.iter() {
                let mut previous_layer_output = input.clone();
                for layer in self.layers.iter() {
                    previous_layer_output = layer.forward(previous_layer_output);
                }

                let final_output = previous_layer_output;

                // Backpropagate
                self.context.borrow_mut().backwards(final_output);
            }
        }
    }

    fn predict(&mut self, data:Vec<f64>) -> Vec<f64> {
        let input = self.context.borrow_mut().new_tensor(vec![data.len()], data).clone();
            let mut previous_layer_output = input.clone();
            for layer in self.layers.iter() {
                previous_layer_output = layer.forward(previous_layer_output);
            }

            let final_output = previous_layer_output;
        self.context.borrow_mut().get_tensor(final_output).data
    }

    fn evaluate(&self, data:Vec<f64>, labels:Vec<f64>) -> (f64, f64) {
        todo!()
    }

    fn save(&self) {
        todo!()
    }
}

pub trait Model {
    fn compile(&mut self, input_shape: Vec<usize>, output_shape:Vec<usize>, optimizer: Optimizer, loss: LossFunction, metrics: Vec<Metric>);
    fn fit(&mut self, data:TensorRef, labels:TensorRef, epochs: usize);
    fn predict(&mut self, data:Vec<f64>) -> Vec<f64>;
    fn evaluate(&self, data:Vec<f64>, labels:Vec<f64>) -> (f64, f64);
    fn save(&self);
}

