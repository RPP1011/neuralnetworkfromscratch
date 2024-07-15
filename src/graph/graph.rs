use std::{cell::RefCell, iter, rc::Rc};

use crate::{
    layers::layers::layers::Layer,
    math::{
        tensor::Tensor,
        tensor_context::{self, TensorContext, TensorRef},
    },
};

use super::{loss_function::LossFunction, network_metric::Metric, optimizer::Optimizer};

pub struct Sequential {
    pub layers: std::vec::Vec<Box<dyn Layer>>,
    pub context: Rc<RefCell<TensorContext>>,
    output_value: Option<TensorRef>,
    loss_function: LossFunction,
}

impl Sequential {
    pub fn new(context: Rc<RefCell<TensorContext>>, layers: Vec<Box<dyn Layer>>) -> Sequential {
        Sequential {
            layers,
            context,
            output_value: None,
            loss_function: LossFunction::MeanSquaredError,
        }
    }
}

impl Model for Sequential {
    fn compile(
        &mut self,
        input_shape: Vec<usize>,
        _output_shape: Vec<usize>,
        _optimizer: Optimizer,
        loss: LossFunction,
        _metrics: Vec<Metric>,
    ) {
        self.loss_function = loss;

        // Iterate thropugh all layers and compile them
        let dummy = self
            .context
            .borrow_mut()
            .new_tensor(input_shape.clone(), vec![0.0; input_shape.iter().product()]);
        let mut last_layer = Some(dummy);
        for layer in self.layers.iter_mut() {
            last_layer = Some(layer.compile(last_layer.unwrap()));
            println!("Compiled Layer: {:?}", last_layer);
        }
        self.output_value = last_layer;
    }

    fn fit(&mut self, data: Tensor, labels: Tensor, epochs: usize) {
        let data_tensor = self.context.borrow_mut().transfer_tensor(data);
        let labels_tensor = self.context.borrow_mut().transfer_tensor(labels);

        // Split tensor into input tensors, assume that first dimension is always the input count
        let data = self.context.borrow_mut().get_tensor(data_tensor);
        let labels = self.context.borrow_mut().get_tensor(labels_tensor);

        let data_shape = data.shape;
        let input_count = data_shape[0];
        let input_shape = data_shape[1..].to_vec();
        let input_shape_product = input_shape.iter().product::<usize>();

        let input_tensors: Vec<TensorRef> = (0..input_count)
            .map(|i| {
                let start = i * input_shape_product;
                let end = start + input_shape_product;
                self.context
                    .borrow_mut()
                    .new_tensor(input_shape.clone(), data.data[start..end].to_vec())
            })
            .collect();

        let labels_shape = labels.shape[1..].to_vec();
        let labels_shape_product = labels_shape.iter().product::<usize>();
        let label_tensors: Vec<TensorRef> = (0..input_count)
            .map(|i| {
                let start = i * labels_shape_product;
                let end = start + labels_shape_product;
                self.context
                    .borrow_mut()
                    .new_tensor(labels_shape.clone(), labels.data[start..end].to_vec())
            })
            .collect();

        let pair: Vec<(&TensorRef, &TensorRef)> =
            input_tensors.iter().zip(label_tensors.iter()).collect();

        for _ in 0..epochs {
            let losses: Vec<TensorRef> = pair
                .iter()
                .enumerate()
                .map(|(i, (input, label))| {
                    let prediction = self.predict_tensor(**input);
                    let loss = self.loss_function.loss(self.context.clone(), prediction, **label);
                    loss
                })
                .collect();
            let full_loss_vector = self.context.borrow_mut().concat(losses);
            let full_loss = self.context.borrow_mut().sum(full_loss_vector);

            self.context.borrow_mut().backwards(full_loss);

            self.layers
                .iter_mut()
                .map(|layer| layer.get_parameters())
                .flatten()
                .for_each(|parameter| {
                    self.context
                        .borrow_mut()
                        .update_data_from_grad(parameter, -0.01);
                    self.context.borrow_mut().reset_grads(parameter);
                });
        }
    }
    fn predict_tensor(&mut self, data: TensorRef) -> TensorRef {
        let mut previous_layer_output = data;
        for layer in self.layers.iter() {
            previous_layer_output = layer.forward(previous_layer_output);
        }
        previous_layer_output
    }

    fn predict(&mut self, data: Vec<f64>) -> Vec<f64> {
        println!("Input: {:?}", data.clone().iter().fold(0, |acc, x| acc + x.round() as i32));

        let input = self.context.borrow_mut().new_tensor(vec![data.len()], data);
        println!("Input Tensor: {:?}", input);


        let mut previous_layer_output = input;
        for layer in self.layers.iter() {
            previous_layer_output = layer.forward(previous_layer_output);
            println!("Layer Output: {:?}", self.context.borrow_mut().get_tensor(previous_layer_output).data.iter().fold(0, |acc, x| acc + x.round() as i32));
        }

        let final_output = previous_layer_output;
        println!("Output Tensor: {:?}", final_output);
        self.context.borrow_mut().get_tensor(final_output).data
    }

    fn evaluate(&self, _data: Vec<f64>, _labels: Vec<f64>) -> (f64, f64) {
        todo!()
    }

    fn save(&self) {
        todo!()
    }
}

pub trait Model {
    fn compile(
        &mut self,
        input_shape: Vec<usize>,
        output_shape: Vec<usize>,
        optimizer: Optimizer,
        loss: LossFunction,
        metrics: Vec<Metric>,
    );
    fn fit(&mut self, data: Tensor, labels: Tensor, epochs: usize);
    fn predict(&mut self, data: Vec<f64>) -> Vec<f64>;
    fn predict_tensor(&mut self, data: TensorRef) -> TensorRef;
    fn evaluate(&self, data: Vec<f64>, labels: Vec<f64>) -> (f64, f64);
    fn save(&self);
}
