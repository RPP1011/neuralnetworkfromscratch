
use file::idx_reader;
use graph::{
    graph::{Model, Sequential},
    loss_function::LossFunction,
    network_metric::Metric,
    optimizer::Optimizer,
};
use layers::{dense::Dense, dropout::Dropout, flatten::Flatten, layers::layers::Layer};
use nuerons::activation_function::ActivationFunction;
use std::{cell::RefCell, rc::Rc, vec};

pub mod file;
pub mod graph;
pub mod layers;
pub mod math;
pub mod nuerons;

fn main() {
    let tensor_context = create_tensor_context!(1024);
    let layers: Vec<Box<dyn Layer>> = vec![
        Box::new(Flatten::new(tensor_context.clone(), vec![28, 28])),
        Box::new(Dense::new(tensor_context.clone(), 10, ActivationFunction::ReLU)),
        // Box::new(Dropout::new(tensor_context.clone(), 0.2, false)),
        // Box::new(Dense::new(tensor_context.clone(), 10, ActivationFunction::ReLU)),
    ];
    let mut network = Sequential::new(tensor_context.clone(), layers);

    let training_data = idx_reader::read_file("data/train-images-idx3-ubyte/train-images.idx3-ubyte").unwrap();
    let training_labels = idx_reader::read_file("data/train-labels-idx1-ubyte/train-labels.idx1-ubyte").unwrap();

    network.compile(
        vec![28, 28],
        vec![10],
        Optimizer::SGD,
        LossFunction::MeanSquaredError,
        vec![Metric::Accuracy],
    );
     // predict first 10 images
     for i in 0..10 {
        let prediction = network.predict(training_data.data[i*28*28..(i+1)*28*28].to_vec());
        let label = training_labels.data[i];
        println!("Prediction: {:?}, Label: {:?}", prediction, label);
    }
    // println!("{:?}", network.predict(training_data.data[0..28*28].to_vec()).len());

    let epochs = 10;
    // network.fit(training_data.clone(), training_labels.clone(), epochs);

    println!("Training done!");

    // network.visualize()
}
