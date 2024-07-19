
use file::idx_reader;
use graph::{
    graph::{Model, Sequential},
    loss_function::LossFunction,
    network_metric::Metric,
    optimizer::Optimizer,
};
use layers::{dense::Dense, dropout::Dropout, flatten::Flatten, layers::layers::Layer};
use nuerons::activation_function::ActivationFunction;
use sample_functions::sine_wave::SineWaveGenerator;
use std::{cell::RefCell, rc::Rc, vec};

pub mod file;
pub mod graph;
pub mod layers;
pub mod math;
pub mod nuerons;
pub mod sample_functions;

fn main() {
    //try_MNIST();
    try_sin();
}

fn try_sin() {
    let tensor_context = create_tensor_context!(4096);
    let layers: Vec<Box<dyn Layer>> = vec![
        Box::new(Dense::new(tensor_context.clone(), 1, ActivationFunction::ReLU)),
        Box::new(Dense::new(tensor_context.clone(), 10, ActivationFunction::ReLU)),
        Box::new(Dense::new(tensor_context.clone(), 10, ActivationFunction::ReLU)),
        Box::new(Dense::new(tensor_context.clone(), 1, ActivationFunction::ReLU)),
    ];

    let mut network = Sequential::new(tensor_context.clone(), layers);

    let sine_wave_generator = SineWaveGenerator {
        amplitude: 3.0,
        frequency: 2.0,
        phase: 0.4,
        noise: 0.05,
    };

    let (training_data, training_labels) = sine_wave_generator.generate_data( 1000);

    network.compile(
        vec![1],
        vec![1],
        Optimizer::SGD,
        LossFunction::MeanSquaredError,
        vec![Metric::Accuracy],
    );

    let epochs = 100;
    
    network.fit(training_data.clone(), training_labels.clone(), epochs);

    println!("Training done!");

     // predict first 10 sine values
     for i in 0..10 {
        let input = training_data.data[i];
        let prediction = 
        network.predict(vec![input]);
        
        let label = training_labels.data[i];

        // Round to 2 decimal places for prediction        
        println!("Input {:?}, Prediction: {:?}, Label: {:?}", input, prediction, label);
    }
}

fn try_MNIST() {
    let tensor_context = create_tensor_context!(1024);
    let layers: Vec<Box<dyn Layer>> = vec![
        Box::new(Flatten::new(tensor_context.clone(), vec![28, 28])),
        Box::new(Dense::new(tensor_context.clone(), 10, ActivationFunction::ReLU)),
        Box::new(Dropout::new(tensor_context.clone(), 0.2, false)),
        Box::new(Dense::new(tensor_context.clone(), 10, ActivationFunction::ReLU)),
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


    let epochs = 10;
    network.fit(training_data.clone(), training_labels.clone(), epochs);

    println!("Training done!");

     // predict first 10 images
     for i in 0..10 {
        let prediction = network.predict(training_data.data[i*28*28..(i+1)*28*28].to_vec()).iter().enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(index, _)| index);
        let label = training_labels.data[i];
        // Round to 2 decimal places for prediction        
        println!("Prediction: {:?}, Label: {:?}", prediction, label);
    }

}