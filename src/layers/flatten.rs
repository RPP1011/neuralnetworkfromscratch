use super::layers::layers::Layer;

pub struct Flatten {
    pub input_shape: Vec<usize>
}

impl Layer for Flatten {
    fn forward(&self) {
        todo!()
    }

    fn backward(&self) {
        todo!()
    }
}

impl Flatten {
    pub fn new(input_vector_shape : Vec<usize>) -> Flatten {
        Flatten {
            input_shape : input_vector_shape
        }
    }
}