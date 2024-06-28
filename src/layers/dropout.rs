use super::layers::layers::Layer;


pub struct Dropout {
    pub rate: f64,
    pub training: bool,
}

impl Layer for Dropout {
    fn forward(&self)  {
        todo!("Implement dropout Forward pass")
    }

    fn backward(&self) {
        todo!("Implement dropout backward pass")
    }
}

impl Dropout {
    pub fn new(rate: f64, training: bool) -> Dropout {
        Dropout {
            rate,
            training
        }
    }
}