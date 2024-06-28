use crate::nuerons::activation_function::ActivationFunction;

#[derive(Debug, Clone)]
pub struct Nueron {
    pub activation_function: ActivationFunction,
}

impl Nueron {
    pub fn new(activation_function: ActivationFunction) -> Nueron{
        Nueron {
            activation_function
        }
    }
}