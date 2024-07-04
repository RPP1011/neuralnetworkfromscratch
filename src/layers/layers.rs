pub mod layers {
    use std::{cell::RefCell, rc::Rc};

    use crate::math::{tensor::Tensor, tensor_context::{TensorContext, TensorRef}};

    
    pub trait Layer {
        fn forward(&self, input: TensorRef) -> TensorRef;
        fn backward(&self, input: TensorRef) -> TensorRef;  
    } 
}