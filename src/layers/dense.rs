mod layers {
    use std::rc::Rc;
    use crate::layers::layers::layers::Layer;

    pub struct DenseLayer {
        pub previous_layer: Rc<dyn Layer>,
        // pub neurons: Vec<Neuron>
    }

    impl Layer for DenseLayer{
        fn forward(&self) {
            todo!()
        }
    
        fn backward(&self) {
            todo!()
        }
    }
}