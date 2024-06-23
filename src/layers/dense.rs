mod layers {
    use crate::layers::layers::layers::Layer;
    use crate::nuerons::nuerons::Nueron;

    pub struct DenseLayer {
        pub neurons: Vec<Nueron>
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