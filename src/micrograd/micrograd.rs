mod micrograd {
    use std::borrow::{Borrow, BorrowMut};
    use std::cell::RefCell;
    use std::rc::{Rc};
    
    #[derive(Clone)]
    pub enum Operation {
        ADD,
        MUL,
        POW,
    }

    impl Operation {
        pub fn forward(&self, a: Rc<RefCell<Value>>, b: Rc<RefCell<Value>>) -> f64 {
            let a_reference : &RefCell<Value> = a.borrow();
            let a_data = a_reference.borrow().data;
            
            let b_reference : &RefCell<Value> = b.borrow();
            let b_data = b_reference.borrow().data;
            
            match self {
                Operation::ADD => a_data + b_data,
                Operation::MUL => a_data * b_data,
                Operation::POW => a_data.powf(b_data),
            }
        }

        // pub fn backwards(&self, a: Rc<RefCell<Value>>, b: Rc<RefCell<Value>>, out: Rc<RefCell<Value>>) {
        //     let mut a = a.borrow().borrow_mut();
        //     let mut b = b.borrow().borrow_mut();
        //     let out = out.borrow().borrow();
        //     match self {
        //         Operation::ADD => {
        //             a.grad += out.clone().grad;
        //             b.grad += out.clone().grad;
        //         },
        //         Operation::MUL => {
        //             a.clone().borrow_mut().grad += b.clone().data * out.clone().grad;
        //             b.clone().borrow_mut().grad += a.clone().data * out.clone().grad;
        //         },
        //         Operation::POW => {
        //             a.clone().borrow_mut().grad += b.clone().data * a.clone().data.powf(b.clone().data - 1.0) * out.clone().grad;
        //             b.clone().borrow_mut().grad += a.clone().data.powf(b.clone().data) * out.clone().grad * (a.clone().data.ln());
        //         },
        //     }
        // }
    }

    pub enum CreateValueOption {
        EMPTY,
        DATA(Value),
    }
   
    pub struct Graph {
        pub values: Vec<Rc<RefCell<Value>>>,
        pub layers: Vec<Layer>
    }

    impl Graph {
        pub fn add_value(&mut self, value: CreateValueOption) -> Rc<RefCell<Value>>{
            let out = Rc::new(RefCell::new(Value {
                data: 0.0,
                grad: 0.0,
                index: self.values.len(),
                previous: vec![],
                op: Operation::ADD,
            }));

            match value {
                CreateValueOption::EMPTY => {
                    self.values.push(out.clone());
                },
                CreateValueOption::DATA(value) => {
                    self.values.push(Rc::new(RefCell::new(value)));
                }          
            };
            
            out.clone()
        }

        pub fn add_layer(&mut self, n:usize) {
            for _ in 0..n {
                self.add_value(CreateValueOption::EMPTY);
            }

            let layer = Layer {
                values: vec![],
                weights: vec![],
                biases: vec![],
            };

            self.layers.push(layer);
        }
    }

    pub struct Layer {
        pub values: Vec<usize>,
        pub weights: Vec<f64>,
        pub biases: Vec<f64>,
    }

    #[derive(Clone)]
    pub struct Value {
        pub data: f64,
        pub grad: f64,
        pub index: usize,
        pub previous: Vec<usize>,
        pub op: Operation,
    }

    // pub fn add(graph: Rc<RefCell<Graph>>, a:Rc<RefCell<Value>>, b:Rc<RefCell<Value>>) -> Rc<RefCell<Value>> {
    //     let graph_reference: Rc<RefCell<Graph>> = graph.clone();

    //     let values: Vec<Rc<RefCell<Value>>> = graph_reference.borrow().values;
    //     let a_index = a.borrow().index;
    //     let b_index = b.borrow().index;

    //     let out = Value {
    //         data: values[a_index].borrow().data + values[b_index].borrow().data,
    //         grad: 0.0,
    //         index: graph_reference.borrow().values.len(),
    //         previous: vec![a_index, b_index],
    //         op: Operation::ADD,
    //     };

    //     let mut mut_graph_reference = graph.clone().borrow().borrow_mut();
    //     mut_graph_reference.add_value(CreateValueOption::DATA(out.clone()))
    // }
}
