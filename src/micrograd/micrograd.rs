mod micrograd {
    use std::borrow::{Borrow, BorrowMut};
    use std::cell::RefCell;
    use std::rc::{Rc};
    
    #[derive(Clone)]
    pub enum Operation {
        ADD,
        MUL,
    }

    impl Operation {
        pub fn forward(&self, a: Rc<RefCell<Value>>, b: Rc<RefCell<Value>>) -> f64 {
            let a_reference : &RefCell<Value> = a.borrow();
            let a_data = a_reference.borrow().data;
            
            let b_reference : &RefCell<Value> = b.borrow();
            let b_data = b_reference.borrow().data;
            
            match self {
                Operation::ADD => a_data + b_data,
                Operation::MUL => a_data * b_data
            }
        }

        pub fn backwards(&self,mut a: Rc<RefCell<Value>>, mut b: Rc<RefCell<Value>>, out: Rc<RefCell<Value>>) {
            let a_binding = a.borrow_mut();
            let b_binding = b.borrow_mut();
            let mut a_mut_reference = a_binding.as_ref().borrow_mut();
            let mut b_mut_reference = b_binding.as_ref().borrow_mut();
            
            let out_ref : &RefCell<Value> = out.borrow();
            let out_grad : f64 = out_ref.borrow().grad;


            match self {
                Operation::ADD => {
                    a_mut_reference.grad += out_grad;
                    b_mut_reference.grad += out_grad;
                },
                Operation::MUL => {
                    a_mut_reference.grad += b_mut_reference.data * out_grad;
                    b_mut_reference.grad += a_mut_reference.data * out_grad;
                }
            }
        }
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

    pub fn add(graph: Rc<RefCell<Graph>>, a:Rc<RefCell<Value>>, b:Rc<RefCell<Value>>) -> Rc<RefCell<Value>> {
        
        let values: &Vec<Rc<RefCell<Value>>> = &graph.as_ref().borrow().values;
        let a_index = a.as_ref().borrow().index;
        let b_index = b.as_ref().borrow().index;

        let a_data = a.as_ref().borrow().data;
        let b_data = b.as_ref().borrow().data;

        let out = Value {
            data: a_data + b_data,
            grad: 0.0,
            index: values.len(),
            previous: vec![a_index, b_index],
            op: Operation::ADD,
        };

        let graph_reference: Rc<RefCell<Graph>> = graph.clone();

        let mut mut_graph_reference = graph_reference.as_ref().borrow_mut();
        mut_graph_reference.add_value(CreateValueOption::DATA(out.clone()))
    }

    pub fn mul(graph: Rc<RefCell<Graph>>, a:Rc<RefCell<Value>>, b:Rc<RefCell<Value>>) -> Rc<RefCell<Value>> {
        
        let values: &Vec<Rc<RefCell<Value>>> = &graph.as_ref().borrow().values;
        let a_index = a.as_ref().borrow().index;
        let b_index = b.as_ref().borrow().index;

        let a_data = a.as_ref().borrow().data;
        let b_data = b.as_ref().borrow().data;

        let out = Value {
            data: a_data * b_data,
            grad: 0.0,
            index: values.len(),
            previous: vec![a_index, b_index],
            op: Operation::MUL,
        };

        let graph_reference: Rc<RefCell<Graph>> = graph.clone();

        let mut mut_graph_reference = graph_reference.as_ref().borrow_mut();
        mut_graph_reference.add_value(CreateValueOption::DATA(out.clone()))
    }
}
