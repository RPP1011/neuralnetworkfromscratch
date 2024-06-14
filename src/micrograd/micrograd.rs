mod micrograd {
    use std::borrow::BorrowMut;
    use std::cell::RefCell;
    use std::rc::{Rc, Weak};
    pub enum Operation {
        ADD,
        MUL,
        POW,
    }
    

    pub struct Value {
        pub data: f64,
        pub grad: f64,

        pub backwards: Option<Box<dyn Fn()>>,
        pub previous: Vec<Rc<RefCell<Value>>>,
        pub op: Operation,
    }

        pub fn add(this: Rc<RefCell<Value>>, other: Rc<RefCell<Value>>) -> Rc<RefCell<Value>> {
            let previous: Vec<Rc<RefCell<Value>>> = vec![
                this.clone(),
                other.clone(),
            ];
            let mut out = Value {
                data: this.borrow().data + other.borrow().data,
                grad: 0.0,
                backwards: None,
                previous,
                op: Operation::ADD,
            };

            let out_reference = Rc::new(RefCell::new(out));


            out.borrow_mut().backwards = Some(Box::new(move || {
                let out_grad: f64 = out_reference.borrow().grad;
                let mut this_instance =  this.borrow().borrow_mut();
               
                this_instance.grad += out_grad;
                
            }));

            out_reference
        }

        // pub fn mul(&self, other: &Value) -> Value {
        //     unsafe {
        //         let previous = vec![Weak::from_raw(self), Weak::from_raw(other)];
        //         let mut out = Value {
        //             data: self.data * other.data,
        //             grad: 0.0,
        //             backwards: None,
        //             previous,
        //             op: Operation::MUL,
        //         };

        //         out.backwards = Some(Box::new(move || -> () {
        //             self.grad *= other.grad * out.grad;
        //             other.grad *= self.grad * out.grad;
        //         }));

        //         return out;
        //     }
        // }

    
}
