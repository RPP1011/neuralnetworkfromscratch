mod micrograd {
    pub struct Value {
        pub data: f64
    }

    impl Value {
        pub fn add(&self, other: &Value) -> Value {
            return Value {
                data: self.data + other.data
            }
        }

        pub fn mul(&self, other: &Value) -> Value {
            return Value {
                data: self.data * other.data
            }
        }

        pub fn repr(&self) -> String {
            return format!("Value({})", self.data);
        }
    }
}