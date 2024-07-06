use std::{fmt::Result, fs::File};

use crate::graph::graph::Model;
 // Add this line to import serde_json crate


pub fn save_model(_model: &dyn Model, path: &str) -> Result{
    let _file = File::create(path);
    // let mut writer = BufWriter::new(file);
    // let mut encoder = serde_json::Serializer::new(&mut writer);
    // model.serialize(&mut encoder)?;
    Ok(())
}