use std::{error::Error, fs::File, io::Read};

use crate::math::tensor::Tensor;

#[derive(Debug)]
enum IDX_DATA_TYPE {
    U8,
    I8,
    I16,
    I32,
    F32,
    F64,
}

impl IDX_DATA_TYPE {
    fn from_u8(value: u8) -> Option<IDX_DATA_TYPE> {
        match value {
            0x08 => Some(IDX_DATA_TYPE::U8),
            0x09 => Some(IDX_DATA_TYPE::I8),
            0x0B => Some(IDX_DATA_TYPE::I16),
            0x0C => Some(IDX_DATA_TYPE::I32),
            0x0D => Some(IDX_DATA_TYPE::F32),
            0x0E => Some(IDX_DATA_TYPE::F64),
            _ => None,
        }
    }

    fn to_type(&self) -> std::any::TypeId {
        match self {
            IDX_DATA_TYPE::U8 => std::any::TypeId::of::<u8>(),
            IDX_DATA_TYPE::I8 => std::any::TypeId::of::<i8>(),
            IDX_DATA_TYPE::I16 => std::any::TypeId::of::<i16>(),
            IDX_DATA_TYPE::I32 => std::any::TypeId::of::<i32>(),
            IDX_DATA_TYPE::F32 => std::any::TypeId::of::<f32>(),
            IDX_DATA_TYPE::F64 => std::any::TypeId::of::<f64>(),
        }
    }
}

fn read_magic_number(data: &[u8]) -> Option<(IDX_DATA_TYPE, usize)> {
    if data.len() < 4 {
        return None;
    }

    let data_type = IDX_DATA_TYPE::from_u8(data[2])?;
    let dimensions = data[3] as usize;

    Some((data_type, dimensions))
}

pub fn read_file(file_path: &str) -> Result<Tensor, &'static str> {
    let buffer = File::open(file_path)
        .and_then(|mut file| {
            let mut buffer = Vec::new();
            file.read_to_end(&mut buffer)?;
            let (data_type, dimensions) = read_magic_number(&buffer).unwrap();
            // Log the data type and dimensions
            println!("Data type: {:?}, Dimensions: {}", data_type, dimensions);

            let dimension_sizes = (0..dimensions).into_iter()
            .map(|i| {
                let start:usize = (4+i*4).into();
                let end:usize = start + 4;
                let slice = &buffer[start..end];
                u32::from_be_bytes(slice.try_into().unwrap()) as usize})
                .collect::<Vec<usize>>();

            println!("Dimension sizes: {:?}", dimension_sizes);
            Ok(Tensor::new(dimension_sizes, buffer.iter().skip(4 + 4*dimensions).map(|&x| x as f64).collect::<Vec<f64>>()))
        });
    
    
    match buffer {
        Ok(tensor) => Ok(tensor),
        Err(_) => Err("Error reading file")
    }
}
    

// Add test to read example data as byte array
// #[cfg(test)]
// mod tests {
//     use super::*;


//     #[test]
//     fn test_read_train_labels() {
//         let output = read_file("data/train-labels-idx1-ubyte/train-labels.idx1-ubyte").unwrap();
//         assert_eq!(output.shape, vec![60000]);
        
//         // Round all values to ints and check if they are between 0 and 9 inclusive
//         output.data.iter().for_each(|&x| {
//             let x = x.round() as u8;
//             assert!(x <= 9);
//         });
//     }

//     #[test]
//     fn test_read_train_data() {
//         let output = read_file("data/train-images-idx3-ubyte/train-images.idx3-ubyte").unwrap();
//         assert_eq!(output.shape, vec![60000, 28, 28]);
        
//         // Round all values to ints and check if they are between 0 and 255 inclusive
//         output.data.iter().for_each(|&x| {
//             let x = x.round() as u16;
//             assert!(x <= 255);     
//         });
//     }
// }