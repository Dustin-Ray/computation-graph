#[cfg(test)]
mod circuit_tests {
    use crate::Circuit;
    use std::sync::Arc;

    // print evaluation results of the
    const DEBUG: bool = true;

    #[test]
    fn test_evaluate_function() {
        let mut circuit = Circuit::new();

        // Initialize placeholder for 'x'
        let x = circuit.init();

        // Compute 'x_squared = x * x'
        let x_squared = circuit.mul(x, x);

        let five = circuit.constant(5);
        // Compute 'x_squared_plus_5 = x_squared + 5'
        let x_squared_plus_5 = circuit.add(x_squared, five);

        // Compute 'y = x_squared_plus_5 + x'
        circuit.add(x_squared_plus_5, x);

        // Evaluate the circuit with a concrete input for 'x'
        assert!(circuit.evaluate(&[3], DEBUG).is_ok());

        // Check constraints
        assert!(circuit.check_constraints().is_ok());
    }

    #[test]
    fn test_division_hint() {
        let mut circuit = Circuit::new();

        // Initialize placeholder for 'a'
        let a = circuit.init();

        // Compute 'b = a + 1'
        let one = circuit.constant(1);
        let b = circuit.add(a, one);

        // Create a hint gate with functionality not supported by the circuit.
        let c = circuit.hint(
            b,
            Arc::new(|x: u32| x / 8) as Arc<dyn Fn(u32) -> u32 + Send + Sync>,
        );

        // Constant '8' to be used in multiplication with 'c'
        let eight = circuit.constant(8);

        // Compute 'c_times_8 = c * 8'
        let c_times_8 = circuit.mul(c, eight);

        // Assert that 'b' is equal to 'c_times_8'
        circuit.assert_equal(b, c_times_8);

        // Evaluate the circuit with a concrete input for 'a'
        assert!(circuit.evaluate(&[7], DEBUG).is_ok());

        // Check constraints
        assert!(circuit.check_constraints().is_ok());
    }

    #[test]
    fn test_sqrt_hint() {
        let mut circuit = Circuit::new();

        // Initialize placeholder for 'x'
        let x = circuit.init();

        let seven = circuit.constant(7);

        let x_plus_seven = circuit.add(x, seven);

        let sqrt_x_plus_seven = circuit.hint(
            x_plus_seven,
            Arc::new(|x: u32| (x as f32).sqrt().round() as u32)
                as Arc<dyn Fn(u32) -> u32 + Send + Sync>,
        );
        let computed_sq = circuit.mul(sqrt_x_plus_seven, sqrt_x_plus_seven);
        circuit.assert_equal(computed_sq, x_plus_seven);

        assert!(circuit.evaluate(&[2], DEBUG).is_ok());

        // Check constraints
        assert!(circuit.check_constraints().is_ok());
    }

    #[test]
    fn test_layerize() {
        let mut circuit = Circuit::new();

        // Test the function (x+y)×z+w
        let x = circuit.init(); // Index 0
        let y = circuit.init(); // Index 1
        let z = circuit.init(); // Index 2
        let w = circuit.init(); // Index 3

        // Operations
        let sum_xy = circuit.add(x, y); // Index 4: x + y
        let product_xyz = circuit.mul(sum_xy, z); // Index 5: (x + y) * z
        let total = circuit.add(product_xyz, w); // Index 6: ((x + y) * z) + w

        // Input values for x, y, z, w
        let input_vals = vec![1, 2, 3, 4]; // Example values: x = 1, y = 2, z = 3, w = 4

        // Evaluate the circuit
        assert!(circuit.evaluate(&input_vals, DEBUG).is_ok());

        // Print the final result (index of 'total' operation)
        println!("The result of the circuit is: {}", circuit.results[total]);

        // Optionally, print layers to see how they were organized
        if let Some(layers) = &circuit.layers {
            println!("Layers of the circuit:");
            for (i, layer) in layers.iter().enumerate() {
                println!("Layer {}: {:?}", i, layer);
            }
        }
    }

    #[test]
    fn test_random_circuits() {
        let mut circuit = Circuit::new();
        let num_gates = 10000000;

        // Generate a large random circuit
        circuit.generate_random(num_gates);

        // Mock input
        let inputs = vec![42; 10];

        // Evaluate the circuit
        assert!(circuit.evaluate(&inputs, DEBUG).is_ok());

        // Check constraints
        assert!(circuit.check_constraints().is_ok());

        println!(
            "Sample of evaluation results: {:?}",
            &circuit.results[..std::cmp::min(10, circuit.results.len())]
        );
    }

    #[test]
    fn test_circuit_initialization() {
        let circuit = Circuit::new();
        assert!(circuit.nodes.is_empty());
        assert!(circuit.equalities.is_empty());
        assert!(circuit.layers.is_none());
        assert!(circuit.results.is_empty());
    }

    #[test]
    fn test_adding_gates() {
        let mut circuit = Circuit::new();
        let idx1 = circuit.init(); // Node 0
        let idx2 = circuit.constant(10); // Node 1
        let add_idx = circuit.add(idx1, idx2); // Node 2
        assert_eq!(add_idx, 2);
        assert_eq!(circuit.nodes.len(), 3);
    }

    #[test]
    fn test_evaluate_simple_circuit() {
        let mut circuit = Circuit::new();
        let idx1 = circuit.init();
        let idx2 = circuit.constant(20);
        let add_idx = circuit.add(idx1, idx2);
        assert!(circuit.evaluate(&[10], DEBUG).is_ok());
        assert_eq!(circuit.results[add_idx], 30);
    }

    #[test]
    fn test_check_constraints() {
        let mut circuit = Circuit::new();
        let idx1 = circuit.constant(10);
        let idx2 = circuit.constant(10);
        circuit.assert_equal(idx1, idx2);
        assert!(circuit.evaluate(&[], DEBUG).is_ok());
        assert!(circuit.check_constraints().is_ok());
    }

    #[test]
    fn test_empty_circuit_eval_is_error() {
        assert!(Circuit::new().evaluate(&[], DEBUG).is_err());
    }

    #[test]
    #[allow(clippy::out_of_bounds_indexing)]
    fn test_hint_panic() {
        let mut circuit = Circuit::new();

        let one = circuit.init();

        // This hint function will attempt to access an out-of-bounds index in an array
        let small_array = [3_u32, 7, 9]; // Small array with only 3 items
        let _c = circuit.hint(
            one,
            Arc::new(move |_| small_array[10]) as Arc<dyn Fn(u32) -> u32 + Send + Sync>,
        );

        assert!(circuit.evaluate(&[0], DEBUG).is_err());
    }

    #[test]
    fn test_saturating_addition() {
        let mut circuit = Circuit::new();
        let max_value = circuit.constant(u32::MAX);
        let one = circuit.constant(1);

        // Operation that should cause saturation
        let result_idx = circuit.add(max_value, one);

        // Evaluate the circuit with no input values since there are no variable nodes
        circuit
            .evaluate(&[], DEBUG)
            .expect("Evaluation should not fail, it should saturate");

        // Fetch the result and check for saturation
        assert_eq!(
            circuit.results[result_idx],
            u32::MAX,
            "Addition should saturate at u32::MAX"
        );
    }

    #[test]
    fn test_saturating_multiplication() {
        let mut circuit = Circuit::new();
        let large_value = circuit.constant(u32::MAX / 2 + 1);
        let two = circuit.constant(2);

        // Operation that should cause saturation
        let result_idx = circuit.mul(large_value, two);

        // Evaluate the circuit with no input values since there are no variable nodes
        circuit
            .evaluate(&[], DEBUG)
            .expect("Evaluation should not fail, it should saturate");

        // Fetch the result and check for saturation
        assert_eq!(
            circuit.results[result_idx],
            u32::MAX,
            "Multiplication should saturate at u32::MAX"
        );
    }

    #[test]
    fn test_division_with_large_denominator() {
        let mut circuit = Circuit::new();
        let numerator = circuit.constant(1);

        // Create a hint gate for division where denominator is much larger than numerator
        let result_idx = circuit.hint(
            numerator,
            Arc::new(|num: u32| num / 1000) as Arc<dyn Fn(u32) -> u32 + Send + Sync>,
        );

        // Evaluate the circuit to check the division result
        circuit
            .evaluate(&[], DEBUG)
            .expect("Evaluation should handle division properly");

        // Fetch the result and verify that it's zero
        assert_eq!(
            circuit.results[result_idx], 0,
            "Division of a smaller number by a larger should result in 0"
        );
    }
}
