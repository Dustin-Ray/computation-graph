use capy_graph::Circuit;
use criterion::{black_box, criterion_group, criterion_main, Criterion};

// Function to setup a complex circuit for the benchmark with variable number of gates
fn setup_circuit(num_gates: usize) -> Circuit {
    let mut circuit = Circuit::new();
    circuit.generate_random(num_gates);
    circuit
}

// Benchmark function
fn circuit_evaluation_benchmark(c: &mut Criterion) {
    let mut group = c.benchmark_group("Circuit Evaluation Scaling");

    // Start at 2^12 (4096 gates) and double until 2^24 (~16 million gates, closest to 10 million)
    let start_power = 12;
    let end_power = 24; // Adjust this value for more or less granularity
    for i in start_power..=end_power {
        let num_gates = 2_usize.pow(i);
        let mut circuit = setup_circuit(num_gates);

        group.bench_function(format!("num_gates_{}", num_gates), |b| {
            b.iter(|| {
                let input_values = vec![42; 10]; // Example input values
                circuit.evaluate(black_box(&input_values), black_box(false))
            });
        });
    }

    group.finish();
}

criterion_group!(benches, circuit_evaluation_benchmark);
criterion_main!(benches);
