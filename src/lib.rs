//! This module provides a framework for constructing and evaluating arithmetic circuits.
//! It supports operations such as addition, multiplication, and custom operations through hint gates.
//! The circuits can be constructed dynamically, evaluated in parallel layers, and verified with equality constraints.
//! This flexible architecture is suitable for applications requiring configurable computational graphs, such as in
//! cryptographic schemes or complex algorithm simulations.
//!
//! # Example Usage:
//! ```rust
//! use capy_graph::Circuit;
//! use std::sync::Arc;
//!
//! let mut circuit = Circuit::new();
//! let x = circuit.constant(10);
//! let y = circuit.add(x, x);
//! let custom_operation = Arc::new(|val: u32| val * 2);
//! let z = circuit.hint(x, custom_operation);
//! circuit.assert_equal(y, z);
//!
//! let input_values = vec![10];
//! let debug = true;
//! assert!(circuit.evaluate(&input_values, debug).is_ok());
//! assert!(circuit.check_constraints().is_ok());
//! ```
//!
//! This example demonstrates creating a circuit with constant inputs, adding two nodes,
//! applying a custom doubling operation, and asserting equality conditions. It also shows how
//! to evaluate the circuit with debugging enabled to trace computation values and performance.
mod tests;

use rand::distributions::{Distribution, Uniform};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::{
    collections::{HashSet, VecDeque},
    fmt,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::{Duration, Instant},
};
use thiserror::Error;

/// Enum to represent errors that can occur within the Circuit operations.
#[derive(Error, Debug)]
pub enum CircuitError {
    #[error("Cannot evaluate an empty circuit")]
    EmptyCircuit,
    #[error("Error evaluating node: {0}")]
    NodeEvaluationError(String),
    #[error("Constraint check failed")]
    ConstraintCheckFailure,
    #[error("Failed to acquire a necessary lock: {0}")]
    LockAcquisitionError(String),
}

/// Represents a gate in an arithmetic circuit.
///
/// Variants:
/// - `Add`: Adds the values from two nodes identified by their indices.
/// - `Multiply`: Multiplies the values from two nodes identified by their indices.
/// - `Hint`: A custom gate which allows for applying any user-defined operation.
///   It takes a single `u32` input and produces a `u32` output, defined by a closure.
#[derive(Clone)]
pub enum Gate {
    Add(usize, usize),
    Multiply(usize, usize),
    Hint(usize, Arc<dyn Fn(u32) -> u32 + Send + Sync>),
}

/// Represents a node within an arithmetic circuit.
///
/// Variants:
/// - `Input(u32)`: A constant input value to the circuit. Set once and used during execution.
/// - `Variable`: Represents a variable whose value is determined during the circuit's execution.
/// - `Operation`: Applies an operation defined by a `Gate` to inputs, dynamically during execution.
#[derive(Clone)]
pub enum Node {
    Input(u32),
    Variable,
    Operation(Gate, Vec<usize>),
}

/// Represents an arithmetic circuit.
/// This struct manages nodes, gates, and the evaluation process of the circuit.
/// It supports adding various types of nodes, including constants, variables, and operations,
/// and provides methods to evaluate the circuit, check constraints, and apply custom operations.
pub struct Circuit {
    nodes: Vec<Node>, // Holds all nodes within the circuit, including inputs, variables, or operations.
    equalities: Vec<(usize, usize)>, // Tracks equality constraints between pairs of node indices.
    layers: Option<Vec<Vec<usize>>>, // Organized layers for efficient evaluation and parallel processing.
    results: Vec<u32>, // Stores the computation results of the circuit nodes after evaluation.
    total_duration: Duration, // Total time taken to evaluate the circuit.
    number_of_layers: usize, // Number of layers used during the parallel evaluation of the circuit.
    number_of_constraints: usize, // Total number of equality constraints defined in the circuit.
    total_hint_gates: AtomicUsize, // Counter for the number of hint gates processed during evaluation.
    total_gates_processed: AtomicUsize, // Counter for the total number of gates processed during evaluation.
    gates_per_second: f64, // Computation throughput: number of gates processed per second.
}

// Default implementation of `Circuit` that clippy realllly wants to keep adding in...
impl Default for Circuit {
    fn default() -> Self {
        Self::new()
    }
}

impl Circuit {
    /// Create a new circuit and initialize all fields to empty.
    pub fn new() -> Self {
        Circuit {
            nodes: Vec::new(),
            equalities: Vec::new(),
            layers: None,
            results: Vec::new(),
            total_gates_processed: AtomicUsize::new(0),
            total_hint_gates: AtomicUsize::new(0),
            total_duration: Duration::new(0, 0),
            number_of_layers: 0,
            number_of_constraints: 0,
            gates_per_second: 0.0,
        }
    }

    // Insert a gate into the circuit. Returns the index
    // of the newly inserted node.
    fn insert_gate(&mut self, gate: Gate) -> usize {
        let dependencies = match &gate {
            Gate::Add(left, right) => vec![*left, *right],
            Gate::Multiply(left, right) => vec![*left, *right],
            Gate::Hint(idx, _) => vec![*idx],
        };

        self.nodes.push(Node::Operation(gate, dependencies));
        self.nodes.len() - 1
    }

    /// Inserts an input node into the circuit. Any number of input
    /// nodes can be inserted. Their values are then set in sequence
    /// by passing a list of `u32`s to `circuit.evaluate(&[])`.
    /// Returns the index of the newly inserted node.
    ///
    /// ### Usage:
    /// ```
    /// let mut circuit = capy_graph::Circuit::new();
    /// let x = circuit.init();
    /// let y = circuit.init();
    /// let z = circuit.init();
    /// let debug = true;
    /// assert!(circuit.evaluate(&[1, 2, 3], debug).is_ok());
    /// ```
    pub fn init(&mut self) -> usize {
        self.nodes.push(Node::Variable);
        self.nodes.len() - 1
    }

    /// Inserts a constant-valued node into the circuit.
    /// Returns the index of the newly inserted node.
    /// ### Usage:
    /// ```
    /// let mut circuit = capy_graph::Circuit::new();
    /// let x = circuit.constant(42);
    /// let y = circuit.mul(x, x);
    /// ```
    pub fn constant(&mut self, value: u32) -> usize {
        self.nodes.push(Node::Input(value));
        self.nodes.len() - 1
    }

    /// Inserts an `addition` node into the circuit. It has max fan-in
    /// of 2 and accepts the indices of the nodes to add. Addition is
    /// saturated; overflow yields the max `u32` value, underflow yields
    /// the minimum.

    ///
    /// ### Usage:
    /// ```
    /// let mut circuit = capy_graph::Circuit::new();
    /// let x = circuit.constant(42);
    /// let y = circuit.add(x, x);
    /// ```
    pub fn add(&mut self, idx: usize, idx2: usize) -> usize {
        self.insert_gate(Gate::Add(idx, idx2))
    }

    /// Inserts a `multiplication` node into the circuit. It has max fan-in
    /// of 2 and accepts the indices of the nodes to multiply. Multiplication
    /// is saturated; overflow yields the max `u32` value, underflow yields
    /// the minimum.
    ///
    /// ### Usage:
    /// ```
    /// let mut circuit = capy_graph::Circuit::new();
    /// let x = circuit.constant(42);
    /// let y = circuit.mul(x, x);
    /// ```
    pub fn mul(&mut self, idx1: usize, idx2: usize) -> usize {
        self.insert_gate(Gate::Multiply(idx1, idx2))
    }

    /// Inserts a custom function into the circuit. This function
    /// is passed as a closure with trait bounds restricted to
    /// `Send` + `Sync` in order to support layerization
    /// and parallel circuit evaluation. The circuit will
    /// catch any panics (i.e. dividing by zero) as a `CircuitError`.
    ///
    /// ### Usage:
    /// ```
    /// use capy_graph::Circuit;
    /// use std::sync::Arc;
    /// let mut circuit = Circuit::new();
    /// let two = circuit.init();
    /// let b = circuit.constant(16);
    /// // the circuit doesn't support division, so we hint it
    /// let c = circuit.hint(
    ///     b,
    ///     Arc::new(|x: u32| x / 8) as Arc<dyn Fn(u32) -> u32 + Send + Sync>
    /// );
    /// // then we establish a constraint to ensure the hint is executed correctly
    /// let constraint = circuit.mul(c, two);
    /// circuit.assert_equal(two, c);
    /// let debug = true;
    /// assert!(circuit.evaluate(&[2], debug).is_ok());
    /// assert!(circuit.check_constraints().is_ok());
    /// ```
    pub fn hint(&mut self, idx: usize, func: Arc<dyn Fn(u32) -> u32 + Send + Sync>) -> usize {
        self.insert_gate(Gate::Hint(idx, func))
    }

    /// Inserts a constraint-check between two nodes into the circuit.
    /// This is useful for asserting that custom functions were executed
    /// correctly.
    ///
    /// ### Usage:
    /// ```
    /// let mut circuit = capy_graph::Circuit::new();
    /// let x = circuit.init();
    /// let y = circuit.constant(42);
    /// circuit.assert_equal(x, y);
    /// circuit.evaluate(&[42], false);
    /// assert!(circuit.check_constraints().is_ok());
    /// ```
    pub fn assert_equal(&mut self, idx1: usize, idx2: usize) {
        self.equalities.push((idx1, idx2));
    }

    /// Checks if all constraints in the circuit are satisfied.
    /// Returns `Ok(())` if all constraints are satisfied, or
    /// `Err(CircuitError::ConstraintCheckFailure)` if any constraint fails.
    /// ### Usage:
    /// ```
    /// let mut circuit = capy_graph::Circuit::new();
    /// let x = circuit.init();
    /// let y = circuit.constant(42);
    /// circuit.assert_equal(x, y);
    /// circuit.evaluate(&[42], false);
    /// assert!(circuit.check_constraints().is_ok());
    /// ```
    pub fn check_constraints(&self) -> Result<(), CircuitError> {
        if self
            .equalities
            .iter()
            .all(|&(idx1, idx2)| self.results[idx1] == self.results[idx2])
        {
            Ok(())
        } else {
            Err(CircuitError::ConstraintCheckFailure)
        }
    }

    /// Evaluates the circuit, initializing all `init` nodes to a list of input `u32` values.
    /// Inputs are initialized sequentially as they appear in the input list.
    /// Gracefully errors on any panic introduced from a custom hint or when attempting
    /// to evaluate an empty circuit.
    /// Optionally print debug information from the evaluation. These details include:
    /// - evaluation circuit evaluation time
    /// - number of layers
    /// - number of gates
    /// - number of hints
    /// - number of constraints
    /// - number of gates processed per second
    ///
    /// # Example Usage:
    /// ```rust
    /// use capy_graph::Circuit;
    /// use std::sync::Arc;
    ///
    /// let mut circuit = Circuit::new();
    /// let x = circuit.constant(10);
    /// let y = circuit.add(x, x);
    /// let custom_operation = Arc::new(|val: u32| val * 2);
    /// let z = circuit.hint(x, custom_operation);
    /// circuit.assert_equal(y, z);
    ///
    /// let input_values = vec![10];
    /// let debug = true;
    /// assert!(circuit.evaluate(&input_values, debug).is_ok());
    /// assert!(circuit.check_constraints().is_ok());
    /// ```
    pub fn evaluate(&mut self, input_vals: &[u32], debug: bool) -> Result<(), CircuitError> {
        if self.nodes.is_empty() {
            return Err(CircuitError::EmptyCircuit);
        }

        let mut results = vec![0; self.nodes.len()];
        let start_time = Instant::now();
        let total_gates_processed = AtomicUsize::new(0);
        let total_hint_gates = AtomicUsize::new(0);

        // Use parallel Kahn's to split the graph into its requisite layers
        self.layerize()?;
        self.number_of_layers = self.layers.as_ref().map_or(0, Vec::len);
        self.number_of_constraints = self.equalities.len();

        if let Some(layers) = &self.layers {
            for (i, layer) in layers.iter().enumerate() {
                let layer_start = Instant::now();

                let layer_results: Result<Vec<_>, CircuitError> = layer
                    .par_iter() // Use Rayon's parallel iterator here
                    .map(|&node_idx| {
                        let node = &self.nodes[node_idx];
                        match node {
                            Node::Input(value) => Ok(*value),
                            Node::Variable => Ok(input_vals[node_idx]),
                            Node::Operation(gate, _) => {
                                if matches!(gate, Gate::Hint(_, _)) {
                                    total_hint_gates.fetch_add(1, Ordering::Relaxed);
                                }
                                total_gates_processed.fetch_add(1, Ordering::Relaxed);
                                self.evaluate_gate(gate, &results)
                            }
                        }
                    })
                    .collect();

                let layer_results = layer_results?;
                let layer_duration = layer_start.elapsed();

                // Update results after processing each layer
                for (&node_idx, &result) in layer.iter().zip(layer_results.iter()) {
                    results[node_idx] = result;
                }

                if debug {
                    println!("Layer {}: Processed in {:?}", i + 1, layer_duration);
                }
            }
        }
        // Collect debug information into self
        self.total_hint_gates = total_hint_gates;
        self.results = results;
        self.total_duration = start_time.elapsed();
        if self.total_duration > Duration::ZERO {
            self.gates_per_second = total_gates_processed.load(Ordering::Relaxed) as f64
                / self.total_duration.as_secs_f64();
        }
        self.total_gates_processed = total_gates_processed;

        if debug {
            println!("{}", self)
        }

        Ok(())
    }

    // Defines the operations of the gates in the circuit and specifies the format of the
    // `hint` gate. Returns `CircuitError` if hint function panics for any reason.
    fn evaluate_gate(&self, gate: &Gate, results: &[u32]) -> Result<u32, CircuitError> {
        match gate {
            Gate::Add(left, right) => Ok(results[*left].saturating_add(results[*right])),
            Gate::Multiply(left, right) => Ok(results[*left].saturating_mul(results[*right])),
            Gate::Hint(idx, func) => {
                let result = catch_unwind(AssertUnwindSafe(|| func(results[*idx])));
                result.map_err(|_| CircuitError::NodeEvaluationError("Function panic".to_string()))
            }
        }
    }

    // Topological sort using parallel Kahn's algorithm: https://dl.acm.org/doi/10.1145/368996.369025
    // Finds and seperates all nodes and gates into layers based on their dependencies. This facilitates
    // parallel evaluation of the circuit, hypothetically increasing evaluation performance and throughput.
    fn layerize(&mut self) -> Result<(), CircuitError> {
        let nodes = Arc::new(self.nodes.clone());
        let num_nodes = nodes.len();
        let in_degree = Arc::new(Mutex::new(vec![0; num_nodes]));
        let graph = Arc::new(Mutex::new(vec![vec![]; num_nodes]));

        // Initialize graph and in_degree using rayon
        (0..num_nodes)
            .into_par_iter()
            .try_for_each(|node_idx| -> Result<(), CircuitError> {
                if let Node::Operation(_, deps) = &nodes[node_idx] {
                    let mut graph_lock = graph.lock().map_err(|e| {
                        CircuitError::LockAcquisitionError(format!("Failed to lock graph: {}", e))
                    })?;
                    let mut in_degree_lock = in_degree.lock().map_err(|e| {
                        CircuitError::LockAcquisitionError(format!(
                            "Failed to lock in_degree: {}",
                            e
                        ))
                    })?;
                    for &dep in deps {
                        graph_lock[dep].push(node_idx);
                        in_degree_lock[node_idx] += 1;
                    }
                }
                Ok(())
            })?;

        // Determine the initial layer of nodes with zero in-degree
        let mut queue = VecDeque::new();
        let mut layers = Vec::new();
        {
            let in_deg = in_degree.lock().map_err(|e| {
                CircuitError::LockAcquisitionError(format!(
                    "Failed to lock in_degree for reading: {}",
                    e
                ))
            })?;
            for (i, &degree) in in_deg.iter().enumerate() {
                if degree == 0 {
                    queue.push_back(i);
                }
            }
        }

        // Process the layers
        while !queue.is_empty() {
            let current_layer: Vec<_> = queue.drain(..).collect();
            layers.push(current_layer.clone());

            let mut next_layer = HashSet::new();
            {
                let graph_lock = graph.lock().map_err(|e| {
                    CircuitError::LockAcquisitionError(format!(
                        "Failed to lock graph for processing: {}",
                        e
                    ))
                })?;
                let mut in_deg_lock = in_degree.lock().map_err(|e| {
                    CircuitError::LockAcquisitionError(format!(
                        "Failed to lock in_degree for updating: {}",
                        e
                    ))
                })?;

                for &node_idx in &current_layer {
                    for &dependent in &graph_lock[node_idx] {
                        in_deg_lock[dependent] -= 1;
                        if in_deg_lock[dependent] == 0 {
                            next_layer.insert(dependent);
                        }
                    }
                }
            }

            for node in next_layer {
                queue.push_back(node);
            }
        }

        self.layers = Some(layers);
        Ok(())
    }
    /// Generate an arbitrary-size, random combination of nodes and gates for testing purposes.
    /// Includes a mix of variants from the `Gate` enum as well as a custom hint. Automatically
    /// constrains the hint and creates random dependencies between nodes.
    /// ### Usage:
    /// ```
    /// use capy_graph::Circuit;
    /// let mut circuit = Circuit::new();
    /// // Generate a large random circuit
    /// let num_gates = 100000;
    /// circuit.generate_random(num_gates);
    /// // Mock input
    /// let inputs = vec![42; 10];
    /// // Evaluate the circuit
    /// assert!(circuit.evaluate(&inputs, true).is_ok());
    /// // check all random constraints
    /// assert!(circuit.check_constraints().is_ok());
    /// ```
    pub fn generate_random(&mut self, num_gates: usize) {
        let num_inputs = 10; // Fixed number of input variables

        // Initialize input nodes with random values
        for _ in 0..num_inputs {
            self.constant(rand::random::<u32>() % 100);
        }

        let custom_funcs: Vec<Arc<dyn Fn(u32) -> u32 + Send + Sync>> =
            vec![Arc::new(|x| (x as f32).sqrt().round() as u32)];

        let mut rng = rand::thread_rng();
        let gate_dist = Uniform::from(0..3); // For Add, Multiply, Custom
        let index_dist = Uniform::from(0..self.nodes.len());
        let func_dist = Uniform::from(0..custom_funcs.len());

        for _ in 0..num_gates {
            // Sample some random gates
            let gate_type = gate_dist.sample(&mut rng);
            let idx1 = index_dist.sample(&mut rng);

            match gate_type {
                0 => {
                    self.add(idx1, index_dist.sample(&mut rng));
                }
                1 => {
                    self.mul(idx1, index_dist.sample(&mut rng));
                }
                2 => {
                    // Hint function
                    let func_idx = func_dist.sample(&mut rng);
                    let func_node = self.hint(idx1, custom_funcs[func_idx].clone());

                    // If we add a hint into the circuit, let's also add an accompanying
                    // equality check to enforce the constraint automatically
                    let verification_node =
                        self.apply_equality_constraint(func_node, func_idx, idx1);

                    // Assert that original and verified are equal
                    self.assert_equal(idx1, verification_node);
                }
                _ => unreachable!(),
            }
        }
    }

    // Helper function applies an equality constraint to a randomly generated hint
    fn apply_equality_constraint(
        &mut self,
        func_node: usize,
        func_idx: usize,
        original_idx: usize,
    ) -> usize {
        match func_idx {
            3 => self.hint(func_node, Arc::new(|x| x * x)), // Check sqrt by squaring
            _ => original_idx,
        }
    }
}

impl fmt::Display for Circuit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Circuit Evaluation Summary:")?;
        writeln!(f, "Total evaluation time: {:?}", self.total_duration)?;
        writeln!(f, "Number of layers: {}", self.number_of_layers)?;
        writeln!(f, "Number of constraints: {}", self.number_of_constraints)?;
        writeln!(
            f,
            "Number of hint gates processed: {}",
            self.total_hint_gates.load(Ordering::Relaxed)
        )?;
        writeln!(
            f,
            "Total gates processed: {}",
            self.total_gates_processed.load(Ordering::Relaxed)
        )?;
        writeln!(
            f,
            "Gates processed per second: {:.2}",
            self.gates_per_second
        )?;

        Ok(())
    }
}
