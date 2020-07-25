// MLP network that performs the exclusive-or logic function on a pair
// of binary inputs
use {
    rand::prelude::*,
};

pub struct Network {
    weights: Vec<Vec<Vec<f64>>>,
}

impl Network {
    pub fn new() -> Network {
        // One hidden layer with 3 nodes
        // 2 - 3 - 2 configuration
        // Bias weights included
        let mut weights = vec![vec![vec![0_f64; 3]; 3],
                               vec![vec![0_f64; 4]; 1]];

        let mut rng = rand::thread_rng();


        for layer in &mut weights {
            for sum in layer {
                for weight in sum {
                    *weight = rng.gen();
                }
            }
        }

        Network {
            weights
        }
    }

    pub fn decide(&self, input: &[f64]) -> Vec<f64> {
        let mut hidden_result: Vec<f64> = input.iter()
            .cloned()
            .collect();

        // Multiply previous activated hidden layer result by the next
        // weight matrix
        // Begins with input nodes
        for (i, layer) in self.weights.iter().enumerate() {
            let mut new_hidden_result = Vec::with_capacity(layer.len());

            for (j, _) in layer.iter().enumerate() {
                new_hidden_result.push(Self::relu(hidden_result.iter()
                    .chain([1_f64].iter())
                    .zip(self.weights[i][j].iter())
                    .map(|(x, w)| x * w)
                    .sum()));
            }

            hidden_result = new_hidden_result;
        }

        hidden_result
    }

    pub fn train(&mut self, input: &[f64], target: f64) {
        // Hidden nodes that have not been activated
        let mut unactivated = vec![vec![0_f64; 3],
                                   vec![0_f64; 1]];

        // Hidden nodes that have been activated
        // Starts with input layer
        let mut activated   = vec![vec![0_f64; 2],
                                   vec![0_f64; 3],
                                   vec![0_f64; 1]];

        activated[0][0] = input[0];
        activated[0][1] = input[1];

        for (i, layer) in unactivated.iter_mut().enumerate() {
            for (j, sum) in layer.iter_mut().enumerate() {
                *sum = activated[i].iter()
                    .chain([1_f64].iter())
                    .zip(self.weights[i][j].iter())
                    .map(|(x, w)| x * w)
                    .sum();

                activated[i + 1][j] = Self::relu(*sum);
            }
        }

        // Set of gradients to optimize the network
        let mut weights_delta = vec![vec![vec![0_f64; 3]; 3],
                                     vec![vec![0_f64; 4]; 1]];

        // Mean Squared Error
        // Start with first few terms of chain rule
        let d_chain = -2_f64 * (target - activated[2][0])
                             * Self::d_relu(unactivated[1][0])
                             // Learning rate of 0.05
                             * 0.05_f64;

        for i in 0..3 {
            weights_delta[1][0][i] = d_chain * activated[1][i];

            let d_chain = d_chain * self.weights[1][0][i]
                                  * Self::d_relu(unactivated[0][i]);

            for j in 0..2 {
                // Scale weights down for [Input -> Hidden Layer] weights
                // because multiple gradients are summated per weight delta
                weights_delta[0][i][j] += d_chain * activated[0][j] / 3_f64;
            }

            weights_delta[0][i][2] += d_chain / 3_f64;
        }

        weights_delta[1][0][3] = d_chain;

        // Subtract the derivative of the error function from weights
        // to optimize for minimum error
        for (i, layer) in self.weights.iter_mut().enumerate() {
            for (j, sum) in layer.iter_mut().enumerate() {
                for (k, weight) in sum.iter_mut().enumerate() {
                    *weight -= weights_delta[i][j][k];
                }
            }
        }
    }

    // Leaky ReLU function used for activation
    fn relu(x: f64) -> f64 {
        x.max(0.015_f64 * x)
    }

    fn d_relu(x: f64) -> f64 {
        if x < 0_f64 {
            return 0.015_f64;
        } else {
            return 1_f64;
        }
    }
}
