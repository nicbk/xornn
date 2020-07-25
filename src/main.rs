use {
    xornn::Network,
};

fn main() {
    let mut network = Network::new();

    let training_data = vec![[0_f64, 0_f64, 0_f64],
                             [0_f64, 1_f64, 1_f64],
                             [1_f64, 0_f64, 1_f64],
                             [1_f64, 1_f64, 0_f64]];

    for _ in 0..100_000 {
        for lesson in &training_data {
            network.train(&lesson[0..2], lesson[2]);
        }
    }

    for lesson in &training_data {
        let decision = network.decide(&lesson[0..2]);
        println!("{:?}", decision);
    }
}
