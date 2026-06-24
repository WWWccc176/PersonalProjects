use clap::Parser;
use matrix::computer::cli::{Cli, run_cli};

fn main() {
    let cli = Cli::parse();
    if let Err(e) = run_cli(cli) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}
