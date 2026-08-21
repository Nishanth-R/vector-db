//! CLI entrypoint (binary name `mara`). Not yet implemented beyond --version.
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--version") {
        println!("mara {}", env!("CARGO_PKG_VERSION"));
        return;
    }
    eprintln!("mara: not yet implemented");
    std::process::exit(1);
}
