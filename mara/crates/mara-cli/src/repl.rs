//! Interactive shell (`mara repl`): one already-connected `MaraClient`,
//! reused across every line, each parsed with the exact same `clap`
//! grammar the non-interactive CLI uses — so `search docs --text "how
//! does x work" -k 5` behaves identically whether typed at a shell prompt
//! or inside the repl. `exit`/`quit` or Ctrl-D leaves; Ctrl-C cancels the
//! current line without leaving.

use crate::request;
use crate::{Cli, Commands};
use clap::Parser;
use mara_client::MaraClient;
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;

pub async fn run(client: &MaraClient) -> Result<(), String> {
    let mut rl = DefaultEditor::new().map_err(|e| e.to_string())?;
    println!("mara repl — one command per line, e.g. `get docs a` or `search docs --text \"...\" -k 5`.");
    println!("`--help` on any command for its flags; `exit`, `quit`, or Ctrl-D to leave.");

    loop {
        let line = match rl.readline("mara> ") {
            Ok(l) => l,
            Err(ReadlineError::Interrupted) => continue,
            Err(ReadlineError::Eof) => break,
            Err(e) => {
                eprintln!("error: {e}");
                break;
            }
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let _ = rl.add_history_entry(trimmed);
        if matches!(trimmed, "exit" | "quit") {
            break;
        }

        let tokens = match shell_words::split(trimmed) {
            Ok(t) => t,
            Err(_) => {
                eprintln!("error: unbalanced quotes");
                continue;
            }
        };
        // clap expects argv[0] to be the program name.
        let cli = match Cli::try_parse_from(std::iter::once("mara".to_string()).chain(tokens)) {
            Ok(c) => c,
            // clap's error `Display` already includes usage/help text —
            // e.g. an unrecognized subcommand, or `--help` itself, which
            // clap intentionally surfaces as an "error" carrying the help
            // text to print.
            Err(e) => {
                println!("{e}");
                continue;
            }
        };

        if let Err(e) = dispatch_one(client, &cli.command).await {
            eprintln!("error: {e}");
        }
    }
    Ok(())
}

async fn dispatch_one(client: &MaraClient, cmd: &Commands) -> Result<(), String> {
    if matches!(cmd, Commands::Repl) {
        return Err("repl cannot be nested inside itself".into());
    }
    let Some(req) = request::command_to_request(cmd)? else {
        return Err("this command needs its own process/connection setup and isn't available inside the repl — run it directly instead (e.g. `mara daemon status`)".into());
    };
    let response = client.call(req).await.map_err(|e| e.to_string())?;
    request::print_response(cmd, response)
}
