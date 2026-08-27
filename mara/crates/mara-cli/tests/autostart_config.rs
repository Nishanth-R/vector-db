//! Regression test for a real bug: autostart used to spawn `mara serve
//! --data-dir <X>` using `[server] data_dir` straight out of the
//! discovered config file, then `mara serve` re-derived `<X>/mara.toml`
//! to decide what config to load — silently missing the *actual* config
//! file whenever `[server] data_dir` isn't the same directory the config
//! itself lives in (a perfectly normal layout: a config file with
//! `data_dir = "./data"` relative to it). The daemon it spawned then
//! silently fell back to `Config::default_for`, discarding every
//! non-`[server]`/`[daemon]` section — `[auth]` included.
//!
//! Fixed by having `endpoint::resolve`'s already-discovered config path
//! flow through `autostart` as `mara serve --config <path>`, used as-is
//! instead of re-derived. This test reproduces the exact layout and
//! checks for the *effect* of the fix rather than internals: with
//! `[auth] enabled = true` (and `allow_local_unauthenticated = false`,
//! since that otherwise exempts local UDS connections regardless) in a
//! config laid out this way, a plain command against the autostarted
//! daemon must fail — if the bug regresses, the spawned daemon silently
//! has auth *disabled* (`default_for`'s value) and the command would
//! instead succeed.

use std::io::Write as _;
use std::path::Path;
use std::process::Command;
use std::time::Duration;

fn mara_cmd() -> Command {
    Command::new(env!("CARGO_BIN_EXE_mara"))
}

fn stop_daemon(data_dir: &Path) {
    let _ = mara_cmd().args(["--data-dir", &data_dir.to_string_lossy(), "daemon", "stop"]).output();
}

#[test]
fn autostart_loads_the_discovered_config_even_when_data_dir_is_a_subdirectory_of_it() {
    let tmp = tempfile::tempdir().unwrap();
    // `data_dir` is a *subdirectory* of where mara.toml itself lives —
    // the exact layout that triggered the bug.
    let data_dir = tmp.path().join("data");
    let config_path = tmp.path().join("mara.toml");
    let mut f = std::fs::File::create(&config_path).unwrap();
    // `allow_local_unauthenticated` must be turned off too — it defaults
    // to `true`, exempting local UDS connections from needing a token
    // even with auth enabled (see the master plan's *Auth toggle and the
    // local case*), which would make this test pass regardless of
    // whether the real config loaded at all.
    writeln!(
        f,
        "[server]\ndata_dir = {:?}\nunix_socket = \"mara.sock\"\n[auth]\nenabled = true\nallow_local_unauthenticated = false\n",
        data_dir
    )
    .unwrap();
    drop(f);

    // Autostart triggers on this first command — nothing is running yet.
    // With the bug present, the spawned daemon has auth *disabled*
    // (`default_for`'s value) and this succeeds; with the fix, `[auth]
    // enabled = true` from the real config is in effect and it's
    // rejected — mara-cli has no way to supply a token, so any command
    // against an auth-enabled daemon fails.
    let output = mara_cmd()
        .args(["--data-dir", &tmp.path().to_string_lossy(), "create-collection", "regression-test", "--dim", "2"])
        .output()
        .unwrap();

    stop_daemon(&data_dir);
    // Give the daemon a moment to release its socket/pid file before the
    // temp dir is dropped — best-effort, not load-bearing for the assertion.
    std::thread::sleep(Duration::from_millis(200));

    assert!(!output.status.success(), "expected the command to be rejected by an auth-enabled daemon, but it succeeded — the real [auth] config was not loaded");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("auth") || stderr.contains("token"),
        "expected an auth-related rejection, got: {stderr}"
    );
}
