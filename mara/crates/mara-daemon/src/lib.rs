//! `marad`'s composition root (master plan Layer 3): tokio bootstrap,
//! config, UDS+TCP listeners, the `Engine` dispatching to storage. `mara
//! serve` (autostart) and `mara-api` both call into this same library —
//! see the master plan's *Why both `marad` and `mara serve`*.

pub mod config;
pub mod engine;
pub mod identity;
pub mod listener;
pub mod lock;
pub mod replication;
pub mod server;

pub use config::{Config, ConfigError};
pub use engine::{Engine, EngineImpl};
pub use listener::DaemonShared;
pub use lock::DataDirLock;
pub use server::{boot, run, BootError};
