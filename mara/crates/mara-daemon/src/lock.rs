//! The data-dir lock (master plan Layer 3, *Autostart & embedded* —
//! *Single-writer safety*): `<data_dir>/.mara.lock` is an advisory
//! exclusive file lock held for the lifetime of whichever process owns the
//! directory. Two writers appending to one WAL would corrupt it, so this
//! is not optional — `marad`, `mara serve`, and `--embedded` all take it
//! before touching storage, and all fail fast and clearly if another
//! process already holds it.

use std::fs::{File, OpenOptions};
use std::io;
use std::path::Path;

pub struct DataDirLock {
    // Holds the write guard for the process's entire lifetime. The
    // `'static` lifetime comes from leaking the boxed `RwLock` — a
    // deliberate, one-time, process-lifetime leak (reclaimed by the OS at
    // exit regardless), not a bug: `fd_lock::RwLock::try_write` borrows
    // `&mut self`, so holding the guard past the function that acquired it
    // requires the lock to outlive that function's stack frame.
    _guard: fd_lock::RwLockWriteGuard<'static, File>,
}

impl DataDirLock {
    /// Acquires the lock, creating `data_dir` if needed. A clear error
    /// (not a hang) if another process already holds it.
    pub fn acquire(data_dir: &Path) -> io::Result<Self> {
        std::fs::create_dir_all(data_dir)?;
        let path = data_dir.join(".mara.lock");
        let file = OpenOptions::new().create(true).write(true).open(&path)?;
        let lock: &'static mut fd_lock::RwLock<File> = Box::leak(Box::new(fd_lock::RwLock::new(file)));
        let guard = lock.try_write().map_err(|_| {
            io::Error::new(
                io::ErrorKind::WouldBlock,
                format!(
                    "data dir {:?} is already locked by another mara process (marad/`mara serve`/--embedded) — only one process may own a data dir at a time",
                    data_dir.display()
                ),
            )
        })?;
        Ok(DataDirLock { _guard: guard })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_second_lock_on_the_same_data_dir_fails_clearly() {
        let dir = tempfile::tempdir().unwrap();
        let _first = DataDirLock::acquire(dir.path()).unwrap();
        let second = DataDirLock::acquire(dir.path());
        assert!(second.is_err(), "a second lock attempt on the same data dir must fail, not hang or silently succeed");
    }

    #[test]
    fn the_lock_is_released_when_the_guard_drops() {
        let dir = tempfile::tempdir().unwrap();
        {
            let _first = DataDirLock::acquire(dir.path()).unwrap();
        }
        let second = DataDirLock::acquire(dir.path());
        assert!(second.is_ok(), "dropping the first lock must release it for a subsequent acquire");
    }

    #[test]
    fn different_data_dirs_never_conflict() {
        let dir_a = tempfile::tempdir().unwrap();
        let dir_b = tempfile::tempdir().unwrap();
        let _a = DataDirLock::acquire(dir_a.path()).unwrap();
        let _b = DataDirLock::acquire(dir_b.path()).unwrap();
    }
}
