//! Resolving a connection's identity when `auth.enabled = false` — see the
//! master plan's *Auth toggle and the local case*: even with auth off, the
//! audit log always has a genuine subject, because UDS peer credentials
//! give the real OS user.

/// Looks up the username for a UID via `getpwuid_r` (POSIX). Falls back to
/// the bare numeric UID (formatted as a string) if the passwd database has
/// no entry — a synthetic principal must never fail to construct just
/// because a lookup came back empty.
#[cfg(unix)]
pub fn username_for_uid(uid: u32) -> String {
    // SAFETY: `buf` is sized generously and `getpwuid_r` never writes past
    // `buf.len()` (it returns ERANGE instead, which we treat as "no
    // username" and fall back). `pwd` is a plain-old-data struct with no
    // invariants beyond what a successful call establishes; we only read
    // `pw_name` after confirming `result` is non-null, i.e. the call
    // populated `pwd` and pointed `result` at it.
    unsafe {
        let mut pwd: libc::passwd = std::mem::zeroed();
        let mut buf = vec![0i8; 4096];
        let mut result: *mut libc::passwd = std::ptr::null_mut();
        let ret = libc::getpwuid_r(uid as libc::uid_t, &mut pwd, buf.as_mut_ptr(), buf.len(), &mut result);
        if ret == 0 && !result.is_null() {
            let name = std::ffi::CStr::from_ptr(pwd.pw_name);
            if let Ok(s) = name.to_str() {
                return s.to_string();
            }
        }
        uid.to_string()
    }
}

#[cfg(not(unix))]
pub fn username_for_uid(uid: u32) -> String {
    uid.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_the_current_process_uid_to_a_non_empty_name() {
        #[cfg(unix)]
        {
            let uid = unsafe { libc::getuid() };
            let name = username_for_uid(uid);
            assert!(!name.is_empty());
        }
    }

    #[test]
    fn an_unlikely_uid_falls_back_to_the_numeric_string_rather_than_panicking() {
        let name = username_for_uid(u32::MAX);
        assert!(!name.is_empty());
    }
}
