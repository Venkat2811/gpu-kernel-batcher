//! # Inter-Process Communication (IPC) via Shared Memory
//!
//! This module handles the communication between the producer (scheduler) and
//! consumer (GPU worker) processes. It uses a POSIX shared memory (SHM) queue
//! to enable high-throughput, low-latency communication of `GemmCommand`s
//! without involving the kernel for every message.
//!
//! ## Key Components
//!
//! - `ShmQueue`: A lock-free, single-producer, single-consumer (SPSC) bounded
//!   queue built on a shared memory segment. It uses atomic operations for head
//!   and tail pointers to ensure thread-safe and process-safe reads and writes
//!   without locks.
//! - `GemmCommand`: A `#[repr(C)]` struct representing a single matrix
//!   multiplication task, designed to be safely sent across process boundaries.
//!
//! ## ASCII Diagram
//!
//! ```text
//!    Producer Process                Shared Memory                Consumer Process
//! +--------------------+           +-----------------+           +--------------------+
//! |                    |           |   SHM Segment   |           |                    |
//! |   Generates        | --Push--> | [ ][ ][ ][ ][ ] | --Pop-->  |   Executes         |
//! |   GemmCommand      |           |   ^         ^   |           |   GEMMs            |
//! |                    |           |  Head      Tail |           |                    |
//! +--------------------+           +-----------------+           +--------------------+
//!                               (Atomics for sync)
//! ```

use std::ffi::c_void;
use std::ffi::CString;
use std::mem;
use std::ptr;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use anyhow::{Context, Result};
use libc::{shm_open, shm_unlink, O_CREAT, O_EXCL, O_RDWR};
use nix::sys::mman::{mmap, munmap, MapFlags, ProtFlags};
use nix::unistd::{close, ftruncate};
use std::os::unix::io::{FromRawFd, OwnedFd};
use thiserror::Error;

use crate::model::GemmCommand;

/// Errors that can occur when operating on the shared memory queue.
#[derive(Error, Debug)]
pub enum QueueError {
    #[error("Queue is full")]
    Full,
    #[error("Queue is empty")]
    Empty,
    #[error("Shared memory error: {0}")]
    ShmError(String),
    #[error("Invalid capacity (must be power of 2)")]
    InvalidCapacity,
}

/// Cache-aligned header for shared memory queue
#[repr(C, align(64))]
struct ShmHeader {
    // Producer cacheline
    tail: AtomicU64,
    _pad1: [u8; 56],

    // Consumer cacheline
    head: AtomicU64,
    _pad2: [u8; 56],

    // Shared metadata
    capacity: u64,
    stop: AtomicU32,
    _pad3: [u8; 52],
}

/// A lock-free, single-producer, single-consumer (SPSC) queue for sending
/// `GemmCommand`s over a POSIX shared memory segment.
///
/// This queue is bounded and uses a ring buffer for command storage. Head and
/// tail pointers are implemented as cache-aligned atomics to prevent false sharing
/// and enable wait-free synchronization between the producer and consumer processes.
pub struct ShmQueue {
    /// Name of shared memory region
    name: String,
    /// Pointer to mapped memory
    base_ptr: *mut u8,
    /// Size of mapped region
    map_size: usize,
    /// Header pointer
    header: *mut ShmHeader,
    /// Commands array pointer
    commands: *mut GemmCommand,
    /// Capacity (must be power of 2)
    capacity: usize,
    /// Whether this process created the SHM
    is_creator: bool,
    /// File descriptor (for creator)
    fd: Option<i32>,
}

impl ShmQueue {
    /// Creates a new shared memory queue. This function should only be called
    /// by the process that is responsible for creating and owning the queue (e.g., the orchestrator).
    pub fn create(name: &str, capacity: usize) -> Result<Self> {
        // Validate capacity is power of 2
        if !capacity.is_power_of_two() {
            return Err(QueueError::InvalidCapacity.into());
        }

        // Calculate total size needed
        let header_size = mem::size_of::<ShmHeader>();
        let commands_size = capacity * mem::size_of::<GemmCommand>();
        let total_size = header_size + commands_size;

        // Create shared memory name
        let shm_name = if name.starts_with('/') {
            name.to_string()
        } else {
            format!("/{}", name)
        };
        let c_name = CString::new(shm_name.clone()).context("Invalid SHM name")?;

        // Try to unlink any existing SHM
        unsafe {
            shm_unlink(c_name.as_ptr());
        }

        // Create new shared memory
        let fd = unsafe { shm_open(c_name.as_ptr(), O_CREAT | O_EXCL | O_RDWR, 0o600) };

        if fd < 0 {
            return Err(QueueError::ShmError(format!(
                "shm_open failed: {}",
                std::io::Error::last_os_error()
            ))
            .into());
        }

        // Set size
        unsafe {
            let owned_fd = OwnedFd::from_raw_fd(fd);
            ftruncate(&owned_fd, total_size as i64)
                .map_err(|e| QueueError::ShmError(format!("ftruncate failed: {}", e)))?;
            std::mem::forget(owned_fd);
        };

        // Map memory
        let base_ptr = unsafe {
            let owned_fd = OwnedFd::from_raw_fd(fd);
            let result = mmap(
                None, // Let kernel choose address
                std::num::NonZeroUsize::new(total_size).unwrap(),
                ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
                MapFlags::MAP_SHARED,
                Some(&owned_fd),
                0,
            )
            .map_err(|e| QueueError::ShmError(format!("mmap failed: {}", e)))?;
            // Don't drop the fd yet, we'll close it later
            std::mem::forget(owned_fd);
            result
        } as *mut u8;

        // Initialize header
        let header = base_ptr as *mut ShmHeader;
        unsafe {
            ptr::write_bytes(header, 0, 1);
            (*header).capacity = capacity as u64;
            (*header).head.store(0, Ordering::Release);
            (*header).tail.store(0, Ordering::Release);
            (*header).stop.store(0, Ordering::Release);
        }

        // Calculate commands pointer
        let commands = unsafe { base_ptr.add(header_size) as *mut GemmCommand };

        Ok(Self {
            name: shm_name,
            base_ptr,
            map_size: total_size,
            header,
            commands,
            capacity,
            is_creator: true,
            fd: Some(fd),
        })
    }

    /// Opens an existing shared memory queue created by another process.
    pub fn open(name: &str) -> Result<Self> {
        // Create shared memory name
        let shm_name = if name.starts_with('/') {
            name.to_string()
        } else {
            format!("/{}", name)
        };
        let c_name = CString::new(shm_name.clone()).context("Invalid SHM name")?;

        // Open existing shared memory
        let fd = unsafe { shm_open(c_name.as_ptr(), O_RDWR, 0) };

        if fd < 0 {
            return Err(QueueError::ShmError(format!(
                "shm_open failed: {}",
                std::io::Error::last_os_error()
            ))
            .into());
        }

        // Get size (read header first to get capacity)
        let header_size = mem::size_of::<ShmHeader>();

        // Map just header first
        let header_ptr = unsafe {
            let owned_fd = OwnedFd::from_raw_fd(fd);
            let result = mmap(
                None,
                std::num::NonZeroUsize::new(header_size).unwrap(),
                ProtFlags::PROT_READ,
                MapFlags::MAP_SHARED,
                Some(&owned_fd),
                0,
            )
            .map_err(|e| QueueError::ShmError(format!("mmap header failed: {}", e)))?;
            std::mem::forget(owned_fd);
            result
        } as *mut u8;

        // Read capacity
        let capacity = unsafe { (*(header_ptr as *const ShmHeader)).capacity as usize };

        // Unmap header
        unsafe {
            munmap(header_ptr as *mut c_void, header_size)
                .map_err(|e| QueueError::ShmError(format!("munmap header failed: {}", e)))?;
        }

        // Calculate total size
        let commands_size = capacity * mem::size_of::<GemmCommand>();
        let total_size = header_size + commands_size;

        // Map full region
        let base_ptr = unsafe {
            let owned_fd = OwnedFd::from_raw_fd(fd);
            let result = mmap(
                None,
                std::num::NonZeroUsize::new(total_size).unwrap(),
                ProtFlags::PROT_READ | ProtFlags::PROT_WRITE,
                MapFlags::MAP_SHARED,
                Some(&owned_fd),
                0,
            )
            .map_err(|e| QueueError::ShmError(format!("mmap full failed: {}", e)))?;
            std::mem::forget(owned_fd);
            result
        } as *mut u8;

        // Set up pointers
        let header = base_ptr as *mut ShmHeader;
        let commands = unsafe { base_ptr.add(header_size) as *mut GemmCommand };

        // Close fd as we don't need it after mapping
        close(fd).map_err(|e| QueueError::ShmError(format!("close failed: {}", e)))?;

        Ok(Self {
            name: shm_name,
            base_ptr,
            map_size: total_size,
            header,
            commands,
            capacity,
            is_creator: false,
            fd: None,
        })
    }

    /// Pushes a command to the queue. This is a non-blocking operation.
    ///
    /// Returns `Err(QueueError::Full)` if the queue is at capacity.
    pub fn push(&self, cmd: GemmCommand) -> Result<(), QueueError> {
        unsafe {
            let tail = (*self.header).tail.load(Ordering::Acquire);
            let head = (*self.header).head.load(Ordering::Acquire);

            // Check if full
            if tail.wrapping_sub(head) >= self.capacity as u64 {
                return Err(QueueError::Full);
            }

            // Write command
            let index = (tail & (self.capacity - 1) as u64) as usize;
            ptr::write(self.commands.add(index), cmd);

            // Update tail
            (*self.header)
                .tail
                .store(tail.wrapping_add(1), Ordering::Release);

            Ok(())
        }
    }

    /// Pops a command from the queue. This is a non-blocking operation.
    ///
    /// Returns `Err(QueueError::Empty)` if the queue is empty.
    pub fn pop(&self) -> Result<GemmCommand, QueueError> {
        unsafe {
            let head = (*self.header).head.load(Ordering::Acquire);
            let tail = (*self.header).tail.load(Ordering::Acquire);

            // Check if empty
            if head == tail {
                return Err(QueueError::Empty);
            }

            // Read command
            let index = (head & (self.capacity - 1) as u64) as usize;
            let cmd = ptr::read(self.commands.add(index));

            // Update head
            (*self.header)
                .head
                .store(head.wrapping_add(1), Ordering::Release);

            Ok(cmd)
        }
    }

    /// Try to pop without blocking
    pub fn try_pop(&self) -> Option<GemmCommand> {
        self.pop().ok()
    }

    /// Check if queue is empty
    pub fn is_empty(&self) -> bool {
        unsafe {
            let head = (*self.header).head.load(Ordering::Acquire);
            let tail = (*self.header).tail.load(Ordering::Acquire);
            head == tail
        }
    }

    /// Get current queue size
    pub fn size(&self) -> usize {
        unsafe {
            let head = (*self.header).head.load(Ordering::Acquire);
            let tail = (*self.header).tail.load(Ordering::Acquire);
            tail.wrapping_sub(head) as usize
        }
    }

    /// Set stop flag (producer)
    pub fn set_stop(&self) {
        unsafe {
            (*self.header).stop.store(1, Ordering::Release);
        }
    }

    /// Check stop flag (consumer)
    pub fn should_stop(&self) -> bool {
        unsafe { (*self.header).stop.load(Ordering::Acquire) != 0 }
    }

    /// Get queue capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Get queue name
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Drop for ShmQueue {
    fn drop(&mut self) {
        // Unmap memory
        if !self.base_ptr.is_null() {
            unsafe {
                let _ = munmap(self.base_ptr as *mut c_void, self.map_size);
            }
        }

        // Close fd if we have one
        if let Some(fd) = self.fd {
            let _ = close(fd);
        }

        // Unlink shared memory if we created it
        if self.is_creator {
            if let Ok(c_name) = CString::new(self.name.clone()) {
                unsafe {
                    shm_unlink(c_name.as_ptr());
                }
            }
        }
    }
}

// Safety: ShmQueue can be sent between threads (but not shared)
unsafe impl Send for ShmQueue {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_create_open() {
        let name = "/test_queue_create";
        let capacity = 1024;

        // Create queue
        let queue1 = ShmQueue::create(name, capacity).unwrap();
        assert_eq!(queue1.capacity(), capacity);
        assert!(queue1.is_empty());

        // Open same queue
        let queue2 = ShmQueue::open(name).unwrap();
        assert_eq!(queue2.capacity(), capacity);
        assert!(queue2.is_empty());
    }

    #[test]
    fn test_push_pop() {
        let name = "/test_push_pop";
        let queue = ShmQueue::create(name, 16).unwrap();

        // Push some commands
        let cmd1 = GemmCommand::new(64, 128, 256);
        let cmd2 = GemmCommand::new(32, 64, 128);

        queue.push(cmd1).unwrap();
        queue.push(cmd2).unwrap();

        assert_eq!(queue.size(), 2);

        // Pop commands
        let popped1 = queue.pop().unwrap();
        assert_eq!(popped1.m, cmd1.m);
        assert_eq!(popped1.n, cmd1.n);
        assert_eq!(popped1.k, cmd1.k);

        let popped2 = queue.pop().unwrap();
        assert_eq!(popped2.m, cmd2.m);

        assert!(queue.is_empty());
    }

    #[test]
    fn test_queue_full() {
        let name = "/test_full";
        let queue = ShmQueue::create(name, 4).unwrap();

        // Fill queue
        for i in 0..4 {
            let cmd = GemmCommand::new(i as u32, i as u32, i as u32);
            queue.push(cmd).unwrap();
        }

        // Should be full
        let cmd = GemmCommand::new(99, 99, 99);
        assert!(matches!(queue.push(cmd), Err(QueueError::Full)));
    }

    #[test]
    fn test_stop_flag() {
        let name = "/test_stop";
        let queue = ShmQueue::create(name, 16).unwrap();

        assert!(!queue.should_stop());
        queue.set_stop();
        assert!(queue.should_stop());
    }

    #[test]
    fn test_wrap_around() {
        let name = "/test_wrap";
        let queue = ShmQueue::create(name, 4).unwrap();

        // Fill and empty multiple times to test wrap-around
        for round in 0..3 {
            for i in 0..4 {
                let cmd = GemmCommand::new(round * 10 + i, i, i);
                queue.push(cmd).unwrap();
            }

            for i in 0..4 {
                let cmd = queue.pop().unwrap();
                assert_eq!(cmd.m, round * 10 + i);
            }

            assert!(queue.is_empty());
        }
    }
}
