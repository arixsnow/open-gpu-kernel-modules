/*******************************************************************************
    Copyright (c) 2015-2025 NVIDIA Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/

#ifndef __UVM_LOCK_H__
#define __UVM_LOCK_H__

#include "uvm_forward_decl.h"
#include "uvm_linux.h"
#include "uvm_common.h"

// --------------------------- UVM Locking Order ---------------------------- //
//
// Any locks described here should have their locking order added to
// uvm_lock_order_t below.
//
// - Global power management lock (g_uvm_global.pm.lock)
//      Order: UVM_LOCK_ORDER_GLOBAL_PM
//      Reader/write lock (rw_semaphore)
//
//      Synchronizes user threads with system power management.
//
//      Taken in read mode by most user-facing UVM driver entry points.  Taken
//      in write mode by uvm_suspend(), only, and held for the duration of
//      sleep cycles.
//
//      This lock is special: while it's taken by user-facing entry points,
//      and may be taken before or after mmap_lock, this apparent violation of
//      lock ordering is permissible because pm_lock may only be taken via
//      trylock in read mode by paths which already hold any lower-level
//      locks, as well as by paths subject to the kernel's freezer.  Paths
//      taking it must be prepared to back off in case of acquisition failures.
//
//      This, in turn, is acceptable because the lock is taken in write mode
//      infrequently, and only as part of to power management.  Starvation is
//      not a concern.
//
//      The mmap_lock deadlock potential aside, the trylock approch is also
//      motivated by the need to prevent user threads making UVM system calls
//      from blocking when UVM is suspended: when the kernel suspends the
//      system, the freezer employed to stop user tasks requires these tasks
//      to be interruptible.
//
// - Global driver state lock (g_uvm_global.global_lock)
//      Order: UVM_LOCK_ORDER_GLOBAL
//      Exclusive lock (mutex)
//
//      This protects state associated with GPUs, such as the P2P table
//      and instance pointer mappings.
//
//      This should be taken whenever global GPU state might need to be modified.
//
// - Access counters VA space enablement state lock
//      Order: UVM_LOCK_ORDER_ACCESS_COUNTERS
//      Exclusive lock (mutex)
//
//      This protects VA space state associated with access counters enablement.
//      Blackwell+ GPUs may have multiple access counters notification buffers
//      and their "atomic" enablement is protected by this lock.
//
//      This should be taken whenever VA space access counters state might need
//      to be modified.
//
// - GPU ISR lock
//      Order: UVM_LOCK_ORDER_ISR
//      Exclusive lock (mutex) per gpu
//
//      Protects:
//      - gpu->parent->isr.replayable_faults.service_lock:
//        Changes to the state of a GPU as it transitions from top-half to
//        bottom-half interrupt handler for replayable faults. This lock is
//        acquired for that GPU, in the ISR top-half. Then a bottom-half is
//        scheduled (to run in a workqueue). Then the bottom-half releases the
//        lock when that GPU's processing appears to be done.
//
//      - gpu->parent->isr.non_replayable_faults.service_lock:
//        Changes to the state of a GPU in the bottom-half for non-replayable
//        faults. Non-replayable faults are handed-off from RM instead of
//        directly from the GPU hardware. This means that we do not keep
//        receiving interrupts after RM pops out the faults from the HW buffer.
//        In order not to miss fault notifications, we will always schedule a
//        bottom-half for non-replayable faults if there are faults ready to be
//        consumed in the buffer, even if there already is some bottom-half
//        running or scheduled. This lock serializes all scheduled bottom halves
//        per GPU which service non-replayable faults.
//
//      - gpu->parent->isr.access_counters.service_lock:
//        Changes to the state of a GPU as it transitions from top-half to
//        bottom-half interrupt handler for access counter notifications. This
//        lock is acquired for that GPU, in the ISR top-half. Then a bottom-half
//        is scheduled (to run in a workqueue). Then the bottom-half releases
//        the lock when that GPU's processing appears to be done.
//
// - mmap_lock (mmap_sem in kernels < 5.8)
//      Order: UVM_LOCK_ORDER_MMAP_LOCK
//      Reader/writer lock (rw_semaphore)
//
//      We're often called with the kernel already holding mmap_lock: mmap,
//      munmap, CPU fault, etc. These operations may have to take any number of
//      UVM locks, so mmap_lock requires special consideration in the lock
//      order, since it's sometimes out of our control.
//
//      We need to hold mmap_lock when calling vm_insert_page, which means that
//      any time an operation (such as an ioctl) might need to install a CPU
//      mapping, it must take mmap_lock in read mode very early on.
//
//      However, current->mm is not necessarily the owning mm of the UVM vma.
//      fork or fd passing via a UNIX doman socket can cause that. Notably, this
//      is also the case when handling GPU faults or doing other operations from
//      a kernel thread. In some cases we have an mm associated with a VA space,
//      and in those cases we lock that mm instead of current->mm. But since we
//      don't always have that luxury, each path specifies the mm to use (either
//      explicitly or via uvm_va_block_context_t::mm). That mm may be NULL.
//      Later on down the stack we look up the UVM vma and compare its mm before
//      operating on that vma.
//
//      With HMM and ATS, the GPU fault handler takes mmap_lock. GPU faults may
//      block forward progress of threads holding the RM GPUs lock until those
//      faults are serviced, which means that mmap_lock cannot be held when the
//      UVM driver calls into RM. In other words, mmap_lock and the RM GPUs lock
//      are mutually exclusive.
//
// - Global VA spaces list lock
//      Order: UVM_LOCK_ORDER_VA_SPACES_LIST
//      Mutex which protects g_uvm_global.va_spaces state.
//
// - VA space writer serialization lock (va_space->serialize_writers_lock)
//      Order: UVM_LOCK_ORDER_VA_SPACE_SERIALIZE_WRITERS
//      Exclusive lock (mutex) per uvm_va_space (UVM struct file)
//
//      This lock prevents a deadlock between RM and UVM by only allowing one
//      writer to queue up on the VA space lock at a time.
//
//      GPU faults are serviced by the UVM bottom half with the VA space lock
//      held in read mode. Until they're serviced, these faults may block
//      forward progress of RM threads.
//
//      This constraint means that the UVM driver cannot call into RM while
//      GPU fault servicing is blocked. We may block GPU fault servicing by:
//      - Taking the VA space lock in write mode
//      - Holding the VA space lock in read mode with a writer pending, since
//        Linux rw_semaphores are fair.
//
//      Example of the second condition:
//      Thread A        Thread B        UVM BH          Thread C
//      UVM API call    UVM API call    GPU fault       RM API call
//      ------------    ------------    ------------    ------------
//      down_read
//                      down_write
//                      // Blocked on A
//                                      down_read
//                                      // Blocked on B
//                                                      RM GPU lock
//                                                      // Blocked on GPU fault
//      RM GPU lock
//      // Deadlock
//
//      The writer serialization lock works around this by biasing the VA space
//      lock towards readers, without causing starvation of writers. Writers and
//      readers which will make RM calls take this lock, which prevents them
//      from queueing up on the VA space rw_semaphore and blocking the UVM
//      bottom half.
//
//      TODO: Bug 1799173: A better long-term approach might be to never allow
//            RM calls under the VA space lock at all, but that will take a
//            larger restructuring.
//
// - VA space serialization of down_read with up_write of the VA space lock
//   (va_space->read_acquire_write_release_lock)
//      Order: UVM_LOCK_ORDER_VA_SPACE_READ_ACQUIRE_WRITE_RELEASE_LOCK
//      Exclusive lock (mutex) per uvm_va_space (UVM struct file)
//
//      This lock prevents a deadlock between RM and UVM by preventing any
//      interleaving of down_reads on the VA space lock with concurrent
//      up_writes/downgrade_writes. The Linux rw_semaphore implementation does
//      not guarantee that two readers will always run concurrently, as shown by
//      the following interleaving:
//
//      Thread A                Thread B
//      UVM API call            UVM BH
//      ------------            ------------
//      down_write
//                              down_read
//                                  // Fails, calls handler
//      up_write
//      down_read
//          // Success
//                                  // Handler sees the lock still active
//                                  // Handler waits for lock to be released
//                                  // Blocked on A
//      RM GPU lock
//      // Blocked on GPU fault
//
//      Given the above interleaving, the kernel's implementation of the
//      down_read failure handler running in thread B does not distinguish
//      between a reader vs writer holding the lock. From the perspective of all
//      other threads, even those which attempt to take the lock for read while
//      thread A's reader holds it, a writer is active. Therefore no other
//      readers can take the lock, and we result in the same deadlock described
//      in the above comments on the VA space writer serialization lock.
//
//      This lock prevents any such interleaving:
//      - Writers take this lock for the duration of the write lock.
//
//      - Readers which do not call into RM only take this lock across the
//        down_read call. If a writer holds the lock, the reader would be
//        blocked on the VA space lock anyway. Concurrent readers will serialize
//        the taking of the VA space lock, but they will not be serialized
//        across their read sections.
//
//      - Readers which call into RM do not need to take this lock. Their
//        down_read is already serialized with a writer's up_write by the
//        serialize_writers_lock.
//
// - VA space lock (va_space->lock)
//      Order: UVM_LOCK_ORDER_VA_SPACE
//      Reader/writer lock (rw_semaphore) per uvm_va_space (UVM struct file)
//
//      This is the UVM equivalent of mmap_lock. It protects all state under
//      that va_space, such as the VA range tree.
//
//      Read mode: Faults (CPU and GPU), mapping creation, prefetches. These
//      will be serialized at the VA block level if necessary. RM calls are
//      allowed only if the VA space serialize_writers_lock is also taken.
//
//      Write mode: Modification of the range state such as mmap and changes to
//      logical permissions or location preferences. RM calls are never allowed.
//
// - External Allocation Tree lock
//      Order: UVM_LOCK_ORDER_EXT_RANGE_TREE
//      Exclusive lock (mutex) per external VA range, per GPU.
//
//      Protects the per-GPU sub-range tree mappings in each external VA range.
//
// - GPU semaphore pool lock (semaphore_pool->mutex)
//      Order: UVM_LOCK_ORDER_GPU_SEMAPHORE_POOL
//      Exclusive lock (mutex) per uvm_gpu_semaphore_pool
//
//      Protects the state of the semaphore pool.
//
// - RM API lock
//      Order: UVM_LOCK_ORDER_RM_API
//      Exclusive lock
//
//      This is an internal RM lock that's acquired by most if not all UVM-RM
//      APIs.
//      Notably this lock is also held on PMA eviction.
//
// - RM GPUs lock
//      Order: UVM_LOCK_ORDER_RM_GPUS
//      Exclusive lock
//
//      This is an internal RM lock that's acquired by most if not all UVM-RM
//      APIs and disables interrupts for the GPUs.
//      Notably this lock is *not* held on PMA eviction.
//
// - VA block lock (va_block->lock)
//      Order: UVM_LOCK_ORDER_VA_BLOCK
//      Exclusive lock (mutex)
//
//      Protects:
//      - CPU and GPU page table mappings for all VAs under the block
//      - Updates to the GPU work tracker for that block (migrations)
//
//      Operations allowed while holding the lock:
//      - CPU allocation (we don't evict CPU memory)
//      - GPU memory allocation which cannot evict
//      - CPU page table mapping/unmapping
//      - Pushing work (GPU page table mapping/unmapping)
//
//      Operations not allowed while holding the lock:
//      - GPU memory allocation which can evict memory (would require nesting
//        block locks)
//
// - GPU DMA Allocation pool lock (gpu->conf_computing.dma_buffer_pool.lock)
//      Order: UVM_LOCK_ORDER_CONF_COMPUTING_DMA_BUFFER_POOL
//      Condition: The Confidential Computing feature is enabled
//      Exclusive lock (mutex)
//
//      Protects:
//      - Protect the state of the uvm_conf_computing_dma_buffer_pool_t
//
// - Chunk mapping lock (gpu->root_chunk_mappings.bitlocks and
//   gpu->sysmem_mappings.bitlock)
//      Order: UVM_LOCK_ORDER_CHUNK_MAPPING
//      Exclusive bitlock (mutex) per each root chunk, or physical sysmem
//      segment.
//
//      A chunk mapping lock is used to enforce serialization when updating
//      kernel mappings of GPU root chunks (vidmem), or CPU chunks (sysmem).
//      The VA block lock is usually held during the mapping operation.
//
//      In the case of vidmem, each lock in the bitlock array serializes the
//      mapping and unmapping of a single GPU root chunk. If serialization
//      is required to update a root chunk, but no mappings are involved, use
//      the PMM root chunk lock (order UVM_LOCK_ORDER_PMM_ROOT_CHUNK) instead.
//
//      In the case of sysmem, each lock in the array serializes the mapping
//      of a large segment of system address space: the locking granularity is
//      significantly coarser than the CPU chunk size.
//
// - Page tree lock
//      Order: UVM_LOCK_ORDER_PAGE_TREE
//      Exclusive lock per GPU page tree
//
//      This protects a page tree.  All modifications to the device's page tree
//      and the host-side cache of that tree must be done under this lock.
//      The host-side cache and device state must be consistent when this lock
//      is released
//
//      Operations allowed while holding this lock
//      - Pushing work
//
//      Operations not allowed while holding this lock
//      - GPU memory allocation which can evict
//
// - Channel pool key rotation lock
//      Order: UVM_LOCK_ORDER_KEY_ROTATION
//      Condition: Confidential Computing is enabled
//      Mutex per channel pool
//
//      The lock ensures mutual exclusion during key rotation affecting all the
//      channels in the associated pool. Key rotation in WLC pools is handled
//      using a separate lock order, see UVM_LOCK_ORDER_KEY_ROTATION_WLC below.
//
// - CE channel CSL channel pool semaphore
//      Order: UVM_LOCK_ORDER_CSL_PUSH
//      Condition: The Confidential Computing feature is enabled
//      Semaphore per CE channel pool
//
//      The semaphore controls concurrent pushes to CE channels that are not WCL
//      channels. Secure work submission depends on channel availability in
//      GPFIFO entries (as in any other channel type) but also on channel
//      locking. Each channel has a lock to enforce ordering of pushes. The
//      channel's CSL lock is taken on channel reservation until uvm_push_end.
//      When the Confidential Computing feature is enabled, channels are
//      stateful, and the CSL lock protects their CSL state/context.
//
//      Operations allowed while holding this lock
//      - Pushing work to CE channels (except for WLC channels)
//
// - WLC channel pool key rotation lock
//      Order: UVM_LOCK_ORDER_KEY_ROTATION_WLC
//      Condition: Confidential Computing is enabled
//      Mutex of WLC channel pool
//
//      The lock has the same purpose as the regular channel pool key rotation
//      lock. Using a different order lock for WLC channels allows key rotation
//      on those channels during indirect work submission.
//
// - WLC CSL channel pool semaphore
//      Order: UVM_LOCK_ORDER_CSL_WLC_PUSH
//      Condition: The Confidential Computing feature is enabled
//      Semaphore per WLC channel pool
//
//      The semaphore controls concurrent pushes to WLC channels. WLC work
//      submission depends on channel availability in GPFIFO entries (as in any
//      other channel type) but also on channel locking. Each WLC channel has a
//      lock to enforce ordering of pushes. The channel's CSL lock is taken on
//      channel reservation until uvm_push_end. SEC2 channels are stateful
//      channels and the CSL lock protects their CSL state/context.
//
//      This lock ORDER is different and sits below the generic channel CSL
//      lock and above the SEC2 CSL lock. This reflects the dual nature of WLC
//      channels; they use SEC2 indirect work launch during initialization,
//      and after their schedule is initialized they provide indirect launch
//      functionality to other CE channels.
//
//      Operations allowed while holding this lock
//      - Pushing work to WLC channels
//
// - SEC2 CSL channel pool semaphore
//      Order: UVM_LOCK_ORDER_SEC2_CSL_PUSH
//      Condition: The Confidential Computing feature is enabled
//      Semaphore per SEC2 channel pool
//
//      The semaphore controls concurrent pushes to SEC2 channels. SEC2 work
//      submission depends on channel availability in GPFIFO entries (as in any
//      other channel type) but also on channel locking. Each SEC2 channel has a
//      lock to enforce ordering of pushes. The channel's CSL lock is taken on
//      channel reservation until uvm_push_end. SEC2 channels are stateful
//      channels and the CSL lock protects their CSL state/context.
//
//      This lock ORDER is different and lower than UVM_LOCK_ORDER_CSL_PUSH
//      to allow secure work submission to use a SEC2 channel to submit
//      work before releasing the CSL lock of the originating channel.
//
//      Operations allowed while holding this lock
//      - Pushing work to SEC2 channels
//
// - Access counters clear operations
//     Order: UVM_LOCK_ACCESS_COUNTERS_CLEAR_OPS
//
//     It protects the parent_gpu's access counters clear tracker.
//
// - Concurrent push semaphore
//      Order: UVM_LOCK_ORDER_PUSH
//      Semaphore (uvm_semaphore_t)
//
//      This is a semaphore limiting the amount of concurrent pushes that is
//      held for the duration of a push (between uvm_push_begin*() and
//      uvm_push_end()).
//
// - PMM GPU lock (pmm->lock)
//      Order: UVM_LOCK_ORDER_PMM
//      Exclusive lock (mutex) per uvm_pmm_gpu_t
//
//      Protects the state of PMM - internal to PMM.
//
// - PMM GPU PMA lock (pmm->pma_lock)
//      Order: UVM_LOCK_ORDER_PMM_PMA
//      Reader/writer lock (rw_semaphore) per per uvm_pmm_gpu_t
//
//      Lock internal to PMM for synchronizing allocations from PMA with
//      PMA eviction.
//
// - PMM root chunk lock (pmm->root_chunks.bitlocks)
//      Order: UVM_LOCK_ORDER_PMM_ROOT_CHUNK
//      Exclusive bitlock (mutex) per each root chunk internal to PMM.
//
// - Channel lock
//      Order: UVM_LOCK_ORDER_CHANNEL
//      Spinlock (uvm_spinlock_t) or exclusive lock (mutex)
//
//      Lock protecting the state of all the channels in a channel pool. The
//      channel pool lock documentation contains the guidelines about which lock
//      type (mutex or spinlock) to use.
//
// - WLC Channel lock
//      Order: UVM_LOCK_ORDER_WLC_CHANNEL
//      Condition: The Confidential Computing feature is enabled
//      Spinlock (uvm_spinlock_t)
//
//      Lock protecting the state of WLC channels in a channel pool. This lock
//      is separate from the generic channel lock (UVM_LOCK_ORDER_CHANNEL)
//      to allow for indirect worklaunch pushes while holding the main channel
//      lock (WLC pushes don't need any of the pushbuffer locks described
//      above)
//
// - Tools global VA space list lock (g_tools_va_space_list_lock)
//      Order: UVM_LOCK_ORDER_TOOLS_VA_SPACE_LIST
//      Reader/writer lock (rw_semaphore)
//
//      This lock protects the list of VA spaces used when broadcasting
//      UVM profiling events.
//
// - VA space events
//      Order: UVM_LOCK_ORDER_VA_SPACE_EVENTS
//      Reader/writer lock (rw_semaphore) per uvm_perf_va_space_events_t.
//      serializes perf callbacks with event register/unregister. It's separate
//      from the VA space lock so it can be taken on the eviction path.
//
// - VA space tools
//      Order: UVM_LOCK_ORDER_VA_SPACE_TOOLS
//      Reader/writer lock (rw_semaphore) per uvm_va_space_t. Serializes tools
//      reporting with tools register/unregister. Since some of the tools
//      events come from perf events, both VA_SPACE_EVENTS and VA_SPACE_TOOLS
//      must be taken to register/report some tools events.
//
// - Tracking semaphores
//      Order: UVM_LOCK_ORDER_SECURE_SEMAPHORE
//      Condition: The Confidential Computing feature is enabled
//
//      CE semaphore payloads are encrypted, and require to take the CSL lock
//      (UVM_LOCK_ORDER_LEAF) to decrypt the payload.
//
// - CSL Context
//      Order: UVM_LOCK_ORDER_CSL_CTX
//      When the Confidential Computing feature is enabled, encrypt/decrypt
//      operations to communicate with GPU are handled by the CSL context.
//      This lock protects RM calls that use this context.
//
// - Leaf locks
//      Order: UVM_LOCK_ORDER_LEAF
//
//      All leaf locks.
//
// -------------------------------------------------------------------------- //

// Remember to add any new lock orders to uvm_lock_order_to_string() in
// uvm_lock.c
typedef enum
{
    UVM_LOCK_ORDER_INVALID = 0,
    UVM_LOCK_ORDER_GLOBAL_PM,
    UVM_LOCK_ORDER_GLOBAL,
    UVM_LOCK_ORDER_ACCESS_COUNTERS,
    UVM_LOCK_ORDER_ISR,
    UVM_LOCK_ORDER_MMAP_LOCK,
    UVM_LOCK_ORDER_VA_SPACES_LIST,
    UVM_LOCK_ORDER_VA_SPACE_SERIALIZE_WRITERS,
    UVM_LOCK_ORDER_VA_SPACE_READ_ACQUIRE_WRITE_RELEASE_LOCK,
    UVM_LOCK_ORDER_VA_SPACE,
    UVM_LOCK_ORDER_EXT_RANGE_TREE,
    UVM_LOCK_ORDER_GPU_SEMAPHORE_POOL,
    UVM_LOCK_ORDER_RM_API,
    UVM_LOCK_ORDER_RM_GPUS,
    UVM_LOCK_ORDER_VA_BLOCK_MIGRATE,
    UVM_LOCK_ORDER_VA_BLOCK,
    UVM_LOCK_ORDER_CONF_COMPUTING_DMA_BUFFER_POOL,
    UVM_LOCK_ORDER_CHUNK_MAPPING,
    UVM_LOCK_ORDER_PAGE_TREE,
    UVM_LOCK_ORDER_KEY_ROTATION,
    UVM_LOCK_ORDER_CSL_PUSH,
    UVM_LOCK_ORDER_KEY_ROTATION_WLC,
    UVM_LOCK_ORDER_CSL_WLC_PUSH,
    UVM_LOCK_ORDER_CSL_SEC2_PUSH,
    UVM_LOCK_ACCESS_COUNTERS_CLEAR_OPS,
    UVM_LOCK_ORDER_PUSH,
    UVM_LOCK_ORDER_PMM,
    UVM_LOCK_ORDER_PMM_PMA,
    UVM_LOCK_ORDER_PMM_ROOT_CHUNK,
    UVM_LOCK_ORDER_CHANNEL,
    UVM_LOCK_ORDER_WLC_CHANNEL,
    UVM_LOCK_ORDER_TOOLS_VA_SPACE_LIST,
    UVM_LOCK_ORDER_VA_SPACE_EVENTS,
    UVM_LOCK_ORDER_VA_SPACE_TOOLS,
    UVM_LOCK_ORDER_SEMA_POOL_TRACKER,
    UVM_LOCK_ORDER_SECURE_SEMAPHORE,

    // TODO: Bug 4184836: [uvm][hcc] Remove UVM_LOCK_ORDER_CSL_CTX
    // This lock order can be removed after RM no longer relies on RPC event
    // notifications.
    UVM_LOCK_ORDER_CSL_CTX,

    UVM_LOCK_ORDER_LEAF,
    UVM_LOCK_ORDER_COUNT,
} uvm_lock_order_t;

const char *uvm_lock_order_to_string(uvm_lock_order_t lock_order);

typedef enum
{
    UVM_LOCK_FLAGS_INVALID          = 0,
    UVM_LOCK_FLAGS_MODE_EXCLUSIVE   = (1 << 0),
    UVM_LOCK_FLAGS_MODE_SHARED      = (1 << 1),
    UVM_LOCK_FLAGS_MODE_ANY         = (UVM_LOCK_FLAGS_MODE_EXCLUSIVE | UVM_LOCK_FLAGS_MODE_SHARED),
    UVM_LOCK_FLAGS_MODE_MASK        = (UVM_LOCK_FLAGS_MODE_EXCLUSIVE | UVM_LOCK_FLAGS_MODE_SHARED),
    UVM_LOCK_FLAGS_OUT_OF_ORDER     = (1 << 2),
    UVM_LOCK_FLAGS_TRYLOCK          = (1 << 3),
    UVM_LOCK_FLAGS_MASK             = (1 << 4) - 1
} uvm_lock_flags_t;

// Record locking a lock of given lock_order in exclusive or shared mode,
// distinguishing between trylock and normal acquisition attempts.
// Returns true if the recorded lock follows all the locking rules and false
// otherwise.
bool __uvm_record_lock(void *lock, uvm_lock_order_t lock_order, uvm_lock_flags_t flags);

// Record unlocking a lock of given lock_order in exclusive or shared mode and
// possibly out of order.
// Returns true if the unlock follows all the locking rules and false otherwise.
bool __uvm_record_unlock(void *lock, uvm_lock_order_t lock_order, uvm_lock_flags_t flags);

bool __uvm_record_downgrade(void *lock, uvm_lock_order_t lock_order);

// Check whether a lock of given lock_order is held in exclusive, shared, or
// either mode by the current thread.
bool __uvm_check_locked(void *lock, uvm_lock_order_t lock_order, uvm_lock_flags_t flags);

// Check that no locks are held with the given lock order
bool __uvm_check_unlocked_order(uvm_lock_order_t lock_order);

// Check that a lock of the given order can be locked, i.e. that no locks are
// held with the given or deeper lock order.  Allow for out-of-order locking
// when checking for a trylock.
bool __uvm_check_lockable_order(uvm_lock_order_t lock_order, uvm_lock_flags_t flags);

// Check that all locks have been released in a thread context lock
bool __uvm_check_all_unlocked(uvm_thread_context_lock_t *context_lock);

// Check that all locks have been released in the current thread context lock
bool __uvm_thread_check_all_unlocked(void);

// Check that the locking infrastructure has been initialized
bool __uvm_locking_initialized(void);

#if UVM_IS_DEBUG()
  // These macros are intended to be expanded on the call site directly and will
  // print the precise location of the violation while the __uvm_record*
  // functions will error print the details.
  #define uvm_record_lock_raw(lock, lock_order, flags) \
      UVM_ASSERT_MSG(__uvm_record_lock((lock), (lock_order), (flags)), "Locking violation\n")
  #define uvm_record_unlock_raw(lock, lock_order, flags) \
      UVM_ASSERT_MSG(__uvm_record_unlock((lock), (lock_order), (flags)), "Locking violation\n")
  #define uvm_record_downgrade_raw(lock, lock_order) \
      UVM_ASSERT_MSG(__uvm_record_downgrade((lock), (lock_order)), "Locking violation\n")

  // Record UVM lock (a lock that has a lock_order member) operation and assert
  // that it's correct
  #define uvm_record_lock(lock, flags) \
      uvm_record_lock_raw((lock), (lock)->lock_order, (flags))
  #define uvm_record_unlock(lock, flags) uvm_record_unlock_raw((lock), (lock)->lock_order, (flags))
  #define uvm_record_unlock_out_of_order(lock, flags) \
            uvm_record_unlock_raw((lock), (lock)->lock_order, (flags) | UVM_LOCK_FLAGS_OUT_OF_ORDER)
  #define uvm_record_downgrade(lock) uvm_record_downgrade_raw((lock), (lock)->lock_order)

  // Check whether a UVM lock (a lock that has a lock_order member) is held in
  // the given mode.
  #define uvm_check_locked(lock, flags) __uvm_check_locked((lock), (lock)->lock_order, (flags))

  // Helpers for recording and asserting mmap_lock
  // (mmap_sem in kernels < 5.8 ) state
  #define uvm_record_lock_mmap_lock_read(mm) \
          uvm_record_lock_raw(nv_mmap_get_lock(mm), UVM_LOCK_ORDER_MMAP_LOCK, UVM_LOCK_FLAGS_MODE_SHARED)

  #define uvm_record_unlock_mmap_lock_read(mm) \
          uvm_record_unlock_raw(nv_mmap_get_lock(mm), UVM_LOCK_ORDER_MMAP_LOCK, UVM_LOCK_FLAGS_MODE_SHARED)

  #define uvm_record_unlock_mmap_lock_read_out_of_order(mm) \
          uvm_record_unlock_raw(nv_mmap_get_lock(mm), UVM_LOCK_ORDER_MMAP_LOCK, \
                                UVM_LOCK_FLAGS_MODE_SHARED | UVM_LOCK_FLAGS_OUT_OF_ORDER)

  #define uvm_record_lock_mmap_lock_write(mm) \
          uvm_record_lock_raw(nv_mmap_get_lock(mm), UVM_LOCK_ORDER_MMAP_LOCK, UVM_LOCK_FLAGS_MODE_EXCLUSIVE)

  #define uvm_record_unlock_mmap_lock_write(mm) \
          uvm_record_unlock_raw(nv_mmap_get_lock(mm), UVM_LOCK_ORDER_MMAP_LOCK, UVM_LOCK_FLAGS_MODE_EXCLUSIVE)

  #define uvm_record_unlock_mmap_lock_write_out_of_order(mm) \
          uvm_record_unlock_raw(nv_mmap_get_lock(mm), UVM_LOCK_ORDER_MMAP_LOCK, \
                                UVM_LOCK_FLAGS_MODE_EXCLUSIVE | UVM_LOCK_FLAGS_OUT_OF_ORDER)

  #define uvm_check_locked_mmap_lock(mm, flags) \
           __uvm_check_locked(nv_mmap_get_lock(mm), UVM_LOCK_ORDER_MMAP_LOCK, (flags))

  // Helpers for recording RM API lock usage around UVM-RM interfaces
  #define uvm_record_lock_rm_api() \
          uvm_record_lock_raw((void*)UVM_LOCK_ORDER_RM_API, UVM_LOCK_ORDER_RM_API, \
                              UVM_LOCK_FLAGS_MODE_EXCLUSIVE)
  #define uvm_record_unlock_rm_api() \
          uvm_record_unlock_raw((void*)UVM_LOCK_ORDER_RM_API, UVM_LOCK_ORDER_RM_API, \
                                UVM_LOCK_FLAGS_MODE_EXCLUSIVE)

  // Helpers for recording RM GPUS lock usage around UVM-RM interfaces
  #define uvm_record_lock_rm_gpus() \
          uvm_record_lock_raw((void*)UVM_LOCK_ORDER_RM_GPUS, UVM_LOCK_ORDER_RM_GPUS, \
                              UVM_LOCK_FLAGS_MODE_EXCLUSIVE)
  #define uvm_record_unlock_rm_gpus() \
          uvm_record_unlock_raw((void*)UVM_LOCK_ORDER_RM_GPUS, UVM_LOCK_ORDER_RM_GPUS, \
                                UVM_LOCK_FLAGS_MODE_EXCLUSIVE)

  // Helpers for recording both RM locks usage around UVM-RM interfaces
  #define uvm_record_lock_rm_all() ({ uvm_record_lock_rm_api(); uvm_record_lock_rm_gpus(); })
  #define uvm_record_unlock_rm_all() ({ uvm_record_unlock_rm_gpus(); uvm_record_unlock_rm_api(); })

#else
  #define uvm_record_lock                               UVM_IGNORE_EXPR2
  #define uvm_record_unlock                             UVM_IGNORE_EXPR2
  #define uvm_record_unlock_out_of_order                UVM_IGNORE_EXPR2
  #define uvm_record_downgrade                          UVM_IGNORE_EXPR

  static bool uvm_check_locked(void *lock, uvm_lock_flags_t flags)
  {
      return false;
  }

  #define uvm_record_lock_mmap_lock_read                 UVM_IGNORE_EXPR
  #define uvm_record_unlock_mmap_lock_read               UVM_IGNORE_EXPR
  #define uvm_record_unlock_mmap_lock_read_out_of_order  UVM_IGNORE_EXPR
  #define uvm_record_lock_mmap_lock_write                UVM_IGNORE_EXPR
  #define uvm_record_unlock_mmap_lock_write              UVM_IGNORE_EXPR
  #define uvm_record_unlock_mmap_lock_write_out_of_order UVM_IGNORE_EXPR

  #define uvm_check_locked_mmap_lock                     uvm_check_locked

  #define uvm_record_lock_rm_api()
  #define uvm_record_unlock_rm_api()

  #define uvm_record_lock_rm_gpus()
  #define uvm_record_unlock_rm_gpus()

  #define uvm_record_lock_rm_all()
  #define uvm_record_unlock_rm_all()
#endif

#define uvm_locking_assert_initialized() UVM_ASSERT(__uvm_locking_initialized())
#define uvm_thread_assert_all_unlocked() UVM_ASSERT(__uvm_thread_check_all_unlocked())
#define uvm_assert_lockable_order(order) UVM_ASSERT(__uvm_check_lockable_order(order, UVM_LOCK_FLAGS_MODE_ANY))
#define uvm_assert_unlocked_order(order) UVM_ASSERT(__uvm_check_unlocked_order(order))

#if UVM_IS_DEBUG()
#define uvm_lock_debug_init(lock, order) ({        \
        uvm_locking_assert_initialized();          \
        (lock)->lock_order = (order);              \
    })
#else
#define uvm_lock_debug_init(lock, order) ((void) order)
#endif

// Helpers for locking mmap_lock (mmap_sem in kernels < 5.8)
// and recording its usage
#define uvm_assert_mmap_lock_locked_mode(mm, flags) ({                                      \
      typeof(mm) _mm = (mm);                                                                \
      UVM_ASSERT(nv_mm_rwsem_is_locked(_mm) && uvm_check_locked_mmap_lock((_mm), (flags))); \
  })

#define uvm_assert_mmap_lock_locked(mm) \
        uvm_assert_mmap_lock_locked_mode((mm), UVM_LOCK_FLAGS_MODE_ANY)
#define uvm_assert_mmap_lock_locked_read(mm) \
        uvm_assert_mmap_lock_locked_mode((mm), UVM_LOCK_FLAGS_MODE_SHARED)
#define uvm_assert_mmap_lock_locked_write(mm) \
        uvm_assert_mmap_lock_locked_mode((mm), UVM_LOCK_FLAGS_MODE_EXCLUSIVE)

#define uvm_down_read_mmap_lock(mm) ({                  \
        typeof(mm) _mm = (mm);                          \
        uvm_record_lock_mmap_lock_read(_mm);            \
        nv_mmap_read_lock(_mm);                         \
    })

// Non-blocking uvm_down_read_mmap_lock. Evaluates to true with the lock held,
// or false with nothing taken and nothing recorded.
//
// A kernel thread that can be the target of kthread_stop cannot afford to block
// here. Teardown holds this same mmap_lock for read, so the two do not contend
// directly, but a writer queued between them puts the new reader behind it and
// the thread never reaches its exit. Callers that can retry should use this and
// come back on the next sweep.
#define uvm_down_read_mmap_lock_trylock(mm) ({          \
        typeof(mm) _mm = (mm);                          \
        bool _locked = nv_mmap_read_trylock(_mm) != 0;  \
        if (_locked)                                    \
            uvm_record_lock_mmap_lock_read(_mm);        \
        _locked;                                        \
    })

#define uvm_up_read_mmap_lock(mm) ({                    \
        typeof(mm) _mm = (mm);                          \
        nv_mmap_read_unlock(_mm);                       \
        uvm_record_unlock_mmap_lock_read(_mm);          \
    })

#define uvm_up_read_mmap_lock_out_of_order(mm) ({           \
        typeof(mm) _mm = (mm);                              \
        nv_mmap_read_unlock(_mm);                           \
        uvm_record_unlock_mmap_lock_read_out_of_order(_mm); \
    })

#define uvm_down_write_mmap_lock(mm) ({                 \
        typeof(mm) _mm = (mm);                          \
        uvm_record_lock_mmap_lock_write(_mm);           \
        nv_mmap_write_lock(_mm);                        \
    })

#define uvm_up_write_mmap_lock(mm) ({                   \
        typeof(mm) _mm = (mm);                          \
        nv_mmap_write_unlock(_mm);                      \
        uvm_record_unlock_mmap_lock_write(_mm);         \
    })

// Helper for calling a UVM-RM interface function with lock recording
#define uvm_rm_locked_call(call) ({                     \
        typeof(call) ret;                               \
        uvm_record_lock_rm_all();                       \
        ret = call;                                     \
        uvm_record_unlock_rm_all();                     \
        ret;                                            \
    })

// Helper for calling a UVM-RM interface function that returns void with lock
// recording
#define uvm_rm_locked_call_void(call) ({                \
        uvm_record_lock_rm_all();                       \
        call;                                           \
        uvm_record_unlock_rm_all();                     \
    })

typedef struct
{
    struct rw_semaphore sem;
#if UVM_IS_DEBUG()
    uvm_lock_order_t lock_order;
#endif
} uvm_rw_semaphore_t;

//
// Note that this is a macro, not an inline or static function so the
// "uvm_sem" argument is subsituted as text. If this is invoked with
// uvm_assert_rwsem_locked_mode(_sem, flags) then we get code "_sem = _sem"
// and _sem is initialized to NULL. Avoid this by using a name unlikely to
// be the same as the string passed to "uvm_sem".
// See uvm_down_read() and uvm_up_read() below as examples.
//
#define uvm_assert_rwsem_locked_mode(uvm_sem, flags) ({                               \
        typeof(uvm_sem) _sem_ = (uvm_sem);                                            \
        UVM_ASSERT(rwsem_is_locked(&_sem_->sem) && uvm_check_locked(_sem_, (flags))); \
    })

#define uvm_assert_rwsem_locked(uvm_sem) \
        uvm_assert_rwsem_locked_mode(uvm_sem, UVM_LOCK_FLAGS_MODE_ANY)
#define uvm_assert_rwsem_locked_read(uvm_sem) \
        uvm_assert_rwsem_locked_mode(uvm_sem, UVM_LOCK_FLAGS_MODE_SHARED)
#define uvm_assert_rwsem_locked_write(uvm_sem) \
        uvm_assert_rwsem_locked_mode(uvm_sem, UVM_LOCK_FLAGS_MODE_EXCLUSIVE)

#define uvm_assert_rwsem_unlocked(uvm_sem) UVM_ASSERT(!rwsem_is_locked(&(uvm_sem)->sem))

#define uvm_init_rwsem(uvm_sem, order) ({                   \
        uvm_rw_semaphore_t *uvm_sem_ ## order = (uvm_sem);  \
        init_rwsem(&uvm_sem_ ## order->sem);                \
        uvm_lock_debug_init(uvm_sem, order);                \
        uvm_assert_rwsem_unlocked(uvm_sem);                 \
    })

#define uvm_down_read(uvm_sem) ({                          \
        typeof(uvm_sem) _sem = (uvm_sem);                  \
        uvm_record_lock(_sem, UVM_LOCK_FLAGS_MODE_SHARED); \
        down_read(&_sem->sem);                             \
        uvm_assert_rwsem_locked_read(_sem);                \
    })

#define uvm_up_read(uvm_sem) ({                              \
        typeof(uvm_sem) _sem = (uvm_sem);                    \
        uvm_assert_rwsem_locked_read(_sem);                  \
        up_read(&_sem->sem);                                 \
        uvm_record_unlock(_sem, UVM_LOCK_FLAGS_MODE_SHARED); \
    })

// Unlock w/o any tracking. This should be extremely rare and *_no_tracking
// helpers will be added only as needed.
//
// TODO: Bug 2594854:
// TODO: Bug 2583279: Remove macro when bugs are fixed
#define uvm_up_read_no_tracking(uvm_sem) ({                  \
        typeof(uvm_sem) _sem = (uvm_sem);                    \
        up_read(&_sem->sem);                                 \
    })

#define uvm_down_write(uvm_sem) ({                            \
        typeof (uvm_sem) _sem = (uvm_sem);                    \
        uvm_record_lock(_sem, UVM_LOCK_FLAGS_MODE_EXCLUSIVE); \
        down_write(&_sem->sem);                               \
        uvm_assert_rwsem_locked_write(_sem);                  \
    })

// trylock for reading: returns 1 if successful, 0 if not.  Out-of-order lock
// acquisition via this function is legal, i.e. the lock order checker will
// allow it.  However, if an out-of-order lock acquisition attempt fails, it is
// the caller's responsibility to back off at least to the point where the
// next held lower-order lock is released.
#define uvm_down_read_trylock(uvm_sem) ({                                           \
        typeof(uvm_sem) _sem = (uvm_sem);                                           \
        int locked;                                                                 \
        uvm_record_lock(_sem, UVM_LOCK_FLAGS_MODE_SHARED | UVM_LOCK_FLAGS_TRYLOCK); \
        locked = down_read_trylock(&_sem->sem);                                     \
        if (locked == 0)                                                            \
            uvm_record_unlock(_sem, UVM_LOCK_FLAGS_MODE_SHARED);                    \
        else                                                                        \
            uvm_assert_rwsem_locked_read(_sem);                                     \
        locked;                                                                     \
    })

// Lock w/o any tracking. This should be extremely rare and *_no_tracking
// helpers will be added only as needed.
//
// TODO: Bug 2594854:
// TODO: Bug 2583279: Remove macro when bugs are fixed
#define uvm_down_read_trylock_no_tracking(uvm_sem) ({                               \
        typeof(uvm_sem) _sem = (uvm_sem);                                           \
        down_read_trylock(&_sem->sem);                                              \
    })

// trylock for writing: returns 1 if successful, 0 if not.  Out-of-order lock
// acquisition via this function is legal, i.e. the lock order checker will
// allow it.  However, if an out-of-order lock acquisition attempt fails, it is
// the caller's responsibility to back off at least to the point where the
// next held lower-order lock is released.
#define uvm_down_write_trylock(uvm_sem) ({                                             \
        typeof(uvm_sem) _sem = (uvm_sem);                                              \
        int locked;                                                                    \
        uvm_record_lock(_sem, UVM_LOCK_FLAGS_MODE_EXCLUSIVE | UVM_LOCK_FLAGS_TRYLOCK); \
        locked = down_write_trylock(&_sem->sem);                                       \
        if (locked == 0)                                                               \
            uvm_record_unlock(_sem, UVM_LOCK_FLAGS_MODE_EXCLUSIVE);                    \
        else                                                                           \
            uvm_assert_rwsem_locked_write(_sem);                                       \
        locked;                                                                        \
    })

#define uvm_up_write(uvm_sem) ({                                \
        typeof(uvm_sem) _sem = (uvm_sem);                       \
        uvm_assert_rwsem_locked_write(_sem);                    \
        up_write(&_sem->sem);                                   \
        uvm_record_unlock(_sem, UVM_LOCK_FLAGS_MODE_EXCLUSIVE); \
    })

#define uvm_downgrade_write(uvm_sem) ({                 \
        typeof(uvm_sem) _sem = (uvm_sem);               \
        uvm_assert_rwsem_locked_write(_sem);            \
        downgrade_write(&_sem->sem);                    \
        uvm_record_downgrade(_sem);                     \
    })

typedef struct
{
    struct mutex m;
#if UVM_IS_DEBUG()
    uvm_lock_order_t lock_order;
#endif
} uvm_mutex_t;

// Note that this is a macro, not an inline or static function so the
// "uvm_macro" argument is subsituted as text. If this is invoked with
// uvm__mutex_is_locked(_mutex) then we get code "_mutex = _mutex" and _mutex is
// initialized to NULL. Avoid this by using a name unlikely to be the same as
// the string passed to "uvm_mutex".
// See uvm_mutex_lock() and uvm_mutex_unlock() below as examples.
//
#define uvm_mutex_is_locked(uvm_mutex) ({                                                           \
        typeof(uvm_mutex) _mutex_ = (uvm_mutex);                                                    \
        (mutex_is_locked(&_mutex_->m) && uvm_check_locked(_mutex_, UVM_LOCK_FLAGS_MODE_EXCLUSIVE)); \
    })

#define uvm_assert_mutex_locked(uvm_mutex) UVM_ASSERT(uvm_mutex_is_locked(uvm_mutex))
#define uvm_assert_mutex_unlocked(uvm_mutex) UVM_ASSERT(!mutex_is_locked(&(uvm_mutex)->m))

//
// Linux kernel mutexes cannot be used with interrupts disabled. Doing so
// can lead to deadlocks.
// To warn about mutex usages with interrupts disabled, the following
// macros and inline functions wrap around the raw kernel mutex operations
// in order to check if the interrupts have been disabled and assert if so.
//
// TODO: Bug 2690258: evaluate whether !irqs_disabled() && !in_interrupt() is
//       enough.
//
#define uvm_assert_mutex_interrupts() ({                                                                        \
        UVM_ASSERT_MSG(!irqs_disabled() && !in_interrupt(), "Mutexes cannot be used with interrupts disabled"); \
    })

#define uvm_mutex_init(mutex, order) ({                \
        uvm_mutex_t *mutex_ ## order = (mutex);        \
        mutex_init(&mutex_ ## order->m);               \
        uvm_lock_debug_init(mutex, order);             \
        uvm_assert_mutex_unlocked(mutex);              \
    })

#define uvm_mutex_lock(mutex) ({                                \
        typeof(mutex) _mutex = (mutex);                         \
        uvm_assert_mutex_interrupts();                          \
        uvm_record_lock(_mutex, UVM_LOCK_FLAGS_MODE_EXCLUSIVE); \
        mutex_lock(&_mutex->m);                                 \
        uvm_assert_mutex_locked(_mutex);                        \
    })

// Lock while already holding a lock of the same order taken with
// uvm_mutex_lock() variant. Note this shouldn't be used if the held lock was
// taken with uvm_mutex_lock_nested() because we only support a single level of
// nesting. This should be extremely rare and *_nested helpers will only be
// added as needed.
#define uvm_mutex_lock_nested(mutex) ({         \
        uvm_assert_mutex_interrupts();          \
        mutex_lock_nested(&(mutex)->m, 1);      \
    })

#define uvm_mutex_trylock(mutex) ({                                                      \
        typeof(mutex) _mutex = (mutex);                                                  \
        int locked;                                                                      \
        uvm_record_lock(_mutex, UVM_LOCK_FLAGS_MODE_EXCLUSIVE | UVM_LOCK_FLAGS_TRYLOCK); \
        locked = mutex_trylock(&_mutex->m);                                              \
        if (locked == 0)                                                                 \
            uvm_record_unlock(_mutex, UVM_LOCK_FLAGS_MODE_EXCLUSIVE);                    \
        else                                                                             \
            uvm_assert_mutex_locked(_mutex);                                             \
        locked;                                                                          \
    })

#define uvm_mutex_unlock(mutex) ({                                \
        typeof(mutex) _mutex = (mutex);                           \
        uvm_assert_mutex_interrupts();                            \
        uvm_assert_mutex_locked(_mutex);                          \
        mutex_unlock(&_mutex->m);                                 \
        uvm_record_unlock(_mutex, UVM_LOCK_FLAGS_MODE_EXCLUSIVE); \
    })
#define uvm_mutex_unlock_out_of_order(mutex) ({                                \
        typeof(mutex) _mutex = (mutex);                                        \
        uvm_assert_mutex_interrupts();                                         \
        uvm_assert_mutex_locked(_mutex);                                       \
        mutex_unlock(&_mutex->m);                                              \
        uvm_record_unlock_out_of_order(_mutex, UVM_LOCK_FLAGS_MODE_EXCLUSIVE); \
    })

// Unlock w/o any tracking.
#define uvm_mutex_unlock_nested(mutex) ({       \
        uvm_assert_mutex_interrupts();          \
        mutex_unlock(&(mutex)->m);              \
    })

typedef struct
{
    struct semaphore sem;
#if UVM_IS_DEBUG()
    uvm_lock_order_t lock_order;
#endif
} uvm_semaphore_t;

#define uvm_sema_init(semaphore, val, order) ({         \
        uvm_semaphore_t *sem_ ## order = (semaphore);   \
        sema_init(&sem_ ## order->sem, (val));          \
        uvm_lock_debug_init(semaphore, order);          \
    })

#define uvm_sem_is_locked(uvm_sem) uvm_check_locked(uvm_sem, UVM_LOCK_FLAGS_MODE_SHARED)

#define uvm_down(uvm_sem) ({                               \
        typeof(uvm_sem) _sem = (uvm_sem);                  \
        uvm_record_lock(_sem, UVM_LOCK_FLAGS_MODE_SHARED); \
        down(&_sem->sem);                                  \
    })

#define uvm_up(uvm_sem) ({                                   \
        typeof(uvm_sem) _sem = (uvm_sem);                    \
        UVM_ASSERT(uvm_sem_is_locked(_sem));                 \
        up(&_sem->sem);                                      \
        uvm_record_unlock(_sem, UVM_LOCK_FLAGS_MODE_SHARED); \
    })
#define uvm_up_out_of_order(uvm_sem) ({                                   \
        typeof(uvm_sem) _sem = (uvm_sem);                                 \
        UVM_ASSERT(uvm_sem_is_locked(_sem));                              \
        up(&_sem->sem);                                                   \
        uvm_record_unlock_out_of_order(_sem, UVM_LOCK_FLAGS_MODE_SHARED); \
    })


// A regular spinlock
// Locked/unlocked with uvm_spin_lock()/uvm_spin_unlock()
typedef struct
{
    spinlock_t lock;
#if UVM_IS_DEBUG()
    uvm_lock_order_t lock_order;
#endif
} uvm_spinlock_t;

// A separate spinlock type for spinlocks that need to disable interrupts. For
// guaranteed correctness and convenience embed the saved and restored irq state
// in the lock itself.
// Locked/unlocked with uvm_spin_lock_irqsave()/uvm_spin_unlock_irqrestore()
typedef struct
{
    spinlock_t lock;
    unsigned long irq_flags;
#if UVM_IS_DEBUG()
    uvm_lock_order_t lock_order;
#endif
} uvm_spinlock_irqsave_t;

// Asserts that the spinlock is held. Notably the macros below support both
// types of spinlocks.
// Note that this is a macro, not an inline or static function so the
// "spinlock" argument is subsituted as text. If this is invoked with
// uvm_assert_spinlock_locked(_lock) then we get code "_lock = _lock"
// and _lock is initialized to NULL. Avoid this by using a name unlikely to
// be the same as the string passed to "spinlock".
// See uvm_spin_lock() and uvm_spin_unlock() below as examples.
//
#define uvm_assert_spinlock_locked(spinlock) ({                              \
        typeof(spinlock) _lock_ = (spinlock);                                \
        UVM_ASSERT(spin_is_locked(&_lock_->lock));                           \
        UVM_ASSERT(uvm_check_locked(_lock_, UVM_LOCK_FLAGS_MODE_EXCLUSIVE)); \
    })

#define uvm_assert_spinlock_unlocked(spinlock) UVM_ASSERT(!spin_is_locked(&(spinlock)->lock))

#define uvm_spin_lock_init(spinlock, order) ({                  \
            uvm_spinlock_t *spinlock_ ## order = (spinlock);    \
            spin_lock_init(&spinlock_ ## order->lock);          \
            uvm_lock_debug_init(spinlock, order);               \
            uvm_assert_spinlock_unlocked(spinlock);             \
    })

#define uvm_spin_lock(uvm_lock) ({                             \
        typeof(uvm_lock) _lock = (uvm_lock);                   \
        uvm_record_lock(_lock, UVM_LOCK_FLAGS_MODE_EXCLUSIVE); \
        spin_lock(&_lock->lock);                               \
        uvm_assert_spinlock_locked(_lock);                     \
    })

#define uvm_spin_unlock(uvm_lock) ({                             \
        typeof(uvm_lock) _lock = (uvm_lock);                     \
        uvm_assert_spinlock_locked(_lock);                       \
        spin_unlock(&_lock->lock);                               \
        uvm_record_unlock(_lock, UVM_LOCK_FLAGS_MODE_EXCLUSIVE); \
    })

#define uvm_spin_lock_irqsave_init(spinlock, order) ({                  \
            uvm_spinlock_irqsave_t *spinlock_ ## order = (spinlock);    \
            spin_lock_init(&spinlock_ ## order->lock);                  \
            uvm_lock_debug_init(spinlock, order);                       \
            uvm_assert_spinlock_unlocked(spinlock);                     \
    })

// Use a temp to not rely on flags being written after acquiring the lock.
#define uvm_spin_lock_irqsave(uvm_lock) ({                     \
        typeof(uvm_lock) _lock = (uvm_lock);                   \
        unsigned long irq_flags;                               \
        uvm_record_lock(_lock, UVM_LOCK_FLAGS_MODE_EXCLUSIVE); \
        spin_lock_irqsave(&_lock->lock, irq_flags);            \
        _lock->irq_flags = irq_flags;                          \
        uvm_assert_spinlock_locked(_lock);                     \
    })

// Use a temp to not rely on flags being read before releasing the lock.
#define uvm_spin_unlock_irqrestore(uvm_lock) ({                  \
        typeof(uvm_lock) _lock = (uvm_lock);                     \
        unsigned long irq_flags = _lock->irq_flags;              \
        uvm_assert_spinlock_locked(_lock);                       \
        spin_unlock_irqrestore(&_lock->lock, irq_flags);         \
        uvm_record_unlock(_lock, UVM_LOCK_FLAGS_MODE_EXCLUSIVE); \
    })

// Wrapper for a reader-writer spinlock that disables and enables interrupts
typedef struct
{
    rwlock_t lock;

    // This flags variable is only used by writers, since concurrent readers may
    // have different values.
    unsigned long irq_flags;

#if UVM_IS_DEBUG()
    uvm_lock_order_t lock_order;

    // The kernel doesn't provide a function to tell if an rwlock_t is locked,
    // so we create our own.
    atomic_t lock_count;
#endif
} uvm_rwlock_irqsave_t;

static bool uvm_rwlock_irqsave_is_locked(uvm_rwlock_irqsave_t *rwlock)
{
#if UVM_IS_DEBUG()
    return atomic_read(&rwlock->lock_count) > 0;
#else
    return false;
#endif
}

static void uvm_rwlock_irqsave_inc(uvm_rwlock_irqsave_t *rwlock)
{
#if UVM_IS_DEBUG()
    atomic_inc(&rwlock->lock_count);
#endif
}

static void uvm_rwlock_irqsave_dec(uvm_rwlock_irqsave_t *rwlock)
{
#if UVM_IS_DEBUG()
    atomic_dec(&rwlock->lock_count);
#endif
}

#define uvm_assert_rwlock_locked(uvm_rwlock) \
    UVM_ASSERT(uvm_rwlock_irqsave_is_locked(uvm_rwlock) && uvm_check_locked(uvm_rwlock, UVM_LOCK_FLAGS_MODE_ANY))
#define uvm_assert_rwlock_locked_read(uvm_rwlock) \
    UVM_ASSERT(uvm_rwlock_irqsave_is_locked(uvm_rwlock) && uvm_check_locked(uvm_rwlock, UVM_LOCK_FLAGS_MODE_SHARED))
#define uvm_assert_rwlock_locked_write(uvm_rwlock) \
    UVM_ASSERT(uvm_rwlock_irqsave_is_locked(uvm_rwlock) && uvm_check_locked(uvm_rwlock, UVM_LOCK_FLAGS_MODE_EXCLUSIVE))

#if UVM_IS_DEBUG()
    #define uvm_assert_rwlock_unlocked(uvm_rwlock) UVM_ASSERT(!uvm_rwlock_irqsave_is_locked(uvm_rwlock))
#else
    #define uvm_assert_rwlock_unlocked(uvm_rwlock)
#endif

#define uvm_rwlock_irqsave_init(rwlock, order) ({               \
            uvm_rwlock_irqsave_t *rwlock_ ## order = rwlock;    \
            rwlock_init(&rwlock_ ## order->lock);               \
            uvm_lock_debug_init(rwlock, order);                 \
            uvm_assert_rwlock_unlocked(rwlock);                 \
        })

// We can't store the irq_flags within the lock itself for readers, so they must
// pass in their flags.
#define uvm_read_lock_irqsave(uvm_rwlock, irq_flags) ({     \
        typeof(uvm_rwlock) _lock = (uvm_rwlock);            \
        uvm_record_lock(_lock, UVM_LOCK_FLAGS_MODE_SHARED); \
        read_lock_irqsave(&_lock->lock, irq_flags);         \
        uvm_rwlock_irqsave_inc(uvm_rwlock);                 \
        uvm_assert_rwlock_locked_read(_lock);               \
    })

#define uvm_read_unlock_irqrestore(uvm_rwlock, irq_flags) ({    \
        typeof(uvm_rwlock) _lock = (uvm_rwlock);                \
        uvm_assert_rwlock_locked_read(_lock);                   \
        uvm_rwlock_irqsave_dec(uvm_rwlock);                     \
        read_unlock_irqrestore(&_lock->lock, irq_flags);        \
        uvm_record_unlock(_lock, UVM_LOCK_FLAGS_MODE_SHARED);   \
    })

// Use a temp to not rely on flags being written after acquiring the lock.
#define uvm_write_lock_irqsave(uvm_rwlock) ({                   \
        typeof(uvm_rwlock) _lock = (uvm_rwlock);                \
        unsigned long irq_flags;                                \
        uvm_record_lock(_lock, UVM_LOCK_FLAGS_MODE_EXCLUSIVE);  \
        write_lock_irqsave(&_lock->lock, irq_flags);            \
        uvm_rwlock_irqsave_inc(uvm_rwlock);                     \
        _lock->irq_flags = irq_flags;                           \
        uvm_assert_rwlock_locked_write(_lock);                  \
    })

// Use a temp to not rely on flags being written after acquiring the lock.
#define uvm_write_unlock_irqrestore(uvm_rwlock) ({                  \
        typeof(uvm_rwlock) _lock = (uvm_rwlock);                    \
        unsigned long irq_flags = _lock->irq_flags;                 \
        uvm_assert_rwlock_locked_write(_lock);                      \
        uvm_rwlock_irqsave_dec(uvm_rwlock);                         \
        write_unlock_irqrestore(&_lock->lock, irq_flags);           \
        uvm_record_unlock(_lock, UVM_LOCK_FLAGS_MODE_EXCLUSIVE);    \
    })

// Bit locks are 'compressed' mutexes which take only 1 bit per lock by virtue
// of using shared waitqueues.
typedef struct
{
    unsigned long *bits;

#if UVM_IS_DEBUG()
    uvm_lock_order_t lock_order;
#endif
} uvm_bit_locks_t;

NV_STATUS uvm_bit_locks_init(uvm_bit_locks_t *bit_locks, size_t count, uvm_lock_order_t lock_order);
void uvm_bit_locks_deinit(uvm_bit_locks_t *bit_locks);

// Asserts that the bit lock is held.
//
// TODO: Bug 1766601:
//  - assert for the right ownership (defining the owner might be tricky in
//    the kernel).
#define uvm_assert_bit_locked(bit_locks, bit) ({                             \
    typeof(bit_locks) _bit_locks = (bit_locks);                              \
    typeof(bit) _bit = (bit);                                                \
    UVM_ASSERT(test_bit(_bit, _bit_locks->bits));                            \
    UVM_ASSERT(uvm_check_locked(_bit_locks, UVM_LOCK_FLAGS_MODE_EXCLUSIVE)); \
})

#define uvm_assert_bit_unlocked(bit_locks, bit) ({                      \
    typeof(bit_locks) _bit_locks = (bit_locks);                         \
    typeof(bit) _bit = (bit);                                           \
    UVM_ASSERT(!test_bit(_bit, _bit_locks->bits));                      \
})

static void __uvm_bit_lock(uvm_bit_locks_t *bit_locks, unsigned long bit)
{
    int res;

    res = wait_on_bit_lock(bit_locks->bits, bit, TASK_UNINTERRUPTIBLE);
    UVM_ASSERT_MSG(res == 0, "Uninterruptible task interrupted: %d\n", res);
    uvm_assert_bit_locked(bit_locks, bit);
}
#define uvm_bit_lock(bit_locks, bit) ({                         \
    typeof(bit_locks) _bit_locks = (bit_locks);                 \
    typeof(bit) _bit = (bit);                                   \
    uvm_record_lock(_bit_locks, UVM_LOCK_FLAGS_MODE_EXCLUSIVE); \
    __uvm_bit_lock(_bit_locks, _bit);                           \
})

static void __uvm_bit_unlock(uvm_bit_locks_t *bit_locks, unsigned long bit)
{
    uvm_assert_bit_locked(bit_locks, bit);

    clear_bit_unlock(bit, bit_locks->bits);
    // Make sure we don't reorder release with wakeup as it would cause
    // deadlocks (other thread checking lock and adding itself to queue
    // in reversed order). clear_bit_unlock has only release semantics.
    smp_mb__after_atomic();
    wake_up_bit(bit_locks->bits, bit);
}
#define uvm_bit_unlock(bit_locks, bit) ({                         \
    typeof(bit_locks) _bit_locks = (bit_locks);                   \
    typeof(bit) _bit = (bit);                                     \
    __uvm_bit_unlock(_bit_locks, _bit);                           \
    uvm_record_unlock(_bit_locks, UVM_LOCK_FLAGS_MODE_EXCLUSIVE); \
})

// ----------------------------------------------------------------------------
// Lock contention instrumentation
// ----------------------------------------------------------------------------
//
// Verbosity of the fault instrumentation:
//   0 = the always-on fault pipeline timers only (stock behaviour)
//   1 = reserved for the log2 histograms
//   2 = the lock contention probes below
//
// Levels above 0 cost two timestamp reads and an atomic add per probed lock
// acquisition, so the default keeps every baseline comparable with the
// campaigns measured before the probes existed.
extern unsigned uvm_perf_fault_stats_level;

// Wait time on the driver-internal locks that GPU fault servicing funnels
// through. These answer where servicing serializes once several worker threads
// are active: every GPU PTE write goes through one page tree mutex per GPU,
// every allocation and free through one PMM mutex, and each va_block through
// its own lock. Global rather than per-GPU for the same reason the host-op
// counters are, several of these paths run without a GPU at hand. Cumulative
// since module load; consumers snapshot-and-diff. Exposed at cpu/lock_stats.
//
// The ns_* fields measure *wait*, the time between asking for a lock and
// holding it, except for ns_evict_call and ns_pma_evict_cb which measure how
// long the eviction paths run. Those two nest: ns_pma_evict_cb contains
// ns_evict_call which contains ns_pmm_lock_wait, so the three must never be
// summed or presented as disjoint shares of anything.
//
// ns_pma_evict_cb is the notable asymmetry: RM takes its own API lock and then
// calls down into UVM, so UVM never waits on that lock and cannot observe
// anyone who does. That counter therefore bounds how much serialization the RM
// API lock could be causing, and cannot attribute any particular stall to it.
typedef struct
{
    // uvm_mutex_lock(&tree->lock) in uvm_mmu.c, via page_tree_lock()
    atomic64_t ns_page_tree_lock_wait;
    atomic64_t n_page_tree_lock_acqs;

    // uvm_mutex_lock(&pmm->lock) in uvm_pmm_gpu.c, via pmm_lock()
    atomic64_t ns_pmm_lock_wait;
    atomic64_t n_pmm_lock_acqs;

    // pick_and_evict_root_chunk_retry: allocation-side eviction on the fault
    // path. One sample per logical eviction, retries included.
    atomic64_t ns_evict_call;
    atomic64_t n_evict_calls;

    // Sub-phases of ns_evict_call. The 07-22 campaign measured ns_evict_call
    // growing 5x across the worker ladder at fixed work, with the two locks
    // the path touches accounting for only 3% of it, so the whole-call sample
    // could not say where the time went. These three partition it:
    //
    //   ns_evict_pick        pick_root_chunk_to_evict, the free-list walk done
    //                        with pmm->lock held
    //   ns_evict_block_lock  the va_block->lock acquisition at the head of
    //                        evict_root_chunk_from_va_block. NOT the same site
    //                        as ns_block_lock_wait_gpu, which is the servicing
    //                        path's acquisition in service_fault_batch_block;
    //                        this one was unprobed, and with several workers
    //                        both servicing and evicting it is the prime
    //                        suspect for the unexplained growth
    //   ns_evict_chunks      uvm_va_block_evict_chunks, the migration setup
    //
    // The pmm re-acquisition on the way out needs no counter here: pmm_lock()
    // already records it into ns_pmm_lock_wait.
    //
    // These three plus ns_pmm_lock_wait's eviction share should account for
    // ns_evict_call. They are contained BY it, so never add them to it.
    //
    // Note the counts are not 1:1 with n_evict_calls. evict_root_chunk walks
    // the root chunk repeatedly, calling evict_root_chunk_from_va_block once
    // per va_block it finds, so n_evict_block_lock_acqs counts va_blocks
    // touched and can exceed n_evict_calls by a lot. Divide the block-lock
    // time by its OWN count for a per-acquisition figure, and by n_evict_calls
    // only when asking what one logical eviction costs.
    //
    // That ratio is itself a result, not just a caveat. It separates three
    // explanations for the 5x growth in ns_evict_call that the whole-call
    // probe could not tell apart:
    //
    //   blocks/evict rises, us/acq flat   each eviction is doing more work
    //                                     because chunks span more va_blocks.
    //                                     No lock is to blame and no lock fix
    //                                     would help
    //   blocks/evict flat, us/acq rises   genuine contention on va_block->lock
    //                                     between servicing and eviction
    //   both flat, ns_evict_chunks rises  the migration setup itself got
    //                                     slower, which points at the pushbuf
    //                                     or the tracker rather than at locks
    //
    // The capture script derives blocks_per_evict for exactly this reason.
    atomic64_t ns_evict_pick;
    atomic64_t ns_evict_block_lock;
    atomic64_t n_evict_block_lock_acqs;
    atomic64_t ns_evict_chunks;

    // Sub-phases of ns_evict_chunks. The 07-24 campaign settled the three-way
    // question posed above: blocks_per_evict was flat at 1.00 and the block
    // lock stayed under 0.11 us at every width, while ns_evict_chunks rose
    // 2.52 -> 13.22 us across the worker ladder at fixed attempt count. That is
    // the third branch, "the migration setup itself got slower", and nothing
    // below ns_evict_chunks was measured. These three partition it:
    //
    //   ns_evict_ctx_alloc   uvm_service_block_context_alloc. Note this runs
    //                        BEFORE the walk that counts chunks_to_evict, so
    //                        every call pays it whether or not there turns out
    //                        to be anything to evict
    //   ns_evict_scan        the chunk walk that decides what to evict
    //   ns_evict_resident    the migration proper, both the HMM and the
    //                        make_resident arm. One probe spans the if/else so
    //                        the three still partition the call on either path
    //
    // Contained BY ns_evict_chunks. Never add them to it.
    //
    // They carry two counts of their own rather than sharing
    // n_evict_block_lock_acqs, because the three phases do NOT all produce a
    // sample on every call and dividing them by one number would compare three
    // quantities against three different denominators:
    //
    //   - three early returns fire before the first probe (!va_space,
    //     !gpu_state, and the inject_eviction_error test path). The block lock
    //     was taken and counted, but no phase ran
    //   - chunks_to_evict == 0 exits after the walk, giving a ctx_alloc and a
    //     scan sample but no resident one
    //
    // So n_evict_ctx_alloc divides ctx_alloc and scan, n_evict_resident divides
    // resident, and n_evict_ctx_alloc/n_evict_block_lock_acqs is itself the
    // readout for how often the early returns fire. Without these the per-call
    // figures are underestimates by an unknown and differing factor, and
    // evict_chunks_accounted_pct would report the shortfall as unprobed work
    // rather than as denominator skew.
    atomic64_t ns_evict_ctx_alloc;
    atomic64_t n_evict_ctx_alloc;
    atomic64_t ns_evict_scan;
    atomic64_t ns_evict_resident;
    atomic64_t n_evict_resident;

    // The push path, and NOT an eviction counter: uvm_pushbuffer_begin_push is
    // on the path of every push in the driver, servicing migrations included.
    // It is probed here because it is the leading suspect for the growth above,
    // but what it answers is larger. The pushbuffer admits at most
    // UVM_PUSHBUFFER_CHUNKS (16) concurrent pushes, and the widest arm runs 15
    // workers plus the dispatcher, so if ns_push_sema grows with worker count
    // then 16 is a driver-wide ceiling on how much parallelism the pool can
    // ever have, whatever the fault path does.
    //
    //   ns_push_reserve push_reserve_channel, which spins in uvm_channel_reserve
    //                   until a GPFIFO entry frees up and takes
    //                   channel_pool_lock on every attempt. This one sits
    //                   BEFORE the other two in the push sequence and was the
    //                   original blind spot: a spin loop on a shared lock is
    //                   exactly the shape being hunted, and without it a queue
    //                   here would read as "both push counters flat", which the
    //                   decision table would wrongly call "not a queue"
    //   ns_push_sema    uvm_down on concurrent_pushes_sema, the 16-slot cap
    //   ns_push_claim   claim_chunk, which takes the pushbuffer spinlock
    //
    // A semaphore with 16 slots should cost nothing below 16 threads, whereas a
    // spinlock held briefly by every pusher grows smoothly from two. The
    // measured ns_evict_chunks curve is smooth from two, so ns_push_claim is
    // the better a-priori fit and ns_push_sema is the more interesting result.
    // Both probes sit AFTER the WLC early return, which bypasses the semaphore
    // entirely; counting those static-pushbuffer pushes would dilute the mean.
    //
    // These overlap ns_evict_resident on the eviction path and nothing else in
    // this struct. Not contained by ns_evict_chunks, so never fold them into
    // evict_chunks_accounted_pct.
    atomic64_t ns_push_reserve;
    atomic64_t n_push_reserve;
    atomic64_t ns_push_sema;
    atomic64_t n_push_acqs;
    atomic64_t ns_push_claim;

    // Which of the two waits inside channel reservation is being paid for.
    //
    // Campaign 20260804_232730 answered the question ns_push_reserve was added
    // for: it is the stage that grows, 0.043 -> 2.925 us across the worker
    // ladder at a push count that moves 0.24%, and the eviction unmap phase
    // picks up 2.77 to 3.13 reservations' worth of waiting. What it cannot say
    // is WHY, because channel_reserve_in_pool has two ways to be slow and one
    // timer around both:
    //
    //   1. every channel is out of GPFIFO entries, so the thread spins until
    //      the device drains one. A hardware ceiling. Widening the pool buys
    //      nothing and the copy engines are the gate.
    //   2. an entry is free, but try_claim_channel takes channel_pool_lock on
    //      every attempt and sixteen threads are queueing on it. A software
    //      problem, and the fix is more channels or a finer lock.
    //
    // The two call for opposite conclusions, so the report cannot recommend
    // anything until they are separated. channel_manager_num_channels gives the
    // copy-engine pool two channels, a constant carrying NVIDIA's own TODO to
    // tune it against real workloads, and whether that TODO is worth acting on
    // is exactly what these decide.
    //
    //   n_push_reserve_slow   reservations where the fast sweep over the pool
    //                         found nothing and the thread entered the spin
    //                         loop. Against n_push_reserve this is the rate.
    //                         Near zero means case 2, near one means case 1.
    //   ns_push_reserve_spins spin-loop iterations, NOT nanoseconds despite the
    //                         ns_ prefix the rest of this struct uses for time.
    //                         Named for placement beside its sibling; the
    //                         derived metric divides it by n_push_reserve_slow
    //                         to give iterations per slow reservation, which
    //                         separates "briefly full" from "badly backed up".
    //                         The unit is one UVM_SPIN_LOOP per CHANNEL
    //                         examined, not per sweep of the pool, so a pool of
    //                         two channels contributes two per full sweep.
    //                         Read it as backoffs, and halve it if you want
    //                         sweeps. Nothing downstream divides it by 1000,
    //                         which is the mistake the ns_ prefix invites.
    //
    // Cost: the increment sits AFTER the fast sweep has already failed, so a
    // reservation that succeeds first time executes nothing extra, not even the
    // stats-level branch. Iterations accumulate in a local and are added once on
    // exit, so the loop itself takes no atomic however long it runs.
    //
    // Containment, which is not obvious and is worth not re-deriving.
    // push_reserve_slow_pct divides n_push_reserve_slow by n_push_reserve, and
    // the two are incremented at different levels: n_push_reserve wraps
    // push_reserve_channel in uvm_push.c, while these live one level down in
    // channel_reserve_in_pool. That function has a second caller,
    // uvm_channel_reserve_type from channel_rotate_and_reserve_launch_channel,
    // which is outside the wrapped scope and would put the ratio above 100% if
    // it ever ran alongside. It cannot. That caller sits inside
    // "if (g_uvm_global.conf_computing_enabled)" in uvm_channel_begin_push, and
    // the code these counters sit in is reached only AFTER
    // channel_reserve_in_pool's own early return for the same condition. The
    // two paths are mutually exclusive by construction, so
    // n_push_reserve_slow <= n_push_reserve holds on any configuration.
    atomic64_t n_push_reserve_slow;
    atomic64_t ns_push_reserve_spins;

    // Sub-phases of uvm_va_block_make_resident_copy, in TWO BANKS chosen by the
    // cause argument. That function is shared: eviction reaches it through
    // uvm_va_block_evict_chunks, and fault servicing reaches it through
    // uvm_va_block_service_copy. Servicing migrations vastly outnumber
    // evictions, so a single set of counters would let servicing swamp the
    // eviction signal and would no longer sum inside ns_evict_resident.
    //
    //   *_unmap     both uvm_va_block_unmap_mask calls, accumulated into one
    //               counter (two begin/end pairs, one destination)
    //   *_populate  block_populate_pages. On the eviction path this allocates
    //               the HOST destination and bottoms out in the Linux page
    //               allocator (block_populate_pages_cpu ->
    //               uvm_cpu_chunk_alloc_page -> alloc_pages). Sixteen threads
    //               each moving 2 MB to host memory is per-zone lock pressure,
    //               and it is NOT a push -- so without this counter an
    //               allocator stall reads as "resident grows, pushes flat" and
    //               gets misattributed to the tracker or the copy
    //   *_copy      block_copy_resident_pages, which issues the push
    //
    // The evict bank is contained BY ns_evict_resident, so containment extends
    // one level and evict_resident_accounted_pct is the check. The svc bank is
    // contained by nothing in this struct -- never fold it into any eviction
    // percentage. Each bank divides by its OWN count, never across banks.
    //
    // Deliberately NOT probed one level deeper: uvm_cpu_chunk_alloc_page runs
    // inside a per-chunk loop and CPU chunks are 2 MB, 64 KB or 4 KB
    // (uvm_pmm_sysmem.h), so a fragmented 2 MB eviction can make up to 512
    // allocation calls. At roughly 60 ns of probe per call that is ~30 us added
    // to a 13.85 us measurement, and even the 64 KB case adds ~14%. The probe
    // would exceed the signal. If *_populate is what grows, drill in then, with
    // the fragmentation known.
    atomic64_t ns_evict_unmap;
    atomic64_t ns_evict_populate;
    atomic64_t ns_evict_copy;
    atomic64_t n_evict_mkres;
    atomic64_t ns_svc_unmap;
    atomic64_t ns_svc_populate;
    atomic64_t ns_svc_copy;
    atomic64_t n_svc_mkres;

    // How each eviction attempt ended. pick_and_evict_root_chunk returns
    // NV_ERR_NO_MEMORY when no candidate exists and
    // NV_ERR_MORE_PROCESSING_REQUIRED when chunks are in flight elsewhere;
    // both leave the retry loop while still counting as an attempt, which is
    // how a 2.7 us mean is possible against a 2 MB chunk copy. If in_flight
    // dominates, the growth is collision churn between workers and the answer
    // is a backoff rather than a lock. These three sum to n_evict_calls.
    atomic64_t n_evict_no_candidate;
    atomic64_t n_evict_in_flight;
    atomic64_t n_evict_success;

    // The two PMA eviction callbacks RM invokes with its API lock held. Hold
    // time, not wait time. See the note above.
    atomic64_t ns_pma_evict_cb;
    atomic64_t n_pma_evict_cbs;

    // va_block->lock in service_fault_batch_block, the GPU fault path shared
    // by the dispatcher and every worker
    atomic64_t ns_block_lock_wait_gpu;
    atomic64_t n_block_lock_acqs_gpu;

    // Whole-call cost of servicing one va_block's faults: service_fault_batch_block
    // from entry to return, so it contains the block-lock wait above, the retry
    // loop and the tracker merge. Divide by n_va_block_service for the mean cost
    // of handling one block.
    //
    // This exists to make one number comparable across mechanisms that are
    // otherwise structured differently. service_fault_batch_block is stock, so
    // the serial path, a worker pool and a stage pipeline all pass through it
    // once per block, and the same probe measures all three. Nothing else in
    // this struct can be divided down to a per-block figure: ns_service is a
    // whole-batch makespan, and the make-resident bank covers only the
    // migration phase.
    //
    // Contains ns_block_lock_wait_gpu, so the two must never be summed.
    atomic64_t ns_va_block_service;
    atomic64_t n_va_block_service;

    // va_block->lock in uvm_va_block_cpu_fault: how long CPU faults stall
    // behind GPU servicing. Strictly the acquisition, not the servicing that
    // follows it, which is why the CPU fault path uses the probed variant of
    // UVM_VA_BLOCK_LOCK_RETRY rather than bracketing the macro.
    atomic64_t ns_block_lock_wait_cpu;
    atomic64_t n_cpu_faults;

    // uvm_va_space_down_read in service_fault_batch_range
    atomic64_t ns_va_space_lock_wait;
    atomic64_t n_va_space_lock_acqs;

    // Top-half trylock failures: interrupts arriving while a bottom half is
    // already servicing. A count only, the failed trylock has no duration.
    atomic64_t n_top_half_trylock_fail;

    // Adaptive worker width, for observability. Without these an adaptive run
    // reports only an end-to-end speedup, and a good number would be
    // indistinguishable from the controller sitting still at a lucky width.
    // sum/decisions gives the mean width actually used, and n_adapt_widen and
    // n_adapt_narrow are the control-action count the stability argument is
    // about (Hellerstein s11.1, "excessive control actions increase overheads").
    //
    // NOT gated on the probe level: this measures the mechanism rather than
    // instrumenting a lock, it fires once per epoch (hundreds of times per
    // run, against millions for the lock probes), and an adaptive run at
    // level 0 still needs to be readable.
    atomic64_t n_adapt_decisions;
    atomic64_t sum_adapt_width;
    atomic64_t n_adapt_widen;
    atomic64_t n_adapt_narrow;
} uvm_lock_contention_stats_t;

extern uvm_lock_contention_stats_t g_uvm_lock_contention_stats;

static inline bool uvm_lock_probes_enabled(void)
{
    return uvm_perf_fault_stats_level >= 2;
}

static inline NvU64 uvm_lock_probe_begin(void)
{
    return uvm_lock_probes_enabled() ? NV_GETTIME() : 0;
}

// ns == NULL disables the probe entirely, which is how the unprobed users of
// UVM_VA_BLOCK_LOCK_RETRY opt out. acqs == NULL accumulates time without
// counting acquisitions. The enable test is repeated here rather than checking
// t0, so that flipping the level between the two halves of a probe drops the
// sample instead of recording an interval measured from zero.
static inline void uvm_lock_probe_end(NvU64 t0, atomic64_t *ns, atomic64_t *acqs)
{
    if (!ns || !uvm_lock_probes_enabled())
        return;

    atomic64_add(NV_GETTIME() - t0, ns);

    if (acqs)
        atomic64_inc(acqs);
}

// Count an event that has no duration, under the same level gate as the timed
// probes. Used for the eviction outcome counters, where what matters is which
// of three exits the attempt took rather than how long it took to get there.
static inline void uvm_lock_probe_count(atomic64_t *n)
{
    if (uvm_lock_probes_enabled())
        atomic64_inc(n);
}

// Add a tally accumulated in a local, for loops where one atomic per iteration
// would be the measurement rather than the thing measured. The caller keeps a
// plain counter and hands it over once on exit. A zero tally is dropped so a
// caller that never entered the loop adds nothing.
static inline void uvm_lock_probe_add(atomic64_t *n, NvU64 count)
{
    if (count && uvm_lock_probes_enabled())
        atomic64_add(count, n);
}

#endif // __UVM_LOCK_H__
