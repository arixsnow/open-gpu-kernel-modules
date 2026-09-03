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

#include "uvm_api.h"
#include "uvm_ats.h"
#include "uvm_global.h"
#include "uvm_gpu_replayable_faults.h"
#include "uvm_mem.h"
#include "uvm_perf_events.h"
#include "uvm_processors.h"
#include "uvm_procfs.h"
#include "uvm_thread_context.h"
#include "uvm_va_range.h"
#include "uvm_kvmalloc.h"
#include "uvm_mmu.h"
#include "uvm_perf_heuristics.h"
#include "uvm_pmm_sysmem.h"
#include "uvm_pmm_gpu.h"
#include "uvm_migrate.h"
#include "uvm_gpu_access_counters.h"
#include "uvm_va_space_mm.h"
#include "nv_uvm_interface.h"
#include "uvm_devmem.h"

uvm_global_t g_uvm_global;
static struct UvmEventsLinux g_exported_uvm_events;
static bool g_ops_registered = false;

unsigned uvm_force_conf_computing = 0;
module_param(uvm_force_conf_computing, uint, S_IRUGO);

// ARIADNE (HPCA'26). Master switch for the Sharing-Degree-aware eviction
// policy. Non-static because uvm_pmm_gpu.c reads it from the victim scan.
// Default on, matching their artifact; their ablation turns it off together
// with uvm_perf_fhp to reach the no-PL-SD configuration.
unsigned uvm_perf_SDaware = 1;
module_param(uvm_perf_SDaware, uint, S_IRUGO);

// ----------------------------------------------------------------------------
// ARIADNE (HPCA'26) gating knobs added by the port
// ----------------------------------------------------------------------------
//
// Their artifact diverges from stock in five always-on ways, and their two
// published knobs, uvm_perf_fhp and uvm_perf_SDaware, reach none of them. So
// their no-PL-SD arm, which the paper presents as both mechanisms disabled, is
// not a stock path. Campaign 20260902_014043 measured what that is worth: on
// MVT at 150 per cent oversubscription it finished in 0.69 s against stock at
// 325 s, migrating 103924 pages against 45928648, with both answers verified
// identical.
//
// Every default below is their shipped behaviour, so ariadne:full is unchanged
// and none of this is a fix to their mechanism. These are ablation knobs of the
// same kind as their own two, and they exist so that placement policy and
// servicing mechanism can be varied one at a time. Without them there is no
// configuration of their build that can be compared against ours.
//
// Declared here beside uvm_perf_SDaware, and externed in uvm_global.h, because
// the sites they gate are spread across uvm_va_block.c, uvm_pmm_gpu.c and
// uvm_gpu_replayable_faults.c.

// Dynamic Zero-copy. At 0 the host-pin loop does not run, the eviction path
// queues no candidates, and the unpin kthread is never started, which is the
// only way to get an ARIADNE build running the stock placement policy.
unsigned uvm_dynzero_enable = 1;
module_param(uvm_dynzero_enable, uint, S_IRUGO);

// Read duplication. Theirs forces may_read_duplicate false for every workload,
// so a page read by both processors is migrated back and forth instead of being
// resident in both places. At 0 the stock can_read_duplicate decision is
// restored. This one changes results, not just cost.
unsigned uvm_ariadne_disable_read_dup = 1;
module_param(uvm_ariadne_disable_read_dup, uint, S_IRUGO);

// Eviction victim selection. Theirs replaces the stock fallback with a scan of
// ALLOC_LIST_USED alone plus a batch-exclusion test; uvm_perf_SDaware only
// chooses between minimum-key and first-eligible within that replacement. At 0
// the stock get_first_allocated_chunk fallback is restored.
unsigned uvm_ariadne_evict_policy = 1;
module_param(uvm_ariadne_evict_policy, uint, S_IRUGO);

// Working Chunk Set Size accounting, the used_blocks list and active_blocks.
// At 0 no used_entry is allocated and nothing is charged. Zero-copy reads
// active_blocks, so uvm_dynzero_enable forces this on; see uvm_global_init.
unsigned uvm_ariadne_wcss = 1;
module_param(uvm_ariadne_wcss, uint, S_IRUGO);

// The 2 MB free-page charge counters that drive the eviction kthread doorbell.
// At 0 nothing is charged. uvm_perf_fhp forces this on for the same reason.
unsigned uvm_ariadne_chg2mb = 1;
module_param(uvm_ariadne_chg2mb, uint, S_IRUGO);

static NV_STATUS uvm_register_callbacks(void)
{
    NV_STATUS status = NV_OK;

    g_exported_uvm_events.isrTopHalf = uvm_isr_top_half_entry;
    g_exported_uvm_events.suspend = uvm_suspend_entry;
    g_exported_uvm_events.resume = uvm_resume_entry;
    g_exported_uvm_events.drainP2P = uvm_suspend_and_drainP2P_entry;
    g_exported_uvm_events.resumeP2P = uvm_resumeP2P_entry;

    // Register the UVM callbacks with the main GPU driver:
    status = uvm_rm_locked_call(nvUvmInterfaceRegisterUvmEvents(&g_exported_uvm_events));
    if (status != NV_OK)
        return status;

    g_ops_registered = true;
    return NV_OK;
}

// Calling this function more than once is harmless:
static void uvm_unregister_callbacks(void)
{
    if (g_ops_registered) {
        uvm_rm_locked_call_void(nvUvmInterfaceDeRegisterUvmEvents());
        g_ops_registered = false;
    }
}

NV_STATUS uvm_global_init(void)
{
    NV_STATUS status;
    UvmPlatformInfo platform_info;

    // Initialization of thread contexts happened already, during registration
    // (addition) of the thread context associated with the UVM module entry
    // point that is calling this function.
    UVM_ASSERT(uvm_thread_context_global_initialized());

    // ARIADNE (HPCA'26) knob dependencies, resolved here rather than left to
    // whoever writes the modprobe line. Both combinations below load without
    // complaint and then quietly do nothing, which is the worst way for an
    // experiment to fail: the arm runs, produces numbers, and the numbers are
    // of a different configuration than the label says.
    //
    // Zero-copy sizes its host-pin batch from active_blocks, which only the
    // working-set accounting maintains. With that accounting off, active_blocks
    // stays zero, to_pin evaluates to zero on every batch, and not one block is
    // ever pinned.
    if (uvm_dynzero_enable && !uvm_ariadne_wcss) {
        UVM_ERR_PRINT("ARIADNE: uvm_dynzero_enable=1 needs uvm_ariadne_wcss=1, "
                      "since Zero-copy sizes its batch from active_blocks. "
                      "Forcing uvm_ariadne_wcss=1.\n");
        uvm_ariadne_wcss = 1;
    }

    // The proactive eviction kthread wakes on a watermark computed from
    // max_rest_2mb_pages minus cur_chg_2mb_pages. With the charge counters off
    // that difference never moves and the thread never has anything to do, so
    // the Populate/Copy pipeline runs without the eviction half it assumes.
    if (uvm_perf_fhp && !uvm_ariadne_chg2mb) {
        UVM_ERR_PRINT("ARIADNE: uvm_perf_fhp=1 needs uvm_ariadne_chg2mb=1, "
                      "since the eviction kthread doorbell reads those counters. "
                      "Forcing uvm_ariadne_chg2mb=1.\n");
        uvm_ariadne_chg2mb = 1;
    }

    uvm_mutex_init(&g_uvm_global.global_lock, UVM_LOCK_ORDER_GLOBAL);
    uvm_init_rwsem(&g_uvm_global.pm.lock, UVM_LOCK_ORDER_GLOBAL_PM);
    uvm_spin_lock_irqsave_init(&g_uvm_global.gpu_table_lock, UVM_LOCK_ORDER_LEAF);
    uvm_mutex_init(&g_uvm_global.va_spaces.lock, UVM_LOCK_ORDER_VA_SPACES_LIST);
    INIT_LIST_HEAD(&g_uvm_global.va_spaces.list);
    uvm_mutex_init(&g_uvm_global.devmem_ranges.lock, UVM_LOCK_ORDER_LEAF);
    INIT_LIST_HEAD(&g_uvm_global.devmem_ranges.list);
    INIT_LIST_HEAD(&g_uvm_global.pci_p2pdma_devices.list);

    status = uvm_kvmalloc_init();
    if (status != NV_OK) {
        UVM_ERR_PRINT("uvm_kvmalloc_init() failed: %s\n", nvstatusToString(status));
        goto error;
    }

    status = errno_to_nv_status(nv_kthread_q_init(&g_uvm_global.global_q, "UVM global queue"));
    if (status != NV_OK) {
        UVM_DBG_PRINT("nv_kthread_q_init() failed: %s\n", nvstatusToString(status));
        goto error;
    }

    status = uvm_procfs_init();
    if (status != NV_OK) {
        UVM_ERR_PRINT("uvm_procfs_init() failed: %s\n", nvstatusToString(status));
        goto error;
    }

    status = uvm_rm_locked_call(nvUvmInterfaceSessionCreate(&g_uvm_global.rm_session_handle, &platform_info));
    if (status != NV_OK) {
        UVM_ERR_PRINT("nvUvmInterfaceSessionCreate() failed: %s\n", nvstatusToString(status));
        return status;
    }

    uvm_ats_init(&platform_info);

    g_uvm_global.cdmm_enabled = platform_info.cdmmEnabled;

    g_uvm_global.num_simulated_devices = 0;

    g_uvm_global.hw_conf_computing_enabled = platform_info.confComputingEnabled;
    g_uvm_global.conf_computing_enabled = g_uvm_global.hw_conf_computing_enabled ||
                                          uvm_force_conf_computing;

    status = uvm_processor_mask_cache_init();
    if (status != NV_OK) {
        UVM_ERR_PRINT("uvm_processor_mask_cache_init() failed: %s\n", nvstatusToString(status));
        goto error;
    }

    status = uvm_devmem_global_init();
    if (status != NV_OK) {
        UVM_ERR_PRINT("uvm_devmem_global_init() failed: %s\n", nvstatusToString(status));
        goto error;
    }

    status = uvm_gpu_init();
    if (status != NV_OK) {
        UVM_ERR_PRINT("uvm_gpu_init() failed: %s\n", nvstatusToString(status));
        goto error;
    }

    status = uvm_pmm_sysmem_init();
    if (status != NV_OK) {
        UVM_ERR_PRINT("uvm_pmm_sysmem_init() failed: %s\n", nvstatusToString(status));
        goto error;
    }

    status = uvm_mmu_init();
    if (status != NV_OK) {
        UVM_ERR_PRINT("uvm_mmu_init() failed: %s\n", nvstatusToString(status));
        goto error;
    }

    status = uvm_mem_global_init();
    if (status != NV_OK) {
        UVM_ERR_PRINT("uvm_mem_gloal_init() failed: %s\n", nvstatusToString(status));
        goto error;
    }

    status = uvm_va_policy_init();
    if (status != NV_OK) {
        UVM_ERR_PRINT("uvm_va_policy_init() failed: %s\n", nvstatusToString(status));
        goto error;
    }

    status = uvm_va_range_init();
    if (status != NV_OK) {
        UVM_ERR_PRINT("uvm_va_range_init() failed: %s\n", nvstatusToString(status));
        goto error;
    }

    status = uvm_migrate_init();
    if (status != NV_OK) {
        UVM_ERR_PRINT("uvm_migrate_init() failed: %s\n", nvstatusToString(status));
        goto error;
    }

    status = uvm_perf_events_init();
    if (status != NV_OK) {
        UVM_ERR_PRINT("uvm_perf_events_init() failed: %s\n", nvstatusToString(status));
        goto error;
    }

    status = uvm_perf_heuristics_init();
    if (status != NV_OK) {
        UVM_ERR_PRINT("uvm_perf_heuristics_init() failed: %s\n", nvstatusToString(status));
        goto error;
    }

    status = uvm_service_block_context_init();
    if (status != NV_OK) {
        UVM_ERR_PRINT("uvm_service_block_context_init failed: %s\n", nvstatusToString(status));
        goto error;
    }

    status = uvm_access_counters_init();
    if (status != NV_OK) {
        UVM_ERR_PRINT("uvm_access_counters_init failed: %s\n", nvstatusToString(status));
        goto error;
    }

    // This sets up the ISR (interrupt service routine), by hooking into RM's
    // top-half ISR callback. As soon as this call completes, GPU interrupts
    // will start arriving, so it's important to be prepared to receive
    // interrupts before this point.
    status = uvm_register_callbacks();
    if (status != NV_OK) {
        UVM_ERR_PRINT("uvm_register_callbacks failed: %s\n", nvstatusToString(status));
        goto error;
    }

    status = errno_to_nv_status(nv_kthread_q_init(&g_uvm_global.deferred_release_q, "UVM deferred release queue"));
    if (status != NV_OK) {
        UVM_DBG_PRINT("nv_kthread_q_init() failed: %s\n", nvstatusToString(status));
        goto error;
    }

    return NV_OK;

error:
    uvm_global_exit();
    return status;
}

void uvm_global_exit(void)
{
    uvm_assert_mutex_unlocked(&g_uvm_global.global_lock);

    nv_kthread_q_stop(&g_uvm_global.deferred_release_q);

    uvm_unregister_callbacks();
    uvm_access_counters_exit();
    uvm_service_block_context_exit();
    uvm_perf_heuristics_exit();
    uvm_perf_events_exit();
    uvm_migrate_exit();
    uvm_va_range_exit();
    uvm_va_policy_exit();
    uvm_mem_global_exit();
    uvm_pmm_sysmem_exit();
    uvm_devmem_exit();
    uvm_devmem_pci_p2pdma_cache_exit();
    uvm_gpu_exit();
    uvm_devmem_global_deinit();
    uvm_processor_mask_cache_exit();

    if (g_uvm_global.rm_session_handle != 0)
        uvm_rm_locked_call_void(nvUvmInterfaceSessionDestroy(g_uvm_global.rm_session_handle));

    uvm_procfs_exit();

    nv_kthread_q_stop(&g_uvm_global.global_q);

    uvm_assert_mutex_unlocked(&g_uvm_global.va_spaces.lock);
    UVM_ASSERT(list_empty(&g_uvm_global.va_spaces.list));

    uvm_thread_context_global_exit();
    uvm_kvmalloc_exit();
}

// Signal to the top-half ISR whether calls from the RM's top-half ISR are to
// be completed without processing.
static void uvm_parent_gpu_set_isr_suspended(uvm_parent_gpu_t *parent_gpu, bool is_suspended)
{
    uvm_spin_lock_irqsave(&parent_gpu->isr.interrupts_lock);

    parent_gpu->isr.is_suspended = is_suspended;

    uvm_spin_unlock_irqrestore(&parent_gpu->isr.interrupts_lock);
}

static NV_STATUS uvm_suspend(void)
{
    uvm_va_space_t *va_space = NULL;
    uvm_gpu_id_t gpu_id;
    uvm_gpu_t *gpu;

    // Upon entry into this function, the following is true:
    //   * GPU interrupts are enabled
    //   * Any number of fault or access counter notifications could
    //     be pending
    //   * No new fault notifications will appear, but new access
    //     counter notifications could
    //   * Any of the bottom halves could be running
    //   * New bottom halves of all types could be scheduled as GPU
    //     interrupts are handled
    // Due to this, the sequence of suspend operations for each GPU is the
    // following:
    //   * Flush the fault buffer to prevent fault interrupts when
    //     the top-half ISR is suspended
    //   * Suspend access counter processing
    //   * Suspend the top-half ISR
    //   * Flush relevant kthread queues (bottom half, etc.)

    // Some locks acquired by this function, such as pm.lock, are released
    // by uvm_resume(). This is contrary to the lock tracking code's
    // expectations, so lock tracking is disabled.
    uvm_thread_context_lock_disable_tracking();

    // Take the global power management lock in write mode to lock out
    // most user-facing entry points.
    uvm_down_write(&g_uvm_global.pm.lock);

    nv_kthread_q_flush(&g_uvm_global.global_q);

    // Though global_lock isn't held here, pm.lock indirectly prevents the
    // addition and removal of GPUs, since these operations can currently
    // only occur in response to ioctl() calls.
    for_each_gpu_id_in_mask(gpu_id, &g_uvm_global.retained_gpus) {
        gpu = uvm_gpu_get(gpu_id);

        // Since fault buffer state may be lost across sleep cycles, UVM must
        // ensure any outstanding replayable faults are dismissed. The RM
        // guarantees that all user channels have been preempted before
        // uvm_suspend() is called, which implies that no user channels can be
        // stalled on faults when this point is reached.
        uvm_gpu_replayable_buffer_flush(gpu);

        // Stop access counter interrupt processing for the duration of this
        // sleep cycle to defend against potential interrupt storms in
        // the suspend path: if rate limiting is applied to access counter
        // interrupts in the bottom half in the future, the bottom half flush
        // below will no longer be able to guarantee that all outstanding
        // notifications have been handled.
        uvm_parent_gpu_access_counters_set_ignore(gpu->parent, true);

        uvm_parent_gpu_set_isr_suspended(gpu->parent, true);

        nv_kthread_q_flush(&gpu->parent->isr.bottom_half_q);

        if (gpu->parent->isr.non_replayable_faults.handling)
            nv_kthread_q_flush(&gpu->parent->isr.kill_channel_q);
    }

    // Acquire each VA space's lock in write mode to lock out VMA open and
    // release callbacks. These entry points do not have feasible early exit
    // options, and so aren't suitable for synchronization with pm.lock.
    uvm_mutex_lock(&g_uvm_global.va_spaces.lock);

    list_for_each_entry(va_space, &g_uvm_global.va_spaces.list, list_node)
        uvm_va_space_down_write(va_space);

    uvm_mutex_unlock(&g_uvm_global.va_spaces.lock);

    uvm_thread_context_lock_enable_tracking();

    g_uvm_global.pm.is_suspended = true;

    return NV_OK;
}

NV_STATUS uvm_suspend_entry(void)
{
    UVM_ENTRY_RET(uvm_suspend());
}

static NV_STATUS uvm_resume(void)
{
    uvm_va_space_t *va_space = NULL;
    uvm_gpu_id_t gpu_id;
    uvm_gpu_t *gpu;

    g_uvm_global.pm.is_suspended = false;

    // Some locks released by this function, such as pm.lock, were acquired
    // by uvm_suspend(). This is contrary to the lock tracking code's
    // expectations, so lock tracking is disabled.
    uvm_thread_context_lock_disable_tracking();

    // Release each VA space's lock.
    uvm_mutex_lock(&g_uvm_global.va_spaces.lock);

    list_for_each_entry(va_space, &g_uvm_global.va_spaces.list, list_node)
        uvm_va_space_up_write(va_space);

    uvm_mutex_unlock(&g_uvm_global.va_spaces.lock);

    // pm.lock is held in lieu of global_lock to prevent GPU addition/removal
    for_each_gpu_id_in_mask(gpu_id, &g_uvm_global.retained_gpus) {
        gpu = uvm_gpu_get(gpu_id);

        // Bring the fault buffer software state back in sync with the
        // hardware state.
        uvm_parent_gpu_fault_buffer_resume(gpu->parent);

        uvm_parent_gpu_set_isr_suspended(gpu->parent, false);

        // Reenable access counter interrupt processing unless notifications
        // have been set to be suppressed.
        uvm_parent_gpu_access_counters_set_ignore(gpu->parent, false);
    }

    uvm_up_write(&g_uvm_global.pm.lock);

    uvm_thread_context_lock_enable_tracking();

    // Force completion of any release callbacks successfully queued for
    // deferred completion while suspended. The deferred release
    // queue is not guaranteed to remain empty following this flush since
    // some threads that failed to acquire pm.lock in uvm_release() may
    // not have scheduled their handlers yet.
    nv_kthread_q_flush(&g_uvm_global.deferred_release_q);

    return NV_OK;
}

NV_STATUS uvm_resume_entry(void)
{
    UVM_ENTRY_RET(uvm_resume());
}

bool uvm_global_is_suspended(void)
{
    return g_uvm_global.pm.is_suspended;
}

void uvm_global_set_fatal_error_impl(NV_STATUS error)
{
    NV_STATUS previous_error;

    UVM_ASSERT(error != NV_OK);

    previous_error = atomic_cmpxchg(&g_uvm_global.fatal_error, NV_OK, error);

    if (previous_error == NV_OK) {
        UVM_ERR_PRINT("Encountered a global fatal error: %s\n", nvstatusToString(error));
    }
    else {
        UVM_ERR_PRINT("Encountered a global fatal error: %s after a global error has been already set: %s\n",
                nvstatusToString(error),
                nvstatusToString(previous_error));
    }

    nvUvmInterfaceReportFatalError(error);
}

NV_STATUS uvm_global_reset_fatal_error(void)
{
    if (!uvm_enable_builtin_tests) {
        UVM_ASSERT_MSG(0, "Resetting global fatal error without tests being enabled\n");
        return NV_ERR_INVALID_STATE;
    }

    return atomic_xchg(&g_uvm_global.fatal_error, NV_OK);
}

void uvm_global_gpu_retain(const uvm_processor_mask_t *mask)
{
    uvm_gpu_t *gpu;

    for_each_gpu_in_mask(gpu, mask)
        uvm_gpu_retain(gpu);
}

void uvm_global_gpu_release(const uvm_processor_mask_t *mask)
{
    uvm_gpu_id_t gpu_id;

    if (uvm_processor_mask_empty(mask))
        return;

    uvm_mutex_lock(&g_uvm_global.global_lock);

    // Do not use for_each_gpu_in_mask as it reads the GPU state and it
    // might get destroyed.
    for_each_gpu_id_in_mask(gpu_id, mask)
        uvm_gpu_release_locked(uvm_gpu_get(gpu_id));

    uvm_mutex_unlock(&g_uvm_global.global_lock);
}

NV_STATUS uvm_global_gpu_check_ecc_error(uvm_processor_mask_t *gpus)
{
    uvm_gpu_t *gpu;

    for_each_gpu_in_mask(gpu, gpus) {
        NV_STATUS status = uvm_gpu_check_ecc_error(gpu);
        if (status != NV_OK)
            return status;
    }

    return NV_OK;
}

static NV_STATUS suspend_and_drainP2P(const NvProcessorUuid *parent_uuid)
{
    NV_STATUS status = NV_OK;
    uvm_parent_gpu_t *parent_gpu;

    uvm_mutex_lock(&g_uvm_global.global_lock);

    // NVLINK STO recovery is not supported in combination with MIG
    parent_gpu = uvm_parent_gpu_get_by_uuid(parent_uuid);
    if (!parent_gpu || parent_gpu->smc.enabled) {
        status = NV_ERR_INVALID_DEVICE;
        goto unlock;
    }

    status = uvm_channel_manager_suspend_p2p(parent_gpu->gpus[0]->channel_manager);

unlock:
    uvm_mutex_unlock(&g_uvm_global.global_lock);
    return status;
}

static NV_STATUS resumeP2P(const NvProcessorUuid *parent_uuid)
{
    NV_STATUS status = NV_OK;
    uvm_parent_gpu_t *parent_gpu;

    uvm_mutex_lock(&g_uvm_global.global_lock);

    // NVLINK STO recovery is not supported in combination with MIG
    parent_gpu = uvm_parent_gpu_get_by_uuid(parent_uuid);
    if (!parent_gpu || parent_gpu->smc.enabled) {
        status = NV_ERR_INVALID_DEVICE;
        goto unlock;
    }

    uvm_channel_manager_resume_p2p(parent_gpu->gpus[0]->channel_manager);

unlock:
    uvm_mutex_unlock(&g_uvm_global.global_lock);
    return status;
}

NV_STATUS uvm_suspend_and_drainP2P_entry(const NvProcessorUuid *uuid)
{
    UVM_ENTRY_RET(suspend_and_drainP2P(uuid));
}

NV_STATUS uvm_resumeP2P_entry(const NvProcessorUuid *uuid)
{
    UVM_ENTRY_RET(resumeP2P(uuid));
}

NV_STATUS uvm_global_gpu_check_nvlink_error(uvm_processor_mask_t *gpus)
{
    uvm_gpu_t *gpu;

    for_each_gpu_in_mask(gpu, gpus) {
        NV_STATUS status = uvm_gpu_check_nvlink_error(gpu);
        if (status != NV_OK)
            return status;
    }

    return NV_OK;
}
