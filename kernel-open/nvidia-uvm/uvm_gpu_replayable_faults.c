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

#include "linux/sort.h"
#include "nv_uvm_interface.h"
#include "uvm_api.h"
#include "uvm_common.h"
#include "uvm_linux.h"
#include "uvm_global.h"
#include "uvm_gpu_replayable_faults.h"
#include "uvm_hal.h"
#include "uvm_kvmalloc.h"
#include "uvm_tools.h"
#include "uvm_va_block.h"
#include "uvm_va_range.h"
#include "uvm_va_space.h"
#include "uvm_va_space_mm.h"
#include "uvm_procfs.h"
#include "uvm_perf_thrashing.h"
#include "uvm_gpu_non_replayable_faults.h"
#include "uvm_gpu_isr.h"
#include "uvm_ats_faults.h"
#include "uvm_test.h"

// The documentation at the beginning of uvm_gpu_non_replayable_faults.c
// provides some background for understanding replayable faults, non-replayable
// faults, and how UVM services each fault type.

// The HW fault buffer flush mode instructs RM on how to flush the hardware
// replayable fault buffer; it is only used in Confidential Computing.
//
// Unless HW_FAULT_BUFFER_FLUSH_MODE_MOVE is functionally required (because UVM
// needs to inspect the faults currently present in the HW fault buffer) it is
// recommended to use HW_FAULT_BUFFER_FLUSH_MODE_DISCARD for performance
// reasons.
typedef enum
{
    // Flush the HW fault buffer, discarding all the resulting faults. UVM never
    // gets to see these faults.
    HW_FAULT_BUFFER_FLUSH_MODE_DISCARD,

    // Flush the HW fault buffer, and move all the resulting faults to the SW
    // fault ("shadow") buffer.
    HW_FAULT_BUFFER_FLUSH_MODE_MOVE,
} hw_fault_buffer_flush_mode_t;

#define UVM_PERF_REENABLE_PREFETCH_FAULTS_LAPSE_MSEC_DEFAULT 1000

// Lapse of time in milliseconds after which prefetch faults can be re-enabled.
// 0 means it is never disabled
static unsigned uvm_perf_reenable_prefetch_faults_lapse_msec = UVM_PERF_REENABLE_PREFETCH_FAULTS_LAPSE_MSEC_DEFAULT;
module_param(uvm_perf_reenable_prefetch_faults_lapse_msec, uint, S_IRUGO);

#define UVM_PERF_FAULT_BATCH_COUNT_MIN 1
#define UVM_PERF_FAULT_BATCH_COUNT_DEFAULT 256

// Number of entries that are fetched from the GPU fault buffer and serviced in
// batch
static unsigned uvm_perf_fault_batch_count = UVM_PERF_FAULT_BATCH_COUNT_DEFAULT;
module_param(uvm_perf_fault_batch_count, uint, S_IRUGO);

#define UVM_PERF_FAULT_REPLAY_POLICY_DEFAULT UVM_PERF_FAULT_REPLAY_POLICY_BATCH_FLUSH

// Policy that determines when to issue fault replays
static uvm_perf_fault_replay_policy_t uvm_perf_fault_replay_policy = UVM_PERF_FAULT_REPLAY_POLICY_DEFAULT;
module_param(uvm_perf_fault_replay_policy, uint, S_IRUGO);

// Ceiling on the pinned pool width. This is a policy cap, not a structural
// one: the worker array is allocated from it at fault buffer init and the only
// fixed-size object that depends on it is the load array in
// fault_service_assign_spans(), which this macro sizes.
//
// Raised from 15 to 23 on measured evidence. Campaign 20260904_142811 shows the
// marginal return per worker over the 4-to-15 range is flat at 0.41 s/worker
// across every rung of the w7 oversubscription sweep rather than decaying, so
// fifteen was cutting the curve off while it was still paying. Twenty-three
// plus the dispatcher is 24 threads, which is the logical CPU count of the
// evaluation box, so anything above this cannot be co-scheduled with the
// workload and there is no reason to express it.
#define UVM_PERF_FAULT_SERVICE_MAX_WORKERS 23

// Number of additional worker threads (beyond the bottom-half dispatcher,
// which services its own share inline) used to service a replayable fault
// batch in parallel across va_blocks. 0 (the default) disables the worker
// pool entirely and keeps the stock serial servicing path.
static unsigned uvm_perf_fault_service_num_workers = 0;
module_param(uvm_perf_fault_service_num_workers, uint, S_IRUGO);

// Adaptive worker width. This is gain scheduling in the sense of Hellerstein
// et al. s11.2, rules that distinguish operating conditions using a scheduling
// variable read off the target system. The scheduling variable is eviction
// attempts per batch, which separates the two kinds of workload cleanly: 0.00
// per batch on every in-memory cell against 6.70-46.49 on the oversubscribed
// ones (campaign 20260724_020443, fifteen workers). Self-tuning regulators
// were rejected on the book's own grounds (s11.3): they are "slow in adapting"
// to abrupt workload change and perform worse than handcrafted gain
// scheduling.
//
// Measured outcome, 20260724_020443: the controller sorts all nineteen cells
// correctly and still wins no cell outright against a pinned pool of fifteen,
// losing five. A wide pinned pool costs an in-memory workload too little here
// for width scheduling to have anything to recover. Worth revisiting on
// hardware where a wide pool actually hurts. Two known gaps: eviction rate
// reads zero on a cell that faults heavily, never evicts and still wants
// width, and on the two-client cell the cost of the pool does not depend on
// its width at all.
//
// Off by default, like the other mechanisms here. Setting
// uvm_perf_fault_service_num_workers explicitly still pins the width, so every
// published measurement stays reproducible.
#define UVM_PERF_FAULT_SERVICE_ADAPT_EPOCH_DEFAULT  64
#define UVM_PERF_FAULT_SERVICE_ADAPT_LO_DEFAULT     1000   // milli-evictions/batch
#define UVM_PERF_FAULT_SERVICE_ADAPT_HI_DEFAULT     4000
#define UVM_PERF_FAULT_SERVICE_ADAPT_STEP_DEFAULT   2
#define UVM_PERF_FAULT_SERVICE_ADAPT_NARROW_DEFAULT 4

static unsigned uvm_perf_fault_service_adapt = 0;
module_param(uvm_perf_fault_service_adapt, uint, S_IRUGO);

// Batches per control decision. 64 batches is about 70 ms at the measured
// batch period, which reaches full width in roughly 420 ms.
static unsigned uvm_perf_fault_service_adapt_epoch = UVM_PERF_FAULT_SERVICE_ADAPT_EPOCH_DEFAULT;
module_param(uvm_perf_fault_service_adapt_epoch, uint, S_IRUGO);

// The hold band, in thousandths of an eviction per batch. Widen above hi,
// narrow below lo, hold in between. The band is TCP Vegas: two thresholds with
// "leave unchanged" between them, so a workload sitting near the boundary
// cannot chatter. Sweeping lo/hi over a tenfold range in adapt_sim.py changes
// no verdict, because the measured signal gap is doing the work.
static unsigned uvm_perf_fault_service_adapt_lo = UVM_PERF_FAULT_SERVICE_ADAPT_LO_DEFAULT;
module_param(uvm_perf_fault_service_adapt_lo, uint, S_IRUGO);

static unsigned uvm_perf_fault_service_adapt_hi = UVM_PERF_FAULT_SERVICE_ADAPT_HI_DEFAULT;
module_param(uvm_perf_fault_service_adapt_hi, uint, S_IRUGO);

// Workers added per widening decision.
static unsigned uvm_perf_fault_service_adapt_step = UVM_PERF_FAULT_SERVICE_ADAPT_STEP_DEFAULT;
module_param(uvm_perf_fault_service_adapt_step, uint, S_IRUGO);

// Narrow by one worker only every Nth epoch below the low threshold. The
// asymmetry is not stylistic: running too narrow costs up to 1.64x on the
// oversubscribed cells against about 3% for running too wide, so widening
// eagerly is the cheap error. The floor of 4 is set by stability rather than
// by that ratio. adapt_sim.py sweeps this against an alternating-regime input
// and values below 4 slam the actuator across its full range, which is the
// limit cycle of Hellerstein Fig 8.9; 4 and 8 pass every combination tried.
static unsigned uvm_perf_fault_service_adapt_narrow_every = UVM_PERF_FAULT_SERVICE_ADAPT_NARROW_DEFAULT;
module_param(uvm_perf_fault_service_adapt_narrow_every, uint, S_IRUGO);

#define UVM_PERF_FAULT_SERVICE_MIN_FAULTS_DEFAULT 32

// Minimum number of coalesced faults in a batch for the worker pool to be
// used. Smaller batches are serviced serially by the dispatcher, where the
// dispatch/join overhead would dominate any parallelism win.
static unsigned uvm_perf_fault_service_min_faults = UVM_PERF_FAULT_SERVICE_MIN_FAULTS_DEFAULT;
module_param(uvm_perf_fault_service_min_faults, uint, S_IRUGO);

// Pipelined batch servicing: under UVM_PERF_FAULT_REPLAY_POLICY_BATCH_FLUSH
// (the default), do not synchronously wait for the flush's replay to complete
// before fetching the next batch. The replay push already acquires the batch
// tracker, so the GPU orders it after every migration of the batch with no
// CPU involvement; the synchronous wait only paces the loop. When enabled,
// the wait is deferred until the fault buffer turns up empty while a replay
// is still pending (then we wait once and re-fetch, preserving the stock
// behavior of servicing replayed faults within the same bottom-half pass).
// 0 (the default) keeps the stock synchronous wait.
static unsigned uvm_perf_fault_service_pipeline = 0;
module_param(uvm_perf_fault_service_pipeline, uint, S_IRUGO);

#define UVM_PERF_FAULT_SERVICE_MAX_INFLIGHT_DEFAULT 1

// How many batches may have un-waited replays outstanding at once when
// pipelining is on. 1, the default, is exactly the stock barrier: push the
// replay, wait for it, then fetch the next batch. 0 removes the bound
// entirely, which is the original pipelined behaviour and is kept only so the
// cost of the bound can be measured.
//
// Why the default is 1. The deferred wait is the only backpressure in the
// servicing path, and without it uvm_perf_fault_max_batches_per_service (20)
// batches of uvm_perf_fault_batch_count (256) faults can populate before
// anything completes. Under oversubscription those pages are evicted before
// the replay that would have let the GPU consume them, the GPU re-faults, and
// the loop feeds itself.
//
// Measured, GESUMMV at 150% oversubscription, five repetitions per point,
// fifteen workers, GPU pass time in seconds and median fault count
// (campaign 20260907_153843):
//
//     bound      min    median      max        faults
//     stock     0.812    1.745    21.163      276,826
//     1         0.525    1.557     4.171      339,624
//     4         3.264   30.808   205.675    7,123,225
//     16       14.051   63.501   118.946   14,411,062
//     unbounded 37.807  69.729   120.402   16,190,040
//
// Monotone in the bound, in both time and fault count, and at 1 the worker
// pool is better than stock on this cell rather than merely safe.
//
// The cost is small and is paid on a workload that cannot thrash. On the
// fault_storm oversubscription sweep, which reproduces to 0.3% across four
// campaigns, bound 1 costs 1.3% at 150% (37.27 s against 36.79 s) and 0.9% at
// 110%. Its fault count barely moves with the bound (8.47M at 1 against 8.69M
// unbounded), which is why the bound is nearly free there.
//
// Note that the dose-response only appears once the worker pool is running. An
// earlier sweep of this same parameter with zero workers showed no ordering at
// all, because a batch serviced by one thread populates too little for the
// in-flight footprint to matter.
static unsigned uvm_perf_fault_service_max_inflight = UVM_PERF_FAULT_SERVICE_MAX_INFLIGHT_DEFAULT;
module_param(uvm_perf_fault_service_max_inflight, uint, S_IRUGO);

#define UVM_PERF_FAULT_REPLAY_UPDATE_PUT_RATIO_DEFAULT 50

// Reading fault buffer GET/PUT pointers from the CPU is expensive. However,
// updating PUT before flushing the buffer helps minimizing the number of
// duplicates in the buffer as it discards faults that were not processed
// because of the batch size limit or because they arrived during servicing.
// If PUT is not updated, the replay operation will make them show up again
// in the buffer as duplicates.
//
// We keep track of the number of duplicates in each batch and we use
// UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT for the fault buffer flush after if the
// percentage of duplicate faults in a batch is greater than the ratio defined
// in the following module parameter. UVM_GPU_BUFFER_FLUSH_MODE_CACHED_PUT is
// used, otherwise.
static unsigned uvm_perf_fault_replay_update_put_ratio = UVM_PERF_FAULT_REPLAY_UPDATE_PUT_RATIO_DEFAULT;
module_param(uvm_perf_fault_replay_update_put_ratio, uint, S_IRUGO);

// The same backlog discard, triggered on redundancy the duplicate counter
// above cannot see. 0, the default, disables it and leaves the stock heuristic
// exactly as it was.
//
// The duplicate ratio is an intra-batch measure: check_fault_entry_duplicate
// compares a fault only against the previous entry of ordered_fault_cache, and
// that cache is rebuilt every batch. Parallel servicing produces cross-batch
// redundancy instead - the GPU raises a fault, we map the page, and the fault
// is fetched in a later batch, by which point it needs no service. That fault
// is exactly what UPDATE_PUT exists to discard, and it is invisible to the
// trigger.
//
// Measured on w7 at 110% oversubscription with 21 workers, against the same
// build with the pool switched off:
//
//                       faults/page   authorized   serviced   authorized %
//     pool off              1.09         218,112   3,775,341      5.5%
//     21 workers            1.95       3,071,204   3,647,176     45.7%
//
// Serviced faults are the same, so the pool creates no extra real work; the
// entire excess is redundant. Meanwhile num_duplicate_faults reads 0.61% and
// the discard never fires.
//
// Expressed as a percentage of the batch's fault instances, like the ratio
// above. Left at 0 until measured: discarding is not free, since every entry
// dropped is re-raised by the replay if it still matters, so an aggressive
// setting can churn.
static unsigned uvm_perf_fault_replay_update_put_authorized_ratio = 0;
module_param(uvm_perf_fault_replay_update_put_authorized_ratio, uint, S_IRUGO);

#define UVM_PERF_FAULT_MAX_BATCHES_PER_SERVICE_DEFAULT 20

#define UVM_PERF_FAULT_MAX_THROTTLE_PER_SERVICE_DEFAULT 5

// Maximum number of batches to be processed per execution of the bottom-half
static unsigned uvm_perf_fault_max_batches_per_service = UVM_PERF_FAULT_MAX_BATCHES_PER_SERVICE_DEFAULT;
module_param(uvm_perf_fault_max_batches_per_service, uint, S_IRUGO);

// Maximum number of batches with thrashing pages per execution of the bottom-half
static unsigned uvm_perf_fault_max_throttle_per_service = UVM_PERF_FAULT_MAX_THROTTLE_PER_SERVICE_DEFAULT;
module_param(uvm_perf_fault_max_throttle_per_service, uint, S_IRUGO);

static unsigned uvm_perf_fault_coalesce = 1;
module_param(uvm_perf_fault_coalesce, uint, S_IRUGO);

// This function is used for both the initial fault buffer initialization and
// the power management resume path.
static void fault_buffer_reinit_replayable_faults(uvm_parent_gpu_t *parent_gpu)
{
    uvm_replayable_fault_buffer_t *replayable_faults = &parent_gpu->fault_buffer.replayable;

    // Read the current get/put pointers, as this might not be the first time
    // we take control of the fault buffer since the GPU was initialized,
    // or since we may need to bring UVM's cached copies back in sync following
    // a sleep cycle.
    replayable_faults->cached_get = parent_gpu->fault_buffer_hal->read_get(parent_gpu);
    replayable_faults->cached_put = parent_gpu->fault_buffer_hal->read_put(parent_gpu);

    // (Re-)enable fault prefetching
    if (parent_gpu->fault_buffer.prefetch_faults_enabled)
        parent_gpu->arch_hal->enable_prefetch_faults(parent_gpu);
    else
        parent_gpu->arch_hal->disable_prefetch_faults(parent_gpu);
}

static void fault_service_worker_entry(void *args);

// Allocate and initialize the worker pool for parallel fault-batch servicing.
// With uvm_perf_fault_service_num_workers == 0 (the default) only the zero
// worker count is recorded: nothing is allocated and fault servicing takes
// the stock serial path.
static NV_STATUS fault_service_pool_init(uvm_parent_gpu_t *parent_gpu)
{
    NV_STATUS status;
    NvU32 i;
    char kthread_name[32];
    uvm_replayable_fault_buffer_t *replayable_faults = &parent_gpu->fault_buffer.replayable;
    NvU32 num_workers = min(uvm_perf_fault_service_num_workers, (unsigned)UVM_PERF_FAULT_SERVICE_MAX_WORKERS);

    if (num_workers != uvm_perf_fault_service_num_workers) {
        UVM_INFO_PRINT("Invalid uvm_perf_fault_service_num_workers value on GPU %s: %u. Valid range [0:%u] Using %u instead\n",
                       uvm_parent_gpu_name(parent_gpu),
                       uvm_perf_fault_service_num_workers,
                       UVM_PERF_FAULT_SERVICE_MAX_WORKERS,
                       num_workers);
    }

    replayable_faults->service_pool.num_workers = num_workers;

    // Cold start. With adaptation off this is the whole pool, which is the
    // stock behaviour of the fixed-width parameter. With it on we start at one
    // worker and widen into the pool, because the starting value is what a
    // workload gets when it ends before the controller ever fires, and one
    // worker is the conservative degradation: near-stock, and correct outright
    // for any workload that fits in memory. Starting wide would be the worst
    // configuration for exactly those workloads, and narrowing is deliberately
    // slower than widening, so the mistake would persist.
    replayable_faults->service_pool.active_workers =
        (uvm_perf_fault_service_adapt && num_workers > 0) ? 1 : num_workers;
    // Snapshot the counters rather than zeroing the baselines. n_evict_calls
    // is a single module-global, cumulative since module load and never reset,
    // so a zero baseline would make the first delta equal every eviction the
    // module has ever seen. That spikes the rate and forces a spurious widen
    // on the first epoch. Invisible in a campaign, where each config reloads
    // the module, and wrong everywhere else.
    replayable_faults->service_pool.adapt_last_evictions =
        uvm_lock_stat_sum(&g_uvm_lock_contention_stats.n_evict_calls);
    replayable_faults->service_pool.adapt_last_batches = replayable_faults->stats.num_batches;
    replayable_faults->service_pool.adapt_ewma_milli = 0;
    replayable_faults->service_pool.adapt_narrow_ticks = 0;

    atomic_set(&replayable_faults->service_pool.outstanding, 0);
    init_waitqueue_head(&replayable_faults->service_pool.done_wq);

    if (num_workers == 0)
        return NV_OK;

    replayable_faults->service_pool.queues =
        uvm_kvmalloc_zero(num_workers * sizeof(*replayable_faults->service_pool.queues));
    if (!replayable_faults->service_pool.queues)
        return NV_ERR_NO_MEMORY;

    replayable_faults->service_pool.workers =
        uvm_kvmalloc_zero((num_workers + 1) * sizeof(*replayable_faults->service_pool.workers));
    if (!replayable_faults->service_pool.workers)
        return NV_ERR_NO_MEMORY;

    // Worst case every coalesced fault is its own span (random access
    // patterns get close to this), so size the span scratch like the fault
    // caches: one entry per possible fault
    replayable_faults->service_pool.spans =
        uvm_kvmalloc_zero(replayable_faults->max_faults * sizeof(*replayable_faults->service_pool.spans));
    if (!replayable_faults->service_pool.spans)
        return NV_ERR_NO_MEMORY;

    // Slot 0 is the dispatcher's own worker state; it runs inline on the
    // bottom-half thread and has no queue
    for (i = 0; i < num_workers + 1; i++) {
        uvm_fault_service_worker_t *worker = &replayable_faults->service_pool.workers[i];

        worker->block_service_context.block_context = uvm_va_block_context_alloc(NULL);
        if (!worker->block_service_context.block_context)
            return NV_ERR_NO_MEMORY;

        uvm_tracker_init(&worker->tracker);
        worker->slot = i;
        worker->parent_gpu = parent_gpu;
        nv_kthread_q_item_init(&worker->q_item, fault_service_worker_entry, worker);
    }

    for (i = 0; i < num_workers; i++) {
        snprintf(kthread_name, sizeof(kthread_name), "UVM GPU%u FSVC%u", uvm_parent_id_value(parent_gpu->id), i + 1);

        status = uvm_isr_init_queue_on_node(&replayable_faults->service_pool.queues[i],
                                            kthread_name,
                                            parent_gpu->closest_cpu_numa_node);
        if (status != NV_OK) {
            UVM_ERR_PRINT("Failed in nv_kthread_q_init for fault service queue %u: %s, GPU %s\n",
                          i,
                          nvstatusToString(status),
                          uvm_parent_gpu_name(parent_gpu));
            return status;
        }
    }

    return NV_OK;
}

// Tear down the worker pool. Safe on a partially-initialized pool: the caller
// runs the deinit path on any init failure.
static void fault_service_pool_deinit(uvm_parent_gpu_t *parent_gpu)
{
    NvU32 i;
    uvm_replayable_fault_buffer_t *replayable_faults = &parent_gpu->fault_buffer.replayable;
    NvU32 num_workers = replayable_faults->service_pool.num_workers;

    UVM_ASSERT(atomic_read(&replayable_faults->service_pool.outstanding) == 0);

    if (replayable_faults->service_pool.queues) {
        // Safe on zero-initialized (never-started) queues, same as the
        // bottom-half queues in uvm_parent_gpu_deinit_isr()
        for (i = 0; i < num_workers; i++)
            nv_kthread_q_stop(&replayable_faults->service_pool.queues[i]);

        uvm_kvfree(replayable_faults->service_pool.queues);
        replayable_faults->service_pool.queues = NULL;
    }

    if (replayable_faults->service_pool.workers) {
        for (i = 0; i < num_workers + 1; i++) {
            uvm_fault_service_worker_t *worker = &replayable_faults->service_pool.workers[i];

            // The tracker is initialized right after a successful
            // block_context alloc, so that alloc doubles as the init marker
            if (worker->block_service_context.block_context) {
                uvm_tracker_deinit(&worker->tracker);
                uvm_va_block_context_free(worker->block_service_context.block_context);
            }
        }

        uvm_kvfree(replayable_faults->service_pool.workers);
        replayable_faults->service_pool.workers = NULL;
    }

    uvm_kvfree(replayable_faults->service_pool.spans);
    replayable_faults->service_pool.spans = NULL;
    replayable_faults->service_pool.num_workers = 0;
    replayable_faults->service_pool.active_workers = 0;
}

// There is no error handling in this function. The caller is in charge of
// calling fault_buffer_deinit_replayable_faults on failure.
static NV_STATUS fault_buffer_init_replayable_faults(uvm_parent_gpu_t *parent_gpu)
{
    NV_STATUS status = NV_OK;
    uvm_replayable_fault_buffer_t *replayable_faults = &parent_gpu->fault_buffer.replayable;
    uvm_fault_service_batch_context_t *batch_context = &replayable_faults->batch_service_context;

    UVM_ASSERT(parent_gpu->fault_buffer.rm_info.replayable.bufferSize %
               parent_gpu->fault_buffer_hal->entry_size(parent_gpu) == 0);

    replayable_faults->max_faults = parent_gpu->fault_buffer.rm_info.replayable.bufferSize /
                                    parent_gpu->fault_buffer_hal->entry_size(parent_gpu);

    // Check provided module parameter value
    parent_gpu->fault_buffer.max_batch_size = max(uvm_perf_fault_batch_count,
                                                  (NvU32)UVM_PERF_FAULT_BATCH_COUNT_MIN);
    parent_gpu->fault_buffer.max_batch_size = min(parent_gpu->fault_buffer.max_batch_size,
                                                  replayable_faults->max_faults);

    if (parent_gpu->fault_buffer.max_batch_size != uvm_perf_fault_batch_count) {
        UVM_INFO_PRINT("Invalid uvm_perf_fault_batch_count value on GPU %s: %u. Valid range [%u:%u] Using %u instead\n",
                       uvm_parent_gpu_name(parent_gpu),
                       uvm_perf_fault_batch_count,
                       UVM_PERF_FAULT_BATCH_COUNT_MIN,
                       replayable_faults->max_faults,
                       parent_gpu->fault_buffer.max_batch_size);
    }

    batch_context->fault_cache = uvm_kvmalloc_zero(replayable_faults->max_faults * sizeof(*batch_context->fault_cache));
    if (!batch_context->fault_cache)
        return NV_ERR_NO_MEMORY;

    // fault_cache is used to signal that the tracker was initialized.
    uvm_tracker_init(&replayable_faults->replay_tracker);

    batch_context->ordered_fault_cache = uvm_kvmalloc_zero(replayable_faults->max_faults *
                                                           sizeof(*batch_context->ordered_fault_cache));
    if (!batch_context->ordered_fault_cache)
        return NV_ERR_NO_MEMORY;

    // This value must be initialized by HAL
    UVM_ASSERT(replayable_faults->utlb_count > 0);

    batch_context->utlbs = uvm_kvmalloc_zero(replayable_faults->utlb_count * sizeof(*batch_context->utlbs));
    if (!batch_context->utlbs)
        return NV_ERR_NO_MEMORY;

    batch_context->max_utlb_id = 0;

    uvm_spin_lock_init(&batch_context->fatal_lock, UVM_LOCK_ORDER_LEAF);

    status = uvm_rm_locked_call(nvUvmInterfaceOwnPageFaultIntr(parent_gpu->rm_device, NV_TRUE));
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to take page fault ownership from RM: %s, GPU %s\n",
                      nvstatusToString(status),
                      uvm_parent_gpu_name(parent_gpu));
        return status;
    }

    replayable_faults->replay_policy = uvm_perf_fault_replay_policy < UVM_PERF_FAULT_REPLAY_POLICY_MAX?
                                           uvm_perf_fault_replay_policy:
                                           UVM_PERF_FAULT_REPLAY_POLICY_DEFAULT;

    if (replayable_faults->replay_policy != uvm_perf_fault_replay_policy) {
        UVM_INFO_PRINT("Invalid uvm_perf_fault_replay_policy value on GPU %s: %d. Using %d instead\n",
                       uvm_parent_gpu_name(parent_gpu),
                       uvm_perf_fault_replay_policy,
                       replayable_faults->replay_policy);
    }

    replayable_faults->replay_update_put_ratio = min(uvm_perf_fault_replay_update_put_ratio, 100u);
    if (replayable_faults->replay_update_put_ratio != uvm_perf_fault_replay_update_put_ratio) {
        UVM_INFO_PRINT("Invalid uvm_perf_fault_replay_update_put_ratio value on GPU %s: %u. Using %u instead\n",
                       uvm_parent_gpu_name(parent_gpu),
                       uvm_perf_fault_replay_update_put_ratio,
                       replayable_faults->replay_update_put_ratio);
    }

    status = fault_service_pool_init(parent_gpu);
    if (status != NV_OK)
        return status;

    // Re-enable fault prefetching just in case it was disabled in a previous run
    parent_gpu->fault_buffer.prefetch_faults_enabled = true;

    fault_buffer_reinit_replayable_faults(parent_gpu);

    return NV_OK;
}

static void fault_buffer_deinit_replayable_faults(uvm_parent_gpu_t *parent_gpu)
{
    uvm_replayable_fault_buffer_t *replayable_faults = &parent_gpu->fault_buffer.replayable;
    uvm_fault_service_batch_context_t *batch_context = &replayable_faults->batch_service_context;

    // Stop the service-pool kthreads before any batch state is freed
    fault_service_pool_deinit(parent_gpu);

    if (batch_context->fault_cache) {
        UVM_ASSERT(uvm_tracker_is_empty(&replayable_faults->replay_tracker));
        uvm_tracker_deinit(&replayable_faults->replay_tracker);
    }

    if (parent_gpu->fault_buffer.rm_info.faultBufferHandle) {
        // Re-enable prefetch faults in case we disabled them
        if (!parent_gpu->fault_buffer.prefetch_faults_enabled)
            parent_gpu->arch_hal->enable_prefetch_faults(parent_gpu);
    }

    uvm_kvfree(batch_context->fault_cache);
    uvm_kvfree(batch_context->ordered_fault_cache);
    uvm_kvfree(batch_context->utlbs);
    batch_context->fault_cache         = NULL;
    batch_context->ordered_fault_cache = NULL;
    batch_context->utlbs               = NULL;
}

NV_STATUS uvm_parent_gpu_fault_buffer_init(uvm_parent_gpu_t *parent_gpu)
{
    NV_STATUS status = NV_OK;

    uvm_assert_mutex_locked(&g_uvm_global.global_lock);

    status = uvm_rm_locked_call(nvUvmInterfaceInitFaultInfo(parent_gpu->rm_device,
                                                            &parent_gpu->fault_buffer.rm_info));
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to init fault buffer info from RM: %s, GPU %s\n",
                      nvstatusToString(status),
                      uvm_parent_gpu_name(parent_gpu));

        // nvUvmInterfaceInitFaultInfo may leave fields in rm_info populated
        // when it returns an error. Set the buffer handle to zero as it is
        // used by the deinitialization logic to determine if it was correctly
        // initialized.
        parent_gpu->fault_buffer.rm_info.faultBufferHandle = 0;
        goto fail;
    }

    status = fault_buffer_init_replayable_faults(parent_gpu);
    if (status != NV_OK)
        goto fail;

    status = uvm_parent_gpu_fault_buffer_init_non_replayable_faults(parent_gpu);
    if (status != NV_OK)
        goto fail;

    return NV_OK;

fail:
    uvm_parent_gpu_fault_buffer_deinit(parent_gpu);

    return status;
}

// Reinitialize state relevant to replayable fault handling after returning
// from a power management cycle.
void uvm_parent_gpu_fault_buffer_resume(uvm_parent_gpu_t *parent_gpu)
{
    fault_buffer_reinit_replayable_faults(parent_gpu);
}

void uvm_parent_gpu_fault_buffer_deinit(uvm_parent_gpu_t *parent_gpu)
{
    NV_STATUS status = NV_OK;

    uvm_assert_mutex_locked(&g_uvm_global.global_lock);

    uvm_parent_gpu_fault_buffer_deinit_non_replayable_faults(parent_gpu);

    fault_buffer_deinit_replayable_faults(parent_gpu);

    if (parent_gpu->fault_buffer.rm_info.faultBufferHandle) {
        status = uvm_rm_locked_call(nvUvmInterfaceOwnPageFaultIntr(parent_gpu->rm_device, NV_FALSE));
        UVM_ASSERT(status == NV_OK);

        uvm_rm_locked_call_void(nvUvmInterfaceDestroyFaultInfo(parent_gpu->rm_device,
                                                               &parent_gpu->fault_buffer.rm_info));

        parent_gpu->fault_buffer.rm_info.faultBufferHandle = 0;
    }
}

bool uvm_parent_gpu_replayable_faults_pending(uvm_parent_gpu_t *parent_gpu)
{
    uvm_replayable_fault_buffer_t *replayable_faults = &parent_gpu->fault_buffer.replayable;

    // Fast path 1: we left some faults unserviced in the buffer in the last
    // pass
    if (replayable_faults->cached_get != replayable_faults->cached_put)
        return true;

    // Fast path 2: read the valid bit of the fault buffer entry pointed by the
    // cached get pointer
    if (!parent_gpu->fault_buffer_hal->entry_is_valid(parent_gpu, replayable_faults->cached_get)) {
        // Slow path: read the put pointer from the GPU register via BAR0
        // over PCIe
        replayable_faults->cached_put = parent_gpu->fault_buffer_hal->read_put(parent_gpu);

        // No interrupt pending
        if (replayable_faults->cached_get == replayable_faults->cached_put)
            return false;
    }

    return true;
}

// Push a fault cancel method on the given client. Any failure during this
// operation may lead to application hang (requiring manual Ctrl+C from the
// user) or system crash (requiring reboot).
// In that case we log an error message.
//
// gpc_id and client_id aren't used if global_cancel is true.
//
// This function acquires both the given tracker and the replay tracker
static NV_STATUS push_cancel_on_gpu(uvm_gpu_t *gpu,
                                    uvm_gpu_phys_address_t instance_ptr,
                                    bool global_cancel,
                                    NvU32 gpc_id,
                                    NvU32 client_id,
                                    uvm_tracker_t *tracker)
{
    NV_STATUS status;
    uvm_push_t push;
    uvm_tracker_t *replay_tracker = &gpu->parent->fault_buffer.replayable.replay_tracker;

    UVM_ASSERT(tracker != NULL);

    status = uvm_tracker_add_tracker_safe(tracker, replay_tracker);
    if (status != NV_OK)
        return status;

    if (global_cancel) {
        status = uvm_push_begin_acquire(gpu->channel_manager,
                                        UVM_CHANNEL_TYPE_MEMOPS,
                                        tracker,
                                        &push,
                                        "Cancel targeting instance_ptr {0x%llx:%s}\n",
                                        instance_ptr.address,
                                        uvm_aperture_string(instance_ptr.aperture));
    }
    else {
        status = uvm_push_begin_acquire(gpu->channel_manager,
                                        UVM_CHANNEL_TYPE_MEMOPS,
                                        tracker,
                                        &push,
                                        "Cancel targeting instance_ptr {0x%llx:%s} gpc %u client %u\n",
                                        instance_ptr.address,
                                        uvm_aperture_string(instance_ptr.aperture),
                                        gpc_id,
                                        client_id);
    }

    UVM_ASSERT(status == NV_OK);
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to create push and acquire trackers before pushing cancel: %s, GPU %s\n",
                      nvstatusToString(status),
                      uvm_gpu_name(gpu));
        return status;
    }

    if (global_cancel)
        gpu->parent->host_hal->cancel_faults_global(&push, instance_ptr);
    else
        gpu->parent->host_hal->cancel_faults_targeted(&push, instance_ptr, gpc_id, client_id);

    // We don't need to put the cancel in the GPU replay tracker since we wait
    // on it immediately.
    status = uvm_push_end_and_wait(&push);

    UVM_ASSERT(status == NV_OK);
    if (status != NV_OK)
        UVM_ERR_PRINT("Failed to wait for pushed cancel: %s, GPU %s\n", nvstatusToString(status), uvm_gpu_name(gpu));

    // The cancellation is complete, so the input trackers must be complete too.
    uvm_tracker_clear(tracker);
    uvm_tracker_clear(replay_tracker);

    return status;
}

static NV_STATUS push_cancel_on_gpu_targeted(uvm_gpu_t *gpu,
                                             uvm_gpu_phys_address_t instance_ptr,
                                             NvU32 gpc_id,
                                             NvU32 client_id,
                                             uvm_tracker_t *tracker)
{
    return push_cancel_on_gpu(gpu, instance_ptr, false, gpc_id, client_id, tracker);
}

static NV_STATUS push_cancel_on_gpu_global(uvm_gpu_t *gpu, uvm_gpu_phys_address_t instance_ptr, uvm_tracker_t *tracker)
{
    UVM_ASSERT(!gpu->parent->smc.enabled);

    return push_cancel_on_gpu(gpu, instance_ptr, true, 0, 0, tracker);
}

static NV_STATUS cancel_fault_precise_va(uvm_fault_buffer_entry_t *fault_entry,
                                         uvm_fault_cancel_va_mode_t cancel_va_mode)
{
    NV_STATUS status;
    uvm_gpu_va_space_t *gpu_va_space;
    uvm_va_space_t *va_space = fault_entry->va_space;
    uvm_gpu_t *gpu = fault_entry->gpu;
    uvm_gpu_phys_address_t pdb;
    uvm_push_t push;
    uvm_replayable_fault_buffer_t *replayable_faults = &gpu->parent->fault_buffer.replayable;
    NvU64 offset;

    UVM_ASSERT(fault_entry->fatal_reason != UvmEventFatalReasonInvalid);
    UVM_ASSERT(!fault_entry->filtered);

    gpu_va_space = uvm_gpu_va_space_get(va_space, gpu);
    UVM_ASSERT(gpu_va_space);
    pdb = uvm_page_tree_pdb_address(&gpu_va_space->page_tables);

    // Record fatal fault event
    uvm_tools_record_gpu_fatal_fault(gpu->id, va_space, fault_entry, fault_entry->fatal_reason);

    status = uvm_push_begin_acquire(gpu->channel_manager,
                                    UVM_CHANNEL_TYPE_MEMOPS,
                                    &replayable_faults->replay_tracker,
                                    &push,
                                    "Precise cancel targeting PDB {0x%llx:%s} VA 0x%llx VEID %u with access type %s",
                                    pdb.address,
                                    uvm_aperture_string(pdb.aperture),
                                    fault_entry->fault_address,
                                    fault_entry->fault_source.ve_id,
                                    uvm_fault_access_type_string(fault_entry->fault_access_type));
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to create push and acquire replay tracker before pushing cancel: %s, GPU %s\n",
                      nvstatusToString(status),
                      uvm_gpu_name(gpu));
        return status;
    }

    // UVM aligns fault addresses to PAGE_SIZE as it is the smallest mapping
    // and coherence tracking granularity. However, the cancel method requires
    // the original address (4K-aligned) reported in the packet, which is lost
    // at this point. Since the access permissions are the same for the whole
    // 64K page, we issue a cancel per 4K range to make sure that the HW sees
    // the address reported in the packet.
    for (offset = 0; offset < PAGE_SIZE; offset += UVM_PAGE_SIZE_4K) {
        gpu->parent->host_hal->cancel_faults_va(&push, pdb, fault_entry, cancel_va_mode);
        fault_entry->fault_address += UVM_PAGE_SIZE_4K;
    }
    fault_entry->fault_address = UVM_PAGE_ALIGN_DOWN(fault_entry->fault_address - 1);

    // We don't need to put the cancel in the GPU replay tracker since we wait
    // on it immediately.
    status = uvm_push_end_and_wait(&push);
    if (status != NV_OK) {
        UVM_ERR_PRINT("Failed to wait for pushed VA global fault cancel: %s, GPU %s\n",
                      nvstatusToString(status), uvm_gpu_name(gpu));
    }

    uvm_tracker_clear(&replayable_faults->replay_tracker);

    return status;
}

static NV_STATUS push_replay_on_gpu(uvm_gpu_t *gpu,
                                    uvm_fault_replay_type_t type,
                                    uvm_fault_service_batch_context_t *batch_context)
{
    NV_STATUS status;
    uvm_push_t push;
    uvm_replayable_fault_buffer_t *replayable_faults = &gpu->parent->fault_buffer.replayable;
    uvm_tracker_t *tracker = NULL;

    if (batch_context)
        tracker = &batch_context->tracker;

    status = uvm_push_begin_acquire(gpu->channel_manager, UVM_CHANNEL_TYPE_MEMOPS, tracker, &push,
                                    "Replaying faults");
    if (status != NV_OK)
        return status;

    gpu->parent->host_hal->replay_faults(&push, type);

    // Do not count REPLAY_TYPE_START_ACK_ALL's toward the replay count.
    // REPLAY_TYPE_START_ACK_ALL's are issued for cancels, and the cancel
    // algorithm checks to make sure that no REPLAY_TYPE_START's have been
    // issued using batch_context->replays.
    if (batch_context && type != UVM_FAULT_REPLAY_TYPE_START_ACK_ALL) {
        uvm_tools_broadcast_replay(gpu, &push, batch_context->batch_id, UVM_FAULT_CLIENT_TYPE_GPC);
        ++batch_context->num_replays;
    }

    uvm_push_end(&push);

    // Add this push to the GPU's replay_tracker so cancel can wait on it.
    status = uvm_tracker_add_push_safe(&replayable_faults->replay_tracker, &push);

    if (uvm_procfs_is_debug_enabled()) {
        if (type == UVM_FAULT_REPLAY_TYPE_START)
            ++replayable_faults->stats.num_replays;
        else
            ++replayable_faults->stats.num_replays_ack_all;
    }

    return status;
}

static NV_STATUS push_replay_on_parent_gpu(uvm_parent_gpu_t *parent_gpu,
                                           uvm_fault_replay_type_t type,
                                           uvm_fault_service_batch_context_t *batch_context)
{
    uvm_gpu_t *gpu = uvm_parent_gpu_find_first_valid_gpu(parent_gpu);

    if (gpu)
        return push_replay_on_gpu(gpu, type, batch_context);

    return NV_OK;
}

static void write_get(uvm_parent_gpu_t *parent_gpu, NvU32 get)
{
    uvm_replayable_fault_buffer_t *replayable_faults = &parent_gpu->fault_buffer.replayable;

    UVM_ASSERT(uvm_sem_is_locked(&parent_gpu->isr.replayable_faults.service_lock));

    // Write get on the GPU only if it's changed.
    if (replayable_faults->cached_get == get)
        return;

    replayable_faults->cached_get = get;

    // Update get pointer on the GPU
    parent_gpu->fault_buffer_hal->write_get(parent_gpu, get);
}

// In Confidential Computing GSP-RM owns the HW replayable fault buffer.
// Flushing the fault buffer implies flushing both the HW buffer (using a RM
// API), and the SW buffer accessible by UVM ("shadow" buffer).
//
// The HW buffer needs to be flushed first. This is because, once that flush
// completes, any faults that were present in the HW buffer have been moved to
// the shadow buffer, or have been discarded by RM.
static NV_STATUS hw_fault_buffer_flush_locked(uvm_parent_gpu_t *parent_gpu, hw_fault_buffer_flush_mode_t flush_mode)
{
    NV_STATUS status;
    NvBool is_flush_mode_move;

    UVM_ASSERT(uvm_sem_is_locked(&parent_gpu->isr.replayable_faults.service_lock));
    UVM_ASSERT((flush_mode == HW_FAULT_BUFFER_FLUSH_MODE_MOVE) || (flush_mode == HW_FAULT_BUFFER_FLUSH_MODE_DISCARD));

    if (!g_uvm_global.conf_computing_enabled)
        return NV_OK;

    is_flush_mode_move = (NvBool) (flush_mode == HW_FAULT_BUFFER_FLUSH_MODE_MOVE);
    status = nvUvmInterfaceFlushReplayableFaultBuffer(&parent_gpu->fault_buffer.rm_info, is_flush_mode_move);

    UVM_ASSERT(status == NV_OK);

    return status;
}

static void fault_buffer_skip_replayable_entry(uvm_parent_gpu_t *parent_gpu, NvU32 index)
{
    UVM_ASSERT(parent_gpu->fault_buffer_hal->entry_is_valid(parent_gpu, index));

    // Flushed faults are never decrypted, but the decryption IV associated with
    // replayable faults still requires manual adjustment so it is kept in sync
    // with the encryption IV on the GSP-RM's side.
    if (g_uvm_global.conf_computing_enabled)
        uvm_conf_computing_fault_increment_decrypt_iv(parent_gpu);

    parent_gpu->fault_buffer_hal->entry_clear_valid(parent_gpu, index);
}

static NV_STATUS fault_buffer_flush_locked(uvm_parent_gpu_t *parent_gpu,
                                           uvm_gpu_t *gpu,
                                           uvm_gpu_buffer_flush_mode_t flush_mode,
                                           uvm_fault_replay_type_t fault_replay,
                                           uvm_fault_service_batch_context_t *batch_context)
{
    NvU32 get;
    NvU32 put;
    uvm_spin_loop_t spin;
    uvm_replayable_fault_buffer_t *replayable_faults = &parent_gpu->fault_buffer.replayable;
    NV_STATUS status;

    UVM_ASSERT(uvm_sem_is_locked(&parent_gpu->isr.replayable_faults.service_lock));

    // Wait for the prior replay to flush out old fault messages
    if (flush_mode == UVM_GPU_BUFFER_FLUSH_MODE_WAIT_UPDATE_PUT) {
        status = uvm_tracker_wait(&replayable_faults->replay_tracker);
        if (status != NV_OK)
            return status;
    }

    // Read PUT pointer from the GPU if requested
    if (flush_mode == UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT || flush_mode == UVM_GPU_BUFFER_FLUSH_MODE_WAIT_UPDATE_PUT) {
        status = hw_fault_buffer_flush_locked(parent_gpu, HW_FAULT_BUFFER_FLUSH_MODE_DISCARD);
        if (status != NV_OK)
            return status;

        replayable_faults->cached_put = parent_gpu->fault_buffer_hal->read_put(parent_gpu);
    }

    get = replayable_faults->cached_get;
    put = replayable_faults->cached_put;

    while (get != put) {
        // Wait until valid bit is set
        UVM_SPIN_WHILE(!parent_gpu->fault_buffer_hal->entry_is_valid(parent_gpu, get), &spin) {
            // Channels might be idle (e.g. in teardown) so check for errors
            // actively. In that case the gpu pointer is valid.
            status = gpu ? uvm_channel_manager_check_errors(gpu->channel_manager) : uvm_global_get_status();
            if (status != NV_OK) {
                write_get(parent_gpu, get);
                return status;
            }
        }

        fault_buffer_skip_replayable_entry(parent_gpu, get);
        ++get;
        if (get == replayable_faults->max_faults)
            get = 0;
    }

    write_get(parent_gpu, get);

    // Issue fault replay
    if (gpu)
        return push_replay_on_gpu(gpu, fault_replay, batch_context);

    return push_replay_on_parent_gpu(parent_gpu, fault_replay, batch_context);
}

NV_STATUS uvm_gpu_replayable_buffer_flush(uvm_gpu_t *gpu)
{
    NV_STATUS status = NV_OK;

    // Disables replayable fault interrupts and fault servicing
    uvm_parent_gpu_replayable_faults_isr_lock(gpu->parent);

    status = fault_buffer_flush_locked(gpu->parent,
                                       gpu,
                                       UVM_GPU_BUFFER_FLUSH_MODE_WAIT_UPDATE_PUT,
                                       UVM_FAULT_REPLAY_TYPE_START,
                                       NULL);

    // This will trigger the top half to start servicing faults again, if the
    // replay brought any back in
    uvm_parent_gpu_replayable_faults_isr_unlock(gpu->parent);
    return status;
}

static inline int cmp_fault_instance_ptr(const uvm_fault_buffer_entry_t *a,
                                         const uvm_fault_buffer_entry_t *b)
{
    int result = uvm_gpu_phys_addr_cmp(a->instance_ptr, b->instance_ptr);
    // We need to sort by {instance_ptr + subctx_id} pair since it can
    // map to a different VA space.
    if (result != 0)
        return result;
    return UVM_CMP_DEFAULT(a->fault_source.ve_id, b->fault_source.ve_id);
}

// Compare two VA spaces
static inline int cmp_va_space(const uvm_va_space_t *a, const uvm_va_space_t *b)
{
    return UVM_CMP_DEFAULT(a, b);
}

// Compare two GPUs
static inline int cmp_gpu(const uvm_gpu_t *a, const uvm_gpu_t *b)
{
    NvU32 id_a = a ? uvm_id_value(a->id) : 0;
    NvU32 id_b = b ? uvm_id_value(b->id) : 0;

    return UVM_CMP_DEFAULT(id_a, id_b);
}

// Compare two virtual addresses
static inline int cmp_addr(NvU64 a, NvU64 b)
{
    return UVM_CMP_DEFAULT(a, b);
}

// Compare two fault access types
static inline int cmp_access_type(uvm_fault_access_type_t a, uvm_fault_access_type_t b)
{
    UVM_ASSERT(a >= 0 && a < UVM_FAULT_ACCESS_TYPE_COUNT);
    UVM_ASSERT(b >= 0 && b < UVM_FAULT_ACCESS_TYPE_COUNT);

    // Check that fault access type enum values are ordered by "intrusiveness"
    BUILD_BUG_ON(UVM_FAULT_ACCESS_TYPE_ATOMIC_STRONG <= UVM_FAULT_ACCESS_TYPE_ATOMIC_WEAK);
    BUILD_BUG_ON(UVM_FAULT_ACCESS_TYPE_ATOMIC_WEAK <= UVM_FAULT_ACCESS_TYPE_WRITE);
    BUILD_BUG_ON(UVM_FAULT_ACCESS_TYPE_WRITE <= UVM_FAULT_ACCESS_TYPE_READ);
    BUILD_BUG_ON(UVM_FAULT_ACCESS_TYPE_READ <= UVM_FAULT_ACCESS_TYPE_PREFETCH);

    return b - a;
}

typedef enum
{
    // Fetch a batch of faults from the buffer. Stop at the first entry that is
    // not ready yet
    FAULT_FETCH_MODE_BATCH_READY,

    // Fetch all faults in the buffer before PUT. Wait for all faults to become
    // ready
    FAULT_FETCH_MODE_ALL,
} fault_fetch_mode_t;

static void fetch_fault_buffer_merge_entry(uvm_fault_buffer_entry_t *current_entry,
                                           uvm_fault_buffer_entry_t *last_entry)
{
    UVM_ASSERT(last_entry->num_instances > 0);

    ++last_entry->num_instances;
    uvm_fault_access_type_mask_set(&last_entry->access_type_mask, current_entry->fault_access_type);

    if (current_entry->fault_access_type > last_entry->fault_access_type) {
        // If the new entry has a higher access type, it becomes the
        // fault to be serviced. Add the previous one to the list of instances
        current_entry->access_type_mask = last_entry->access_type_mask;
        current_entry->num_instances = last_entry->num_instances;
        last_entry->filtered = true;

        // We only merge faults from different uTLBs if the new fault has an
        // access type with the same or lower level of intrusiveness.
        UVM_ASSERT(current_entry->fault_source.utlb_id == last_entry->fault_source.utlb_id);

        list_replace(&last_entry->merged_instances_list, &current_entry->merged_instances_list);
        list_add(&last_entry->merged_instances_list, &current_entry->merged_instances_list);
    }
    else {
        // Add the new entry to the list of instances for reporting purposes
        current_entry->filtered = true;
        list_add(&current_entry->merged_instances_list, &last_entry->merged_instances_list);
    }
}

static bool fetch_fault_buffer_try_merge_entry(uvm_fault_buffer_entry_t *current_entry,
                                               uvm_fault_service_batch_context_t *batch_context,
                                               uvm_fault_utlb_info_t *current_tlb,
                                               bool is_same_instance_ptr)
{
    uvm_fault_buffer_entry_t *last_tlb_entry = current_tlb->last_fault;
    uvm_fault_buffer_entry_t *last_global_entry = batch_context->last_fault;

    // Check the last coalesced fault and the coalesced fault that was
    // originated from this uTLB
    const bool is_last_tlb_fault = current_tlb->num_pending_faults > 0 &&
                                   cmp_fault_instance_ptr(current_entry, last_tlb_entry) == 0 &&
                                   current_entry->fault_address == last_tlb_entry->fault_address;

    // We only merge faults from different uTLBs if the new fault has an
    // access type with the same or lower level of intrusiveness. This is to
    // avoid having to update num_pending_faults on both uTLBs and recomputing
    // last_fault.
    const bool is_last_fault = is_same_instance_ptr &&
                               current_entry->fault_address == last_global_entry->fault_address &&
                               current_entry->fault_access_type <= last_global_entry->fault_access_type;

    if (is_last_tlb_fault) {
        fetch_fault_buffer_merge_entry(current_entry, last_tlb_entry);
        if (current_entry->fault_access_type > last_tlb_entry->fault_access_type)
            current_tlb->last_fault = current_entry;

        return true;
    }
    else if (is_last_fault) {
        fetch_fault_buffer_merge_entry(current_entry, last_global_entry);
        if (current_entry->fault_access_type > last_global_entry->fault_access_type)
            batch_context->last_fault = current_entry;

        return true;
    }

    return false;
}

// Fetch entries from the fault buffer, decode them and store them in the batch
// context. We implement the fetch modes described above.
//
// When possible, we coalesce duplicate entries to minimize the fault handling
// overhead. Basically, we merge faults with the same instance pointer and page
// virtual address. We keep track of the last fault per uTLB to detect
// duplicates due to local reuse and the last fault in the whole batch to
// detect reuse across CTAs.
//
// We will service the first fault entry with the most "intrusive" (atomic >
// write > read > prefetch) access type*. That fault entry is called the
// "representative". The rest of filtered faults have the "filtered" flag set
// and are added to a list in the representative fault entry for reporting
// purposes. The representative fault entry also contains a mask with all the
// access types that produced a fault on the page.
//
// *We only merge faults from different uTLBs if the new fault has an access
// type with the same or lower level of intrusiveness.
static NV_STATUS fetch_fault_buffer_entries(uvm_parent_gpu_t *parent_gpu,
                                            uvm_fault_service_batch_context_t *batch_context,
                                            fault_fetch_mode_t fetch_mode)
{
    NvU32 get;
    NvU32 put;
    NvU32 fault_index;
    NvU32 num_coalesced_faults;
    NvU32 utlb_id;
    uvm_fault_buffer_entry_t *fault_cache;
    uvm_spin_loop_t spin;
    NV_STATUS status = NV_OK;
    uvm_replayable_fault_buffer_t *replayable_faults = &parent_gpu->fault_buffer.replayable;
    const bool may_filter = uvm_perf_fault_coalesce;

    UVM_ASSERT(uvm_sem_is_locked(&parent_gpu->isr.replayable_faults.service_lock));

    fault_cache = batch_context->fault_cache;

    get = replayable_faults->cached_get;

    // Read put pointer from GPU and cache it
    if (get == replayable_faults->cached_put)
        replayable_faults->cached_put = parent_gpu->fault_buffer_hal->read_put(parent_gpu);

    put = replayable_faults->cached_put;

    batch_context->is_single_instance_ptr = true;
    batch_context->last_fault = NULL;

    fault_index = 0;
    num_coalesced_faults = 0;

    // Clear uTLB counters
    for (utlb_id = 0; utlb_id <= batch_context->max_utlb_id; ++utlb_id) {
        batch_context->utlbs[utlb_id].num_pending_faults = 0;
        batch_context->utlbs[utlb_id].has_fatal_faults = false;
    }
    batch_context->max_utlb_id = 0;

    if (get == put)
        goto done;

    // Parse until get != put and have enough space to cache.
    while ((get != put) &&
           (fetch_mode == FAULT_FETCH_MODE_ALL || fault_index < parent_gpu->fault_buffer.max_batch_size)) {
        bool is_same_instance_ptr = true;
        uvm_fault_buffer_entry_t *current_entry = &fault_cache[fault_index];
        uvm_fault_utlb_info_t *current_tlb;

        // We cannot just wait for the last entry (the one pointed by put) to
        // become valid, we have to do it individually since entries can be
        // written out of order
        UVM_SPIN_WHILE(!parent_gpu->fault_buffer_hal->entry_is_valid(parent_gpu, get), &spin) {
            // We have some entry to work on. Let's do the rest later.
            if (fetch_mode == FAULT_FETCH_MODE_BATCH_READY && fault_index > 0)
                goto done;

            status = uvm_global_get_status();
            if (status != NV_OK)
                goto done;
        }

        // Prevent later accesses being moved above the read of the valid bit
        smp_mb__after_atomic();

        // Got valid bit set. Let's cache.
        status = parent_gpu->fault_buffer_hal->parse_replayable_entry(parent_gpu, get, current_entry);
        if (status != NV_OK)
            goto done;

        // The GPU aligns the fault addresses to 4k, but all of our tracking is
        // done in PAGE_SIZE chunks which might be larger.
        current_entry->fault_address = UVM_PAGE_ALIGN_DOWN(current_entry->fault_address);

        // Make sure that all fields in the entry are properly initialized
        current_entry->is_fatal = (current_entry->fault_type >= UVM_FAULT_TYPE_FATAL);

        if (current_entry->is_fatal) {
            // Record the fatal fault event later as we need the va_space locked
            current_entry->fatal_reason = UvmEventFatalReasonInvalidFaultType;
        }
        else {
            current_entry->fatal_reason = UvmEventFatalReasonInvalid;
        }

        current_entry->va_space = NULL;
        current_entry->gpu = NULL;
        current_entry->filtered = false;
        current_entry->replayable.cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_ALL;

        if (current_entry->fault_source.utlb_id > batch_context->max_utlb_id) {
            UVM_ASSERT(current_entry->fault_source.utlb_id < replayable_faults->utlb_count);
            batch_context->max_utlb_id = current_entry->fault_source.utlb_id;
        }

        current_tlb = &batch_context->utlbs[current_entry->fault_source.utlb_id];

        if (fault_index > 0) {
            UVM_ASSERT(batch_context->last_fault);
            is_same_instance_ptr = cmp_fault_instance_ptr(current_entry, batch_context->last_fault) == 0;

            // Coalesce duplicate faults when possible
            if (may_filter && !current_entry->is_fatal) {
                bool merged = fetch_fault_buffer_try_merge_entry(current_entry,
                                                                 batch_context,
                                                                 current_tlb,
                                                                 is_same_instance_ptr);
                if (merged)
                    goto next_fault;
            }
        }

        if (batch_context->is_single_instance_ptr && !is_same_instance_ptr)
            batch_context->is_single_instance_ptr = false;

        current_entry->num_instances = 1;
        current_entry->access_type_mask = uvm_fault_access_type_mask_bit(current_entry->fault_access_type);
        INIT_LIST_HEAD(&current_entry->merged_instances_list);

        ++current_tlb->num_pending_faults;
        current_tlb->last_fault = current_entry;
        batch_context->last_fault = current_entry;

        ++num_coalesced_faults;

    next_fault:
        ++fault_index;
        ++get;
        if (get == replayable_faults->max_faults)
            get = 0;
    }

done:
    write_get(parent_gpu, get);

    batch_context->num_cached_faults = fault_index;
    batch_context->num_coalesced_faults = num_coalesced_faults;

    return status;
}

// Sort comparator for pointers to fault buffer entries that sorts by
// instance pointer
static int cmp_sort_fault_entry_by_instance_ptr(const void *_a, const void *_b)
{
    const uvm_fault_buffer_entry_t **a = (const uvm_fault_buffer_entry_t **)_a;
    const uvm_fault_buffer_entry_t **b = (const uvm_fault_buffer_entry_t **)_b;

    return cmp_fault_instance_ptr(*a, *b);
}

// Sort comparator for pointers to fault buffer entries that sorts by va_space,
// GPU ID, fault address, and fault access type.
static int cmp_sort_fault_entry_by_va_space_gpu_address_access_type(const void *_a, const void *_b)
{
    const uvm_fault_buffer_entry_t **a = (const uvm_fault_buffer_entry_t **)_a;
    const uvm_fault_buffer_entry_t **b = (const uvm_fault_buffer_entry_t **)_b;

    int result;

    result = cmp_va_space((*a)->va_space, (*b)->va_space);
    if (result != 0)
        return result;

    result = cmp_gpu((*a)->gpu, (*b)->gpu);
    if (result != 0)
        return result;

    result = cmp_addr((*a)->fault_address, (*b)->fault_address);
    if (result != 0)
        return result;

    return cmp_access_type((*a)->fault_access_type, (*b)->fault_access_type);
}

// Translate all instance pointers to a VA space and GPU instance. Since the
// buffer is ordered by instance_ptr, we minimize the number of translations.
//
// This function returns NV_WARN_MORE_PROCESSING_REQUIRED if a fault buffer
// flush occurred and executed successfully, or the error code if it failed.
// NV_OK otherwise.
static NV_STATUS translate_instance_ptrs(uvm_parent_gpu_t *parent_gpu,
                                         uvm_fault_service_batch_context_t *batch_context)
{
    NvU32 i;
    NV_STATUS status;

    for (i = 0; i < batch_context->num_coalesced_faults; ++i) {
        uvm_fault_buffer_entry_t *current_entry;

        current_entry = batch_context->ordered_fault_cache[i];

        // If this instance pointer matches the previous instance pointer, just
        // copy over the already-translated va_space and move on.
        if (i != 0 && cmp_fault_instance_ptr(current_entry, batch_context->ordered_fault_cache[i - 1]) == 0) {
            current_entry->va_space = batch_context->ordered_fault_cache[i - 1]->va_space;
            current_entry->gpu = batch_context->ordered_fault_cache[i - 1]->gpu;
            continue;
        }

        status = uvm_parent_gpu_fault_entry_to_va_space(parent_gpu,
                                                        current_entry,
                                                        &current_entry->va_space,
                                                        &current_entry->gpu);
        if (status != NV_OK) {
            uvm_gpu_t *gpu = NULL;

            if (status == NV_ERR_PAGE_TABLE_NOT_AVAIL) {
                // The channel is valid but the subcontext is not. This can only
                // happen if the subcontext is torn down before its work is
                // complete while other subcontexts in the same TSG are still
                // executing which means some GPU is still valid under that
                // parent GPU. This is a violation of the programming model. We
                // have limited options since the VA space is gone, meaning we
                // can't target the PDB for cancel even if we wanted to. So
                // we'll just throw away precise attribution and cancel this
                // fault using the SW method, which validates that the intended
                // context (TSG) is still running so we don't cancel an innocent
                // context.
                gpu = uvm_parent_gpu_find_first_valid_gpu(parent_gpu);

                UVM_ASSERT(!current_entry->va_space);
                UVM_ASSERT(gpu);
                UVM_ASSERT(gpu->max_subcontexts > 0);

                if (parent_gpu->smc.enabled) {
                    status = push_cancel_on_gpu_targeted(gpu,
                                                         current_entry->instance_ptr,
                                                         current_entry->fault_source.gpc_id,
                                                         current_entry->fault_source.client_id,
                                                         &batch_context->tracker);
                }
                else {
                    status = push_cancel_on_gpu_global(gpu, current_entry->instance_ptr, &batch_context->tracker);
                }

                if (status != NV_OK)
                    return status;

                // Fall through and let the flush restart fault processing
            }
            else {
                UVM_ASSERT(status == NV_ERR_INVALID_CHANNEL);
            }

            // If the channel is gone then we're looking at a stale fault entry.
            // The fault must have been resolved already (serviced or
            // cancelled), so we can just flush the fault buffer.
            //
            // No need to use UVM_GPU_BUFFER_FLUSH_MODE_WAIT_UPDATE_PUT since
            // there was a context preemption for the entries we want to flush,
            // meaning PUT must reflect them.
            status = fault_buffer_flush_locked(parent_gpu,
                                               gpu,
                                               UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT,
                                               UVM_FAULT_REPLAY_TYPE_START,
                                               batch_context);
            if (status != NV_OK)
                 return status;

            return NV_WARN_MORE_PROCESSING_REQUIRED;
        }
        else {
            UVM_ASSERT(current_entry->va_space);
            UVM_ASSERT(current_entry->gpu);
        }
    }

    return NV_OK;
}

// Fault cache preprocessing for fault coalescing
//
// This function generates an ordered view of the given fault_cache in which
// faults are sorted by VA space, fault address (aligned to 4K) and access type
// "intrusiveness". In order to minimize the number of instance_ptr to VA space
// translations we perform a first sort by instance_ptr.
//
// This function returns NV_WARN_MORE_PROCESSING_REQUIRED if a fault buffer
// flush occurred during instance_ptr translation and executed successfully, or
// the error code if it failed. NV_OK otherwise.
//
// Current scheme:
// 1) sort by instance_ptr
// 2) translate all instance_ptrs to VA spaces
// 3) sort by va_space, GPU ID, fault address (fault_address is page-aligned at
//    this point) and access type.
static NV_STATUS preprocess_fault_batch(uvm_parent_gpu_t *parent_gpu,
                                        uvm_fault_service_batch_context_t *batch_context)
{
    NV_STATUS status;
    NvU32 i, j;
    uvm_fault_buffer_entry_t **ordered_fault_cache = batch_context->ordered_fault_cache;

    UVM_ASSERT(batch_context->num_coalesced_faults > 0);
    UVM_ASSERT(batch_context->num_cached_faults >= batch_context->num_coalesced_faults);

    // Generate an ordered view of the fault cache in ordered_fault_cache.
    // We sort the pointers, not the entries in fault_cache

    // Initialize pointers before they are sorted. We only sort one instance per
    // coalesced fault
    for (i = 0, j = 0; i < batch_context->num_cached_faults; ++i) {
        if (!batch_context->fault_cache[i].filtered)
            ordered_fault_cache[j++] = &batch_context->fault_cache[i];
    }
    UVM_ASSERT(j == batch_context->num_coalesced_faults);

    // 1) if the fault batch contains more than one, sort by instance_ptr
    if (!batch_context->is_single_instance_ptr) {
        sort(ordered_fault_cache,
             batch_context->num_coalesced_faults,
             sizeof(*ordered_fault_cache),
             cmp_sort_fault_entry_by_instance_ptr,
             NULL);
    }

    // 2) translate all instance_ptrs to VA spaces
    status = translate_instance_ptrs(parent_gpu, batch_context);
    if (status != NV_OK)
        return status;

    // 3) sort by va_space, GPU ID, fault address (GPU already reports
    // 4K-aligned address), and access type.
    sort(ordered_fault_cache,
         batch_context->num_coalesced_faults,
         sizeof(*ordered_fault_cache),
         cmp_sort_fault_entry_by_va_space_gpu_address_access_type,
         NULL);

    return NV_OK;
}

static bool check_fault_entry_duplicate(const uvm_fault_buffer_entry_t *current_entry,
                                        const uvm_fault_buffer_entry_t *previous_entry)
{
    bool is_duplicate = false;

    if (previous_entry) {
        is_duplicate = (current_entry->va_space == previous_entry->va_space) &&
                       (current_entry->fault_address == previous_entry->fault_address);
    }

    return is_duplicate;
}

static void update_batch_and_notify_fault(uvm_gpu_t *gpu,
                                          uvm_fault_service_batch_context_t *batch_context,
                                          uvm_va_block_t *va_block,
                                          uvm_processor_id_t preferred_location,
                                          uvm_fault_buffer_entry_t *current_entry,
                                          bool is_duplicate)
{
    if (is_duplicate)
        atomic_add(current_entry->num_instances, &batch_context->num_duplicate_faults);
    else
        atomic_add(current_entry->num_instances - 1, &batch_context->num_duplicate_faults);

    uvm_perf_event_notify_gpu_fault(&current_entry->va_space->perf_events,
                                    va_block,
                                    gpu->id,
                                    preferred_location,
                                    current_entry,
                                    batch_context->batch_id,
                                    is_duplicate);
}

static void mark_fault_invalid_prefetch(uvm_fault_service_batch_context_t *batch_context,
                                        uvm_fault_buffer_entry_t *fault_entry)
{
    fault_entry->is_invalid_prefetch = true;

    // For block faults, the following counter might be updated more than once
    // for the same fault if block_context->num_retries > 0. As a result, this
    // counter might be higher than the actual count. In order for this counter
    // to be always accurate, block_context needs to passed down the stack from
    // all callers. But since num_retries > 0 case is uncommon and imprecise
    // invalid_prefetch counter doesn't affect functionality (other than
    // disabling prefetching if the counter indicates lots of invalid prefetch
    // faults), this is ok.
    atomic_add(fault_entry->num_instances, &batch_context->num_invalid_prefetch_faults);
}

static void mark_fault_throttled(uvm_fault_service_batch_context_t *batch_context,
                                 uvm_fault_buffer_entry_t *fault_entry)
{
    fault_entry->is_throttled = true;
    batch_context->has_throttled_faults = true;
}

// First-wins publication of the fatal (va_space, gpu) pair. The spinlock
// keeps the pair coherent when parallel servicing workers mark fatal faults
// concurrently; the serial path takes it uncontended.
static void fault_batch_publish_fatal_va_space(uvm_fault_service_batch_context_t *batch_context,
                                               uvm_va_space_t *va_space,
                                               uvm_gpu_t *gpu)
{
    UVM_ASSERT(va_space);

    uvm_spin_lock(&batch_context->fatal_lock);

    if (!batch_context->fatal_va_space) {
        batch_context->fatal_va_space = va_space;
        batch_context->fatal_gpu = gpu;
    }

    uvm_spin_unlock(&batch_context->fatal_lock);
}

static void mark_fault_fatal(uvm_fault_service_batch_context_t *batch_context,
                             uvm_fault_buffer_entry_t *fault_entry,
                             UvmEventFatalReason fatal_reason,
                             uvm_fault_cancel_va_mode_t cancel_va_mode)
{
    uvm_fault_utlb_info_t *utlb = &batch_context->utlbs[fault_entry->fault_source.utlb_id];

    fault_entry->is_fatal = true;
    fault_entry->fatal_reason = fatal_reason;
    fault_entry->replayable.cancel_va_mode = cancel_va_mode;

    utlb->has_fatal_faults = true;

    fault_batch_publish_fatal_va_space(batch_context, fault_entry->va_space, fault_entry->gpu);
}

static void fault_entry_duplicate_flags(uvm_fault_service_batch_context_t *batch_context,
                                        uvm_fault_buffer_entry_t *current_entry,
                                        const uvm_fault_buffer_entry_t *previous_entry)
{
    UVM_ASSERT(previous_entry);
    UVM_ASSERT(check_fault_entry_duplicate(current_entry, previous_entry));

    // Propagate the is_invalid_prefetch flag across all prefetch faults
    // on the page
    if (previous_entry->is_invalid_prefetch)
        mark_fault_invalid_prefetch(batch_context, current_entry);

    // If a page is throttled, all faults on the page must be skipped
    if (previous_entry->is_throttled)
        mark_fault_throttled(batch_context, current_entry);
}

// This function computes the maximum access type that can be serviced for the
// reported fault instances given the logical permissions of the VA range. If
// none of the fault instances can be serviced UVM_FAULT_ACCESS_TYPE_COUNT is
// returned instead.
//
// In the case that there are faults that cannot be serviced, this function
// also sets the flags required for fault cancellation. Prefetch faults do not
// need to be cancelled since they disappear on replay.
//
// The UVM driver considers two scenarios for logical permissions violation:
// - All access types are invalid. For example, when faulting from a processor
// that doesn't have access to the preferred location of a range group when it
// is not migratable. In this case all accesses to the page must be cancelled.
// - Write/atomic accesses are invalid. Basically, when trying to modify a
// read-only VA range. In this case we restrict fault cancelling to those types
// of accesses.
//
// Return values:
// - service_access_type: highest access type that can be serviced.
static uvm_fault_access_type_t check_fault_access_permissions(uvm_gpu_t *gpu,
                                                              uvm_fault_service_batch_context_t *batch_context,
                                                              uvm_va_block_t *va_block,
                                                              uvm_service_block_context_t *service_block_context,
                                                              uvm_fault_buffer_entry_t *fault_entry)
{
    NV_STATUS perm_status;
    UvmEventFatalReason fatal_reason;
    uvm_fault_cancel_va_mode_t cancel_va_mode;
    uvm_fault_access_type_t ret = UVM_FAULT_ACCESS_TYPE_COUNT;
    uvm_va_block_context_t *va_block_context = service_block_context->block_context;

    perm_status = uvm_va_block_check_logical_permissions(va_block,
                                                         va_block_context,
                                                         gpu->id,
                                                         uvm_va_block_cpu_page_index(va_block,
                                                                                     fault_entry->fault_address),
                                                         fault_entry->fault_access_type);
    if (perm_status == NV_OK)
        return fault_entry->fault_access_type;

    if (fault_entry->fault_access_type == UVM_FAULT_ACCESS_TYPE_PREFETCH) {
        // Only update the count the first time since logical permissions cannot
        // change while we hold the VA space lock
        // TODO: Bug 1750144: That might not be true with HMM.
        if (service_block_context->num_retries == 0)
            mark_fault_invalid_prefetch(batch_context, fault_entry);

        return ret;
    }

    // At this point we know that some fault instances cannot be serviced
    fatal_reason = uvm_tools_status_to_fatal_fault_reason(perm_status);

    if (fault_entry->fault_access_type > UVM_FAULT_ACCESS_TYPE_READ) {
        cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_WRITE_AND_ATOMIC;

        // If there are pending read accesses on the same page, we have to
        // service them before we can cancel the write/atomic faults. So we
        // retry with read fault access type.
        if (uvm_fault_access_type_mask_test(fault_entry->access_type_mask, UVM_FAULT_ACCESS_TYPE_READ)) {
            perm_status = uvm_va_block_check_logical_permissions(va_block,
                                                                 va_block_context,
                                                                 gpu->id,
                                                                 uvm_va_block_cpu_page_index(va_block,
                                                                                             fault_entry->fault_address),
                                                                 UVM_FAULT_ACCESS_TYPE_READ);
            if (perm_status == NV_OK) {
                ret = UVM_FAULT_ACCESS_TYPE_READ;
            }
            else {
                // Read accesses didn't succeed, cancel all faults
                cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_ALL;
                fatal_reason = uvm_tools_status_to_fatal_fault_reason(perm_status);
            }
        }
    }
    else {
        cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_ALL;
    }

    mark_fault_fatal(batch_context, fault_entry, fatal_reason, cancel_va_mode);

    return ret;
}

// We notify the fault event for all faults within the block so that the
// performance heuristics are updated. Then, all required actions for the block
// data are performed by the performance heuristics code.
//
// Fatal faults are flagged as fatal for later cancellation. Servicing is not
// interrupted on fatal faults due to insufficient permissions or invalid
// addresses.
//
// Return codes:
// - NV_OK if all faults were handled (both fatal and non-fatal)
// - NV_ERR_MORE_PROCESSING_REQUIRED if servicing needs allocation retry
// - NV_ERR_NO_MEMORY if the faults could not be serviced due to OOM
// - Any other value is a UVM-global error
static NV_STATUS service_fault_batch_block_locked(uvm_gpu_va_space_t *gpu_va_space,
                                                  uvm_va_block_t *va_block,
                                                  uvm_va_block_retry_t *va_block_retry,
                                                  uvm_fault_service_batch_context_t *batch_context,
                                                  uvm_service_block_context_t *block_context,
                                                  NvU32 first_fault_index,
                                                  const bool hmm_migratable,
                                                  NvU32 *block_faults)
{
    NV_STATUS status = NV_OK;
    NvU32 i;
    uvm_page_index_t first_page_index;
    uvm_page_index_t last_page_index;
    NvU32 page_fault_count = 0;
    uvm_gpu_t *gpu = gpu_va_space->gpu;
    uvm_fault_buffer_entry_t **ordered_fault_cache = batch_context->ordered_fault_cache;
    uvm_fault_buffer_entry_t *first_fault_entry = ordered_fault_cache[first_fault_index];
    uvm_va_space_t *va_space = uvm_va_block_get_va_space(va_block);
    const uvm_va_policy_t *policy;
    NvU64 end;

    // Check that all uvm_fault_access_type_t values can fit into an NvU8
    BUILD_BUG_ON(UVM_FAULT_ACCESS_TYPE_COUNT > (int)(NvU8)-1);

    uvm_assert_mutex_locked(&va_block->lock);

    *block_faults = 0;

    first_page_index = PAGES_PER_UVM_VA_BLOCK;
    last_page_index = 0;

    // Initialize fault service block context
    uvm_processor_mask_zero(&block_context->resident_processors);
    block_context->thrashing_pin_count = 0;
    block_context->read_duplicate_count = 0;

    // The first entry is guaranteed to fall within this block
    UVM_ASSERT(first_fault_entry->va_space == va_space);
    UVM_ASSERT(first_fault_entry->gpu == gpu);
    UVM_ASSERT(first_fault_entry->fault_address >= va_block->start);
    UVM_ASSERT(first_fault_entry->fault_address <= va_block->end);

    if (uvm_va_block_is_hmm(va_block)) {
        policy = uvm_hmm_find_policy_end(va_block,
                                         block_context->block_context->hmm.vma,
                                         first_fault_entry->fault_address,
                                         &end);
    }
    else {
        policy = &va_block->managed_range->policy;
        end = va_block->end;
    }

    // Scan the sorted array and notify the fault event for all fault entries
    // in the block
    for (i = first_fault_index;
         i < batch_context->num_coalesced_faults &&
         ordered_fault_cache[i]->va_space == va_space &&
         ordered_fault_cache[i]->gpu == gpu &&
         ordered_fault_cache[i]->fault_address <= end;
         ++i) {
        uvm_fault_buffer_entry_t *current_entry = ordered_fault_cache[i];
        const uvm_fault_buffer_entry_t *previous_entry = NULL;
        bool read_duplicate;
        uvm_processor_id_t new_residency;
        uvm_perf_thrashing_hint_t thrashing_hint;
        uvm_page_index_t page_index = uvm_va_block_cpu_page_index(va_block, current_entry->fault_address);
        bool is_duplicate = false;
        uvm_fault_access_type_t service_access_type;
        NvU32 service_access_type_mask;

        UVM_ASSERT(current_entry->fault_access_type ==
                   uvm_fault_access_type_mask_highest(current_entry->access_type_mask));

        // Unserviceable faults were already skipped by the caller. There are no
        // unserviceable fault types that could be in the same VA block as a
        // serviceable fault.
        UVM_ASSERT(!current_entry->is_fatal);
        current_entry->is_throttled        = false;
        current_entry->is_invalid_prefetch = false;

        if (i > first_fault_index) {
            previous_entry = ordered_fault_cache[i - 1];
            is_duplicate = check_fault_entry_duplicate(current_entry, previous_entry);
        }

        // Only update counters the first time since logical permissions cannot
        // change while we hold the VA space lock.
        // TODO: Bug 1750144: That might not be true with HMM.
        if (block_context->num_retries == 0) {
            update_batch_and_notify_fault(gpu,
                                          batch_context,
                                          va_block,
                                          policy->preferred_location,
                                          current_entry,
                                          is_duplicate);
        }

        // Service the most intrusive fault per page, only. Waive the rest
        if (is_duplicate) {
            fault_entry_duplicate_flags(batch_context, current_entry, previous_entry);

            // The previous fault was non-fatal so the page has been already
            // serviced
            if (!previous_entry->is_fatal)
                continue;
        }

        service_access_type = check_fault_access_permissions(gpu,
                                                             batch_context,
                                                             va_block,
                                                             block_context,
                                                             current_entry);

        // Do not exit early due to logical errors such as access permission
        // violation.
        if (service_access_type == UVM_FAULT_ACCESS_TYPE_COUNT)
            continue;

        if (service_access_type != current_entry->fault_access_type) {
            // Some of the fault instances cannot be serviced due to invalid
            // access permissions. Recompute the access type service mask to
            // service the rest.
            UVM_ASSERT(service_access_type < current_entry->fault_access_type);
            service_access_type_mask = uvm_fault_access_type_mask_bit(service_access_type);
        }
        else {
            service_access_type_mask = current_entry->access_type_mask;
        }

        // If the GPU already has the necessary access permission, the fault
        // does not need to be serviced
        if (uvm_va_block_page_is_gpu_authorized(va_block,
                                                page_index,
                                                gpu->id,
                                                uvm_fault_access_type_to_prot(service_access_type))) {
            // A fault the GPU raised on work already done. Counted because the
            // whole measured gap against ARIADNE is these: on w7 at 110% all
            // three builds deliver the same pages but need 1.09 (stock), 1.51
            // (ARIADNE) and 1.95 (ours) faults per page to do it, and the
            // excess ratio tracks the wall gap. Nothing else in the statistics
            // distinguishes a fault that moved data from one that did not.
            uvm_lock_probe_count(&g_uvm_lock_contention_stats.n_fault_authorized);

            // Not gated on the stats level: this one drives the flush-mode
            // decision below, not a report, so it has to be counted in every
            // build. atomic because parallel workers reach it concurrently.
            atomic_inc(&batch_context->num_authorized_faults);
            continue;
        }

        // Not authorized for what this fault wants. Separate the case where the
        // page is already readable, which is an ordinary read-to-write upgrade
        // and would be counted once by stock too, from the case where the GPU
        // has no access at all, which is genuine new demand. Which of the three
        // carries our excess is what names the cause: authorized means the GPU
        // retried before our mappings were visible and the defect is ours,
        // upgrade means we split a permission change stock does once, and
        // serviced means the extra faults are real work the mechanism creates.
        //
        // Same predicate as above, so it is a second cheap region test under a
        // lock already held, not a new kind of query.
        if (uvm_lock_probes_enabled()) {
            if (uvm_va_block_page_is_gpu_authorized(va_block, page_index, gpu->id, UVM_PROT_READ_ONLY))
                uvm_lock_stat_inc(&g_uvm_lock_contention_stats.n_fault_upgrade);
            else
                uvm_lock_stat_inc(&g_uvm_lock_contention_stats.n_fault_serviced);
        }

        thrashing_hint = uvm_perf_thrashing_get_hint(va_block,
                                                     block_context->block_context,
                                                     current_entry->fault_address,
                                                     gpu->id);
        if (thrashing_hint.type == UVM_PERF_THRASHING_HINT_TYPE_THROTTLE) {
            // Throttling is implemented by sleeping in the fault handler on
            // the CPU and by continuing to process faults on other pages on
            // the GPU
            //
            // Only update the flag the first time since logical permissions
            // cannot change while we hold the VA space lock.
            // TODO: Bug 1750144: That might not be true with HMM.
            if (block_context->num_retries == 0)
                mark_fault_throttled(batch_context, current_entry);

            continue;
        }
        else if (thrashing_hint.type == UVM_PERF_THRASHING_HINT_TYPE_PIN) {
            if (block_context->thrashing_pin_count++ == 0)
                uvm_page_mask_zero(&block_context->thrashing_pin_mask);

            uvm_page_mask_set(&block_context->thrashing_pin_mask, page_index);
        }

        // Compute new residency and update the masks
        new_residency = uvm_va_block_select_residency(va_block,
                                                      block_context->block_context,
                                                      page_index,
                                                      gpu->id,
                                                      service_access_type_mask,
                                                      policy,
                                                      &thrashing_hint,
                                                      UVM_SERVICE_OPERATION_REPLAYABLE_FAULTS,
                                                      hmm_migratable,
                                                      &read_duplicate);

        // If this is a ATS processor and the page is already resident in the
        // correct location then it should already be mapped on the CPU so handle this as a
        // minor fault.
        if (uvm_va_block_is_hmm(va_block) && gpu->parent->ats_supported) {
            uvm_va_block_page_resident_processors(va_block, page_index,
                                                  &block_context->block_context->scratch_processor_mask);
            if (uvm_processor_mask_test(&block_context->block_context->scratch_processor_mask, gpu->id)) {
                unsigned int flags = FAULT_FLAG_REMOTE;

                if (service_access_type >= UVM_FAULT_ACCESS_TYPE_WRITE)
                    flags |= FAULT_FLAG_WRITE;

                UVM_HANDLE_MM_FAULT(block_context->block_context->hmm.vma,
                                    uvm_va_block_cpu_page_address(va_block, page_index), flags);
                continue;
            }
        }

        if (!uvm_processor_mask_test_and_set(&block_context->resident_processors, new_residency))
            uvm_page_mask_zero(&block_context->per_processor_masks[uvm_id_value(new_residency)].new_residency);

        uvm_page_mask_set(&block_context->per_processor_masks[uvm_id_value(new_residency)].new_residency, page_index);

        if (read_duplicate) {
            if (block_context->read_duplicate_count++ == 0)
                uvm_page_mask_zero(&block_context->read_duplicate_mask);

            uvm_page_mask_set(&block_context->read_duplicate_mask, page_index);
        }

        ++page_fault_count;

        block_context->access_type[page_index] = service_access_type;

        // Since mixed-coherency now involves the va_block to handle pageable
        // memory ats faults, we need to ensure that 4K pages are handled the
        // same way as in the ats faulting path for numa. See the comment at the
        // end of uvm_ats_service_faults_region() in the numa path for why this
        // flush is necessary.
        if (PAGE_SIZE == UVM_PAGE_SIZE_4K && gpu->parent->ats_supported && uvm_va_block_is_hmm(va_block))
            uvm_flush_tlb_va_region(gpu_va_space, current_entry->fault_address, UVM_PAGE_SIZE_4K, UVM_FAULT_CLIENT_TYPE_GPC);

        if (page_index < first_page_index)
            first_page_index = page_index;
        if (page_index > last_page_index)
            last_page_index = page_index;
    }

    // Apply the changes computed in the fault service block context, if there
    // are pages to be serviced
    if (page_fault_count > 0) {
        block_context->region = uvm_va_block_region(first_page_index, last_page_index + 1);
        status = uvm_va_block_service_locked(gpu, va_block, va_block_retry, block_context);
    }

    *block_faults = i - first_fault_index;

    ++block_context->num_retries;

    return status;
}

// We notify the fault event for all faults within the block so that the
// performance heuristics are updated. The VA block lock is taken for the whole
// fault servicing although it might be temporarily dropped and re-taken if
// memory eviction is required.
//
// See the comments for function service_fault_batch_block_locked for
// implementation details and error codes.
static NV_STATUS service_fault_batch_block(uvm_gpu_va_space_t *gpu_va_space,
                                           uvm_va_block_t *va_block,
                                           uvm_fault_service_batch_context_t *batch_context,
                                           uvm_service_block_context_t *fault_block_context,
                                           uvm_tracker_t *tracker,
                                           NvU32 first_fault_index,
                                           const bool hmm_migratable,
                                           NvU32 *block_faults)
{
    NV_STATUS status;
    uvm_va_block_retry_t va_block_retry;
    NV_STATUS tracker_status;
    NvU64 block_lock_wait_start;
    NvU64 block_service_start;

    // Whole-call cost of servicing this block, taken before the HMM wait so it
    // covers everything the caller pays for one va_block. This function is stock,
    // so every servicing mechanism passes through it once per block and the same
    // number is comparable across them.
    block_service_start = uvm_lock_probe_begin();

    fault_block_context->operation = UVM_SERVICE_OPERATION_REPLAYABLE_FAULTS;
    fault_block_context->num_retries = 0;

    if (uvm_va_block_is_hmm(va_block))
        uvm_hmm_migrate_begin_wait(va_block);

    // The dispatcher and every worker thread converge on this one acquisition,
    // so it is the single place the pool can serialize against itself or
    // against a CPU fault holding the same block. The timestamp is taken after
    // uvm_hmm_migrate_begin_wait above, which can block for reasons that have
    // nothing to do with this lock.
    block_lock_wait_start = uvm_lock_probe_begin();

    uvm_mutex_lock(&va_block->lock);

    uvm_lock_probe_end(block_lock_wait_start,
                       &g_uvm_lock_contention_stats.ns_block_lock_wait_gpu,
                       &g_uvm_lock_contention_stats.n_block_lock_acqs_gpu);

    status = UVM_VA_BLOCK_RETRY_LOCKED(va_block, &va_block_retry,
                                       service_fault_batch_block_locked(gpu_va_space,
                                                                        va_block,
                                                                        &va_block_retry,
                                                                        batch_context,
                                                                        fault_block_context,
                                                                        first_fault_index,
                                                                        hmm_migratable,
                                                                        block_faults));

    tracker_status = uvm_tracker_add_tracker_safe(tracker, &va_block->tracker);

    uvm_mutex_unlock(&va_block->lock);

    if (uvm_va_block_is_hmm(va_block))
        uvm_hmm_migrate_finish(va_block);

    uvm_lock_probe_end(block_service_start,
                       &g_uvm_lock_contention_stats.ns_va_block_service,
                       &g_uvm_lock_contention_stats.n_va_block_service);

    return status == NV_OK? tracker_status: status;
}

typedef enum
{
    // Use this mode when calling from the normal fault servicing path
    FAULT_SERVICE_MODE_REGULAR,

    // Use this mode when servicing faults from the fault cancelling algorithm.
    // In this mode no replays are issued
    FAULT_SERVICE_MODE_CANCEL,
} fault_service_mode_t;

static void service_fault_batch_fatal(uvm_fault_service_batch_context_t *batch_context,
                                      NvU32 first_fault_index,
                                      NV_STATUS status,
                                      uvm_fault_cancel_va_mode_t cancel_va_mode,
                                      NvU32 *block_faults)
{
    uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[first_fault_index];
    const uvm_fault_buffer_entry_t *previous_entry = first_fault_index > 0 ?
                                                       batch_context->ordered_fault_cache[first_fault_index - 1] : NULL;
    bool is_duplicate = check_fault_entry_duplicate(current_entry, previous_entry);

    if (is_duplicate)
        fault_entry_duplicate_flags(batch_context, current_entry, previous_entry);

    if (current_entry->fault_access_type == UVM_FAULT_ACCESS_TYPE_PREFETCH)
        mark_fault_invalid_prefetch(batch_context, current_entry);
    else
        mark_fault_fatal(batch_context, current_entry, uvm_tools_status_to_fatal_fault_reason(status), cancel_va_mode);

    (*block_faults)++;
}

static void service_fault_batch_fatal_notify(uvm_gpu_t *gpu,
                                             uvm_fault_service_batch_context_t *batch_context,
                                             NvU32 first_fault_index,
                                             NV_STATUS status,
                                             uvm_fault_cancel_va_mode_t cancel_va_mode,
                                             NvU32 *block_faults)
{
    uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[first_fault_index];
    const uvm_fault_buffer_entry_t *previous_entry = first_fault_index > 0 ?
                                                       batch_context->ordered_fault_cache[first_fault_index - 1] : NULL;
    bool is_duplicate = check_fault_entry_duplicate(current_entry, previous_entry);

    service_fault_batch_fatal(batch_context, first_fault_index, status, cancel_va_mode, block_faults);

    update_batch_and_notify_fault(gpu, batch_context, NULL, UVM_ID_INVALID, current_entry, is_duplicate);
}

static NV_STATUS service_fault_batch_ats_sub_vma(uvm_gpu_va_space_t *gpu_va_space,
                                                 struct vm_area_struct *vma,
                                                 NvU64 base,
                                                 uvm_fault_service_batch_context_t *batch_context,
                                                 uvm_ats_fault_context_t *ats_context,
                                                 NvU32 fault_index_start,
                                                 NvU32 fault_index_end,
                                                 NvU32 *block_faults)
{
    NvU32 i;
    NV_STATUS status = NV_OK;
    const uvm_page_mask_t *read_fault_mask = &ats_context->faults.read_fault_mask;
    const uvm_page_mask_t *write_fault_mask = &ats_context->faults.write_fault_mask;
    const uvm_page_mask_t *reads_serviced_mask = &ats_context->faults.reads_serviced_mask;
    uvm_page_mask_t *faults_serviced_mask = &ats_context->faults.faults_serviced_mask;
    uvm_page_mask_t *accessed_mask = &ats_context->faults.accessed_mask;

    UVM_ASSERT(vma);

    ats_context->client_type = UVM_FAULT_CLIENT_TYPE_GPC;

    uvm_page_mask_or(accessed_mask, write_fault_mask, read_fault_mask);

    status = uvm_ats_service_faults(gpu_va_space, vma, base, ats_context);

    // Remove SW prefetched pages from the serviced mask since fault servicing
    // failures belonging to prefetch pages need to be ignored.
    uvm_page_mask_and(faults_serviced_mask, faults_serviced_mask, accessed_mask);

    UVM_ASSERT(uvm_page_mask_subset(faults_serviced_mask, accessed_mask));

    if ((status != NV_OK) || uvm_page_mask_equal(faults_serviced_mask, accessed_mask)) {
        (*block_faults) += (fault_index_end - fault_index_start);
        return status;
    }

    // Check faults_serviced_mask and reads_serviced_mask for precise fault
    // attribution after calling the ATS servicing routine. The
    // errors returned from ATS servicing routine should only be
    // global errors such as OOM or ECC.
    // uvm_parent_gpu_service_replayable_faults() handles global errors by
    // calling cancel_fault_batch(). Precise attribution isn't currently
    // supported in such cases.
    //
    // Precise fault attribution for global errors can be handled by
    // servicing one fault at a time until fault servicing encounters an
    // error.
    // TODO: Bug 3989244: Precise ATS fault attribution for global errors.
    for (i = fault_index_start; i < fault_index_end; i++) {
        uvm_page_index_t page_index;
        uvm_fault_cancel_va_mode_t cancel_va_mode;
        uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];
        uvm_fault_access_type_t access_type = current_entry->fault_access_type;

        page_index = (current_entry->fault_address - base) / PAGE_SIZE;

        if (uvm_page_mask_test(faults_serviced_mask, page_index)) {
            (*block_faults)++;
            continue;
        }

        if (access_type <= UVM_FAULT_ACCESS_TYPE_READ) {
            cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_ALL;
        }
        else {
            UVM_ASSERT(access_type >= UVM_FAULT_ACCESS_TYPE_WRITE);
            if (uvm_fault_access_type_mask_test(current_entry->access_type_mask, UVM_FAULT_ACCESS_TYPE_READ) &&
                !uvm_page_mask_test(reads_serviced_mask, page_index))
                cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_ALL;
            else
                cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_WRITE_AND_ATOMIC;
        }

        service_fault_batch_fatal(batch_context, i, NV_ERR_INVALID_ADDRESS, cancel_va_mode, block_faults);
    }

    return status;
}

static void start_new_sub_batch(NvU64 *sub_batch_base,
                                NvU64 address,
                                NvU32 *sub_batch_fault_index,
                                NvU32 fault_index,
                                uvm_ats_fault_context_t *ats_context)
{
    uvm_page_mask_zero(&ats_context->faults.read_fault_mask);
    uvm_page_mask_zero(&ats_context->faults.write_fault_mask);
    uvm_page_mask_zero(&ats_context->faults.prefetch_only_fault_mask);

    *sub_batch_fault_index = fault_index;
    *sub_batch_base = UVM_VA_BLOCK_ALIGN_DOWN(address);
}

static NV_STATUS service_fault_batch_ats_sub(uvm_gpu_va_space_t *gpu_va_space,
                                             struct vm_area_struct *vma,
                                             uvm_fault_service_batch_context_t *batch_context,
                                             uvm_ats_fault_context_t *ats_context,
                                             NvU32 fault_index,
                                             NvU64 outer,
                                             NvU32 *block_faults)
{
    NV_STATUS status = NV_OK;
    NvU32 i = fault_index;
    NvU32 sub_batch_fault_index;
    NvU64 sub_batch_base;
    uvm_fault_buffer_entry_t *previous_entry = NULL;
    uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];
    uvm_page_mask_t *read_fault_mask = &ats_context->faults.read_fault_mask;
    uvm_page_mask_t *write_fault_mask = &ats_context->faults.write_fault_mask;
    uvm_page_mask_t *prefetch_only_fault_mask = &ats_context->faults.prefetch_only_fault_mask;
    uvm_gpu_t *gpu = gpu_va_space->gpu;
    bool replay_per_va_block =
                        (gpu->parent->fault_buffer.replayable.replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BLOCK);

    UVM_ASSERT(vma);

    outer = min(outer, (NvU64) vma->vm_end);

    start_new_sub_batch(&sub_batch_base, current_entry->fault_address, &sub_batch_fault_index, i, ats_context);

    do {
        uvm_page_index_t page_index;
        NvU64 fault_address = current_entry->fault_address;
        uvm_fault_access_type_t access_type = current_entry->fault_access_type;
        bool is_duplicate = check_fault_entry_duplicate(current_entry, previous_entry);

        // ATS faults can't be unserviceable, since unserviceable faults require
        // GMMU PTEs.
        UVM_ASSERT(!current_entry->is_fatal);
        UVM_ASSERT(current_entry->gpu == gpu);

        i++;

        update_batch_and_notify_fault(gpu,
                                      batch_context,
                                      NULL,
                                      UVM_ID_INVALID,
                                      current_entry,
                                      is_duplicate);

        // End of sub-batch. Service faults gathered so far.
        if (fault_address >= (sub_batch_base + UVM_VA_BLOCK_SIZE)) {
            UVM_ASSERT(!uvm_page_mask_empty(read_fault_mask) ||
                       !uvm_page_mask_empty(write_fault_mask) ||
                       !uvm_page_mask_empty(prefetch_only_fault_mask));

            status = service_fault_batch_ats_sub_vma(gpu_va_space,
                                                     vma,
                                                     sub_batch_base,
                                                     batch_context,
                                                     ats_context,
                                                     sub_batch_fault_index,
                                                     i - 1,
                                                     block_faults);
            if (status != NV_OK || replay_per_va_block)
                break;

            start_new_sub_batch(&sub_batch_base, fault_address, &sub_batch_fault_index, i - 1, ats_context);
        }

        page_index = (fault_address - sub_batch_base) / PAGE_SIZE;

        // Do not check for coalesced access type. If there are multiple
        // different accesses to an address, we can disregard the prefetch one.
        if ((access_type == UVM_FAULT_ACCESS_TYPE_PREFETCH) &&
            (uvm_fault_access_type_mask_highest(current_entry->access_type_mask) == UVM_FAULT_ACCESS_TYPE_PREFETCH))
            uvm_page_mask_set(prefetch_only_fault_mask, page_index);

        if ((access_type == UVM_FAULT_ACCESS_TYPE_READ) ||
            uvm_fault_access_type_mask_test(current_entry->access_type_mask, UVM_FAULT_ACCESS_TYPE_READ))
            uvm_page_mask_set(read_fault_mask, page_index);

        if (access_type >= UVM_FAULT_ACCESS_TYPE_WRITE)
            uvm_page_mask_set(write_fault_mask, page_index);

        previous_entry = current_entry;
        current_entry = i < batch_context->num_coalesced_faults ? batch_context->ordered_fault_cache[i] : NULL;

    } while (current_entry &&
             (current_entry->fault_address < outer) &&
             (previous_entry->gpu == current_entry->gpu) &&
             (previous_entry->va_space == current_entry->va_space));

    // Service the last sub-batch.
    if ((status == NV_OK) &&
        (!uvm_page_mask_empty(read_fault_mask) ||
         !uvm_page_mask_empty(write_fault_mask) ||
         !uvm_page_mask_empty(prefetch_only_fault_mask))) {
        status = service_fault_batch_ats_sub_vma(gpu_va_space,
                                                 vma,
                                                 sub_batch_base,
                                                 batch_context,
                                                 ats_context,
                                                 sub_batch_fault_index,
                                                 i,
                                                 block_faults);
    }

    return status;
}

static NV_STATUS service_fault_batch_ats(uvm_gpu_va_space_t *gpu_va_space,
                                         struct mm_struct *mm,
                                         uvm_fault_service_batch_context_t *batch_context,
                                         uvm_ats_fault_context_t *ats_context,
                                         NvU32 first_fault_index,
                                         NvU64 outer,
                                         NvU32 *block_faults)
{
    NvU32 i;
    NV_STATUS status = NV_OK;

    for (i = first_fault_index; i < batch_context->num_coalesced_faults;) {
        uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];
        const uvm_fault_buffer_entry_t *previous_entry = i > first_fault_index ?
                                                                       batch_context->ordered_fault_cache[i - 1] : NULL;
        NvU64 fault_address = current_entry->fault_address;
        struct vm_area_struct *vma;
        NvU32 num_faults_before = (*block_faults);

        if (previous_entry &&
            (previous_entry->va_space != current_entry->va_space || previous_entry->gpu != current_entry->gpu))
            break;

        if (fault_address >= outer)
            break;

        vma = find_vma_intersection(mm, fault_address, fault_address + 1);
        if (!vma) {
            // Since a vma wasn't found, cancel all accesses on the page since
            // cancelling write and atomic accesses will not cancel pending read
            // faults and this can lead to a deadlock since read faults need to
            // be serviced first before cancelling write faults.
            service_fault_batch_fatal_notify(gpu_va_space->gpu,
                                             batch_context,
                                             i,
                                             NV_ERR_INVALID_ADDRESS,
                                             UVM_FAULT_CANCEL_VA_MODE_ALL,
                                             block_faults);

            // Do not fail due to logical errors.
            status = NV_OK;

            break;
        }

        status = service_fault_batch_ats_sub(gpu_va_space, vma, batch_context, ats_context, i, outer, block_faults);
        if (status != NV_OK)
            break;

        i += ((*block_faults) - num_faults_before);
    }

    return status;
}

static NV_STATUS service_fault_batch_dispatch(uvm_va_space_t *va_space,
                                              uvm_gpu_va_space_t *gpu_va_space,
                                              uvm_fault_service_batch_context_t *batch_context,
                                              uvm_service_block_context_t *service_context,
                                              uvm_ats_fault_context_t *ats_context,
                                              uvm_tracker_t *tracker,
                                              NvU32 fault_index,
                                              NvU32 *block_faults,
                                              bool replay_per_va_block,
                                              const bool hmm_migratable)
{
    NV_STATUS status;
    uvm_va_range_t *va_range = NULL;
    uvm_va_range_t *va_range_next = NULL;
    uvm_va_block_t *va_block;
    uvm_gpu_t *gpu = gpu_va_space->gpu;
    uvm_va_block_context_t *va_block_context = service_context->block_context;
    uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[fault_index];
    struct mm_struct *mm = va_block_context->mm;
    NvU64 fault_address = current_entry->fault_address;

    (*block_faults) = 0;

    va_range_next = uvm_va_space_iter_gmmu_mappable_first(va_space, fault_address);
    if (va_range_next && (fault_address >= va_range_next->node.start)) {
        UVM_ASSERT(fault_address < va_range_next->node.end);

        va_range = va_range_next;
        va_range_next = uvm_va_range_gmmu_mappable_next(va_range);
    }

    if (va_range)
        status = uvm_va_block_find_create_in_range(va_space, va_range, fault_address, &va_block);
    else if (mm)
        status = uvm_hmm_va_block_find_create(va_space, fault_address, &va_block_context->hmm.vma, &va_block);
    else
        status = NV_ERR_INVALID_ADDRESS;

    if (status == NV_OK) {
        if (uvm_va_block_is_hmm(va_block) &&
            gpu->parent->ats_supported &&
            uvm_ats_check_in_gmmu_region(va_space, fault_address, va_range_next)) {

            service_fault_batch_fatal_notify(gpu,
                                             batch_context,
                                             fault_index,
                                             NV_ERR_INVALID_ADDRESS,
                                             UVM_FAULT_CANCEL_VA_MODE_ALL,
                                             block_faults);
            status = NV_OK;
        }
        else {
            status = service_fault_batch_block(gpu_va_space,
                                               va_block,
                                               batch_context,
                                               service_context,
                                               tracker,
                                               fault_index,
                                               hmm_migratable,
                                               block_faults);
        }
    }
    else if ((status == NV_ERR_INVALID_ADDRESS) && uvm_ats_can_service_faults(gpu_va_space, mm)) {
        NvU64 outer = ~0ULL;

         UVM_ASSERT(replay_per_va_block ==
                    (gpu->parent->fault_buffer.replayable.replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BLOCK));

        // Limit outer to the minimum of next va_range.start and first
        // fault_address' next UVM_GMMU_ATS_GRANULARITY alignment so that it's
        // enough to check whether the first fault in this dispatch belongs to a
        // GMMU region.
        if (va_range_next) {
            outer = min(va_range_next->node.start,
                           UVM_ALIGN_DOWN(fault_address + UVM_GMMU_ATS_GRANULARITY, UVM_GMMU_ATS_GRANULARITY));
        }

        // ATS lookups are disabled on all addresses within the same
        // UVM_GMMU_ATS_GRANULARITY as existing GMMU mappings (see documentation
        // in uvm_mmu.h). User mode is supposed to reserve VAs as appropriate to
        // prevent any system memory allocations from falling within the NO_ATS
        // range of other GMMU mappings, so this shouldn't happen during normal
        // operation. However, since this scenario may lead to infinite fault
        // loops, we handle it by canceling the fault.
        if (uvm_ats_check_in_gmmu_region(va_space, fault_address, va_range_next)) {
            service_fault_batch_fatal_notify(gpu,
                                             batch_context,
                                             fault_index,
                                             NV_ERR_INVALID_ADDRESS,
                                             UVM_FAULT_CANCEL_VA_MODE_ALL,
                                             block_faults);

            // Do not fail due to logical errors
            status = NV_OK;
        }
        else {
            status = service_fault_batch_ats(gpu_va_space,
                                             mm,
                                             batch_context,
                                             ats_context,
                                             fault_index,
                                             outer,
                                             block_faults);
        }
    }
    else {
        service_fault_batch_fatal_notify(gpu,
                                         batch_context,
                                         fault_index,
                                         status,
                                         UVM_FAULT_CANCEL_VA_MODE_ALL,
                                         block_faults);

        // Do not fail due to logical errors
        status = NV_OK;
    }

    return status;
}

// Called when a fault in the batch has been marked fatal. Flush the buffer
// under the VA and mmap locks to remove any potential stale fatal faults, then
// service all new faults for just that VA space and cancel those which are
// fatal. Faults in other VA spaces are replayed when done and will be processed
// when normal fault servicing resumes.
static NV_STATUS service_fault_batch_for_cancel(uvm_fault_service_batch_context_t *batch_context)
{
    NV_STATUS status = NV_OK;
    NvU32 i;
    uvm_va_space_t *va_space = batch_context->fatal_va_space;
    uvm_gpu_t *gpu = batch_context->fatal_gpu;
    uvm_gpu_va_space_t *gpu_va_space = NULL;
    struct mm_struct *mm;
    uvm_replayable_fault_buffer_t *replayable_faults = &gpu->parent->fault_buffer.replayable;
    uvm_service_block_context_t *service_context = &gpu->parent->fault_buffer.replayable.block_service_context;
    uvm_va_block_context_t *va_block_context = service_context->block_context;

    UVM_ASSERT(va_space);
    UVM_ASSERT(gpu);

    // Perform the flush and re-fetch while holding the mmap_lock and the
    // VA space lock. This avoids stale faults because it prevents any vma
    // modifications (mmap, munmap, mprotect) from happening between the time HW
    // takes the fault and we cancel it.
    mm = uvm_va_space_mm_retain_lock(va_space);
    uvm_va_block_context_init(va_block_context, mm);
    uvm_va_space_down_read(va_space);

    // We saw fatal faults in this VA space before. Flush while holding
    // mmap_lock to make sure those faults come back (aren't stale).
    //
    // We need to wait until all old fault messages have arrived before
    // flushing, hence UVM_GPU_BUFFER_FLUSH_MODE_WAIT_UPDATE_PUT.
    status = fault_buffer_flush_locked(gpu->parent,
                                       gpu,
                                       UVM_GPU_BUFFER_FLUSH_MODE_WAIT_UPDATE_PUT,
                                       UVM_FAULT_REPLAY_TYPE_START,
                                       batch_context);
    if (status != NV_OK)
        goto done;

    // Wait for the flush's replay to finish to give the legitimate faults a
    // chance to show up in the buffer again.
    status = uvm_tracker_wait(&replayable_faults->replay_tracker);
    if (status != NV_OK)
        goto done;

    // We expect all replayed faults to have arrived in the buffer so we can re-
    // service them. The replay-and-wait sequence above will ensure they're all
    // in the HW buffer. When GSP owns the HW buffer, we also have to wait for
    // GSP to copy all available faults from the HW buffer into the shadow
    // buffer.
    status = hw_fault_buffer_flush_locked(gpu->parent, HW_FAULT_BUFFER_FLUSH_MODE_MOVE);
    if (status != NV_OK)
        goto done;

    // If there is no GPU VA space for the GPU, ignore all faults in the VA
    // space. This can happen if the GPU VA space has been destroyed since we
    // unlocked the VA space in service_fault_batch. That means the fatal faults
    // are stale, because unregistering the GPU VA space requires preempting the
    // context and detaching all channels in that VA space. Restart fault
    // servicing from the top.
    gpu_va_space = uvm_gpu_va_space_get(va_space, gpu);
    if (!gpu_va_space)
        goto done;

    // Re-parse the new faults
    atomic_set(&batch_context->num_invalid_prefetch_faults, 0);
    atomic_set(&batch_context->num_duplicate_faults, 0);
    batch_context->num_replays                 = 0;
    batch_context->fatal_va_space              = NULL;
    batch_context->fatal_gpu                   = NULL;
    batch_context->has_throttled_faults        = false;

    status = fetch_fault_buffer_entries(gpu->parent, batch_context, FAULT_FETCH_MODE_ALL);
    if (status != NV_OK)
        goto done;

    // No more faults left. Either the previously-seen fatal entry was stale, or
    // RM killed the context underneath us.
    if (batch_context->num_cached_faults == 0)
        goto done;

    ++batch_context->batch_id;

    status = preprocess_fault_batch(gpu->parent, batch_context);
    if (status != NV_OK) {
        if (status == NV_WARN_MORE_PROCESSING_REQUIRED) {
            // Another flush happened due to stale faults or a context-fatal
            // error. The previously-seen fatal fault might not exist anymore,
            // so restart fault servicing from the top.
            status = NV_OK;
        }

        goto done;
    }

    // Search for the target VA space and GPU.
    for (i = 0; i < batch_context->num_coalesced_faults; i++) {
        uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];
        UVM_ASSERT(current_entry->va_space);
        if (current_entry->va_space == va_space && current_entry->gpu == gpu)
            break;
    }

    while (i < batch_context->num_coalesced_faults) {
        uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];

        if (current_entry->va_space != va_space || current_entry->gpu != gpu)
            break;

        // service_fault_batch_dispatch() doesn't expect unserviceable faults.
        // Just cancel them directly.
        if (current_entry->is_fatal) {
            status = cancel_fault_precise_va(current_entry, UVM_FAULT_CANCEL_VA_MODE_ALL);
            if (status != NV_OK)
                break;

            ++i;
        }
        else {
            uvm_ats_fault_invalidate_t *ats_invalidate = &gpu->parent->fault_buffer.replayable.ats_invalidate;
            NvU32 block_faults;
            const bool hmm_migratable = true;

            ats_invalidate->tlb_batch_pending = false;

            // Service all the faults that we can. We only really need to search
            // for fatal faults, but attempting to service all is the easiest
            // way to do that.
            status = service_fault_batch_dispatch(va_space,
                                                  gpu_va_space,
                                                  batch_context,
                                                  service_context,
                                                  &batch_context->ats_context,
                                                  &batch_context->tracker,
                                                  i,
                                                  &block_faults,
                                                  false,
                                                  hmm_migratable);
            if (status != NV_OK) {
                // TODO: Bug 3900733: clean up locking in service_fault_batch().
                // We need to drop lock and retry. That means flushing and
                // starting over.
                if (status == NV_WARN_MORE_PROCESSING_REQUIRED || status == NV_WARN_MISMATCHED_TARGET)
                    status = NV_OK;

                break;
            }

            // Invalidate TLBs before cancel to ensure that fatal faults don't
            // get stuck in HW behind non-fatal faults to the same line.
            status = uvm_ats_invalidate_tlbs(gpu_va_space, ats_invalidate, &batch_context->tracker);
            if (status != NV_OK)
                break;

            while (block_faults-- > 0) {
                current_entry = batch_context->ordered_fault_cache[i];
                if (current_entry->is_fatal) {
                    status = cancel_fault_precise_va(current_entry, current_entry->replayable.cancel_va_mode);
                    if (status != NV_OK)
                        break;
                }

                ++i;
            }
        }
    }

done:
    uvm_va_space_up_read(va_space);
    uvm_va_space_mm_release_unlock(va_space, mm);

    if (status == NV_OK) {
        // There are two reasons to flush the fault buffer here.
        //
        // 1) Functional. We need to replay both the serviced non-fatal faults
        //    and the skipped faults in other VA spaces. The former need to be
        //    restarted and the latter need to be replayed so the normal fault
        //    service mechanism can fetch and process them.
        //
        // 2) Performance. After cancelling the fatal faults, a flush removes
        //    any potential duplicated fault that may have been added while
        //    processing the faults in this batch. This flush also avoids doing
        //    unnecessary processing after the fatal faults have been cancelled,
        //    so all the rest are unlikely to remain after a replay because the
        //    context is probably in the process of dying.
        status = fault_buffer_flush_locked(gpu->parent,
                                           gpu,
                                           UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT,
                                           UVM_FAULT_REPLAY_TYPE_START,
                                           batch_context);
    }

    return status;
}

// Service a contiguous range [first_index, outer_index) of the ordered fault
// cache: scan it, group faults by va_block (managed) and service each block
// in batch; service non-managed faults as they are encountered during the
// scan. Fatal faults are marked for later processing by the caller.
//
// This is the loop the stock driver runs over the whole batch on the
// bottom-half thread. Under parallel servicing every worker runs it over its
// assigned spans with its own privatized scratch state (service_context,
// ats_context, ats_invalidate, tracker), so invocations never share mutable
// service state. Everything the loop takes - va_space/mmap read locks,
// va_block locks, the retry protocol - is already safe under concurrent
// invocation: two parent GPUs' bottom halves service faults of the same
// va_space concurrently in the stock driver today. Ranges must never split a
// va_block (see uvm_fault_service_span_t).
static NV_STATUS service_fault_batch_range(uvm_parent_gpu_t *parent_gpu,
                                           fault_service_mode_t service_mode,
                                           uvm_fault_service_batch_context_t *batch_context,
                                           uvm_service_block_context_t *service_context,
                                           uvm_ats_fault_context_t *ats_context,
                                           uvm_ats_fault_invalidate_t *ats_invalidate,
                                           uvm_tracker_t *tracker,
                                           NvU32 first_index,
                                           NvU32 outer_index)
{
    NV_STATUS status = NV_OK;
    NvU32 i;
    uvm_va_space_t *va_space = NULL;
    uvm_gpu_va_space_t *prev_gpu_va_space = NULL;
    struct mm_struct *mm = NULL;
    const bool replay_per_va_block = service_mode != FAULT_SERVICE_MODE_CANCEL &&
                                     parent_gpu->fault_buffer.replayable.replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BLOCK;
    uvm_va_block_context_t *va_block_context = service_context->block_context;
    bool hmm_migratable = true;
    NvU64 va_space_lock_wait_start;

    ats_invalidate->tlb_batch_pending = false;

    for (i = first_index; i < outer_index;) {
        NvU32 block_faults;
        uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];
        uvm_fault_utlb_info_t *utlb = &batch_context->utlbs[current_entry->fault_source.utlb_id];
        uvm_gpu_va_space_t *gpu_va_space;

        UVM_ASSERT(current_entry->va_space);

        if (current_entry->va_space != va_space) {
            if (prev_gpu_va_space) {
                // TLB entries are invalidated per GPU VA space
                status = uvm_ats_invalidate_tlbs(prev_gpu_va_space, ats_invalidate, tracker);
                if (status != NV_OK)
                    goto fail;

                prev_gpu_va_space = NULL;
            }

            // Fault on a different va_space, drop the lock of the old one...
            if (va_space) {
                uvm_va_space_up_read(va_space);
                uvm_va_space_mm_release_unlock(va_space, mm);
                mm = NULL;
            }

            va_space = current_entry->va_space;

            // ... and take the lock of the new one

            // If an mm is registered with the VA space, we have to retain it
            // in order to lock it before locking the VA space. It is guaranteed
            // to remain valid until we release. If no mm is registered, we
            // can only service managed faults, not ATS/HMM faults.
            mm = uvm_va_space_mm_retain_lock(va_space);
            uvm_va_block_context_init(va_block_context, mm);

            // Brackets the VA space lock only. uvm_va_space_mm_retain_lock
            // above takes mmap_lock, whose wait is often the larger of the
            // two, so this counter under-reports the total stall at a va_space
            // transition rather than over-reporting it.
            va_space_lock_wait_start = uvm_lock_probe_begin();

            uvm_va_space_down_read(va_space);

            uvm_lock_probe_end(va_space_lock_wait_start,
                               &g_uvm_lock_contention_stats.ns_va_space_lock_wait,
                               &g_uvm_lock_contention_stats.n_va_space_lock_acqs);
        }

        // Some faults could be already fatal if they cannot be handled by
        // the UVM driver
        if (current_entry->is_fatal) {
            ++i;
            fault_batch_publish_fatal_va_space(batch_context, va_space, current_entry->gpu);

            utlb->has_fatal_faults = true;
            UVM_ASSERT(utlb->num_pending_faults > 0);
            continue;
        }

        gpu_va_space = uvm_gpu_va_space_get(va_space, current_entry->gpu);

        if (prev_gpu_va_space && prev_gpu_va_space != gpu_va_space) {
            status = uvm_ats_invalidate_tlbs(prev_gpu_va_space, ats_invalidate, tracker);
            if (status != NV_OK)
                goto fail;
        }

        prev_gpu_va_space = gpu_va_space;

        // If there is no GPU VA space for the GPU, ignore the fault. This
        // can happen if a GPU VA space is destroyed without explicitly
        // freeing all memory ranges and there are stale entries in the
        // buffer that got fixed by the servicing in a previous batch.
        if (!gpu_va_space) {
            ++i;
            continue;
        }

        status = service_fault_batch_dispatch(va_space,
                                              gpu_va_space,
                                              batch_context,
                                              service_context,
                                              ats_context,
                                              tracker,
                                              i,
                                              &block_faults,
                                              replay_per_va_block,
                                              hmm_migratable);
        // TODO: Bug 3900733: clean up locking in service_fault_batch().
        if (status == NV_WARN_MORE_PROCESSING_REQUIRED || status == NV_WARN_MISMATCHED_TARGET) {
            if (status == NV_WARN_MISMATCHED_TARGET)
                hmm_migratable = false;
            uvm_va_space_up_read(va_space);
            uvm_va_space_mm_release_unlock(va_space, mm);
            mm = NULL;
            va_space = NULL;
            prev_gpu_va_space = NULL;
            status = NV_OK;
            continue;
        }

        if (status != NV_OK)
            goto fail;

        hmm_migratable = true;
        i += block_faults;

        // A dispatch call must never consume faults beyond the range it was
        // given. Spans are cut on va_block boundaries and every dispatch path
        // stops at the end of a va_block, so this holds for a span as it does
        // for the whole batch. If it ever fires under the worker pool, two
        // workers are servicing the same faults.
        UVM_ASSERT(i <= outer_index);

        // Don't issue replays in cancel mode
        if (replay_per_va_block && !batch_context->fatal_va_space) {
            status = push_replay_on_gpu(gpu_va_space->gpu, UVM_FAULT_REPLAY_TYPE_START, batch_context);
            if (status != NV_OK)
                goto fail;

            // Increment the batch id if UVM_PERF_FAULT_REPLAY_POLICY_BLOCK
            // is used, as we issue a replay after servicing each VA block
            // and we can service a number of VA blocks before returning.
            ++batch_context->batch_id;
        }
    }

    if (prev_gpu_va_space) {
        NV_STATUS invalidate_status = uvm_ats_invalidate_tlbs(prev_gpu_va_space, ats_invalidate, tracker);
        if (invalidate_status != NV_OK)
            status = invalidate_status;
    }

fail:
    if (va_space) {
        uvm_va_space_up_read(va_space);
        uvm_va_space_mm_release_unlock(va_space, mm);
    }

    return status;
}

// Build the span partition of ordered_fault_cache[0..num_coalesced_faults) in
// service_pool.spans: one span per maximal run with equal
// (va_space, gpu, UVM_VA_BLOCK_SIZE-aligned address) key. Returns the span
// count. Runs on the dispatcher before workers are kicked.
static NvU32 fault_service_build_spans(uvm_parent_gpu_t *parent_gpu,
                                       uvm_fault_service_batch_context_t *batch_context)
{
    uvm_fault_service_span_t *spans = parent_gpu->fault_buffer.replayable.service_pool.spans;
    NvU32 num_spans = 0;
    NvU32 i;

    for (i = 0; i < batch_context->num_coalesced_faults; i++) {
        uvm_fault_buffer_entry_t *entry = batch_context->ordered_fault_cache[i];

        if (num_spans > 0) {
            uvm_fault_buffer_entry_t *prev = batch_context->ordered_fault_cache[i - 1];

            if (entry->va_space == prev->va_space &&
                entry->gpu == prev->gpu &&
                UVM_VA_BLOCK_ALIGN_DOWN(entry->fault_address) == UVM_VA_BLOCK_ALIGN_DOWN(prev->fault_address)) {
                spans[num_spans - 1].end = i + 1;
                continue;
            }
        }

        spans[num_spans].begin = i;
        spans[num_spans].end = i + 1;
        num_spans++;
    }

    return num_spans;
}

// One control decision. Called by the dispatcher at batch boundaries, which
// the ISR service_lock serialises per GPU, so this needs no locking and
// touches only dispatcher-private state plus one integer the dispatcher also
// owns.
//
// Fixed-point throughout: milli-evictions per batch, so there is no floating
// point in kernel context. The EWMA is the standard first-order filter
// (Hellerstein s8.4.3, w(k+1) = c*w(k) + (1-c)*y(k)) with c = 0.8, which is
// what adapt_sim.py settled on.
static void fault_service_adapt_tick(uvm_replayable_fault_buffer_t *replayable_faults)
{
    NvU64 evictions, batches, d_evict, d_batch;
    NvU32 rate_milli, lo, hi, width, step, narrow_every;

    if (!uvm_perf_fault_service_adapt || replayable_faults->service_pool.num_workers == 0)
        return;

    batches = replayable_faults->stats.num_batches;
    d_batch = batches - replayable_faults->service_pool.adapt_last_batches;

    // max(epoch, 1): the parameter is writable by an operator and a zero epoch
    // would both tick every batch and divide by zero below.
    if (d_batch < max(uvm_perf_fault_service_adapt_epoch, 1u))
        return;

    // The signal. Module-global rather than per-GPU: this box has one GPU, and
    // the multi-GPU case (one GPU's memory pressure widening every pool) is a
    // stated scope limit rather than an oversight.
    //
    // It also counts evictions from the RM PMA callback path
    // (uvm_pmm_gpu_pma_evict_pages, PMM_CONTEXT_PMA_EVICTION), not only the
    // fault path. That is defensible, since RM asking UVM to free memory is
    // genuine memory pressure and is exactly what should widen the pool, and it
    // is moot in practice: the 07-22 campaign measured n_pma_evict_cbs = 0 on
    // every cell. Worth knowing if that ever stops being true.
    evictions = uvm_lock_stat_sum(&g_uvm_lock_contention_stats.n_evict_calls);
    d_evict = evictions - replayable_faults->service_pool.adapt_last_evictions;

    replayable_faults->service_pool.adapt_last_batches = batches;
    replayable_faults->service_pool.adapt_last_evictions = evictions;

    // Fixed point, and do_div rather than a bare 64-bit divide: this driver
    // has no general 64-bit division helper (only uvm_div_pow2_*, which needs
    // a power-of-two divisor), and do_div is what nv-linux.h already provides.
    {
        NvU64 scaled = d_evict * 1000;
        // do_div wants a 32-bit divisor. d_batch is the batch delta over one
        // epoch, so it is bounded by the epoch plus one bottom half's worth of
        // batches: thousands, never near 2^32.
        NvU32 divisor = (d_batch > 0xFFFFFFFFULL) ? 0xFFFFFFFFu : (NvU32)d_batch;

        do_div(scaled, divisor);

        // Clamp before it reaches the filter. The EWMA below multiplies by 4,
        // so an unbounded sample could overflow NvU32; 1e6 milli is a
        // thousand evictions per batch, twenty times the heaviest rate any
        // campaign has produced, so the clamp is a guard and not a limit.
        rate_milli = (scaled > 1000000ULL) ? 1000000u : (NvU32)scaled;
    }

    // EWMA at c = 0.8: new = (4*old + 1*sample) / 5 (Hellerstein s8.4.3)
    replayable_faults->service_pool.adapt_ewma_milli =
        (replayable_faults->service_pool.adapt_ewma_milli * 4 + rate_milli) / 5;

    // Clamp the operator-settable parameters here rather than trusting them.
    // module_param does no range checking, and each of these has a value that
    // misbehaves rather than merely performing badly:
    //
    //   step 0            never widens, so the pool is decorative
    //   step near UINT_MAX  width + step wraps and min() picks the wrapped
    //                     value, which can shrink the pool on a widen
    //   narrow_every 0    "++ticks >= 0" is always true, so it narrows every
    //                     epoch. That is exactly the configuration adapt_sim.py
    //                     shows slamming the actuator across its full range,
    //                     the limit cycle of Hellerstein Fig 8.9
    //   lo > hi           both branches would qualify; widen wins by ordering,
    //                     so it would only ever grow
    //
    // Values below narrow_every 4 are permitted but known unstable under an
    // alternating-regime workload; they are left reachable so the failure can
    // be reproduced deliberately.
    // min/max rather than clamp: those two are already used in this file for
    // exactly this kind of bound (see the num_workers clamp at pool init),
    // clamp is not used anywhere in the driver.
    step = min(max(uvm_perf_fault_service_adapt_step, 1u),
               (unsigned)UVM_PERF_FAULT_SERVICE_MAX_WORKERS);
    narrow_every = max(uvm_perf_fault_service_adapt_narrow_every, 1u);
    hi = uvm_perf_fault_service_adapt_hi;
    lo = min(uvm_perf_fault_service_adapt_lo, hi);
    width = replayable_faults->service_pool.active_workers;

    if (replayable_faults->service_pool.adapt_ewma_milli > hi) {
        width = min(width + step, replayable_faults->service_pool.num_workers);
        replayable_faults->service_pool.adapt_narrow_ticks = 0;
    }
    else if (replayable_faults->service_pool.adapt_ewma_milli < lo) {
        if (++replayable_faults->service_pool.adapt_narrow_ticks >= narrow_every) {
            if (width > 1)
                width--;
            replayable_faults->service_pool.adapt_narrow_ticks = 0;
        }
    }
    // else: hold. The deadband, and the reason a boundary workload is quiet.

    // Observability. Counted per decision including the holds, so
    // sum/decisions is the mean width the workload actually ran at, and the
    // widen/narrow counts show how much the controller moved to get there.
    if (width > replayable_faults->service_pool.active_workers)
        uvm_lock_stat_inc(&g_uvm_lock_contention_stats.n_adapt_widen);
    else if (width < replayable_faults->service_pool.active_workers)
        uvm_lock_stat_inc(&g_uvm_lock_contention_stats.n_adapt_narrow);

    uvm_lock_stat_inc(&g_uvm_lock_contention_stats.n_adapt_decisions);
    uvm_lock_stat_add(&g_uvm_lock_contention_stats.sum_adapt_width, width);

    replayable_faults->service_pool.active_workers = width;
}

// Cut the span array into num_bins CONTIGUOUS ranges of approximately equal
// fault weight and hand one to each worker slot.
//
// Contiguous rather than a scatter, for three reasons that all follow from the
// span array being in ascending (va_space, gpu, address) order:
//
//   - A worker's whole share is one service_fault_batch_range() call, so the
//     va_space and mmap locks are taken once per worker per batch instead of
//     once per va_block.
//   - Servicing walks ascending addresses, which is the order
//     preprocess_fault_batch() established and which the prefetch predictor
//     and the copy path are both built around. The previous shape sorted this
//     array by weight in place and destroyed that order.
//   - A worker finds its work in O(1) instead of scanning every span looking
//     for the ones it owns, which was O(spans * workers) per batch.
//
// The cut points are chosen to minimise the heaviest bin exactly, not
// greedily. A span is one va_block and cannot be split, so the makespan can
// never beat the heaviest single span, but between that floor and an even
// share there is real room: measured against the optimum over 200,000 random
// partitions, filling each bin to a running even-share target is 1.21x worse
// on average and up to 1.97x worse when the weight distribution has a heavy
// tail. On w7 the mean span is 1.08 faults and the two agree, but the
// real-application workloads fault whole tensors into one va_block and are
// exactly the heavy-tail case.
//
// Ranges are half-open indices into ordered_fault_cache, not into the span
// array, because that is what service_fault_batch_range() takes. Slots that
// get nothing are given an empty range rather than left stale.

// Smallest per-bin weight that still admits a partition into at most num_bins
// contiguous parts. Binary search on the capacity, with the feasibility test
// being one first-fit walk: standard linear partitioning, O(num_spans * log
// total_weight). At about 170 spans and 180 faults per batch that is roughly
// 1,400 integer operations once per batch on the dispatcher, against the
// 0.285 ms of CPU a batch costs it.
static NvU32 fault_service_span_capacity(const uvm_fault_service_span_t *spans,
                                         NvU32 num_spans,
                                         NvU32 num_bins)
{
    NvU32 lo = 0;
    NvU32 hi = 0;
    NvU32 i;

    // The search range: no bin can hold less than the heaviest single span,
    // and one bin holding everything is always feasible.
    for (i = 0; i < num_spans; i++) {
        NvU32 weight = spans[i].end - spans[i].begin;

        hi += weight;
        if (weight > lo)
            lo = weight;
    }

    while (lo < hi) {
        NvU32 cap = lo + (hi - lo) / 2;
        NvU32 bins = 1;
        NvU32 load = 0;

        for (i = 0; i < num_spans; i++) {
            NvU32 weight = spans[i].end - spans[i].begin;

            // cap >= lo >= the heaviest span, so a fresh bin always admits the
            // span that overflowed the previous one and this cannot loop.
            if (load + weight > cap) {
                if (++bins > num_bins)
                    break;
                load = weight;
            }
            else {
                load += weight;
            }
        }

        if (bins <= num_bins)
            hi = cap;
        else
            lo = cap + 1;
    }

    return lo;
}

static void fault_service_assign_spans(uvm_replayable_fault_buffer_t *replayable_faults,
                                       NvU32 num_spans,
                                       NvU32 num_bins)
{
    const uvm_fault_service_span_t *spans = replayable_faults->service_pool.spans;
    uvm_fault_service_worker_t *workers = replayable_faults->service_pool.workers;
    NvU32 cap;
    NvU32 load = 0;
    NvU32 bin = 0;
    NvU32 i;

    UVM_ASSERT(num_spans > 0);
    UVM_ASSERT(num_bins > 0);
    UVM_ASSERT(num_bins <= replayable_faults->service_pool.num_workers + 1);

    cap = fault_service_span_capacity(spans, num_spans, num_bins);

    // Emit the cuts with the same first-fit walk the capacity was chosen for,
    // so the partition it produces is the one proved feasible above.
    workers[0].range_begin = spans[0].begin;

    for (i = 0; i < num_spans; i++) {
        NvU32 weight = spans[i].end - spans[i].begin;

        if (load > 0 && load + weight > cap) {
            workers[bin].range_end = spans[i].begin;
            bin++;
            UVM_ASSERT(bin < num_bins);
            workers[bin].range_begin = spans[i].begin;
            load = 0;
        }

        load += weight;
    }

    workers[bin].range_end = spans[num_spans - 1].end;

    // Fewer parts than bins, which happens whenever the heaviest span alone
    // sets the capacity. Those slots are scheduled but return immediately.
    for (bin++; bin < num_bins; bin++) {
        workers[bin].range_begin = 0;
        workers[bin].range_end = 0;
    }
}

// Service this worker's range of the batch. Runs inline on the bottom-half
// thread for slot 0 and on a service_pool queue for the other slots. An error
// is recorded for this worker but does not stop the others; the first error
// wins at join.
//
// One call, not one per va_block: the range is contiguous and cut on span
// boundaries, so service_fault_batch_range() groups it by va_block internally
// exactly as the serial path groups the whole batch, and takes the va_space
// and mmap locks once for the whole range.
static void fault_service_worker_run(uvm_fault_service_worker_t *worker)
{
    if (worker->range_begin == worker->range_end)
        return;

    worker->status = service_fault_batch_range(worker->parent_gpu,
                                               FAULT_SERVICE_MODE_REGULAR,
                                               worker->batch_context,
                                               &worker->block_service_context,
                                               &worker->ats_context,
                                               &worker->ats_invalidate,
                                               &worker->tracker,
                                               worker->range_begin,
                                               worker->range_end);
}

static void fault_service_worker_entry_internal(void *args)
{
    uvm_fault_service_worker_t *worker = (uvm_fault_service_worker_t *)args;
    uvm_replayable_fault_buffer_t *replayable_faults = &worker->parent_gpu->fault_buffer.replayable;

    UVM_ASSERT(worker->slot != 0);

    fault_service_worker_run(worker);

    if (atomic_dec_and_test(&replayable_faults->service_pool.outstanding))
        wake_up(&replayable_faults->service_pool.done_wq);
}

static void fault_service_worker_entry(void *args)
{
    UVM_ENTRY_VOID(fault_service_worker_entry_internal(args));
}

// Scan the ordered view of faults and group them by different va_blocks
// (managed faults) and service faults for each va_block, in batch.
// Service non-managed faults one at a time as they are encountered during the
// scan.
//
// Fatal faults are marked for later processing by the caller.
//
// When the worker pool is enabled the batch is partitioned into va_block
// spans, spread across the pool with LPT, and serviced concurrently: the
// dispatcher services its own share inline, then joins the workers and merges
// their trackers into the batch tracker BEFORE the caller acquires it for the
// replay push. Otherwise this reduces to one service_fault_batch_range() call
// over the whole batch with the same shared scratch state the stock driver
// uses - the exact serial path.
static NV_STATUS service_fault_batch(uvm_parent_gpu_t *parent_gpu,
                                     fault_service_mode_t service_mode,
                                     uvm_fault_service_batch_context_t *batch_context)
{
    NV_STATUS status;
    NvU32 k;
    NvU32 num_spans;
    NvU32 not_queued = 0;
    uvm_replayable_fault_buffer_t *replayable_faults = &parent_gpu->fault_buffer.replayable;
    NvU32 pool_workers = replayable_faults->service_pool.num_workers;
    NvU32 num_workers;

    // One control decision per epoch, before this batch is partitioned, so the
    // width used below is the one the controller just chose.
    fault_service_adapt_tick(replayable_faults);

    // The width THIS batch may use. Equal to the allocated pool width unless
    // the adaptive policy has narrowed it. Slots above it get no spans and are
    // never scheduled, but they remain allocated and their state is still
    // reset and merged below, so a later widening finds them clean.
    num_workers = replayable_faults->service_pool.active_workers;

    // not_queued is a bitmask over worker slots
    BUILD_BUG_ON(UVM_PERF_FAULT_SERVICE_MAX_WORKERS >= 8 * sizeof(not_queued));

    // Serial fallbacks: pool disabled, cancel mode (rare, correctness-first),
    // per-block replay policy (would need ordered replay pushes from the
    // workers), confidential computing (CE encryption state is
    // single-threaded), ATS (see below), and batches too small to amortize the
    // dispatch/join.
    //
    // ATS: a span is one va_block, but service_fault_batch_ats_sub() consumes
    // faults up to the next UVM_GMMU_ATS_GRANULARITY (512MB) or vma boundary,
    // whichever comes first, so one dispatch call can reach past its span and
    // into faults another worker owns. gpu_va_space->ats.enabled implies
    // g_uvm_global.ats.enabled, so this covers every ATS-capable VA space.
    if (num_workers == 0 ||
        service_mode == FAULT_SERVICE_MODE_CANCEL ||
        replayable_faults->replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BLOCK ||
        g_uvm_global.conf_computing_enabled ||
        g_uvm_global.ats.enabled ||
        batch_context->num_coalesced_faults < uvm_perf_fault_service_min_faults) {
        return service_fault_batch_range(parent_gpu,
                                         service_mode,
                                         batch_context,
                                         &replayable_faults->block_service_context,
                                         &batch_context->ats_context,
                                         &replayable_faults->ats_invalidate,
                                         &batch_context->tracker,
                                         0,
                                         batch_context->num_coalesced_faults);
    }

    num_spans = fault_service_build_spans(parent_gpu, batch_context);

    // A single span is one va_block and cannot be split; service it serially
    if (num_spans < 2) {
        return service_fault_batch_range(parent_gpu,
                                         service_mode,
                                         batch_context,
                                         &replayable_faults->block_service_context,
                                         &batch_context->ats_context,
                                         &replayable_faults->ats_invalidate,
                                         &batch_context->tracker,
                                         0,
                                         batch_context->num_coalesced_faults);
    }

    replayable_faults->service_pool.num_spans = num_spans;

    // Reset the WHOLE pool, not just the active part, and do it BEFORE the
    // assignment so the assignment is what survives. A slot the controller has
    // narrowed away gets no range and is never scheduled, but resetting it
    // anyway means a later widening finds it clean rather than carrying a
    // status, or a range, from whenever it last ran. The loop is a handful of
    // stores.
    for (k = 0; k < pool_workers + 1; k++) {
        uvm_fault_service_worker_t *worker = &replayable_faults->service_pool.workers[k];

        worker->status = NV_OK;
        worker->batch_context = batch_context;
        worker->range_begin = 0;
        worker->range_end = 0;
        UVM_ASSERT(uvm_tracker_is_empty(&worker->tracker));
    }

    fault_service_assign_spans(replayable_faults, num_spans, num_workers + 1);

    // outstanding must account for every worker before the first one is
    // queued: a worker can run to completion and decrement it while this loop
    // is still handing out items.
    atomic_set(&replayable_faults->service_pool.outstanding, num_workers);

    for (k = 1; k < num_workers + 1; k++) {
        uvm_fault_service_worker_t *worker = &replayable_faults->service_pool.workers[k];

        // The item must be off its queue's list here or the refusal fallback
        // below would be wrong: a still-queued item WILL eventually run, so
        // re-running it inline would double-service its spans and decrement
        // outstanding twice. The previous batch's join guarantees the node is
        // empty: the queue thread removes the item from the list before
        // invoking it, and outstanding reaches zero only after the run
        // returns.
        UVM_ASSERT(list_empty(&worker->q_item.q_list_node));

        if (nv_kthread_q_schedule_q_item(&replayable_faults->service_pool.queues[k - 1], &worker->q_item))
            continue;

        // With the item known not to be queued, the only way the schedule can
        // be refused is the queue being stopped, and then nothing will ever
        // run the item or decrement outstanding on its behalf: without this
        // the join below would sleep forever and this GPU would never service
        // another fault. Take the slot's spans back and run them inline.
        not_queued |= 1u << k;
        atomic_dec(&replayable_faults->service_pool.outstanding);
    }

    // The dispatcher is worker 0: service our own share while the pool runs
    fault_service_worker_run(&replayable_faults->service_pool.workers[0]);

    for (k = 1; not_queued != 0 && k < num_workers + 1; k++) {
        if (not_queued & (1u << k))
            fault_service_worker_run(&replayable_faults->service_pool.workers[k]);
    }

    // Join BEFORE replay: the caller acquires the batch tracker for the
    // replay push, so every worker's migrations must be merged in first. The
    // dispatcher holds only the ISR-order service_lock here (no va_space or
    // mmap locks), so workers can never be blocked waiting on us.
    wait_event(replayable_faults->service_pool.done_wq,
               atomic_read(&replayable_faults->service_pool.outstanding) == 0);

    // Pairs with the atomic_dec_and_test() each worker performs after writing
    // its tracker and status. wait_event()'s fast path reads the counter
    // without a barrier, so order that read against the reads below.
    smp_rmb();

    status = NV_OK;

    // Merge across the WHOLE pool for the same reason the reset above spans it.
    // An inactive slot contributes an empty tracker and NV_OK, so this is a
    // no-op for it, and it guarantees nothing is left behind if the controller
    // narrows between one batch and the next.
    for (k = 0; k < pool_workers + 1; k++) {
        uvm_fault_service_worker_t *worker = &replayable_faults->service_pool.workers[k];
        NV_STATUS tracker_status = uvm_tracker_add_tracker_safe(&batch_context->tracker, &worker->tracker);

        uvm_tracker_clear(&worker->tracker);

        if (status == NV_OK)
            status = worker->status == NV_OK ? tracker_status : worker->status;
    }

    return status;
}

// Tells if the given fault entry is the first one in its uTLB
static bool is_first_fault_in_utlb(uvm_fault_service_batch_context_t *batch_context, NvU32 fault_index)
{
    NvU32 i;
    NvU32 utlb_id = batch_context->fault_cache[fault_index].fault_source.utlb_id;

    for (i = 0; i < fault_index; ++i) {
        uvm_fault_buffer_entry_t *current_entry = &batch_context->fault_cache[i];

        // We have found a prior fault in the same uTLB
        if (current_entry->fault_source.utlb_id == utlb_id)
            return false;
    }

    return true;
}

// Compute the number of fatal and non-fatal faults for a page in the given uTLB
static void faults_for_page_in_utlb(uvm_fault_service_batch_context_t *batch_context,
                                    uvm_va_space_t *va_space,
                                    NvU64 addr,
                                    NvU32 utlb_id,
                                    NvU32 *fatal_faults,
                                    NvU32 *non_fatal_faults)
{
    uvm_gpu_t *gpu = NULL;
    NvU32 i;

    *fatal_faults = 0;
    *non_fatal_faults = 0;

    // Fault filtering is not allowed in the TLB-based fault cancel path
    UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

    for (i = 0; i < batch_context->num_cached_faults; ++i) {
        uvm_fault_buffer_entry_t *current_entry = &batch_context->fault_cache[i];

        if (!gpu)
            gpu = current_entry->gpu;
        else
            UVM_ASSERT(current_entry->gpu == gpu);

        if (current_entry->fault_source.utlb_id == utlb_id &&
            current_entry->va_space == va_space &&
            current_entry->fault_address == addr) {
            // We have found the page
            if (current_entry->is_fatal)
                ++(*fatal_faults);
            else
                ++(*non_fatal_faults);
        }
    }
}

// Function that tells if there are addresses (reminder: they are aligned to 4K)
// with non-fatal faults only
static bool no_fatal_pages_in_utlb(uvm_fault_service_batch_context_t *batch_context,
                                   NvU32 start_index,
                                   NvU32 utlb_id)
{
    NvU32 i;

    // Fault filtering is not allowed in the TLB-based fault cancel path
    UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

    for (i = start_index; i < batch_context->num_cached_faults; ++i) {
        uvm_fault_buffer_entry_t *current_entry = &batch_context->fault_cache[i];

        if (current_entry->fault_source.utlb_id == utlb_id) {
            // We have found a fault for the uTLB
            NvU32 fatal_faults;
            NvU32 non_fatal_faults;

            faults_for_page_in_utlb(batch_context,
                                    current_entry->va_space,
                                    current_entry->fault_address,
                                    utlb_id,
                                    &fatal_faults,
                                    &non_fatal_faults);

            if (non_fatal_faults > 0 && fatal_faults == 0)
                return true;
        }
    }

    return false;
}

static void record_fatal_fault_helper(uvm_fault_buffer_entry_t *entry, UvmEventFatalReason reason)
{
    uvm_va_space_t *va_space = entry->va_space;
    uvm_gpu_t *gpu = entry->gpu;

    UVM_ASSERT(va_space);
    UVM_ASSERT(gpu);

    uvm_va_space_down_read(va_space);
    // Record fatal fault event
    uvm_tools_record_gpu_fatal_fault(gpu->id, va_space, entry, reason);
    uvm_va_space_up_read(va_space);
}

// This function tries to find and issue a cancel for each uTLB that meets
// the requirements to guarantee precise fault attribution:
// - No new faults can arrive on the uTLB (uTLB is in lockdown)
// - The first fault in the buffer for a specific uTLB is fatal
// - There are no other addresses in the uTLB with non-fatal faults only
//
// This function and the related helpers iterate over faults as read from HW,
// not through the ordered fault view
//
// TODO: Bug 1766754
// This is very costly, although not critical for performance since we are
// cancelling.
// - Build a list with all the faults within a uTLB
// - Sort by uTLB id
static NV_STATUS try_to_cancel_utlbs(uvm_fault_service_batch_context_t *batch_context)
{
    NvU32 i;

    // Fault filtering is not allowed in the TLB-based fault cancel path
    UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

    for (i = 0; i < batch_context->num_cached_faults; ++i) {
        uvm_fault_buffer_entry_t *current_entry = &batch_context->fault_cache[i];
        uvm_fault_utlb_info_t *utlb = &batch_context->utlbs[current_entry->fault_source.utlb_id];
        NvU32 gpc_id = current_entry->fault_source.gpc_id;
        NvU32 utlb_id = current_entry->fault_source.utlb_id;
        NvU32 client_id = current_entry->fault_source.client_id;
        uvm_gpu_t *gpu = current_entry->gpu;

        // Only fatal faults are considered
        if (!current_entry->is_fatal)
            continue;

        // Only consider uTLBs in lock-down
        if (!utlb->in_lockdown)
            continue;

        // Issue a single cancel per uTLB
        if (utlb->cancelled)
            continue;

        if (is_first_fault_in_utlb(batch_context, i) &&
            !no_fatal_pages_in_utlb(batch_context, i + 1, utlb_id)) {
            NV_STATUS status;

            record_fatal_fault_helper(current_entry, current_entry->fatal_reason);

            status = push_cancel_on_gpu_targeted(gpu,
                                                 current_entry->instance_ptr,
                                                 gpc_id,
                                                 client_id,
                                                 &batch_context->tracker);
            if (status != NV_OK)
                return status;

            utlb->cancelled = true;
        }
    }

    return NV_OK;
}

static NvU32 find_fatal_fault_in_utlb(uvm_fault_service_batch_context_t *batch_context,
                                      NvU32 utlb_id)
{
    NvU32 i;

    // Fault filtering is not allowed in the TLB-based fault cancel path
    UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

    for (i = 0; i < batch_context->num_cached_faults; ++i) {
        if (batch_context->fault_cache[i].is_fatal &&
            batch_context->fault_cache[i].fault_source.utlb_id == utlb_id)
            return i;
    }

    return i;
}

static NvU32 is_fatal_fault_in_buffer(uvm_fault_service_batch_context_t *batch_context,
                                      uvm_fault_buffer_entry_t *fault)
{
    NvU32 i;

    // Fault filtering is not allowed in the TLB-based fault cancel path
    UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

    for (i = 0; i < batch_context->num_cached_faults; ++i) {
        uvm_fault_buffer_entry_t *current_entry = &batch_context->fault_cache[i];
        if (cmp_fault_instance_ptr(current_entry, fault) == 0 &&
            current_entry->fault_address == fault->fault_address &&
            current_entry->fault_access_type == fault->fault_access_type &&
            current_entry->fault_source.utlb_id == fault->fault_source.utlb_id) {
            return true;
        }
    }

    return false;
}

// Cancel all faults in the given fault service batch context, even those not
// marked as fatal.
static NV_STATUS cancel_faults_all(uvm_fault_service_batch_context_t *batch_context, UvmEventFatalReason reason)
{
    NV_STATUS status = NV_OK;
    NV_STATUS fault_status;
    uvm_gpu_t *gpu = NULL;
    NvU32 i = 0;

    UVM_ASSERT(reason != UvmEventFatalReasonInvalid);
    UVM_ASSERT(batch_context->num_coalesced_faults > 0);

    while (i < batch_context->num_coalesced_faults && status == NV_OK) {
        uvm_fault_buffer_entry_t *current_entry = batch_context->ordered_fault_cache[i];
        uvm_va_space_t *va_space = current_entry->va_space;
        bool skip_gpu_va_space;

        gpu = current_entry->gpu;
        UVM_ASSERT(gpu);
        UVM_ASSERT(va_space);

        uvm_va_space_down_read(va_space);

        // If there is no GPU VA space for the GPU, ignore all faults in
        // that GPU VA space. This can happen if the GPU VA space has been
        // destroyed since we unlocked the VA space in service_fault_batch.
        // Ignoring the fault avoids targetting a PDB that might have been
        // reused by another process.
        skip_gpu_va_space = !uvm_gpu_va_space_get(va_space, gpu);

        for (;
             i < batch_context->num_coalesced_faults &&
                 current_entry->va_space == va_space &&
                 current_entry->gpu == gpu;
             current_entry = batch_context->ordered_fault_cache[++i]) {
            uvm_fault_cancel_va_mode_t cancel_va_mode;

            if (skip_gpu_va_space)
                continue;

            if (current_entry->is_fatal) {
                UVM_ASSERT(current_entry->fatal_reason != UvmEventFatalReasonInvalid);
                cancel_va_mode = current_entry->replayable.cancel_va_mode;
            }
            else {
                current_entry->fatal_reason = reason;
                cancel_va_mode = UVM_FAULT_CANCEL_VA_MODE_ALL;
            }

            status = cancel_fault_precise_va(current_entry, cancel_va_mode);
            if (status != NV_OK)
                break;
        }

        uvm_va_space_up_read(va_space);
    }

    // Because each cancel itself triggers a replay, there may be a large number
    // of new duplicated faults in the buffer after cancelling all the known
    // ones. Flushing the buffer discards them to avoid unnecessary processing.
    // Note that we are using one of the GPUs with a fault, but the choice of
    // which one is arbitrary.
    fault_status = fault_buffer_flush_locked(gpu->parent,
                                             gpu,
                                             UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT,
                                             UVM_FAULT_REPLAY_TYPE_START,
                                             batch_context);

    // We report the first encountered error.
    if (status == NV_OK)
        status = fault_status;

    return status;
}

// Function called when the system has found a global error and needs to
// trigger RC in RM.
static void cancel_fault_batch_tlb(uvm_fault_service_batch_context_t *batch_context, UvmEventFatalReason reason)
{
    NvU32 i;

    for (i = 0; i < batch_context->num_coalesced_faults; ++i) {
        NV_STATUS status = NV_OK;
        uvm_fault_buffer_entry_t *current_entry;
        uvm_fault_buffer_entry_t *coalesced_entry;
        uvm_va_space_t *va_space;
        uvm_gpu_t *gpu;

        current_entry = batch_context->ordered_fault_cache[i];
        va_space = current_entry->va_space;
        gpu = current_entry->gpu;

        // The list iteration below skips the entry used as 'head'.
        // Report the 'head' entry explicitly.
        uvm_va_space_down_read(va_space);
        uvm_tools_record_gpu_fatal_fault(gpu->id, va_space, current_entry, reason);

        list_for_each_entry(coalesced_entry, &current_entry->merged_instances_list, merged_instances_list)
            uvm_tools_record_gpu_fatal_fault(gpu->id, va_space, coalesced_entry, reason);
        uvm_va_space_up_read(va_space);

        // We need to cancel each instance pointer to correctly handle faults
        // from multiple contexts.
        status = push_cancel_on_gpu_global(gpu, current_entry->instance_ptr, &batch_context->tracker);
        if (status != NV_OK)
            break;
    }
}

static void cancel_fault_batch(uvm_parent_gpu_t *parent_gpu,
                               uvm_fault_service_batch_context_t *batch_context,
                               UvmEventFatalReason reason)
{
    // Return code is ignored since we're on a global error path and wouldn't be
    // able to recover anyway.
    cancel_faults_all(batch_context, reason);
}


// Current fault cancel algorithm
//
// 1- Disable prefetching to avoid new requests keep coming and flooding the
// buffer.
// LOOP
//   2- Record one fatal fault per uTLB to check if it shows up after the replay
//   3- Flush fault buffer (REPLAY_TYPE_START_ACK_ALL to prevent new faults from
//      coming to TLBs with pending faults)
//   4- Wait for replay to finish
//   5- Fetch all faults from buffer
//   6- Check what uTLBs are in lockdown mode and can be cancelled
//   7- Preprocess faults (order per va_space, fault address, access type)
//   8- Service all non-fatal faults and mark all non-serviceable faults as
//      fatal.
//      8.1- If fatal faults are not found, we are done
//   9- Search for a uTLB which can be targeted for cancel, as described in
//      try_to_cancel_utlbs. If found, cancel it.
// END LOOP
// 10- Re-enable prefetching
//
// NOTE: prefetch faults MUST NOT trigger fault cancel. We make sure that no
// prefetch faults are left in the buffer by disabling prefetching and
// flushing the fault buffer afterwards (prefetch faults are not replayed and,
// therefore, will not show up again)
static NV_STATUS cancel_faults_precise_tlb(uvm_gpu_t *gpu, uvm_fault_service_batch_context_t *batch_context)
{
    NV_STATUS status;
    NV_STATUS tracker_status;
    uvm_replayable_fault_buffer_t *replayable_faults = &gpu->parent->fault_buffer.replayable;
    bool first = true;

    // 1) Disable prefetching to avoid new requests keep coming and flooding
    //    the buffer
    if (gpu->parent->fault_buffer.prefetch_faults_enabled)
        gpu->parent->arch_hal->disable_prefetch_faults(gpu->parent);

    while (1) {
        NvU32 utlb_id;

        // 2) Record one fatal fault per uTLB to check if it shows up after
        // the replay. This is used to handle the case in which the uTLB is
        // being cancelled from behind our backs by RM. See the comment in
        // step 6.
        for (utlb_id = 0; utlb_id <= batch_context->max_utlb_id; ++utlb_id) {
            uvm_fault_utlb_info_t *utlb = &batch_context->utlbs[utlb_id];

            if (!first && utlb->has_fatal_faults) {
                NvU32 idx = find_fatal_fault_in_utlb(batch_context, utlb_id);
                UVM_ASSERT(idx < batch_context->num_cached_faults);

                utlb->prev_fatal_fault = batch_context->fault_cache[idx];
            }
            else {
                utlb->prev_fatal_fault.fault_address = (NvU64)-1;
            }
        }
        first = false;

        // 3) Flush fault buffer. After this call, all faults from any of the
        // faulting uTLBs are before PUT. New faults from other uTLBs can keep
        // arriving. Therefore, in each iteration we just try to cancel faults
        // from uTLBs that contained fatal faults in the previous iterations
        // and will cause the TLB to stop generating new page faults after the
        // following replay with type UVM_FAULT_REPLAY_TYPE_START_ACK_ALL.
        //
        // No need to use UVM_GPU_BUFFER_FLUSH_MODE_WAIT_UPDATE_PUT since we
        // don't care too much about old faults, just new faults from uTLBs
        // which faulted before the replay.
        status = fault_buffer_flush_locked(gpu->parent,
                                           gpu,
                                           UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT,
                                           UVM_FAULT_REPLAY_TYPE_START_ACK_ALL,
                                           batch_context);
        if (status != NV_OK)
            break;

        // 4) Wait for replay to finish
        status = uvm_tracker_wait(&replayable_faults->replay_tracker);
        if (status != NV_OK)
            break;

        atomic_set(&batch_context->num_invalid_prefetch_faults, 0);
        batch_context->num_replays                 = 0;
        batch_context->fatal_va_space              = NULL;
        batch_context->fatal_gpu                   = NULL;
        batch_context->has_throttled_faults        = false;

        // 5) Fetch all faults from buffer
        status = fetch_fault_buffer_entries(gpu->parent, batch_context, FAULT_FETCH_MODE_ALL);
        if (status != NV_OK)
            break;

        ++batch_context->batch_id;

        UVM_ASSERT(batch_context->num_cached_faults == batch_context->num_coalesced_faults);

        // No more faults left, we are done
        if (batch_context->num_cached_faults == 0)
            break;

        // 6) Check what uTLBs are in lockdown mode and can be cancelled
        for (utlb_id = 0; utlb_id <= batch_context->max_utlb_id; ++utlb_id) {
            uvm_fault_utlb_info_t *utlb = &batch_context->utlbs[utlb_id];

            utlb->in_lockdown = false;
            utlb->cancelled   = false;

            if (utlb->prev_fatal_fault.fault_address != (NvU64)-1) {
                // If a previously-reported fault shows up again we can "safely"
                // assume that the uTLB that contains it is in lockdown mode
                // and no new translations will show up before cancel.
                // A fatal fault could only be removed behind our backs by RM
                // issuing a cancel, which only happens when RM is resetting the
                // engine. That means the instance pointer can't generate any
                // new faults, so we won't have an ABA problem where a new
                // fault arrives with the same state.
                if (is_fatal_fault_in_buffer(batch_context, &utlb->prev_fatal_fault))
                    utlb->in_lockdown = true;
            }
        }

        // 7) Preprocess faults
        status = preprocess_fault_batch(gpu->parent, batch_context);
        if (status == NV_WARN_MORE_PROCESSING_REQUIRED)
            continue;
        else if (status != NV_OK)
            break;

        // 8) Service all non-fatal faults and mark all non-serviceable faults
        // as fatal
        status = service_fault_batch(gpu->parent, FAULT_SERVICE_MODE_CANCEL, batch_context);
        UVM_ASSERT(batch_context->num_replays == 0);
        if (status == NV_ERR_NO_MEMORY)
            continue;
        else if (status != NV_OK)
            break;

        // No more fatal faults left, we are done
        if (!batch_context->fatal_va_space)
            break;

        // 9) Search for uTLBs that contain fatal faults and meet the
        // requirements to be cancelled
        try_to_cancel_utlbs(batch_context);
    }

    // 10) Re-enable prefetching
    if (gpu->parent->fault_buffer.prefetch_faults_enabled)
        gpu->parent->arch_hal->enable_prefetch_faults(gpu->parent);

    if (status == NV_OK)
        status = push_replay_on_gpu(gpu, UVM_FAULT_REPLAY_TYPE_START, batch_context);

    tracker_status = uvm_tracker_wait(&batch_context->tracker);

    return status == NV_OK? tracker_status: status;
}

static NV_STATUS cancel_faults_precise(uvm_fault_service_batch_context_t *batch_context)
{
    uvm_gpu_t *gpu;

    UVM_ASSERT(batch_context->fatal_va_space);
    UVM_ASSERT(batch_context->fatal_gpu);

    gpu = batch_context->fatal_gpu;
    return service_fault_batch_for_cancel(batch_context);
}

static void enable_disable_prefetch_faults(uvm_parent_gpu_t *parent_gpu,
                                           uvm_fault_service_batch_context_t *batch_context)
{
    // If more than 66% of faults are invalid prefetch accesses, disable
    // prefetch faults for a while.
    // num_invalid_prefetch_faults may be higher than the actual count. See the
    // comment in mark_fault_invalid_prefetch(..).
    // Some tests rely on this logic (and ratio) to correctly disable prefetch
    // fault reporting. If the logic changes, the tests will have to be changed.
    if (parent_gpu->fault_buffer.prefetch_faults_enabled &&
        uvm_perf_reenable_prefetch_faults_lapse_msec > 0 &&
        (((NvU32)atomic_read(&batch_context->num_invalid_prefetch_faults) * 3 >
          parent_gpu->fault_buffer.max_batch_size * 2) ||
         (uvm_enable_builtin_tests &&
          parent_gpu->rm_info.isSimulated &&
          atomic_read(&batch_context->num_invalid_prefetch_faults) > 5))) {
        uvm_parent_gpu_disable_prefetch_faults(parent_gpu);
    }
    else if (!parent_gpu->fault_buffer.prefetch_faults_enabled) {
        NvU64 lapse = NV_GETTIME() - parent_gpu->fault_buffer.disable_prefetch_faults_timestamp;

        // Reenable prefetch faults after some time
        if (lapse > ((NvU64)uvm_perf_reenable_prefetch_faults_lapse_msec * (1000 * 1000)))
            uvm_parent_gpu_enable_prefetch_faults(parent_gpu);
    }
}

// ----------------------------------------------------------------------------
// Dynamic Zero-copy, ARIADNE's mechanism (HPCA'26)
// ----------------------------------------------------------------------------
//
// Everything below is inert unless uvm_dynzero_enable is 1, and its default in
// this build is 0. See uvm_global.c for why the mechanism is carried here and
// uvm_va_block_types.h for what is and is not carried.
//
// The shape: the eviction path decides a block that just left GPU memory is
// worth keeping reachable and queues it; the fault path pins as many queued
// blocks as the demand surplus calls for, giving each a GPU-to-sysmem mapping
// with a deadline; a kthread revokes mappings whose deadline has passed so the
// next access refaults and placement is reconsidered.

// The single live sub-GPU behind a parent. The fault loop is per parent GPU on
// 610 and has no uvm_gpu_t in scope, while the queues and counters are per
// uvm_gpu_t. Under SMC several sub-GPUs would share the parent's fault buffer
// and silently share that state, so this asserts there is exactly one.
static uvm_gpu_t *zc_gpu(uvm_parent_gpu_t *parent_gpu)
{
    uvm_gpu_t *gpu;

    UVM_ASSERT(parent_gpu);

    if (parent_gpu->smc.enabled) {
        NvU32 sub_processor_index;

        UVM_ASSERT_MSG(bitmap_weight(parent_gpu->valid_gpus, UVM_PARENT_ID_MAX_SUB_PROCESSORS) <= 1,
                       "Zero-copy is single-GPU only; %u sub-GPUs are live\n",
                       bitmap_weight(parent_gpu->valid_gpus, UVM_PARENT_ID_MAX_SUB_PROCESSORS));

        sub_processor_index = find_first_bit(parent_gpu->valid_gpus, UVM_PARENT_ID_MAX_SUB_PROCESSORS);
        gpu = (sub_processor_index < UVM_PARENT_ID_MAX_SUB_PROCESSORS) ?
              parent_gpu->gpus[sub_processor_index] : NULL;
    }
    else {
        gpu = parent_gpu->gpus[0];
    }

    return gpu;
}

// Non-blocking va_space read lock, respecting the read-acquire-write-release
// ordering the va_space lock requires. For the unpin kthread, which must never
// block against the teardown trying to stop it.
static bool zc_va_space_tryread(uvm_va_space_t *va_space)
{
    if (!uvm_mutex_trylock(&va_space->read_acquire_write_release_lock))
        return false;

    if (!uvm_down_read_trylock(&va_space->lock)) {
        uvm_mutex_unlock(&va_space->read_acquire_write_release_lock);
        return false;
    }

    uvm_mutex_unlock_out_of_order(&va_space->read_acquire_write_release_lock);
    return true;
}

// Move the lock-free inbox onto spl_blocks. Caller holds zc_lock.
//
// llist_del_all reverses the batch, and the order within one drain carries no
// meaning: the pin walk takes some prefix of a queue whose entries were all
// evicted at about the same time.
static void zc_drain_pending(uvm_gpu_t *gpu)
{
    struct llist_node *pending = llist_del_all(&gpu->spl_pending);
    struct llist_node *pnext;

    while (pending) {
        uvm_pl_entry *pe = container_of(pending, uvm_pl_entry, pll);

        pnext = pending->next;
        list_add_tail(&pe->spln, &gpu->spl_blocks);
        pending = pnext;
    }
}

// Zero-copy expiry. Revokes the GPU mapping of any block whose pin time has
// passed, so the next access refaults and the driver can reconsider where the
// block belongs.
//
// Detach then process. The expired entries are moved onto a local list under
// zc_lock, the lock is dropped, and only then is anything mapped, unmapped or
// waited on. Nothing that can sleep runs with a Zero-copy lock held, which is
// what lets zc_lock be a LEAF spinlock with no ordering question against
// mmap_lock or the va_space lock.
static int uvm_zc_unpin_period(void *data)
{
    uvm_gpu_t *gpu = (uvm_gpu_t *)data;

    while (!kthread_should_stop()) {
        LIST_HEAD(expired);
        uvm_pl_entry *entry, *next;
        NvU64 now;
        NvU32 decay;
        NvU32 budget;
        bool busy;

        msleep_interruptible(uvm_dynzero_unpin_period);

        if (kthread_should_stop())
            break;

        now = NV_GETTIME();

        uvm_spin_lock(&gpu->zc_lock);

        // Self-throttle. The more blocks are pinned, the fewer are released per
        // sweep, so a large pinned set does not turn into a burst of refaults
        // and a PCIe storm. Theirs, including the constants.
        decay = (gpu->num_spled >> 8) + (NvU32)(((NvU64)gpu->num_spled * gpu->num_spled) >> 21);
        budget = (decay < 99) ? (100 - decay) : 1;

        list_for_each_entry_safe(entry, next, &gpu->spled_blocks, spln) {
            if (budget == 0)
                break;

            if (entry->endtime > now)
                continue;

            list_move_tail(&entry->spln, &expired);

            if (gpu->num_spled > 0)
                gpu->num_spled--;

            budget--;
        }

        // Same reason as the pin walk: these entries are off the queues and
        // invisible to teardown until they are freed or requeued below.
        // uvm_zc_gpu_va_space_put only stops this thread when the LAST user
        // goes, so with two clients it keeps sweeping while one tears down.
        busy = !list_empty(&expired);
        if (busy)
            gpu->zc_walk_busy++;

        uvm_spin_unlock(&gpu->zc_lock);

        // Off the queue and owned by this thread alone, so zc_lock is not held
        // here. The va_space and mm acquisitions are still TRYLOCKS, and that
        // is a second deadlock, distinct from the one the detach above solves.
        //
        // zc_stop_thread runs from remove_gpu_va_space, which holds the
        // va_space WRITE lock, and calls kthread_stop on this thread.
        // kthread_stop waits for the thread to exit. If the thread were sitting
        // in down_read on that same va_space it could never get the lock, never
        // exit, and both sides would be stuck with the write lock held. Every
        // process exit is a chance to hit it.
        //
        // A failed attempt costs one sweep: the entry goes back on the queue
        // below with its charge restored, and is reconsidered on the next pass.
        list_for_each_entry_safe(entry, next, &expired, spln) {
            uvm_va_block_t *block = NULL;
            uvm_va_block_context_t *block_context;
            uvm_tracker_t local_tracker = UVM_TRACKER_INIT();
            struct mm_struct *mm;
            bool unpinned = false;

            if (!uvm_va_space_mm_retain_trylock(entry->va_space, &mm))
                continue;

            if (zc_va_space_tryread(entry->va_space)) {
                // Checked before the block is used. The va_space can have
                // dropped the range while the block sat pinned.
                if (uvm_va_block_find(entry->va_space, entry->start, &block) == NV_OK && block) {
                    block_context = uvm_va_space_block_context(entry->va_space, mm);

                    block->prefetch_info.is_spled = 0;

                    uvm_mutex_lock(&block->lock);
                    uvm_va_block_unmap(block,
                                       block_context,
                                       gpu->id,
                                       uvm_va_block_region_from_block(block),
                                       NULL,
                                       &local_tracker);
                    uvm_mutex_unlock(&block->lock);

                    uvm_tracker_wait_deinit(&local_tracker);
                }

                uvm_va_space_up_read(entry->va_space);
                unpinned = true;
            }

            uvm_va_space_mm_release_unlock(entry->va_space, mm);

            if (unpinned) {
                list_del_init(&entry->spln);
                NV_KFREE(entry, sizeof(uvm_pl_entry));
            }
        }

        // Whatever could not be unpinned this sweep goes back on the queue with
        // its charge restored, rather than being dropped, which would leak the
        // entry and leave the block mapped forever.
        if (busy) {
            uvm_spin_lock(&gpu->zc_lock);
            list_for_each_entry_safe(entry, next, &expired, spln) {
                list_move_tail(&entry->spln, &gpu->spled_blocks);
                gpu->num_spled++;
            }
            gpu->zc_walk_busy--;
            uvm_spin_unlock(&gpu->zc_lock);
        }
    }

    return 0;
}

// Drop every queue entry belonging to one va_space, or all of them when dying
// is NULL.
//
// Detach under zc_lock, free outside it. The counters come down per entry
// rather than being zeroed, because the queues may still hold other clients'
// entries.
static void zc_drop_va_space_entries(uvm_gpu_t *gpu, uvm_va_space_t *dying)
{
    LIST_HEAD(doomed);
    uvm_pl_entry *pl_entry, *pl_next;
    uvm_used_entry *used_entry, *used_next;

    // Wait for any walk that currently holds entries detached from the queues.
    // Those entries are invisible here, and the walk puts them back when it is
    // done, so scanning past a live walk would leave an entry naming a
    // va_space that is about to be freed. The single mutex ARIADNE uses gives
    // this exclusion for free; detaching gives it up and this takes it back.
    //
    // Bounded, and cannot deadlock. This runs holding the va_space write lock,
    // and a walk holding entries never blocks on a va_space or mm lock: it
    // trylocks and requeues.
    //
    // The break KEEPS zc_lock, deliberately, and the rest of this function runs
    // under it down to the single unlock below. Do not "tidy" this into a
    // check-then-lock: releasing the lock between seeing zc_walk_busy at zero
    // and scanning the queues lets a walk start in that window and detach the
    // very entries this is here to find, which is the bug the counter exists
    // for. The test and the scan have to be one atomic section.
    while (1) {
        uvm_spin_lock(&gpu->zc_lock);
        if (!gpu->zc_walk_busy)
            break;
        uvm_spin_unlock(&gpu->zc_lock);
        cond_resched();
    }

    // The inbox is drained first, so a candidate published by the eviction path
    // but not yet picked up by the fault path is considered here too. Without
    // this a va_space could go away leaving entries in the inbox, and the next
    // fault batch would splice them onto spl_blocks and walk a dead va_space.
    zc_drain_pending(gpu);

    list_for_each_entry_safe(pl_entry, pl_next, &gpu->spl_blocks, spln) {
        if (dying && pl_entry->va_space != dying)
            continue;

        list_move_tail(&pl_entry->spln, &doomed);
    }

    list_for_each_entry_safe(pl_entry, pl_next, &gpu->spled_blocks, spln) {
        if (dying && pl_entry->va_space != dying)
            continue;

        list_move_tail(&pl_entry->spln, &doomed);

        if (gpu->num_spled)
            gpu->num_spled--;
    }

    uvm_spin_unlock(&gpu->zc_lock);

    list_for_each_entry_safe(pl_entry, pl_next, &doomed, spln) {
        list_del_init(&pl_entry->spln);
        NV_KFREE(pl_entry, sizeof(uvm_pl_entry));
    }

    // block_kill already unlinks a used_entry when its block dies, and tearing
    // down a va_space kills its blocks, so this is a safety net rather than the
    // primary mechanism. Clearing the block back-pointer matters: a surviving
    // block would otherwise dereference freed memory from the eviction path.
    //
    // used_lock is a different LEAF spinlock covering a different list, and the
    // two are never held together.
    uvm_spin_lock(&gpu->used_lock);

    list_for_each_entry_safe(used_entry, used_next, &gpu->used_blocks, spln) {
        // The NULL block test belongs to the filter, not to the loop. An entry
        // with no block cannot be matched against a va_space, so it is skipped
        // while filtering, but the unfiltered sweep still has to free it or GPU
        // removal leaks it.
        if (dying) {
            if (!used_entry->block || uvm_va_block_get_va_space_maybe_dead(used_entry->block) != dying)
                continue;
        }

        if (used_entry->block)
            used_entry->block->prefetch_info.used_entry = NULL;

        list_del_init(&used_entry->spln);
        NV_KFREE(used_entry, sizeof(uvm_used_entry));

        if (gpu->active_blocks)
            gpu->active_blocks--;
    }

    uvm_spin_unlock(&gpu->used_lock);
}

// Stop the unpin kthread. Caller holds zc_lifetime_lock.
//
// kthread_stop waits for the thread to leave its loop. The thread takes only
// zc_lock and the va_space and mm locks, never zc_lifetime_lock, so it cannot
// be waiting on the lock this is held under. Its sleep is interruptible and
// kthread_stop wakes it.
static void zc_stop_thread(uvm_gpu_t *gpu)
{
    uvm_assert_mutex_locked(&gpu->zc_lifetime_lock);

    if (gpu->async_unpin) {
        kthread_stop(gpu->async_unpin);
        gpu->async_unpin = NULL;
    }
}

void uvm_zc_gpu_va_space_get(uvm_gpu_t *gpu)
{
    if (!gpu)
        return;

    uvm_mutex_lock(&gpu->zc_lifetime_lock);
    gpu->zc_users++;
    uvm_mutex_unlock(&gpu->zc_lifetime_lock);
}

void uvm_zc_gpu_va_space_put(uvm_gpu_t *gpu, uvm_va_space_t *dying)
{
    bool last_user = false;

    if (!gpu)
        return;

    // Called from remove_gpu_va_space, which already holds mmap_lock and the
    // va_space write lock. Only zc_lifetime_lock is taken here, and nothing is
    // taken under it, so there is no order to get wrong.
    uvm_mutex_lock(&gpu->zc_lifetime_lock);

    if (gpu->zc_users == 0) {
        UVM_ERR_PRINT("Zero-copy: va_space put without a matching get.\n");
    }
    else {
        gpu->zc_users--;
        last_user = (gpu->zc_users == 0);
    }

    // The thread is stopped before this va_space's entries are dropped, so it
    // cannot be holding one of them when they are freed.
    if (last_user)
        zc_stop_thread(gpu);

    uvm_mutex_unlock(&gpu->zc_lifetime_lock);

    zc_drop_va_space_entries(gpu, dying);
}

void uvm_zc_gpu_deinit(uvm_gpu_t *gpu)
{
    if (!gpu)
        return;

    uvm_mutex_lock(&gpu->zc_lifetime_lock);

    if (gpu->zc_users != 0) {
        UVM_ERR_PRINT("Zero-copy: GPU removed with %u va_space users still counted. "
                      "A get without its put.\n",
                      gpu->zc_users);
        gpu->zc_users = 0;
    }

    zc_stop_thread(gpu);

    uvm_mutex_unlock(&gpu->zc_lifetime_lock);

    // NULL dying: take everything, whoever it belonged to.
    zc_drop_va_space_entries(gpu, NULL);

    uvm_spin_lock(&gpu->used_lock);
    gpu->active_blocks = 0;
    uvm_spin_unlock(&gpu->used_lock);

    uvm_spin_lock(&gpu->zc_lock);
    gpu->num_spled = 0;
    gpu->man_size = 0;
    uvm_spin_unlock(&gpu->zc_lock);
}

// Age the working set.
//
// A block that is out of GPU memory, not host-pinned, and untouched for the
// retention window has stopped being demand, so it leaves the working set.
// Without this the estimate only grows and every workload eventually looks
// oversubscribed. The window is 500 ms, theirs, hardcoded.
#define UVM_ZC_WCSS_RETAIN_NS (500ULL * 1000ULL * 1000ULL)

static void zc_age_working_set(uvm_parent_gpu_t *parent_gpu)
{
    uvm_gpu_t *gpu = zc_gpu(parent_gpu);
    uvm_used_entry *entry, *next;
    NvU64 now;

    if (!gpu)
        return;

    now = NV_GETTIME();

    // Under used_lock. The populate path appends to this list and the eviction
    // path deletes from it, both without any Zero-copy lock, so an unlocked
    // reap could free an entry another thread was standing on. Nothing in the
    // loop sleeps: the test reads fields, and NV_KFREE is kfree.
    uvm_spin_lock(&gpu->used_lock);
    list_for_each_entry_safe(entry, next, &gpu->used_blocks, spln) {
        if (!entry->is_in_gpu &&
            entry->block &&
            !entry->block->prefetch_info.is_spled &&
            entry->block->prefetch_info.last_migration_time + UVM_ZC_WCSS_RETAIN_NS < now) {
            list_del_init(&entry->spln);
            entry->block->prefetch_info.used_entry = NULL;
            NV_KFREE(entry, sizeof(uvm_used_entry));

            if (gpu->active_blocks > 0)
                gpu->active_blocks--;
        }
    }
    uvm_spin_unlock(&gpu->used_lock);
}

// Turn working-set pressure into Zero-copy.
//
// Demand minus what is resident minus what is already pinned is the surplus
// that cannot fit. That many blocks are taken off the candidate queue and given
// a GPU-to-sysmem remote mapping, so the next access reads them across PCIe
// instead of dragging them back and evicting something else.
//
// Three phases, and the split is what keeps the locking honest. Under zc_lock:
// drain the inbox, size the surplus, detach that many candidates. With no
// Zero-copy lock held: map each one, which needs mmap_lock, the va_space lock
// and the block lock. Under zc_lock again: put the pinned ones on spled_blocks
// and any that could not be mapped back on spl_blocks.
//
// man_size is recounted rather than tracked, which is theirs.
static void zc_pin_from_pressure(uvm_parent_gpu_t *parent_gpu)
{
    uvm_gpu_t *gpu = zc_gpu(parent_gpu);
    LIST_HEAD(candidates);
    LIST_HEAD(pinned);
    LIST_HEAD(returned);
    uvm_pl_entry *entry, *next_entry;
    struct list_head *cur;
    NvU32 resident = 0;
    NvU32 pinned_count = 0;
    NvU32 to_pin;
    NvU64 now;
    bool busy = false;

    if (!gpu)
        return;

    now = NV_GETTIME();

    uvm_spin_lock(&gpu->pmm.list_lock);
    list_for_each(cur, &gpu->pmm.root_chunks.alloc_list[UVM_PMM_ALLOC_LIST_USED])
        resident++;
    uvm_spin_unlock(&gpu->pmm.list_lock);

    uvm_spin_lock(&gpu->zc_lock);

    zc_drain_pending(gpu);

    list_for_each(cur, &gpu->spled_blocks)
        pinned_count++;

    gpu->man_size = resident;
    gpu->num_spled = pinned_count;

    to_pin = (gpu->active_blocks > resident + pinned_count) ?
             (gpu->active_blocks - resident - pinned_count) : 0;

    list_for_each_entry_safe(entry, next_entry, &gpu->spl_blocks, spln) {
        if (to_pin == 0)
            break;

        list_move_tail(&entry->spln, &candidates);
        to_pin--;
    }

    // Entries are off the queues from here until they are filed below, so
    // teardown cannot see them. It waits on this instead.
    busy = !list_empty(&candidates);
    if (busy)
        gpu->zc_walk_busy++;

    uvm_spin_unlock(&gpu->zc_lock);

    list_for_each_entry_safe(entry, next_entry, &candidates, spln) {
        uvm_va_block_t *block = NULL;
        uvm_va_block_context_t *block_context;
        struct mm_struct *block_mm;
        uvm_va_space_t *entry_va_space = entry->va_space;
        NvU64 pintime;
        bool mapped = false;

        // Trylocks, for two reasons. Teardown waits for zc_walk_busy while
        // holding the va_space write lock, so blocking on that same lock here
        // would deadlock. And this walk can be handed an entry for a va_space
        // that is being torn down right now, where the tryread failing is the
        // correct answer.
        if (!uvm_va_space_mm_retain_trylock(entry_va_space, &block_mm))
            continue;

        if (!zc_va_space_tryread(entry_va_space)) {
            uvm_va_space_mm_release_unlock(entry_va_space, block_mm);
            continue;
        }

        // Checked before use: the range can have gone away while the block sat
        // on this queue.
        if (uvm_va_block_find(entry_va_space, entry->start, &block) == NV_OK && block) {
            block_context = uvm_va_space_block_context(entry_va_space, block_mm);

            // The Zero-copy mapping itself. On a CPU-resident block this builds
            // a GPU-to-sysmem remote mapping over the whole 2 MB block with no
            // migration.
            if (uvm_va_block_set_accessed_by(block, block_context, gpu->id) == NV_OK) {
                // 50 us per active block, floored at the parameter, times five
                // for a repeat offender. Theirs, and it is NvU32 nanoseconds in
                // their code, which wraps past about 4.3 s. Widened to NvU64
                // here, which only removes the wrap.
                pintime = (NvU64)uvm_dynzero_pintime * 500ULL * gpu->active_blocks;
                if (pintime < (NvU64)uvm_dynzero_pintime * 1000000ULL)
                    pintime = (NvU64)uvm_dynzero_pintime * 1000000ULL;

                block->prefetch_info.thr_count++;
                if (block->prefetch_info.thr_count > 1)
                    pintime *= 5;

                block->prefetch_info.last_migration_time = now;
                block->prefetch_info.is_spled = 1;
                entry->endtime = now + pintime;
                mapped = true;
            }
        }

        uvm_va_space_up_read(entry_va_space);
        uvm_va_space_mm_release_unlock(entry_va_space, block_mm);

        // The block is gone, so the entry has nothing left to name. Freed here
        // rather than requeued. entry->va_space was cached above because this
        // free is what makes reading it afterwards a use-after-free, which is
        // the shape ARIADNE's walk has.
        if (block)
            list_move_tail(&entry->spln, mapped ? &pinned : &returned);
        else {
            list_del_init(&entry->spln);
            NV_KFREE(entry, sizeof(uvm_pl_entry));
        }
    }

    uvm_spin_lock(&gpu->zc_lock);

    list_for_each_entry_safe(entry, next_entry, &pinned, spln) {
        list_move_tail(&entry->spln, &gpu->spled_blocks);
        gpu->num_spled++;
    }

    // Could not be mapped this time. Back on the candidate queue so a later
    // batch can retry, rather than dropped, which would lose the candidate.
    list_for_each_entry_safe(entry, next_entry, &returned, spln)
        list_move_tail(&entry->spln, &gpu->spl_blocks);

    // Whatever is still on candidates is what a failed trylock skipped over.
    // It goes back too. Leaving it on a stack list would leak the entry and
    // lose the candidate, and this loop is the only thing standing between
    // "the trylock failed" and exactly that.
    list_for_each_entry_safe(entry, next_entry, &candidates, spln)
        list_move_tail(&entry->spln, &gpu->spl_blocks);

    if (busy)
        gpu->zc_walk_busy--;

    uvm_spin_unlock(&gpu->zc_lock);

    // Brought up lazily, so a GPU nobody is faulting on has no thread waking
    // every uvm_dynzero_unpin_period. Under zc_lifetime_lock, which is the same
    // lock uvm_zc_gpu_va_space_put stops the thread under, so a thread cannot
    // be started just after the last user has gone and then live until GPU
    // teardown. ARIADNE starts it unlocked and has that race.
    uvm_mutex_lock(&gpu->zc_lifetime_lock);
    if (!gpu->async_unpin && gpu->zc_users > 0) {
        gpu->async_unpin = kthread_run(uvm_zc_unpin_period, gpu, "uvm_zc_unpin");
        if (IS_ERR(gpu->async_unpin))
            gpu->async_unpin = NULL;
    }
    uvm_mutex_unlock(&gpu->zc_lifetime_lock);
}

void uvm_parent_gpu_service_replayable_faults(uvm_parent_gpu_t *parent_gpu)
{
    NvU32 num_replays = 0;
    NvU32 num_batches = 0;
    NvU32 num_throttled = 0;
    NvU64 batch_start_time = 0;
    NvU64 time_stamp;
    NV_STATUS status = NV_OK;

    // Batches whose replay has been pushed but not waited on. Was a bool, which
    // is the same thing with no upper bound; the count is what lets
    // uvm_perf_fault_service_max_inflight cap how far population may run ahead
    // of consumption. One wait drains the whole replay tracker, so any wait
    // returns this to zero rather than decrementing it.
    NvU32 pending_replays = 0;
    uvm_replayable_fault_buffer_t *replayable_faults = &parent_gpu->fault_buffer.replayable;
    uvm_fault_service_batch_context_t *batch_context = &replayable_faults->batch_service_context;

    // Snapshot the per-GPU servicing-pipeline counters so we can fold this
    // invocation's delta into the module-lifetime global (cpu/fault_stats) at
    // the single function exit. The per-GPU node vanishes on GPU unregister;
    // the global persists so the external capture can always diff it.
    // (ns_bh_queue_delay is updated in the ISR before this call, so it is
    // mirrored there, not here.)
    NvU64 fold_start_num_batches          = replayable_faults->stats.num_batches;
    NvU64 fold_start_num_cached_faults    = replayable_faults->stats.num_cached_faults;
    NvU64 fold_start_num_coalesced_faults = replayable_faults->stats.num_coalesced_faults;
    NvU64 fold_start_ns_fetch             = replayable_faults->stats.ns_fetch;
    NvU64 fold_start_ns_preprocess        = replayable_faults->stats.ns_preprocess;
    NvU64 fold_start_ns_service           = replayable_faults->stats.ns_service;
    NvU64 fold_start_ns_replay            = replayable_faults->stats.ns_replay;
    NvU64 fold_start_ns_tracker_wait      = replayable_faults->stats.ns_tracker_wait;
    NvU64 fold_start_ns_batch_total       = replayable_faults->stats.ns_batch_total;

    uvm_tracker_init(&batch_context->tracker);

    // Zero-copy, ARIADNE's (HPCA'26), off by default in this build. Age the
    // working set and bring the unpin thread up before any batch is serviced,
    // so the pin decision at the end of the loop reads a current estimate.
    if (uvm_dynzero_enable)
        zc_age_working_set(parent_gpu);

    // Process all faults in the buffer
    while (1) {
        if (num_throttled >= uvm_perf_fault_max_throttle_per_service ||
            num_batches >= uvm_perf_fault_max_batches_per_service) {
            break;
        }

        // Admission control on how far population may run ahead of the GPU.
        //
        // Taken before the fetch rather than after the replay push so that the
        // wait paces the next batch's population, which is the thing being
        // bounded. At a limit of 1 this is exactly the stock barrier: push the
        // replay, come back here, wait, then fetch.
        if (uvm_perf_fault_service_max_inflight != 0 &&
            pending_replays >= uvm_perf_fault_service_max_inflight) {
            time_stamp = NV_GETTIME();
            status = uvm_tracker_wait(&replayable_faults->replay_tracker);
            replayable_faults->stats.ns_tracker_wait += NV_GETTIME() - time_stamp;
            pending_replays = 0;
            if (status != NV_OK)
                break;
        }

        atomic_set(&batch_context->num_invalid_prefetch_faults, 0);
        atomic_set(&batch_context->num_duplicate_faults, 0);
        atomic_set(&batch_context->num_authorized_faults, 0);
        batch_context->num_replays                 = 0;
        batch_context->fatal_va_space              = NULL;
        batch_context->fatal_gpu                   = NULL;
        batch_context->has_throttled_faults        = false;

        batch_start_time = NV_GETTIME();

        status = fetch_fault_buffer_entries(parent_gpu, batch_context, FAULT_FETCH_MODE_BATCH_READY);
        replayable_faults->stats.ns_fetch += NV_GETTIME() - batch_start_time;
        if (status != NV_OK)
            break;

        if (batch_context->num_cached_faults == 0) {
            // Pipelined servicing: the buffer may be empty only because a
            // deferred replay hasn't executed yet and its faults haven't
            // repopulated the buffer. Take the (deferred) wait now and fetch
            // once more, preserving the stock behavior of servicing replayed
            // faults within the same bottom-half pass.
            if (pending_replays > 0) {
                pending_replays = 0;
                time_stamp = NV_GETTIME();
                status = uvm_tracker_wait(&replayable_faults->replay_tracker);
                replayable_faults->stats.ns_tracker_wait += NV_GETTIME() - time_stamp;
                if (status != NV_OK)
                    break;

                continue;
            }

            break;
        }

        ++batch_context->batch_id;

        replayable_faults->stats.num_cached_faults += batch_context->num_cached_faults;
        replayable_faults->stats.num_coalesced_faults += batch_context->num_coalesced_faults;

        time_stamp = NV_GETTIME();
        status = preprocess_fault_batch(parent_gpu, batch_context);
        replayable_faults->stats.ns_preprocess += NV_GETTIME() - time_stamp;

        num_replays += batch_context->num_replays;

        if (status == NV_WARN_MORE_PROCESSING_REQUIRED)
            continue;
        else if (status != NV_OK)
            break;

        time_stamp = NV_GETTIME();
        status = service_fault_batch(parent_gpu, FAULT_SERVICE_MODE_REGULAR, batch_context);
        replayable_faults->stats.ns_service += NV_GETTIME() - time_stamp;

        // We may have issued replays even if status != NV_OK if
        // UVM_PERF_FAULT_REPLAY_POLICY_BLOCK is being used or the fault buffer
        // was flushed
        num_replays += batch_context->num_replays;

        enable_disable_prefetch_faults(parent_gpu, batch_context);

        if (status != NV_OK) {
            // Unconditionally cancel all faults to trigger RC. This will not
            // provide precise attribution, but this case handles global
            // errors such as OOM or ECC where it's not reasonable to
            // guarantee precise attribution. We ignore the return value of
            // the cancel operation since this path is already returning an
            // error code.
            cancel_fault_batch(parent_gpu, batch_context, uvm_tools_status_to_fatal_fault_reason(status));
            break;
        }

        if (batch_context->fatal_va_space) {
            time_stamp = NV_GETTIME();
            status = uvm_tracker_wait(&batch_context->tracker);
            replayable_faults->stats.ns_tracker_wait += NV_GETTIME() - time_stamp;
            if (status == NV_OK) {
                status = cancel_faults_precise(batch_context);
                if (status == NV_OK) {
                    // Cancel handling should've issued at least one replay
                    UVM_ASSERT(batch_context->num_replays > 0);
                    ++num_batches;
                    ++replayable_faults->stats.num_batches;
                    replayable_faults->stats.ns_batch_total += NV_GETTIME() - batch_start_time;
                    continue;
                }
            }

            break;
        }

        // Zero-copy, at the end of the batch and before the replay, which is
        // where their tree puts it: the surplus is computed from what this
        // batch just made resident, and any block pinned here is mapped before
        // the faults are replayed.
        if (uvm_dynzero_enable)
            zc_pin_from_pressure(parent_gpu);

        if (replayable_faults->replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BATCH) {
            time_stamp = NV_GETTIME();
            status = push_replay_on_parent_gpu(parent_gpu, UVM_FAULT_REPLAY_TYPE_START, batch_context);
            replayable_faults->stats.ns_replay += NV_GETTIME() - time_stamp;
            if (status != NV_OK)
                break;
            ++num_replays;
        }
        else if (replayable_faults->replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_BATCH_FLUSH) {
            uvm_gpu_buffer_flush_mode_t flush_mode = UVM_GPU_BUFFER_FLUSH_MODE_CACHED_PUT;

            if ((NvU32)atomic_read(&batch_context->num_duplicate_faults) * 100 >
                batch_context->num_cached_faults * replayable_faults->replay_update_put_ratio) {
                flush_mode = UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT;
            }

            // Second, independent trigger on cross-batch redundancy, which the
            // duplicate ratio above is structurally unable to observe. Off at
            // 0, which is the default, so the stock decision is unchanged
            // unless this is asked for.
            if (uvm_perf_fault_replay_update_put_authorized_ratio != 0 &&
                (NvU32)atomic_read(&batch_context->num_authorized_faults) * 100 >
                batch_context->num_cached_faults * uvm_perf_fault_replay_update_put_authorized_ratio) {
                flush_mode = UVM_GPU_BUFFER_FLUSH_MODE_UPDATE_PUT;
            }

            time_stamp = NV_GETTIME();
            status = fault_buffer_flush_locked(parent_gpu, NULL, flush_mode, UVM_FAULT_REPLAY_TYPE_START, batch_context);
            replayable_faults->stats.ns_replay += NV_GETTIME() - time_stamp;
            if (status != NV_OK)
                break;
            ++num_replays;

            // The replay push acquired the batch tracker, so the GPU already
            // orders it after every migration of this batch. Pipelined mode
            // defers this CPU-side wait and overlaps the next batch's
            // fetch/service with this batch's copies and replay; the wait
            // moves to the empty-fetch path above, or to the in-flight bound
            // at the top of the loop, whichever comes first. Serial mode keeps
            // the stock synchronous wait.
            if (uvm_perf_fault_service_pipeline != 0) {
                ++pending_replays;
            }
            else {
                time_stamp = NV_GETTIME();
                status = uvm_tracker_wait(&replayable_faults->replay_tracker);
                replayable_faults->stats.ns_tracker_wait += NV_GETTIME() - time_stamp;
                if (status != NV_OK)
                    break;
            }
        }

        if (batch_context->has_throttled_faults)
            ++num_throttled;

        ++num_batches;
        ++replayable_faults->stats.num_batches;
        replayable_faults->stats.ns_batch_total += NV_GETTIME() - batch_start_time;
    }

    if (status == NV_WARN_MORE_PROCESSING_REQUIRED)
        status = NV_OK;

    // Make sure that we issue at least one replay if no replay has been
    // issued yet to avoid dropping faults that do not show up in the buffer
    if ((status == NV_OK && replayable_faults->replay_policy == UVM_PERF_FAULT_REPLAY_POLICY_ONCE) ||
        num_replays == 0) {
        time_stamp = NV_GETTIME();
        status = push_replay_on_parent_gpu(parent_gpu, UVM_FAULT_REPLAY_TYPE_START, batch_context);
        replayable_faults->stats.ns_replay += NV_GETTIME() - time_stamp;
    }

    uvm_tracker_deinit(&batch_context->tracker);

    // Fold this invocation's per-GPU pipeline delta into the module-lifetime
    // global aggregate (cpu/fault_stats), which persists across GPU unregister.
    atomic64_add(replayable_faults->stats.num_batches          - fold_start_num_batches,
                 &g_uvm_fault_pipeline_stats.num_batches);
    atomic64_add(replayable_faults->stats.num_cached_faults    - fold_start_num_cached_faults,
                 &g_uvm_fault_pipeline_stats.num_cached_faults);
    atomic64_add(replayable_faults->stats.num_coalesced_faults - fold_start_num_coalesced_faults,
                 &g_uvm_fault_pipeline_stats.num_coalesced_faults);
    atomic64_add(replayable_faults->stats.ns_fetch             - fold_start_ns_fetch,
                 &g_uvm_fault_pipeline_stats.ns_fetch);
    atomic64_add(replayable_faults->stats.ns_preprocess        - fold_start_ns_preprocess,
                 &g_uvm_fault_pipeline_stats.ns_preprocess);
    atomic64_add(replayable_faults->stats.ns_service           - fold_start_ns_service,
                 &g_uvm_fault_pipeline_stats.ns_service);
    atomic64_add(replayable_faults->stats.ns_replay            - fold_start_ns_replay,
                 &g_uvm_fault_pipeline_stats.ns_replay);
    atomic64_add(replayable_faults->stats.ns_tracker_wait      - fold_start_ns_tracker_wait,
                 &g_uvm_fault_pipeline_stats.ns_tracker_wait);
    atomic64_add(replayable_faults->stats.ns_batch_total       - fold_start_ns_batch_total,
                 &g_uvm_fault_pipeline_stats.ns_batch_total);

    if (status != NV_OK)
        UVM_DBG_PRINT("Error servicing replayable faults on GPU: %s\n", uvm_parent_gpu_name(parent_gpu));
}

void uvm_parent_gpu_enable_prefetch_faults(uvm_parent_gpu_t *parent_gpu)
{
    UVM_ASSERT(parent_gpu->isr.replayable_faults.handling);

    if (!parent_gpu->fault_buffer.prefetch_faults_enabled) {
        parent_gpu->arch_hal->enable_prefetch_faults(parent_gpu);
        parent_gpu->fault_buffer.prefetch_faults_enabled = true;
    }
}

void uvm_parent_gpu_disable_prefetch_faults(uvm_parent_gpu_t *parent_gpu)
{
    UVM_ASSERT(parent_gpu->isr.replayable_faults.handling);

    if (parent_gpu->fault_buffer.prefetch_faults_enabled) {
        parent_gpu->arch_hal->disable_prefetch_faults(parent_gpu);
        parent_gpu->fault_buffer.prefetch_faults_enabled = false;
        parent_gpu->fault_buffer.disable_prefetch_faults_timestamp = NV_GETTIME();
    }
}

const char *uvm_perf_fault_replay_policy_string(uvm_perf_fault_replay_policy_t replay_policy)
{
    BUILD_BUG_ON(UVM_PERF_FAULT_REPLAY_POLICY_MAX != 4);

    switch (replay_policy) {
        UVM_ENUM_STRING_CASE(UVM_PERF_FAULT_REPLAY_POLICY_BLOCK);
        UVM_ENUM_STRING_CASE(UVM_PERF_FAULT_REPLAY_POLICY_BATCH);
        UVM_ENUM_STRING_CASE(UVM_PERF_FAULT_REPLAY_POLICY_BATCH_FLUSH);
        UVM_ENUM_STRING_CASE(UVM_PERF_FAULT_REPLAY_POLICY_ONCE);
        UVM_ENUM_STRING_DEFAULT();
    }
}

NV_STATUS uvm_test_get_prefetch_faults_reenable_lapse(UVM_TEST_GET_PREFETCH_FAULTS_REENABLE_LAPSE_PARAMS *params,
                                                      struct file *filp)
{
    params->reenable_lapse = uvm_perf_reenable_prefetch_faults_lapse_msec;

    return NV_OK;
}

NV_STATUS uvm_test_set_prefetch_faults_reenable_lapse(UVM_TEST_SET_PREFETCH_FAULTS_REENABLE_LAPSE_PARAMS *params,
                                                      struct file *filp)
{
    uvm_perf_reenable_prefetch_faults_lapse_msec = params->reenable_lapse;

    return NV_OK;
}

NV_STATUS uvm_test_drain_replayable_faults(UVM_TEST_DRAIN_REPLAYABLE_FAULTS_PARAMS *params, struct file *filp)
{
    uvm_gpu_t *gpu;
    NV_STATUS status = NV_OK;
    uvm_spin_loop_t spin;
    bool pending = true;
    uvm_va_space_t *va_space = uvm_va_space_get(filp);

    gpu = uvm_va_space_retain_gpu_by_uuid(va_space, &params->gpu_uuid);
    if (!gpu)
        return NV_ERR_INVALID_DEVICE;

    uvm_spin_loop_init(&spin);

    do {
        uvm_parent_gpu_replayable_faults_isr_lock(gpu->parent);
        pending = uvm_parent_gpu_replayable_faults_pending(gpu->parent);
        uvm_parent_gpu_replayable_faults_isr_unlock(gpu->parent);

        if (!pending)
            break;

        if (fatal_signal_pending(current)) {
            status = NV_ERR_SIGNAL_PENDING;
            break;
        }

        UVM_SPIN_LOOP(&spin);
    } while (uvm_spin_loop_elapsed(&spin) < params->timeout_ns);

    if (pending && status == NV_OK)
        status = NV_ERR_TIMEOUT;

    uvm_gpu_release(gpu);

    return status;
}
