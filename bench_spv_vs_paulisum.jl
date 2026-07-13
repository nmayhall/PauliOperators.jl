using Distributed
using Printf
using Random
using Statistics

const PROJECT_ROOT = @__DIR__
PROJECT_ROOT in LOAD_PATH || pushfirst!(LOAD_PATH, PROJECT_ROOT)

using PauliOperators

const MODE = get(ENV, "PO_BENCH_MODE", get(ARGS, 1, "local"))
const N = parse(Int, get(ENV, "PO_BENCH_N", "16"))
const INIT_TERMS = parse(Int, get(ENV, "PO_BENCH_INIT_TERMS", "400"))
const NGENS = parse(Int, get(ENV, "PO_BENCH_NGENS", "14"))
const THRESH = parse(Float64, get(ENV, "PO_BENCH_THRESH", "1e-8"))
const NSAMPLES = parse(Int, get(ENV, "PO_BENCH_SAMPLES", "3"))
const WORKERS = parse(Int, get(ENV, "PO_BENCH_WORKERS", "2"))
const WORKER_THREADS = parse(Int, get(ENV, "PO_BENCH_WORKER_THREADS", "2"))
const WINDOW = parse(Int, get(ENV, "PO_BENCH_WINDOW", "1"))
const STORAGES = Symbol.(split(get(ENV, "PO_BENCH_STORAGE", "dict,spv"), ","))

function make_problem(N::Int, init_terms::Int, ngens::Int)
    rng = MersenneTwister(12345)
    W = PauliOperators.uinttype(N)
    mask = PauliOperators._bitmask(W, N)
    rand_basis() = PauliBasis{N,W}(rand(rng, W) & mask, rand(rng, W) & mask)
    O = PauliSum(N, ComplexF64)
    while length(O) < init_terms
        pb = rand_basis()
        O[pb] = (randn(rng) + 1im * randn(rng)) / sqrt(init_terms)
    end

    gens = PauliBasis{N}[]
    while length(gens) < ngens
        push!(gens, rand_basis())
    end
    angles = [0.03 * randn(rng) for _ in 1:ngens]
    return O, gens, angles
end

function evolve_dict_local!(O, gens, angles)
    for (G, θ) in zip(gens, angles)
        evolve!(O, G, θ)
        coeff_clip!(O, THRESH)
    end
    return O
end

function rotate_dict_windowed!(O::PauliSum{N,T}, pending, G::PauliBasis{N},
                               θ::Real) where {N,T}
    _cos = cos(θ)
    _sin = 1im * sin(θ)
    hi = length(pending)

    for (p, c) in O
        commute(p, G) && continue
        tmp = c * _sin * G * p
        push!(pending, (PauliBasis(tmp), convert(T, coeff(tmp))))
        O[p] = c * _cos
    end

    for i in 1:hi
        p, c = pending[i]
        commute(p, G) && continue
        tmp = c * _sin * G * p
        push!(pending, (PauliBasis(tmp), convert(T, coeff(tmp))))
        pending[i] = (p, c * _cos)
    end
    return O
end

function merge_dict_pending!(O::PauliSum{N,T}, pending) where {N,T}
    for (pb, c) in pending
        O[pb] = get(O, pb, zero(T)) + c
    end
    empty!(pending)
    coeff_clip!(O, THRESH)
    return O
end

function evolve_dict_windowed!(O::PauliSum{N,T}, gens, angles) where {N,T}
    pending = Tuple{PauliBasis{N},T}[]
    for (i, (G, θ)) in enumerate(zip(gens, angles))
        rotate_dict_windowed!(O, pending, G, θ)
        (i % WINDOW == 0 || i == length(gens)) && merge_dict_pending!(O, pending)
    end
    return O
end

function run_local(storage::Symbol, O0, gens, angles)
    GC.gc()
    if storage == :dict
        O = deepcopy(O0)
        timing = @timed begin
            WINDOW == 1 ? evolve_dict_local!(O, gens, angles) :
                          evolve_dict_windowed!(O, gens, angles)
        end
        return timing.time, timing.bytes, Base.summarysize(O), length(O)
    elseif storage == :spv
        O = SparsePauliVector(O0)
        timing = @timed evolve!(O, gens, angles; window=WINDOW,
                                 truncation=CoeffTruncation(THRESH),
                                 threaded=Threads.nthreads() > 1)
        return timing.time, timing.bytes, Base.summarysize(O), length(O)
    else
        error("unknown storage $storage")
    end
end

@everywhere function _po_bench_shard_bytes(id::Symbol)
    return Base.summarysize(PauliOperators._dps_get(id))
end

function setup_workers!(mode::String)
    mode == "local" && return Int[]
    if mode == "single_node_multithread" && Threads.nthreads() > 1
        return [myid()]
    end
    nworkers = occursin("multinode", mode) ? WORKERS : 1
    nthreads = occursin("multithread", mode) ? WORKER_THREADS : 1
    real_workers = filter(!=(1), workers())
    if length(real_workers) < nworkers
        addprocs(nworkers - length(real_workers); exeflags="--threads=$nthreads")
    end
    pids = filter(!=(1), workers())[1:nworkers]
    ensure_pauli_workers!(workers=pids)
    @sync for pid in pids
        @async remotecall_fetch(Core.eval, pid, Main, quote
            function _po_bench_shard_bytes(id::Symbol)
                return Base.summarysize(PauliOperators._dps_get(id))
            end
        end)
    end
    return pids
end

function run_distributed(storage::Symbol, O0, gens, angles, pids)
    GC.gc()
    input = storage == :spv ? SparsePauliVector(O0) : O0
    dO = distribute(input; workers=pids, storage=storage)
    timing = @timed evolve!(dO, gens, angles; truncation_thresh=THRESH,
                            threaded=true, window=WINDOW)
    live_bytes = sum(remotecall_fetch(_po_bench_shard_bytes, pid, dO.id) for pid in pids)
    nterms = length(dO)
    destroy!(dO)
    return timing.time, timing.bytes, live_bytes, nterms
end

function summarize(samples)
    times = [s[1] for s in samples]
    allocs = [s[2] for s in samples]
    lives = [s[3] for s in samples]
    terms = [s[4] for s in samples]
    return median(times), median(allocs), median(lives), median(terms)
end

function main()
    O0, gens, angles = make_problem(N, INIT_TERMS, NGENS)
    pids = setup_workers!(MODE)

    println("# mode=$MODE N=$N init_terms=$INIT_TERMS ngens=$NGENS thresh=$THRESH window=$WINDOW samples=$NSAMPLES")
    if !isempty(pids)
        println("# workers=$(length(pids)) worker_threads=$(remotecall_fetch(Threads.nthreads, first(pids))) pids=$(join(pids, ','))")
    else
        println("# local_threads=$(Threads.nthreads())")
    end
    println("mode,storage,median_seconds,median_alloc_mb,median_live_mb,median_terms")

    for storage in STORAGES
        # Warm-up compilation/run.
        if MODE == "local"
            run_local(storage, O0, gens, angles)
        else
            run_distributed(storage, O0, gens, angles, pids)
        end

        samples = Tuple{Float64,Int,Int,Int}[]
        for _ in 1:NSAMPLES
            result = MODE == "local" ?
                run_local(storage, O0, gens, angles) :
                run_distributed(storage, O0, gens, angles, pids)
            push!(samples, result)
        end
        t, alloc, live, nterms = summarize(samples)
        @printf("%s,%s,%.6f,%.3f,%.3f,%d\n",
                MODE, storage, t, alloc / 2.0^20, live / 2.0^20, nterms)
    end
end

main()
