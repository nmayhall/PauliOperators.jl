# Before/after benchmark for the word-type parameterization (v4).
# Run on any tree:   julia --project=. -O3 bench/bench_words.jl
# Compares Dict-path and kernel hot spots across N. Prints a markdown table.
using PauliOperators
using BenchmarkTools
using Random
using Printf

Random.seed!(42)

_rand_ps(N, nterms) = begin
    ps = PauliSum(N)
    while length(ps) < nterms
        ps[PauliBasis(rand(Pauli{N}))] = randn() + 0im
    end
    ps
end

function _bench_pauli_mul(N)
    p1 = rand(Pauli{N}); p2 = rand(Pauli{N})
    @belapsed $p1 * $p2
end

function _bench_ps_mul(N, nterms)
    a = _rand_ps(N, nterms); b = _rand_ps(N, nterms)
    @belapsed $a * $b
end

function _bench_evolve(N, nterms, nrot)
    O0 = _rand_ps(N, nterms)
    gens = [PauliBasis(rand(Pauli{N})) for _ in 1:nrot]
    @belapsed begin
        O = deepcopy($O0)
        for g in $gens
            evolve!(O, g, 0.13)
            coeff_clip!(O, 1e-8)
        end
    end
end

function _bench_dict_getset(N, nterms)
    ps = _rand_ps(N, nterms)
    ks = collect(keys(ps))
    @belapsed begin
        s = 0.0 + 0im
        for k in $ks
            s += $ps[k]
        end
        s
    end
end

function _bench_rand_pb(N)
    @belapsed rand(PauliBasis{$N})
end

rows = []
for N in (10, 20, 60, 64, 100, 127)
    t_mul   = _bench_pauli_mul(N)
    t_psmul = _bench_ps_mul(N, 300)
    t_ev    = _bench_evolve(N, 200, 30)
    t_get   = _bench_dict_getset(N, 1000)
    t_rand  = _bench_rand_pb(N)
    push!(rows, (N, t_mul, t_psmul, t_ev, t_get, t_rand))
    @printf(stderr, "done N=%d\n", N)
end

# Wide-word rows have no v3 baseline; recorded for the release notes.
for N in (200, 500, 1000)
    try
        t_mul   = _bench_pauli_mul(N)
        t_ev    = _bench_evolve(N, 200, 30)
        t_get   = _bench_dict_getset(N, 1000)
        t_rand  = _bench_rand_pb(N)
        push!(rows, (N, t_mul, NaN, t_ev, t_get, t_rand))
        @printf(stderr, "done N=%d (wide)\n", N)
    catch e
        e isa ArgumentError || rethrow()   # v3 baseline tree: >128 unsupported
        break
    end
end

println("| N | Pauli*Pauli (ns) | PauliSum*PauliSum 300x300 (ms) | evolve!+clip 200t x 30rot (ms) | Dict get x1000 (us) | rand(PauliBasis) (ns) |")
println("|---|---|---|---|---|---|")
for (N, t1, t2, t3, t4, t5) in rows
    @printf("| %d | %.1f | %s | %.2f | %.1f | %.1f |\n",
            N, t1*1e9, isnan(t2) ? "—" : @sprintf("%.2f", t2*1e3), t3*1e3, t4*1e6, t5*1e9)
end
