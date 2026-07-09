# Sweep driver for examples/scaling_study.jl: rank counts × merge windows.
# Collects the per-run CSV rows and prints a combined table. Local runs on a
# laptop measure correctness and communication volume, not real scaling —
# populations must be far larger before distribution pays; run on a cluster
# for the actual study.
#
#     julia --project examples/run_scaling_sweep.jl [N n_trotter dt]
using MPI

N         = length(ARGS) >= 1 ? ARGS[1] : "20"
n_trotter = length(ARGS) >= 2 ? ARGS[2] : "10"
dt        = length(ARGS) >= 3 ? ARGS[3] : "0.1"

script = joinpath(@__DIR__, "scaling_study.jl")
exe = MPI.mpiexec()

header = "np,N,rotations,r,window,strict,wall_s,terms,shipped_records,shipped_KB,rebalances,rel_err"
println(header)
for np in (1, 2, 4, 8), window in (1, 8, 32)
    r = string(trailing_zeros(np) + 4)
    out = read(`$exe -n $np $(Base.julia_cmd()) --project=$(Base.active_project()) $script $N $n_trotter $dt $r $window`,
               String)
    for line in split(out, '\n')
        startswith(line, "$np,") && println(line)
    end
end
