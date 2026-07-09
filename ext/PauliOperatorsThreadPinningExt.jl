module PauliOperatorsThreadPinningExt

using PauliOperators
using ThreadPinning

# Backend for PauliOperators.pin_engine!: compact pinning keeps the sharded
# engine's worker threads on fixed cores so first-touch page placement stays
# meaningful. ThreadPinning only acts on Linux; elsewhere this is a no-op.
function _pin!(S::ShardedPauliSum)
    if Sys.islinux()
        ThreadPinning.pinthreads(:cores)
    else
        @info "thread pinning is only supported on Linux; skipped"
    end
    return S
end

end
