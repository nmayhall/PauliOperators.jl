using PauliOperators
using Documenter

DocMeta.setdocmeta!(PauliOperators, :DocTestSetup, :(using PauliOperators); recursive=true)

makedocs(;
    modules=[PauliOperators],
    authors="Nick Mayhall",
    repo="https://github.com/nmayhall/PauliOperators.jl/blob/{commit}{path}#{line}",
    sitename="PauliOperators.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://nmayhall.github.io/PauliOperators.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Internals" => [
            "Pauli Representation" => "representation.md",
            "Data Structures & Performance" => "data_structures.md",
            "Truncation" => "truncation.md",
        ],
        "Migration Guides" => [
            "DBF.jl" => "migration_DBF.md",
            "DissipativePauliGroundState.jl" => "migration_DissipativePauliGroundState.md",
            "OpenSCI.jl" => "migration_OpenSCI.md",
        ],
        "Reference" => [
            "Types" => "types.md",
            "Functions" => "functions.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/nmayhall/PauliOperators.jl",
    devbranch="main",
)
