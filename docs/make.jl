using QCMaterial
using Documenter

makedocs(;
    modules=[QCMaterial],
    authors="Hiroshi Shinaoka <h.shinaoka@gmail.com> and contributors",
    repo="https://github.com/shinaoka/QCMaterial.jl/blob/{commit}{path}#L{line}",
    sitename="QCMaterial.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://shinaoka.github.io/QCMaterial.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/shinaoka/QCMaterial.jl",
)
