using QCMaterial
using Documenter

makedocs(;
    modules=[QCMaterial],
    authors="Rihito Sakurai <sakurairihito@gmail.com> and contributors",
    repo="https://github.com/sakurairihito/QCMaterialNew/blob/{commit}{path}#L{line}",
    sitename="QCMaterialNew",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://github.com/sakurairihito/QCMaterialNew",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Chapter1"=>[
            "Introduction" => "chapter1/intro.md",
            "About Goma-chan" => "chapter1/goma.md",
        ],
        "Chapter2"=>"chapter2/azarashi.md",
    ],
)

deploydocs(;
    repo="github.com/sakurairihito/QCMaterialNew",
    versions=nothing
)
