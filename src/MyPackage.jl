module MyPackage

using MAT
using Images
using FileIO

export generate_grid

function generate_grid()
    matpath = joinpath(@__DIR__, "..", "SyntheticImages500.mat")
    
    if !isfile(matpath)
        @warn "Le fichier MAT est manquant. Fonction ignorée."
        return nothing
    end

    data = matread(matpath)
    raw = data["syntheticImages"]
    images = reshape(raw, 32, 32, 500)

    first64 = images[:, :, 1:64]
    img_size = size(first64, 1), size(first64, 2)
    canvas = zeros(Float32, 8 * img_size[1], 8 * img_size[2])

    for i in 0:7, j in 0:7
        idx = i * 8 + j + 1
        canvas[i*img_size[1]+1:(i+1)*img_size[1],
               j*img_size[2]+1:(j+1)*img_size[2]] .= first64[:, :, idx]
    end

    canvas = clamp01.(canvas)
    save("grille.png", colorview(Gray, canvas))
end

end
