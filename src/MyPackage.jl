module MyPackage
export generate_grid, visualize_forward_diffusion_grid


using MAT
using Images
using FileIO

function generate_grid()
    data = matread(joinpath(@__DIR__, "..", "SyntheticImages500.mat"))
    raw = data["syntheticImages"]         # 32x32x1x500
    images = reshape(raw, 32, 32, 500)     # 32x32x500

    first64 = images[:, :, 1:64]
    img_size = size(first64, 1), size(first64, 2)
    canvas = zeros(Float32, 8 * img_size[1], 8 * img_size[2])

    for i in 0:7, j in 0:7
        idx = i * 8 + j + 1
        canvas[i*img_size[1]+1:(i+1)*img_size[1],
               j*img_size[2]+1:(j+1)*img_size[2]] .= first64[:, :, idx]
    end

    canvas2 = clamp01.(canvas)
    save("grille.png", colorview(Gray, canvas2))
    return canvas
end

function visualize_forward_diffusion_grid(x0; num_steps=500, beta_min=1e-4, beta_max=0.02)
    betas = collect(range(beta_min, beta_max, length=num_steps))
    alphas = 1 .- betas
    alpha_bars = cumprod(alphas)

    ε = randn(Float32, size(x0))  # même bruit pour tout
    save_steps = [1, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    n = length(save_steps)

    h, w = size(x0)
    canvas = zeros(Float32, h, w * n)

    for (i, t) in enumerate(save_steps)
        ᾱ_t = alpha_bars[t]
        x_t = sqrt(ᾱ_t) .* x0 .+ sqrt(1 - ᾱ_t) .* ε
        canvas[:, (w*(i-1)+1):(w*i)] .= x_t
    end

    canvas = clamp01.(canvas)
    save("grille_denoise.png", colorview(Gray, canvas))
end


end