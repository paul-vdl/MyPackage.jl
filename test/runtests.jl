using MyPackage
using Test
using FileIO
using Images

@testset "MyPackage.jl" begin
    mat_file = joinpath(@__DIR__, "..", "SyntheticImages500.mat")
    output_file = "grille_denoise.png"

    try
        img = fill(0.7f0, 64, 64)
        MyPackage.visualize_forward_diffusion_grid(img)
        @test isfile(output_file)
    catch e
        @error "Erreur lors de l'exécution de visualize_forward_diffusion_grid : $e"
        @test false
    end

    @testset "visualize_forward_diffusion_grid" begin
        img = fill(0.7f0, 64, 64)
        noisy_img = load(output_file)

        @test size(noisy_img)[1] == 64
        @test size(noisy_img)[2] == 64 * 11  # 11 steps

        patch = Float32.(channelview(noisy_img[:, 1:64]))  # assure cohérence type
        diff = maximum(abs.(patch .- img))
        @test diff > 0.001  # bruit visible à l’œil (≠ juste epsilon numérique)
    end
end
