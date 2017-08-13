require("Duplicate.jl")
require("SpectralClustering.jl")

# Each row of ZScores is a feature vector
# ie.  each column has mean 0 and variance 1 and corresponds to a specific feature
function experiment(K::Int, ZScores::Array{Float64, 2}, Sigma::Float64, NumSamples::Int, FileName::String)

	P, N = size(ZScores)

	### Compute the Probability Distribution and the Fast Fourier Transform of that distribution for each subject ###
	println("Computing FFTs")
	YOR, PT = yor(N)
	FTA = Array(Array{Array{Float64, 2}, 1}, P)
	for i = 1:P
		SNF, NZL = zscores_pdf(vec(ZScores[i,:]), Sigma, NumSamples)
		FT = sn_fft_sp(N, SNF, NZL, YOR, PT)
		FTA[i] = FT
	end


	### Compute the Distance Matrix ###
	println("Computing Distance Matrix")
	Weights = weights_coset(N)
	Distances = Array(Float64, P, P)
	for i = 1:P
		Distances[i, i] = 0.0
		for j = (i + 1):P
			dist = ft_distance(FTA[i], FTA[j], Weights)
			Distances[i, j] = dist
			Distances[j, i] = dist
		end
	end

	### Run the clustering method ###
	Clustering, Parameter_best = cluster_spectral(Distances, K)

	### Save the output ###
	save(string(FileName, ".jld"), "A", Clustering.assignments, "Parameter_best", Parameter_best, "Distances", Distances)

end
