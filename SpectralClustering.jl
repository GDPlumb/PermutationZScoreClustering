using Clustering

function cluster_spectral(Dists::Array{Float64, 2}, K::Int)

	# Normalize the dissimilarities between objects: makes using a default Sigma_array make sense
	Dists = Dists ./ maximum(Dists)

	# Define the default Sigma_array (0.05, 0.1, 0.15, 0.2)
	Sigma_array = [0.215, 0.22, 0.225, 0.23, 0.235, 0.24, 0.245, 0.25, 0.275, 0.3, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 5.0];
	L = length(Sigma_array)

	# Run the clustering procedure for all of the values in Sigma_array
	println("Sigma     Cost")
	Clusterings = Array(KmeansResult, L)
	for i = 1:L
		Clusterings[i] = cluster_spectral(Dists, K, Sigma_array[i])
		println(Sigma_array[i], "          ", Clusterings[i].totalcost)
	end

	# Find the sigma value that produced the lowest clustering cost: this is theoretically sound
	best_index = 0
	best_cost = Inf
	for i = 1:L
		if Clusterings[i].totalcost < best_cost
			best_index = i
			best_cost = Clusterings[i].totalcost
		end
	end

	return Clusterings[best_index], Sigma_array[best_index]
end

function cluster_spectral(Dists::Array{Float64, 2}, K::Int, Sigma::Float64)
	
	P = size(Dists, 1)

	# Compute the Affinity matrix
	A = Array(Float64, P, P)
	scale = -1.0 / (2 * Sigma * Sigma)
	for i = 1:P
		A[i, i] = 0.0
		for j = (i + 1):P
			affinity = exp(scale * Dists[i, j])
			A[i, j] = affinity
			A[j, i] = affinity
		end
	end

	# Compute the diagonal matrix D^-0.05
	D = zeros(Float64, P)
	for i = 1:P
		sum = 0.0
		for j = 1:P
			sum += A[i, j]
		end
		D[i] = sum ^ -0.5
	end
	D = Diagonal(D)

	# Compute L = D^-0.5 * A * D^-0.5
	L = D * A * D
	L = Symmetric(L)

	# Compute the K largest eigenvectors of L
	Eig = eigfact(L, (P - K + 1):P)

	# Compute the data matrix that will be passed to K-Means
	X = transpose(Eig[:vectors]) #the top K eigenvectors are stored in the columns of Eig[:vectors]
	for i = 1:P 
		col_norm = norm(X[:, i])
		X[:, i] = X[:, i] ./ col_norm
	end

	# Find the absolute value of the cosine of the angles between the columns of X
	CS = Array(Float64, P, P)
	for i = 1:P
		CS[i, i] = 1.0
		for j = (i + 1):P
			cosine = abs(dot(X[:, i], X[:, j]))
			CS[i, j] = cosine
			CS[j, i] = cosine
		end
	end 

	# Find the initial cluster centers for K-Means
	Centers_init = zeros(Int, K)
	Centers_init[1] = rand(1:P)
	for i = 2:K
		index_best = 0
		angle_best = 1.0
		for j = 1:P
			CS[j, Centers_init[1:(i - 1)]]
			angle = maximum(CS[j, Centers_init[1:(i - 1)]])
			if angle < angle_best
				index_best = j
				angle_best = angle
			end
		end
		Centers_init[i] = index_best
	end

	return kmeans(X, K, init = Centers_init)
end