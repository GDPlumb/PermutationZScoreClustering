using SnFFT
using Distributions

########################
# Clustering Algorithm #
########################

function cluster(K::Int, Data::Array{Float64, 2}, Sigma::Float64, NumSamples::Int)
	P, N = size(Data)

	YOR, PT = yor(N)
	
	FTA = Array(Array{Array{Float64, 2}, 1}, P)
	ZScores = Array(Float64, N)
	for p = 1:P
		for n = 1:N
			ZScores[n] = Data[p, n]
		end
		SNF, NZL = zscores_pdf(ZScores, Sigma, NumSamples)
		FTA[p] = sn_fft_sp(N, SNF, NZL, YOR, PT)
	end	

	Weights = weights_coset(N)
	

	A, M, D, DD = kmeans(K, FTA, Weights)

	CI = Array(Float64, P, K)
	for p = 1:P
		for k = 1:K
			CI[p,k] = ft_distance(FTA[p], M[k], Weights)
		end
	end

	return A, M, CI, DD
end

function cluster_bl(K::Int, Data::Array{Float64, 2}, BL::Int, Sigma::Float64, NumSamples::Int)
	P, N = size(Data)

	YOR, PT, PIA = yor_bl(N, BL)
	
	FTA = Array(Array{Array{Float64, 2}, 1}, P)
	ZScores = Array(Float64, N)
	for p = 1:P
		for n = 1:N
			ZScores[n] = Data[p, n]
		end
		SNF = zscores_pdf_bl(ZScores, BL, Sigma, NumSamples)
		FTA[p] = sn_fft_bl(N, BL, SNF, YOR, PT, PIA)
	end	

	Weights = weights_coset_bl(N, BL)
	

	A, M, D, DD = kmeans(K, FTA, Weights)

	CI = Array(Float64, P, K)
	for p = 1:P
		for k = 1:K
			CI[p,k] = ft_distance(FTA[p], M[k], Weights)
		end
	end

	return A, M, CI, DD
end

###################
# Fourier K-Means #
###################

function kmeans(K::Int, Data::Array{Array{Array{Float64, 2}, 1}, 1}, Weights::Array{Float64, 1})
	NP = length(Weights)
	Dims = Array(Int, NP)
	for p = 1:NP
		Dims[p] = size(Data[1][p], 1)
	end

	Means, DD = kmeans_farthest(K, Data, Dims, Weights)
	A, D = kmeans_assign(Data, Means, Weights)
	cost = sum(D)
	println(cost)
	while true
		Means = kmeans_update(K, Data, Dims, A, D)
		A_new, D_new = kmeans_assign(Data, Means, Weights)
		cost_new = sum(D_new)
		if cost - cost_new < 1e-3
			A = A_new
			D = D_new
			break
		else
			A = A_new
			D = D_new
			cost = cost_new
		end
		println(cost)
	end
	return A, Means, D, DD
end

function kmeans_assign(Data::Array{Array{Array{Float64, 2}, 1}, 1}, Means::Array{Array{Array{Float64, 2}, 1}, 1}, Weights::Array{Float64, 1})
	N = length(Data)
	K = length(Means)

	A = Array(Int, N)
	D = Array(Float64, N)
	for n = 1:N
		a_best = 0
		d_best = Inf
		for k = 1:K
			d_cur = ft_distance(Data[n], Means[k], Weights)
			if d_cur < d_best
				a_best = k
				d_best = d_cur
			end
		end
		A[n] = a_best
		D[n] = d_best
	end
	return A, D
end

function kmeans_update(K::Int, Data::Array{Array{Array{Float64, 2}, 1}, 1}, Dims::Array{Int, 1}, A::Array{Int, 1}, D::Array{Float64, 1})
	N = length(Data)
	NP = length(Dims)
	
	Means = Array(Array{Array{Float64, 2}, 1}, K)
	for k = 1:K
		M = Array(Array{Float64, 2}, NP)
		for p = 1:NP
			M[p] = zeros(Float64, Dims[p], Dims[p])
		end
		Means[k] = M
	end

	Means_size = zeros(Int, K)
	for n = 1:N
		a = A[n]
		Means_size[a] += 1
		ft_increment!(Means[a], Data[n], 1.0)
	end
	for k = 1:K
		s = Means_size[k]
		if s != 0
			Means[k] = Means[k] / s
		else
			d_worst = -Inf
			n_worst = 0
			for n = 1:N
				d_cur = D[n]
				if d_cur > d_worst
					d_worst = d_cur
					n_worst = n
				end
			end
			Means[k] = Means[k] + Data[n_worst]
		end
	end
	
	return Means
end

function kmeans_partition(K::Int, Data::Array{Array{Array{Float64, 2}, 1}, 1}, Dims::Array{Int, 1})
	N = length(Data)
	
	A = zeros(Int, N)
	for k = 1:K
		n = rand(1:N)
		while A[n] != 0
			n = rand(1:N)
		end
		A[n] = k
	end
	for n = 1:N
		if A[n] == 0
			A[n] = rand(1:K)
		end
	end

	Means = Array(Array{Array{Float64, 2}, 1}, K)
	NP = length(Dims)
	for k = 1:K
		M = Array(Array{Float64, 2}, NP)
		for p = 1:NP
			M[p] = zeros(Float64, Dims[p], Dims[p])
		end
		Means[k] = M
	end
	
	Means_size = zeros(Int, K)
	for n = 1:N
		a = A[n]
		Means_size[a] += 1
		ft_increment!(Means[a], Data[n], 1.0)
	end
	for k = 1:K
		s = Means_size[k]
		Means[k] = Means[k] / s
	end

	return Means
end

function kmeans_farthest(K::Int, FTA::Array{Array{Array{Float64, 2}, 1}, 1}, Dims::Array{Int, 1}, Weights::Array{Float64, 1})
	P = length(FTA)

	DD = Array(Float64, P, P)
	for i = 1:P
		DD[i, i] = 0.0
		for j = (i + 1):P
			dist = ft_distance(FTA[i], FTA[j], Weights)
			DD[i, j] = dist
			DD[j, i] = dist
		end
	end

	MI = Array(Int, K)
	MI[1] = rand(1:P)
	for k = 2:K
		i_best = 0
		d_best = 0.0
		
		for p = 1:P
			d = 0.0
			for i = 1:(k - 1)
				d += DD[p, MI[i]]
			end
			if d > d_best
				d_best = d
				i_best = p
			end
		end

		MI[k] = i_best
	end

	M = Array(Array{Array{Float64, 2}, 1}, K)
	for k = 1:K
		M[k] = copy(FTA[MI[k]])
	end
	return M, DD
end
	

############################################
# Converting from ZScores to a PDF over Sn #
############################################

function zscores_pdf(ZScores::Array{Float64, 1}, Sigma::Float64, NumSamples::Int)
	N = length(ZScores)

	Distributions = Array(Normal, N)
	for n = 1:N
		Distributions[n] = Normal(ZScores[n], Sigma)
	end

	PDF = zeros(Float64, factorial(N))
	Sample = Array(Float64, N)
	for s = 1:NumSamples

		for n = 1:N
			Sample[n] = rand(Distributions[n])
		end

		Permutation = find_permutation(Sample)

		Index = permutation_index(Permutation)
		PDF[Index] += 1.0
	end
	for i = 1:length(PDF)
		PDF[i] /= NumSamples
	end

	num = 0
	for i = 1:length(PDF)
		if PDF[i] != 0
			num += 1
		end
	end
	SNF = Array(Float64, num)
	NZL = Array(Int, num)
	index = 1
	for i = 1:length(PDF)
		v = PDF[i]
		if v != 0
			SNF[index] = v
			NZL[index] = i
			index += 1
		end
	end
	return SNF, NZL
end

function zscores_pdf_bl(ZScores::Array{Float64, 1}, K::Int, Sigma::Float64, NumSamples::Int)
	N = length(ZScores)

	Distributions = Array(Normal, N)
	for n = 1:N
		Distributions[n] = Normal(ZScores[n], Sigma)
	end

	NF = factorial(N)
	BS = factorial(N - K)
	
	PDF = zeros(int(NF/ BS))
	Sample = Array(Float64, N)
	for s = 1:NumSamples

		for n = 1:N
			Sample[n] = rand(Distributions[n])
		end

		Permutation = find_permutation(Sample)

		Index = permutation_index(Permutation)
		PDF[ceil(Index / BS)] += 1.0
	end
	for i = 1:length(PDF)
		PDF[i] /= NumSamples
	end
	return PDF
end

function find_permutation(Scores::Array{Float64, 1})
	N = length(Scores)
	Permutation = Array(Int, N)
	Values = Array(Float64, N)
	
	Permutation[1] = 1
	Values[1] = Scores[1]
	for n = 2:N
		Score = Scores[n]
		i = 1
		while i <= (n - 1)
			if Score < Values[i]
				break
			end
			i += 1
		end
		for j = (n - 1):-1:i
			Permutation[j + 1] = Permutation[j]
			Values[j + 1] = Values[j]
		end
		Permutation[i] = n
		Values[i] = Score
	end
	return Permutation
end

############################################
# Converting from a PDF over Sn to ZScores #
############################################

function pdf_zscores(N::Int, SNF::Array{Float64, 1}, Sigma::Float64, Epsilon::Float64)
	P = zeros(Float64, N, N)
	for p = 1:length(SNF)
		permutation = index_permutation(N, p)
		probability = SNF[p]
		for n = 1:N
			i = 1
			while true
				j = permutation[i]
				if j == n
					break
				else
					P[n, j] += probability
				end
				i += 1
			end
		end
	end
	Sigma_bar = sqrt(2) * Sigma
	ZScores = Array(Float64, N)
	ZScores[1] = 0.0
	for i = 1:(N - 1)
		probability = P[i, i + 1]
		ZScores[i + 1] = ZScores[i] + find_zscore(probability, Sigma_bar, Epsilon)
	end
	return ZScores
end

function pdf_zscores_bl(N::Int, K::Int, SNF::Array{Float64, 1}, Sigma::Float64, Epsilon::Float64)
	P = zeros(Float64, N, N)
	BS = factorial(N - K)
	Info = Array(Int, N)
	for p = 1:length(SNF)
		index = BS * (p - 1) + 1
		permutation =  index_permutation(N, index)
		for i = 1:(N - K)
			Info[permutation[i]] = 0
		end
		for i = (N - K + 1):N
			Info[permutation[i]] = 1
		end
		probability = SNF[p]
		for n = 1:N
			if Info[n] == 0
				for i = 1:(N - K)
					j = permutation[i]
					if j != n
						P[n,j] += probability / 2
					end
				end
			else
				for i = 1:(N - K)
					P[n, permutation[i]] += probability
				end
				i = N - K + 1
				while true
					j = permutation[i]
					if j == n
						break
					else
						P[n, j] += probability
					end
					i += 1
				end
			end
		end
	end
	ZScores = Array(Float64, N)
	ZScores[1] = 0.0
	for i = 1:(N - 1)
		probability = P[i, i + 1]
		ZScores[i + 1] = ZScores[i] + find_zscore(probability, Sigma, Epsilon)
	end
	return ZScores
end

function find_zscore(CD::Float64, Sigma::Float64, Epsilon::Float64)
	g = 0.0
	ss = 0.1
	f = 0
	while true
		CD_g = cdf(Normal(g, Sigma), 0)
		if abs(CD - CD_g) < Epsilon
			break
		else
			if CD_g < CD
				if f == 1
					ss /= 2
				end
				g -= ss
				f = -1
			else
				if f == -1
					ss /= 2
				end
				g += ss
				f = 1
			end
		end
	end
	return g
end

################################
# Fourier Transform Operations #
################################

###
# Fourier Transform Composition
###

function ft_increment!(FT_sum::Array{Array{Float64, 2}, 1}, FT_addend::Array{Array{Float64, 2}, 1}, Weight::Float64)
	for p = 1:length(FT_sum)
		FT_sum[p] += Weight * FT_addend[p]
	end
end 

###
# Fourier Transform Distance
###

function ft_distance(FT1::Array{Array{Float64, 2}, 1}, FT2::Array{Array{Float64, 2}, 1}, Weights::Array{Float64, 1})
	dist = 0.0
	for p = 1:length(FT1)
		dist += Weights[p] * matrixnorm_hs(FT1[p] - FT2[p])
	end
	return dist
end

###
# Weight Functions
###

function weights_coset(N::Int)
	P, WI = partitions(N)
	Pn = P[N]
	NP = length(Pn)
	Weights = Array(Float64, NP)
	for p = 1:NP
		Weights[p] = factorial(Pn[p][1])
	end
	return Weights
end

function weights_coset_bl(N::Int, K::Int)
	P, WI = partitions(N)
	Pn = P[N]
	NP = length(Pn)
	i = 1
	while true
		if P[N][i][1] >= N - K
			break
		else
			i += 1
		end
	end
	Weights = Array(Float64, length(P[N]) - i + 1)
	c = 1
	for p = i:NP
		Weights[c] = factorial(Pn[p][1])
		c += 1
	end
	return Weights
end

################
# Matrix Norms #
################

# squared Hilbert-Schmidt norm for a real-valued matrix
function matrixnorm_hs(A::Array{Float64, 2})
	Dim = size(A, 1)
	norm = 0.0
	for i = 1:Dim
		for j = 1:Dim
			norm += A[i,j] * A[i,j]
		end
	end
	return sqrt(norm)
end

###################
# Data Processing #
###################

function normalize(Data::Array{Float64, 2})
	P, N = size(Data)
	NData = Array(Float64, P, N)
	for n = 1:N
		mean = 0.0
		for p = 1:P
			mean += Data[p, n]
		end
		mean /= P

		var = 0.0
		for p = 1:P
			var += (Data[p,n] - mean) * (Data[p,n] - mean)
		end
		var /= P

		for p = 1:P
			NData[p,n] = (Data[p,n] - mean) / var
		end
	end
	return NData
end