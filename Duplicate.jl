using Distributions

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

function ft_distance(FT1::Array{Array{Float64, 2}, 1}, FT2::Array{Array{Float64, 2}, 1}, Weights::Array{Float64, 1})
	dist = 0.0
	for p = 1:length(FT1)
		dist += Weights[p] * matrixnorm_hs(FT1[p] - FT2[p])
	end
	return dist
end

function weights_coset(N::Int)
	P, WI = SnFFT.partitions(N)
	Pn = P[N]
	NP = length(Pn)
	Weights = Array(Float64, NP)
	for p = 1:NP
		Weights[p] = factorial(Pn[p][1])
	end
	return Weights
end

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