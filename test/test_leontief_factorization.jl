@testset "solve_leontief" begin
    indices = DataFrame(CountryCode = ["AUT", "DEU"], Sector = ["A", "B"])
    coefficients = [0.2 0.1; 0.3 0.2]
    entry = MatrixEntry(coefficients, indices, indices)
    factorization = IO.calculate_leontief_factorization(entry)

    vector_demand = [10.0, 20.0]
    expected_vector = (I - coefficients) \ vector_demand
    @test solve_leontief(factorization, vector_demand) ≈ expected_vector
    @test solve_leontief(factorization, vector_demand) isa Vector{Float64}

    matrix_demand = [10 2; 20 4]
    expected_matrix = (I - coefficients) \ matrix_demand
    @test solve_leontief(factorization, matrix_demand) ≈ expected_matrix
    @test size(solve_leontief(factorization, matrix_demand)) == size(matrix_demand)

    @test_throws DimensionMismatch solve_leontief(factorization, ones(3))
    @test_throws MethodError solve_leontief(factorization, ["10", "20"])
end

@testset "row and column sums" begin
    values = [1 2 3; 4 5 6]

    @test sum_rows(values) == [6, 15]
    @test sum_cols(values) == [5, 7, 9]
    @test sum_rows(zeros(0, 3)) == Float64[]
    @test sum_cols(zeros(2, 0)) == Float64[]

    @test_throws MethodError sum_rows(["a" "b"])
    @test_throws MethodError sum_cols(["a" "b"])
end

@testset "Leontief inverse cache (H4)" begin
    indices = DataFrame(CountryCode = ["AUT", "DEU"], Sector = ["A", "B"])
    coefficients = [0.2 0.1; 0.3 0.2]
    entry = MatrixEntry(coefficients, indices, indices)
    factorization = IO.calculate_leontief_factorization(entry)

    # The 3-arg constructor keeps working and starts with an empty cache.
    @test factorization.inverse_cache[] === nothing

    # First access materializes F\I; later accesses return the identical object.
    d1 = factorization.data
    @test d1 ≈ (I - coefficients) \ Matrix{Float64}(I, 2, 2)
    @test factorization.data === d1
    @test factorization.data === d1

    # Labeled accessors route through the cached matrix.
    @test factorization[(CountryCode = "AUT", Sector = "A"), (CountryCode = "AUT", Sector = "A")] ≈ d1[1, 1]
    @test factorization.data === d1

    # Direct 3-arg construction (used by mrio/aggregation) also caches.
    lf2 = IO.LeontiefFactorization(lu(I - coefficients), indices, indices)
    @test lf2.data ≈ d1
    @test lf2.data === lf2.data
end

@testset "Leontief sum_rows/sum_cols (H4)" begin
    indices = DataFrame(CountryCode = ["AUT", "DEU"], Sector = ["A", "B"])
    coefficients = [0.2 0.1; 0.3 0.2]
    entry = MatrixEntry(coefficients, indices, indices)
    factorization = IO.calculate_leontief_factorization(entry)

    @test sum_rows(factorization) ≈ vec(sum(factorization.data; dims = 2))
    @test sum_cols(factorization) ≈ vec(sum(factorization.data; dims = 1))
    @test sum_rows(factorization) isa Vector{Float64}
    @test sum_cols(factorization) isa Vector{Float64}
end
