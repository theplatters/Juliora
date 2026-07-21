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
