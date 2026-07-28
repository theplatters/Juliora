function sector_test_mrio(indices::DataFrame)
    n = nrow(indices)
    transactions = MatrixEntry(zeros(n, n), indices, indices)
    final_demand_indices = DataFrame(Category = ["Households"])
    final_demand = MatrixEntry(ones(n, 1), final_demand_indices, indices)
    value_added_indices = DataFrame(Category = ["Value added"])
    value_added = MatrixEntry(ones(1, n), indices, value_added_indices)
    return MRIO(Z = transactions, Y = final_demand, VA = value_added)
end

@testset "Eora and Gloria use the same sector representation" begin
    gloria_indices = DataFrame(
        CountryCode = ["AUT", "AUT", "DEU", "DEU"],
        Sector = ["Agriculture", "Manufacturing", "Agriculture", "Manufacturing"],
    )
    eora_indices = DataFrame(
        CountryCode = ["AUT", "AUT", "DEU", "DEU"],
        Industry = ["01", "02", "01", "02"],
        Sector = ["Agriculture", "Manufacturing", "Agriculture", "Manufacturing"],
    )

    gloria = sector_test_mrio(gloria_indices)
    eora = sector_test_mrio(eora_indices)

    expected = ["Agriculture", "Manufacturing"]
    @test (@inferred sectors(gloria)) == expected
    @test (@inferred sectors(eora)) == expected
    @test sectors(gloria) == sectors(eora)
    @test eltype(sectors(gloria)) === String
    @test eltype(sectors(eora)) === String
    @test sector(gloria) == expected
end

@testset "sectors preserves order and normalizes metadata to strings" begin
    indices = DataFrame(Sector = Any["Services", missing, :Agriculture, "Services"])

    @test sectors(indices) == ["Services", "Agriculture"]
    @test sectors(DataFrame(Industry = ["A", "B", "A"])) == ["A", "B"]
    @test sectors(DataFrame(CountryCode = ["AUT"])) == String[]
end
