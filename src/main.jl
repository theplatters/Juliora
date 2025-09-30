using Juliora

a = Juliora.Eora("data/2017/");

a.env.A[:, [(CountryCode = "AFG", Industry = "Industries", Sector = "Agriculture"), (CountryCode = "USA", Industry = "Industries", Sector = "Agriculture")]]

a.env.A[:, (CountryCode = "USA", Sector = "Agriculture")]


