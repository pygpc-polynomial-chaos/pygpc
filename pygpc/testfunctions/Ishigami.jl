function Ishigami(x1, x2, x3, a, b)
    return sin.(x1) .- a .* sin.(x1).^2 .+ b .* x3.^4 .* sin.(x1)
end

#= x1 = rand(1, 5)
x2 = rand(1, 5)
x3 = rand(1, 5)
a = 1.0
b = 0.7

y = Ishigami(x1, x2, x3, a, b) =#