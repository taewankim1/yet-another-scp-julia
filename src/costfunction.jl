using LinearAlgebra
include("dynamics.jl")


function get_cost(dynamics::Unicycle,x::Vector,u::Vector,idx::Int,N::Int)
    return dot(u,u)
end

function get_cost(dynamics::Rocket,x::Vector,u::Vector,idx::Int,N::Int)
    if idx == N+1 
        return -x[1]
    else
        return 0
    end
end