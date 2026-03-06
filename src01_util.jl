## ---
## Utility functions
## ---
"""
A speedier findall variant
"""
function findallF(f, a::Array{T, N}) where {T, N}
    j = 1
    b = Vector{Int}(undef, length(a))
    @inbounds for i in eachindex(a)
        @inbounds if f(a[i])
            b[j] = i
            j += 1
        end
    end
    resize!(b, j-1)
    sizehint!(b, length(b))
    return b
end

"""
naLess : remove na's from array
"""
function naLess(a::Array)
    a = filter(!isnan, a)
    return a
end

"""
Color agents
"""
function colorAgents(sR::Array{Float64}, sP::Array{Float64}, pal₁, pal₂, cThresh, cLim)
    n  = length(sR)
    co = []
    for i ∈ 1:n
        if abs(sP[i]) > cThresh
            if sP[i] > cThresh
                push!(co, get(pal₁, sP[i], (cThresh, cLim)))
            elseif sP[i] < -cThresh
                push!(co, get(reverse(pal₂), sP[i], (-cLim, -cThresh)))
            end
        else
            push!(co, "#1f1e1e")
        end
    end
    return co
end

"""
Angle on a torus
"""
function torusAngles(x::Float64, y::Float64, h::Float64, neighborX::Vector{Float64}, neighborY::Vector{Float64}, habitatSize::Int)
    ## Compute relative positions of neighbors with respect to focal agent
    dx = neighborX .- x
    dy = neighborY .- y
    ## Wrap distances around the torus
    dx[dx .> habitatSize  / 2] .-= habitatSize
    dx[dx .< -habitatSize / 2] .+= habitatSize
    dy[dy .> habitatSize  / 2] .-= habitatSize
    dy[dy .< -habitatSize / 2] .+= habitatSize
    ## Compute angles
    angles = atan.(dy, dx) .- h
    ## Wrap angles to [-π, π]
    angles[angles .> π]  .-= 2π
    angles[angles .< -π] .+= 2π
    return angles
end

"""
Angle mod
"""
function wrapAngle(h)
    ## Compute modulus
    return mod(h + π, 2π) - π
end

"""
Bearing angle between predator and prey
"""
function computeBearingAngle(pred_x, pred_y, pred_h, prey_x, prey_y)
    ## Line-of-sight angle: angle from predator to prey
    lineOfSight = atan(prey_y - pred_y, prey_x - pred_x)
    ## Bearing angle: difference between line-of-sight and predator's current heading
    bearingAngle = wrapAngle(lineOfSight - pred_h)
    return bearingAngle
end

"""
Make the arena a torus
"""
function torusReflect(xStep::Float64, yStep::Float64, habitatSize::Int)
    ## Wrap around x-axis
    if xStep < 0
        xStep += habitatSize
    elseif xStep > habitatSize
        xStep -= habitatSize
    end
    ## Wrap around y-axis
    if yStep < 0
        yStep += habitatSize
    elseif yStep > habitatSize
        yStep -= habitatSize
    end
    return xStep, yStep
end

"""
Rotate punchouts
"""
function getCurrentAndTargetTile(neighborhood::Matrix{Float64}, heading::Float64)
    ## The center of the grid (the agent's position is at [2, 2])
    agent_pos = (2, 2)
    ## Current tile (agent's tile)
    current_tile = neighborhood[agent_pos[1], agent_pos[2]]
    ## Calculate direction vector based on the heading
    dx = cos(heading)
    dy = sin(heading)
    ## Step through the grid along the ray
    x, y = agent_pos
    target_tile = nothing
    while true
        ## Move in the direction of the ray
        x += dx
        y += dy
        ## Round x and y to the nearest valid grid position (1-based)
        x_grid = round(Int, x)
        y_grid = round(Int, y)
        ## Clamp to grid boundaries (ensures no out-of-bounds access)
        if x_grid >= 1 && x_grid <= 3 && y_grid >= 1 && y_grid <= 3
            # First tile that the ray collides with
            target_tile = neighborhood[y_grid, x_grid]
            break
        end
        ## Optionally add a safeguard to stop after a certain number of steps if no valid tile is found
        ## This prevents an infinite loop in case there's an error in ray direction or grid size
        if abs(x) > 3 || abs(y) > 3
            break
        end
    end
    ## Check if target_tile was set; if not, return some default or error value
    if target_tile == nothing
        throw(ArgumentError("Ray did not hit a valid tile within the grid bounds"))
    end
    return current_tile, target_tile, sum(neighborhood)
end

"""
generate_orthogonal_gaussian_vector(v::Vector{<:Real})

Generates a new random Gaussian vector that is orthogonal to the given vector `v`.

Args:
    v (Vector{<:Real}): The input vector to which the new vector should be orthogonal.

Returns:
    Vector{Float64}: A new vector that is orthogonal to `v`, with Gaussian marginals.
"""
function generateOrthogonalGaussianBlock(dimension::Int, num_vectors::Int)
    # Sanity check: You cannot generate more mutually orthogonal vectors than the dimension itself.
    if num_vectors > dimension
        error("Cannot generate $num_vectors mutually orthogonal vectors in $dimension dimensions within a single block. " *
              "The 'num_vectors' argument for a block should be <= 'dimension'.")
    end
    # Step 1: Generate a random Gaussian matrix of full rank (dimension x dimension).
    # Its QR decomposition (specifically the Q matrix) will give us an orthonormal basis.
    rand_matrix = randn(dimension, dimension)
    Q, _ = qr(rand_matrix) # Q's columns form an orthonormal basis
    # Step 2: Select `num_vectors` orthogonal directions from Q's columns.
    # Step 3: Scale each selected direction by an independent standard Gaussian scalar (randn()).
    # This crucial step ensures that each *element* of the resulting perturbation vector
    # also has a Gaussian marginal distribution, while maintaining orthogonality.
    orthogonal_gaussian_vectors = [Q[:, i] * randn() for i in 1:num_vectors]
    return orthogonal_gaussian_vectors
end

;
