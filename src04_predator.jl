include(joinpath(@__DIR__, "src03_prey.jl"))

## import Pkg; Pkg.add(["IterTools", "Distributions", "StatsBase", "Makie", "CairoMakie", "LinearAlgebra", "Distances", "Random", "Flux", "JLD2", "CSV", "ColorSchemes", "Plots", "Crayons", "DataFrames", "Tidier"])

# Required imports
using Distributions, StatsBase, LinearAlgebra, Distances, Random

# Required imports for neural networks
using Flux
using Flux: Dense, Chain, relu, tanh, sigmoid, softplus, gradient
using JLD2

"""
Define the Predator neural network with encoder-policy architecture
"""
struct PredatorNetwork
    encoder::Chain          # Encodes sensory inputs into higher dimensional feature-rich embedding
    attention_scorer::Chain # Encodes attention from sensory embedding
    policy_head::Chain      # Action outputs from attention-weighted sensory embeddings
end

## Make the network differentiable for gradient-based learning
Flux.@layer PredatorNetwork

## Forward pass through the network
function (m::PredatorNetwork)(x)
    ## Encode each input individually
    encodedInputs    = [m.encoder(i) for i ∈ x]
    ## Convert to matrix for batch processing
    hMatrix          = reduce(hcat, encodedInputs)
    ## Attention scores
    rawScores        = dropdims(m.attention_scorer(hMatrix); dims = 1)
    ## Compute attention
    attentionWeights = Flux.softmax(rawScores)
    ## Pool by weighted sum using attention weights
    pooled           = dropdims(sum(hMatrix .* attentionWeights'; dims = 2); dims = 2)
    ## Generate action parameters
    actions          = m.policy_head(pooled)
    return actions
end

"""
Initialize a new predator network
"""
function init_predator_network(n_input = 3, n_hidden = 64, n_hidden₂ = 8, n_hidden₃ = 48, n_output = 4, n_features = 1)
    ## Encoder
    encoder = Chain(
        Dense(n_input   => n_hidden, relu, bias = false; init = Flux.kaiming_normal(gain = 2.0)),
        Dense(n_hidden  => n_hidden₂, tanh; init = Flux.kaiming_normal(gain = 1.0)))
    ## Attention
    attentionScorer = Chain(
        Dense(n_hidden₂ => n_features; init = Flux.kaiming_normal(gain = 1.0)))
    ## Policy head
    policyHead = Chain(
        Dense(n_hidden₂ => n_hidden₃, relu, bias = false; init = Flux.kaiming_normal(gain = 1.0)),
        Dense(n_hidden₃ => n_output, tanh; init = Flux.kaiming_normal(gain = 1.0)))
    ## Output
    PredatorNetwork(encoder, attentionScorer, policyHead)
end

"""
Predator : Define our Predator agent composite data object
"""
mutable struct Predator
    x::Float64      # x location
    y::Float64      # y location
    h::Float64      # heading angle in radians
    e::Float64      # energy level
    s::Int          # state (0=searching, 1=attacking)
    tx::Float64     # prey target x (when attacking)
    ty::Float64     # prey target y (when attacking)
    age::Int        # how much time since last meal (refractory period)
    ## Gradient specific fields
    prevReward::Float64             # Previous action parameters
    ## Misc fields
    u::Int                          # Time since last learning update
    p::Int                          # Number of updates so far
    network::PredatorNetwork        # Neural network for decision making
end

"""
Freezes current state as a "ghost state" for rollout in ES
"""
struct ghostState
    land::Array{Float64}
    prey::Vector{Prey}
    predators::Vector{Predator}
end

"""
initPredator : Initialize the predators
"""
function initPredator(; n::Int, habitatSize::Int, initialE = 0.0)
    ## Empty vector to hold structs
    predators = Vector{Predator}(undef, n)
    ## Single network initialization, all agents start from common policy
    # network   = init_predator_network()
    ## Initialize predator structs
    for i ∈ 1:n
        x = Float64(rand(Uniform(0.5, habitatSize - 0.5)))
        y = Float64(rand(Uniform(0.5, habitatSize - 0.5)))
        h = Float64(rand(Uniform(-π, π)))
        ## Populate
        predators[i] = Predator(
            x,
            y,
            h,
            initialE,
            0,                    # state
            0.0, 0.0,             # target coords
            0,                    # age
            0.0,                  # prev reward
            0, 0,                 # u, p
            init_predator_network()
            # deepcopy(network)     # neural network
        )
    end
    ## Return vector
    return predators
end

"""
initPredator : Initialize the predators
"""
function initPredatorTrained(; n::Int, habitatSize::Int, networks::Vector{PredatorNetwork}, initialE = 0.0)
    ## Empty vector to hold structs
    predators = Vector{Predator}(undef, n)
    ## Single network initialization, all agents start from common policy
    # network   = init_predator_network()
    ## Initialize predator structs
    for i ∈ 1:n
        x = Float64(rand(Uniform(0.5, habitatSize - 0.5)))
        y = Float64(rand(Uniform(0.5, habitatSize - 0.5)))
        h = Float64(rand(Uniform(-π, π)))
        ## Populate
        predators[i] = Predator(
            x,
            y,
            h,
            initialE,
            0,                    # state
            0.0, 0.0,             # target coords
            0,                    # age
            0.0,                  # prev reward
            0, 0,                 # u, p
            networks[i]
        )
    end
    ## Return vector
    return predators
end

"""
updateAllPredators! : Update function to handle prey capture and add capture reward

MODIFIED: This version immediately removes captured prey from the prey vector
rather than just marking them with negative energy
"""
function updateAllPredators!(ΔPrey, predators::Vector{Predator}, prey::Vector{Prey}, collisionDistance = 0.065, refractoryPeriod = 10, conversionFactor = 7.0)
    ## Reset targeting state for each prey
    for pIdx in eachindex(prey) ## Use eachindex to avoid collect(enumerate) allocation
    pa = prey[pIdx]
    prey[pIdx] = Prey(pa.x, pa.y, pa.h, pa.e, 0, pa.tx, pa.ty) ## Creates a new Prey object if immutable
    end
    ## Track which prey have been captured during this timestep using a Set for O(1) average-case lookup
    capturedPreyIndicesSet = Set{Int}()
    ## Enter main loop
    for i in eachindex(predators) ## Use eachindex to avoid collect(enumerate) allocation
        predator = predators[i] ## Access predator object
        ## Distances to prey for this predator
        ΔPreyFocal = ΔPrey[:, i]
        ## Manually find closest prey and check if any are in range, avoiding tmp copy and allocations
        minDist = Inf
        targetIdx = 0
        for pIdx in eachindex(ΔPreyFocal)
            currentDist = ΔPreyFocal[pIdx]
            if currentDist > 0.0 && currentDist < minDist ## currentDist > 0.0 implies in range and valid
                minDist = currentDist
                targetIdx = pIdx
            end
        end
        ## Only proceed if any prey were found and predator is out of refractory period
        if targetIdx != 0 && predator.age > refractoryPeriod
            ## Only proceed if this prey hasn't been captured already in this timestep
            if !(targetIdx in capturedPreyIndicesSet)
                if prey[targetIdx].t != 1 ## Check if this prey is already targeted by another predator
                    ## Mark this prey as being targeted
                    prey[targetIdx] = Prey(prey[targetIdx].x, prey[targetIdx].y, prey[targetIdx].h, prey[targetIdx].e, 1, predator.x, predator.y)
                end
                ## Prey captured! Add energy bonus and reward
                if minDist < collisionDistance
                    absorbedEnergy = conversionFactor * prey[targetIdx].e
                    ## Add to capturedPreyIndicesSet for removal after the loop
                    push!(capturedPreyIndicesSet, targetIdx)
                    ## Reset age (time since last meal)
                    newAge = 0
                ## Otherwise notch one to age
                else
                    absorbedEnergy = 0.0
                    newAge = predator.age + 1
                end
                ## Update energy
                newE = predator.e + absorbedEnergy
                ## Create updated predator with new state
                predators[i] = Predator(predator.x, predator.y, predator.h, newE, 1, prey[targetIdx].x, prey[targetIdx].y, newAge, predator.prevReward, predator.u + 1, predator.p, predator.network)
            ## Target prey is already captured by another predator, switch to searching
            else
                ## Update predator
                predators[i] = Predator(predator.x, predator.y, predator.h, predator.e, 0, predator.tx, predator.ty, predator.age + 1, predator.prevReward, predator.u + 1, predator.p, predator.network)
            end
        ## No prey in range or in refractory period - maintain searching state
        else
            ## Update predator
            predators[i] = Predator(predator.x, predator.y, predator.h, predator.e, 0, predator.tx, predator.ty, predator.age + 1, predator.prevReward, predator.u + 1, predator.p, predator.network)
        end
    end
    ## Efficiently remove all captured prey after updating all predators
    ## Create a new vector containing only uncaptured prey by filtering
    if !isempty(capturedPreyIndicesSet)
        uncapturedPrey = Vector{Prey}(undef, length(prey) - length(capturedPreyIndicesSet))
        uncapturedCount = 1
        for pIdx in eachindex(prey)
            if !(pIdx in capturedPreyIndicesSet)
                uncapturedPrey[uncapturedCount] = prey[pIdx]
                uncapturedCount += 1
            end
        end
        empty!(prey) ## Clear the original prey vector
        append!(prey, uncapturedPrey) ## Append the uncaptured prey to it (modifies in-place)
    end
end

"""
movePredator : Move one predator agent one step with hybrid learning

Debug:
a = predatorVec[1]; w₀ = w; w₁ = w; u = predatorTurnFactor*u; ΔPredatorsFocal = ΔPredators[:, 1]; ηPredatorsFocal = ηPredators[:, 1]; ΔPreyFocal = ΔPrey[:, 1]; ηPreyFocal = ηPrey[:, 1]; cost = 0.001; collisionZone = 0.065; sensoryRange = predatorSensoryRange; λ = 0.1; lr = 0.01; γ = 0.99; predHeadings = (p->p.h).(predatorVec); σ₀ = 0.1; ρ = 0.15; β = 0.0; α = 1.0; episodeLength = 10; globalStep = 11
"""
function movePredator(; a::Predator, w₀::Float64, w₁::Float64, u::Float64, habitatSize::Int, 
    ΔPredatorsFocal::AbstractVector{Float64}, ηPredatorsFocal::AbstractVector{Float64}, ΔPreyFocal::AbstractVector{Float64}, ηPreyFocal::AbstractVector{Float64}, 
    land::Array{Float64}, carryingCapacity::Float64, sensoryRange, predatorSensingFactor, pid, predHeadings, predEnergys, 
    preyVec, episodeLength, cost = 0.008, β = 0.005, σ₀ = 0.24, globalStep = 0)
    ## Some check that we're still physical
    if a.x ≥ 0
        ## Compute zone sizes
        predatorZone = predatorSensingFactor * sensoryRange
        ## Process sensory information - predators (with 2nd nearest neighbor)
        ## Manually find 1st and 2nd nearest predators to avoid tmp array allocations
        minDist1Pred = Inf ## Stores the distance of the 1st nearest predator
        focus1Pred = 0 ## Index of the 1st nearest predator
        minDist2Pred = Inf ## Stores the distance of the 2nd nearest predator
        focus2Pred = 0 ## Index of the 2nd nearest predator
        for pIdx in eachindex(ΔPredatorsFocal)
            currentDist = ΔPredatorsFocal[pIdx]
            if 0 < currentDist ≤ predatorZone ## Check if predator is within sensing range and not 0 (meaning outside range)
                if currentDist < minDist1Pred
                    minDist2Pred = minDist1Pred ## Shift current 1st to 2nd
                    focus2Pred = focus1Pred
                    minDist1Pred = currentDist ## New 1st nearest
                    focus1Pred = pIdx
                elseif currentDist < minDist2Pred ## Not the 1st, but could be the 2nd nearest
                    minDist2Pred = currentDist
                    focus2Pred = pIdx
                end
            end
        end
        ## Assign values based on whether neighbors were found
        if focus1Pred != 0 ## Some neighbors present
            nnPredator = minDist1Pred / predatorZone
            nnpredatorAngle = ηPredatorsFocal[focus1Pred] / π
            nnPredatorHeading = wrapAngle(a.h - predHeadings[focus1Pred]) / π
            predatorOn = 1.0f0
            if focus2Pred != 0 ## 2nd nearest neighbor present
                nnPredator2 = minDist2Pred / predatorZone
                nnpredatorAngle2 = ηPredatorsFocal[focus2Pred] / π
                nnPredatorHeading2 = wrapAngle(a.h - predHeadings[focus2Pred]) / π
                predatorOn2 = 1.0f0
            else ## Only 1st nearest neighbor present, or no 2nd within range
                nnPredator2, nnpredatorAngle2, nnPredatorHeading2, predatorOn2 = 0.0f0, 0.0f0, 0.0f0, 0.0f0
            end
        else ## No neighbors present
            nnPredator, nnpredatorAngle, nnPredatorHeading, predatorOn = 0.0f0, 0.0f0, 0.0f0, 0.0f0
            nnPredator2, nnpredatorAngle2, nnPredatorHeading2, predatorOn2 = 0.0f0, 0.0f0, 0.0f0, 0.0f0
        end
        ## Process sensory data - environment
        xPunch = punchOut(a.x, habitatSize)
        yPunch = punchOut(a.y, habitatSize)
        producerNeighborhood = land[xPunch, yPunch]
        grassHere, grassAhead, grassTotal = getCurrentAndTargetTile(producerNeighborhood, a.h)
        grassHere, grassAhead, grassTotal = grassHere / carryingCapacity, grassAhead / carryingCapacity, grassTotal / (carryingCapacity * 9)
        ## Process sensory data - prey with 2 nearest neighbors
        if a.s == 1 ## If predator is in searching/targeting state
            preyHeadings = (p->p.h).(preyVec)
            ## Manually find 1st and 2nd nearest prey to avoid tmp array allocations
            minDist1Prey = Inf ## Stores the distance of the 1st nearest prey
            focus1Prey = 0 ## Index of the 1st nearest prey
            minDist2Prey = Inf ## Stores the distance of the 2nd nearest prey
            focus2Prey = 0 ## Index of the 2nd nearest prey
            for pIdx in eachindex(ΔPreyFocal)
                currentDist = ΔPreyFocal[pIdx]
                if 0 < currentDist ≤ sensoryRange ## Check if prey is within sensing range and not 0
                    if currentDist < minDist1Prey
                        minDist2Prey = minDist1Prey ## Shift current 1st to 2nd
                        focus2Prey = focus1Prey
                        minDist1Prey = currentDist ## New 1st nearest
                        focus1Prey = pIdx
                    elseif currentDist < minDist2Prey ## Not the 1st, but could be the 2nd nearest
                        minDist2Prey = currentDist
                        focus2Prey = pIdx
                    end
                end
            end
            ## Assign values based on whether prey neighbors were found
            if focus1Prey != 0
                nnPrey = minDist1Prey / sensoryRange
                nnpreyAngle = ηPreyFocal[focus1Prey] / π
                nnPreyHeading = wrapAngle(a.h - preyHeadings[focus1Prey]) / π
                preyOn = 1.0f0
                if focus2Prey != 0 ## 2nd nearest neighbor present
                    nnPrey2 = minDist2Prey / sensoryRange
                    nnpreyAngle2 = ηPreyFocal[focus2Prey] / π
                    nnPreyHeading2 = wrapAngle(a.h - preyHeadings[focus2Prey]) / π
                    preyOn2 = 1.0f0
                else ## Only 1st nearest neighbor present, or no 2nd within range
                    nnPrey2, nnpreyAngle2, nnPreyHeading2, preyOn2 = 0.0f0, 0.0f0, 0.0f0, 0.0f0
                end
            else ## No prey neighbors present
                nnPrey, nnpreyAngle, nnPreyHeading, preyOn = 0.0f0, 0.0f0, 0.0f0, 0.0f0
                nnPrey2, nnpreyAngle2, nnPreyHeading2, preyOn2 = 0.0f0, 0.0f0, 0.0f0, 0.0f0
            end
        else ## Predator is not in a searching/targeting state
            nnPrey, nnpreyAngle, nnPreyHeading, preyOn = 0.0f0, 0.0f0, 0.0f0, 0.0f0
            nnPrey2, nnpreyAngle2, nnPreyHeading2, preyOn2 = 0.0f0, 0.0f0, 0.0f0, 0.0f0
        end 
        ## Create expanded state vector
        networkInput = [
            Float32[nnPrey, nnpreyAngle, nnPreyHeading],                     ## 13 - 16, nearest prey
            Float32[nnPrey2, nnpreyAngle2, nnPreyHeading2],                  ## 2nd nearest prey
            Float32[nnPredator, nnpredatorAngle, nnPredatorHeading],         ## nearest predator
            Float32[nnPredator2, nnpredatorAngle2, nnPredatorHeading2],      ## 2nd nearest predator
            Float32[grassHere, grassAhead, grassTotal]                       ## 10 - 12, grass
        ]
        ## Get action parameters and value estimate from network
        outputActivations = a.network(networkInput)
        ## Process network output
        turnMean = outputActivations[1]
        turnSD = σ₀ * exp(outputActivations[2])
        accelMean = outputActivations[3]
        accelSD = σ₀ * exp(outputActivations[4])
        ## Reparameterization trick
        ϵ₁ = randn()
        ϵ₂ = randn()
        turnSampleRaw = turnMean + turnSD * ϵ₁
        accelSampleRaw = accelMean + accelSD * ϵ₂
        turnSample = clamp(turnSampleRaw, -1, 1)
        accelSample = clamp(accelSampleRaw, -1, 1)
        ## Scale actions to environment ranges
        turn = u * turnSample
        accel = w₀ * accelSample
        ## Apply actions to update position
        ϕ = wrapAngle(a.h + turn)    ## New heading
        stepSize = w₀ + accel               ## Step size (speed)
        ## Take the step
        xStep = a.x + (stepSize * cos(ϕ))   ## New x position
        yStep = a.y + (stepSize * sin(ϕ))   ## New y position
        ## Wrap around habitat boundaries
        xStep, yStep = torusReflect(xStep, yStep, habitatSize)
        ## Reduce movement cost to not punish exploration too much
        myTurnCost = clamp(cost * abs(turnSample), 0, cost)
        myMoveCost = clamp(cost * stepSize / (2 * w₀), cost / 5, cost)
        myCost = myMoveCost + myTurnCost
        ## Apply rewards
        approach = ifelse(preyOn == 1, 1 - nnPrey, 0)
        approachBonus = β * max(0, (approach - a.prevReward))
        newE = a.e - myCost + approachBonus
        reward = newE
        ## Update fields
        newPrevReward = approach
        newP = a.p
        newU = a.u + 1
        ## Drop a line
        if globalStep % 500 == 1 && pid == 1
            println(Crayon(foreground = (208, 141, 64)), "○- Predator 1 :  o′ = ", round.([turnMean, turnSD, accelMean, accelSD], digits = 3), " -○")
        end
        ## Create and return updated predator
        return Predator(xStep, yStep, ϕ, newE, a.s, a.tx, a.ty, a.age, newPrevReward, newU, newP, a.network)
    end
end

"""
Move all of em
"""
function moveAllPredators!(predators::Vector{Predator}, w₀::Float64, w₁::Float64, u::Float64, habitatSize::Int, 
    ΔPredators::Array{Float64}, ηPredators::Array{Float64},
    land::Matrix{Float64}, carryingCapacity::Float64, sensoryRange::Float64, predatorSensingFactor::Float64, lr::Float64,
    predHeadings, predEnergys, preyVec, productivity, cohereStrength, preyCarryingCapacity,
    ΔPrey::Array{Float64}, ηPrey::Array{Float64}, episodeLength;
    globalStep::Int = 0)
    ## Predator loop
    for i in eachindex(predators) ## Use eachindex to avoid collect(enumerate) allocation
        predator = predators[i] ## Access the predator object by index
        predators[i] = movePredator(
            a = predator, 
            w₀ = w₀,
            w₁ = w₁,
            u = u, 
            habitatSize = habitatSize,
            ΔPredatorsFocal = view(ΔPredators, :, i), ## Pass a view instead of a copy
            ηPredatorsFocal = view(ηPredators, :, i), ## Pass a view instead of a copy
            ΔPreyFocal = view(ΔPrey, :, i), ## Pass a view instead of a copy
            ηPreyFocal = view(ηPrey, :, i), ## Pass a view instead of a copy
            land = land,
            carryingCapacity = carryingCapacity,
            sensoryRange = sensoryRange,
            predatorSensingFactor = predatorSensingFactor,
            pid = i,
            predHeadings = predHeadings,
            predEnergys = predEnergys,
            preyVec = preyVec,
            episodeLength = episodeLength,
            globalStep = globalStep
        )
    end
end

"""
Calculate distance and angles for predator agents within sensory radius.
"""
function neighborDistances(predators::Vector{Predator}, prey::Vector{Prey}, habitatSize::Int, sensoryRange::Float64, predatorSensingFactor::Float64)
    nPredators = length(predators)
    nPrey      = length(prey)
    ΔPredators = zeros(nPredators, nPredators)
    ηPredators = zeros(nPredators, nPredators)
    ΔPrey      = zeros(nPrey, nPredators)
    ηPrey      = zeros(nPrey, nPredators)
    ## Create a lock for thread-safe writing
    arrayLock = ReentrantLock()
    coordPredators  = transpose([(p->p.x).(predators) (p->p.y).(predators)])
    coordPrey       = transpose([(p->p.x).(prey) (p->p.y).(prey)])
    heads           = (p->p.h).(predators)
    predator_sensing_range_sq = (predatorSensingFactor * sensoryRange)^2
    prey_sensing_range_sq     = sensoryRange^2
    Threads.@threads for i ∈ 1:nPredators
        point_x = coordPredators[1, i]
        point_y = coordPredators[2, i]
        head_i  = heads[i]
        idxsPredators_found = Int[]
        for j_candidate ∈ 1:nPredators
            if i == j_candidate
                continue
            end
            neighbor_x_j = coordPredators[1, j_candidate]
            neighbor_y_j = coordPredators[2, j_candidate]
            dx = abs(point_x - neighbor_x_j)
            dy = abs(point_y - neighbor_y_j)
            dx = min(dx, habitatSize - dx)
            dy = min(dy, habitatSize - dy)
            dist_sq = dx^2 + dy^2
            if dist_sq <= predator_sensing_range_sq
                push!(idxsPredators_found, j_candidate)
            end
        end
        idxsPredators = idxsPredators_found
        idxsPrey_found = Int[]
        for k_candidate ∈ 1:nPrey
            prey_x_k = coordPrey[1, k_candidate]
            prey_y_k = coordPrey[2, k_candidate]
            dx = abs(point_x - prey_x_k)
            dy = abs(point_y - prey_y_k)
            dx = min(dx, habitatSize - dx)
            dy = min(dy, habitatSize - dy)
            dist_sq = dx^2 + dy^2
            if dist_sq <= prey_sensing_range_sq
                push!(idxsPrey_found, k_candidate)
            end
        end
        idxsPrey = idxsPrey_found
        lock(arrayLock) do
            if !isempty(idxsPrey)
                for k ∈ idxsPrey
                    prey_x_k = coordPrey[1, k]
                    prey_y_k = coordPrey[2, k]
                    dx = abs(point_x - prey_x_k)
                    dy = abs(point_y - prey_y_k)
                    dx = min(dx, habitatSize - dx)
                    dy = min(dy, habitatSize - dy)
                    dist = sqrt(dx^2 + dy^2)
                    ΔPrey[k, i] = dist
                end
                ηPrey[idxsPrey, i] = torusAngles(point_x, point_y, head_i, coordPrey[1, idxsPrey], coordPrey[2, idxsPrey], habitatSize)
            end
            if !isempty(idxsPredators)
                ηPredators[idxsPredators, i] = torusAngles(point_x, point_y, head_i, coordPredators[1, idxsPredators], coordPredators[2, idxsPredators], habitatSize)
            end
            for j ∈ idxsPredators
                if j > i
                    neighbor_x_j = coordPredators[1, j]
                    neighbor_y_j = coordPredators[2, j]
                    dx = abs(point_x - neighbor_x_j)
                    dy = abs(point_y - neighbor_y_j)
                    dx = min(dx, habitatSize - dx)
                    dy = min(dy, habitatSize - dy)
                    dist = sqrt(dx^2 + dy^2)
                    ΔPredators[i, j] = dist
                    ΔPredators[j, i] = dist
                end
            end
        end
    end
    return ΔPredators, ΔPrey, ηPredators, ηPrey
end

"""
Calculate distance and angles for predator agents within sensory radius.
"""
function neighborDistancesSerial(predators::Vector{Predator}, prey::Vector{Prey}, habitatSize::Int, sensoryRange::Float64, predatorSensingFactor::Float64)
    ## Generate empty arrays to fill in
    nPredators = length(predators)
    nPrey      = length(prey)
    ΔPredators = zeros(nPredators, nPredators)
    ηPredators = zeros(nPredators, nPredators)
    ΔPrey      = zeros(nPrey, nPredators)
    ηPrey      = zeros(nPrey, nPredators)
    ## Extract agent coordinates from agent objects
    coordPredators  = transpose([(p->p.x).(predators) (p->p.y).(predators)])
    coordPrey       = transpose([(p->p.x).(prey) (p->p.y).(prey)])
    heads           = (p->p.h).(predators)
    ## Pre-calculate squared ranges for efficiency
    predator_sensing_range_sq = (predatorSensingFactor * sensoryRange)^2
    prey_sensing_range_sq     = sensoryRange^2
    ## Loop through predators and compute neighbor features (Serial version - no Threads.@threads)
    for i ∈ 1:nPredators
        ## Focal agent's position
        point_x = coordPredators[1, i]
        point_y = coordPredators[2, i]
        head_i  = heads[i]
        ## Manually find idxsPredators (replacing BallTree's inrange)
        idxsPredators_found = Int[]
        for j_candidate ∈ 1:nPredators
            if i == j_candidate  # Skip self
                continue
            end
            neighbor_x_j = coordPredators[1, j_candidate]
            neighbor_y_j = coordPredators[2, j_candidate]
            # Calculate periodic Euclidean distance squared
            dx = abs(point_x - neighbor_x_j)
            dy = abs(point_y - neighbor_y_j)
            dx = min(dx, habitatSize - dx) # Periodic wrap
            dy = min(dy, habitatSize - dy) # Periodic wrap
            dist_sq = dx^2 + dy^2
            if dist_sq <= predator_sensing_range_sq
                push!(idxsPredators_found, j_candidate)
            end
        end
        idxsPredators = idxsPredators_found # Assign to the variable name used in original function
        ## Manually find idxsPrey (replacing BallTree's inrange)
        idxsPrey_found = Int[]
        for k_candidate ∈ 1:nPrey
            prey_x_k = coordPrey[1, k_candidate]
            prey_y_k = coordPrey[2, k_candidate]
            # Calculate periodic Euclidean distance squared
            dx = abs(point_x - prey_x_k)
            dy = abs(point_y - prey_y_k)
            dx = min(dx, habitatSize - dx)
            dy = min(dy, habitatSize - dy)
            dist_sq = dx^2 + dy^2
            if dist_sq <= prey_sensing_range_sq
                push!(idxsPrey_found, k_candidate)
            end
        end
        idxsPrey = idxsPrey_found # Assign to the variable name used in original function
        ## Distances to prey (replaces colwise, using found idxsPrey)
        if !isempty(idxsPrey)
            # Iterate through identified prey neighbors and calculate their distance
            for k ∈ idxsPrey
                prey_x_k = coordPrey[1, k]
                prey_y_k = coordPrey[2, k]
                dx = abs(point_x - prey_x_k)
                dy = abs(point_y - prey_y_k)
                dx = min(dx, habitatSize - dx)
                dy = min(dy, habitatSize - dy)
                dist = sqrt(dx^2 + dy^2) # Take sqrt here as we need the distance itself
                ΔPrey[k, i] = dist
            end
        end
        ## Angles to other predators (uses your original vectorized torusAngles)
        if !isempty(idxsPredators)
            ηPredators[idxsPredators, i] = torusAngles(point_x, point_y, head_i, coordPredators[1, idxsPredators], coordPredators[2, idxsPredators], habitatSize)
        end
        ## Angles to prey (uses your original vectorized torusAngles)
        if !isempty(idxsPrey)
            ηPrey[idxsPrey, i] = torusAngles(point_x, point_y, head_i, coordPrey[1, idxsPrey], coordPrey[2, idxsPrey], habitatSize)
        end
        ## Only populate the upper triangle for predator distances (replaces evaluate)
        for j ∈ idxsPredators
            if j > i # This condition from your original code is preserved
                neighbor_x_j = coordPredators[1, j]
                neighbor_y_j = coordPredators[2, j]
                dx = abs(point_x - neighbor_x_j)
                dy = abs(point_y - neighbor_y_j)
                dx = min(dx, habitatSize - dx)
                dy = min(dy, habitatSize - dy)
                dist = sqrt(dx^2 + dy^2) # Take sqrt here as we need the distance itself
                ΔPredators[i, j] = dist
                ΔPredators[j, i] = dist # Symmetric fill as in original code
            end
        end
    end
    ## Dump it all
    return ΔPredators, ΔPrey, ηPredators, ηPrey
end

"""
Get current tile and target tile grass values for simplified sensor input
"""
function getCurrentAndTargetTile(producerNeighborhood, heading)
    ## Get the grass in the current tile (center of 3x3 grid)
    grassHere = producerNeighborhood[2, 2]
    ## Calculate direction to check based on heading
    dx = round(Int, cos(heading))
    dy = round(Int, sin(heading))
    ## Ensure we stay within bounds (3x3 grid)
    cx = 2 + clamp(dx, -1, 1)
    cy = 2 + clamp(dy, -1, 1)
    ## Get grass in the direction we're facing
    grassAhead = producerNeighborhood[cx, cy]
    ## Total grass in the neighborhood
    grassTotal = sum(producerNeighborhood)
    ## Output
    return grassHere, grassAhead, grassTotal
end

"""
Perform rollouts for ES learning
"""
function esRollouts(ghost, episodeLength, currentPredator, 
                    habitatSize, predatorSensoryRange, predatorSensingFactor, productivity, carryingCapacity,
                    w, u, cohereStrength, turnMaxima, learnRate, preyCarryingCapacity, fakeTime = 0,
                    nPerturbations = 40, noise_σ = 1.65, elite = 4)
    ## Parameter length
    parm, _ = Flux.destructure(ghost.predators[1].network)
    nParm   = length(parm)
    ## Empty vector to populate with parameter gradient
    ogPerturbation    = generateOrthogonalGaussianBlock(nParm, nPerturbations)
    perturbLog_plus   = noise_σ .* ogPerturbation
    ## The negative set
    perturbLog_minus  = -perturbLog_plus
    rewardLog         = Vector{Float64}(undef, 2 * nPerturbations)
    perturbLog        = [x for t ∈ zip(perturbLog_plus, perturbLog_minus) for x ∈ t]
    ## Loop through n perturbations
    Threads.@threads for p ∈ 1:lastindex(perturbLog)
        ## We're copying things so we don't mutate our frozen state
        tempGhost     = deepcopy(ghost)
        ghostPredator = tempGhost.predators[currentPredator]
        preNetwork    = ghostPredator.network
        ## Perturb the network parameters with noise
        paramVec, reconstructor = Flux.destructure(preNetwork)
        perturbation            = perturbLog[p]
        perturbedNetwork        = reconstructor(paramVec .+ perturbation)
        ghostPredator.network   = perturbedNetwork
        ## Inject ghost into the frozen state clone
        tempGhost.predators[currentPredator] = ghostPredator
        ## Prep for rollout
        tempLand      = tempGhost.land
        tempPrey      = tempGhost.prey
        tempPredators = tempGhost.predators
        ## Initial energy
        initE = ghostPredator.e
        ## Rollout episode
        for _ ∈ 1:episodeLength
            # ΔPreyPrey, ηPreyPrey = preyDistances(tempPrey, habitatSize)
            # ΔPredators, ΔPrey, ηPredators, ηPrey = neighborDistances(tempPredators, tempPrey, habitatSize, predatorSensoryRange, predatorSensingFactor)
            ΔPreyPrey, ηPreyPrey = preyDistancesSerial(tempPrey, habitatSize)
            ΔPredators, ΔPrey, ηPredators, ηPrey = neighborDistancesSerial(tempPredators, tempPrey, habitatSize, predatorSensoryRange, predatorSensingFactor)
            tempLand = growProducer(tempLand, habitatSize, productivity, carryingCapacity)
            moveAllPrey!(tempPrey, w, u, habitatSize, ΔPreyPrey, ηPreyPrey, cohereStrength)
            updateAllPrey!(tempLand, tempPrey)
            moveAllPredators!(tempPredators, w, w, turnMaxima, habitatSize, ΔPredators, ηPredators, tempLand, carryingCapacity,
                predatorSensoryRange, predatorSensingFactor, learnRate, (p->p.h).(tempPredators), (p->p.e).(tempPredators),
                tempPrey, productivity, cohereStrength, preyCarryingCapacity, ΔPrey, ηPrey, episodeLength, globalStep = fakeTime)
            updateAllPredators!(ΔPrey, tempPredators, tempPrey)
            reproducePrey!(agents = tempPrey, habitatSize = habitatSize, carryingCapacity = preyCarryingCapacity)
            killPrey!(agents = tempPrey)
        end
        ## Log reward for our focal predator
        rewardLog[p]  = tempPredators[currentPredator].e - initE
    end
    ## openAI recommends reward normalization
    rewardStd   = std(rewardLog)
    n_rewardLog = rewardLog ./ rewardStd
    ## openAI recommends a rank transformation of the fitness difference values
    rPlus    = n_rewardLog[1:2:end]
    rMinus   = n_rewardLog[2:2:end]
    rDiff    = rPlus .- rMinus
    rF       = sortperm(rDiff, rev = true)
    elites   = findall(rF .≤ elite)
    # eliteStd = std(rDiff[elites])
    ## Compute updated parameters
    θ_t, reconstructor = Flux.destructure(ghost.predators[currentPredator].network)
    α            = learnRate / (rewardStd)
    update       = α .* sum(rDiff[elites] .* reduce(hcat, ogPerturbation[elites])', dims = 1)[1, :]
    θ_t_plus_1   = θ_t .+ update
    newNetwork   = reconstructor(θ_t_plus_1)
    ## Drop a line
    if ghost.predators[currentPredator].p % 64 == 0
        println("○- Predator ", currentPredator, " : |Ν| = ", round(norm(θ_t), digits = 2), " -○")
        println("              : |▿| = ", round(norm(update), digits = 3))
    end
    ## Return our new network
    return newNetwork
end

"""
updateAllNetworks! : Apply ES update to all predators
"""
function updateAllNetworks!(predators, ghost, episodeLength, habitatSize, predatorSensoryRange, predatorSensingFactor, productivity, 
                            carryingCapacity, w, u, cohereStrength, turnMaxima, learnRate, preyCarryingCapacity, 
                            ρ = 0.0045, f = 4.5,
                            minTimeSince = 1,
                            restart = -100.0)
    ## Predator loop
    for (i, predator) ∈ collect(enumerate(predators))
        ## Updates happen probabilistically and asynchronously
        willUpdate = predator.u > minTimeSince && (rand(Bernoulli(ρ)) || (predator.s == 1 && rand(Bernoulli(f * ρ))))
        ## Compute gradient
        if willUpdate
            newNetwork = esRollouts(ghost, episodeLength, i, 
                                    habitatSize, predatorSensoryRange, predatorSensingFactor, productivity, carryingCapacity,
                                    w, u, cohereStrength, turnMaxima, learnRate, preyCarryingCapacity)
            predators[i].network = newNetwork
            predators[i].p += 1
            predators[i].u  = 0
        elseif predator.e < restart
            predators[i].e = 0
            predators[i].network = init_predator_network()
            predators[i].p += 1
            predators[i].u  = 0
            ## Drop a line
            println("○- Agent ", i, " was bad, so it started over -○")
        end
    end
end

"""
Load a trained predator network from a JLD2 file
"""
function load_predator_network(filepath)
    model_state = JLD2.load(filepath, "modelState")
    architecture = init_predator_network()
    Flux.loadmodel!(architecture, model_state)
    return architecture
end



## ---
## That's all
## ---

;