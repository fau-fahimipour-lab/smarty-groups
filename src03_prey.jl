"""
Prey : Define our Prey agent composite data object
"""
struct Prey
    x::Float64  ## x location
    y::Float64  ## y location
    h::Float64  ## heading
    e::Float64  ## energy level
    t::Int      ## being targeted?
    tx::Float64 ## targeting predator x
    ty::Float64 ## targeting predator y
end

"""
Some parameters are set here!
----
"""
preySensoryRange     = 1.5
preyReproductiveRate = 0.008

## ---
## Prey functions
## ---
"""
initPrey : Initialize the prey
"""
function initPrey(; n::Int, habitatSize::Int, initialE = 0.5)
    preys = [Prey(
        Float64.(rand(Uniform(0.5, habitatSize - 0.5))),  ## x location
        Float64.(rand(Uniform(0.5, habitatSize - 0.5))),  ## y location
        Float64.(rand(Uniform(-π, π))),                   ## heading
        initialE,                                         ## energy
        0, 0.0, 0.0                                       ## not being targeted
        ) for i ∈ 1:n]                                    ## for each agent...    
    return preys
end

"""
killPrey : Remove if you lose too much energy
"""
function killPrey!(; agents::Vector{Prey}, deathEnergy = 1e-4)
    dying = findallF(y->y.e<deathEnergy, agents)
    deleteat!(agents, dying)
end

"""
movePrey : Move prey one step
"""
function movePrey(; a::Prey, w::Float64, u::Float64, habitatSize::Int, 
    ΔFocal::AbstractVector{Float64}, ## Changed to AbstractVector for flexibility
    ηFocal::AbstractVector{Float64}, ## Changed to AbstractVector for flexibility
    xFocal::AbstractVector{Float64}, ## Also changed to AbstractVector
    yFocal::AbstractVector{Float64}, ## Also changed to AbstractVector
    hFocal::AbstractVector{Float64}, ## Also changed to AbstractVector
    cost = 0.056,
    repulsionZone = 0.10,
    alignStrength = 3.63,
    cohereStrength::Float64, ## Save this flocking parameter set: [0.15, 0.20, 0.055, 0.045]
    repulseStrength = 0.81,
    evadeStrength = 0.28)
    ## If you still have energy
    if a.e > 0.0
        ## Current heading
        ϕ = a.h
        ## If you have neighbors and are not being targeted
        if a.t == 0
            ## Random nudge
            ϕ += rand(Normal(0, u))
            ## Count your neighbors without allocating a new array for findall
            ## We will iterate directly through ΔFocal
            nNeighbors = 0
            for val in ΔFocal
                if val > 0
                    nNeighbors += 1
                end
            end
            ## If you have neighbors
            if nNeighbors > 0
                ## Calculate alignment and cohesion without temporary arrays
                sumX, sumY, sumH = 0.0, 0.0, 0.0
                for i in eachindex(ΔFocal)
                    if ΔFocal[i] > 0 ## If it's a valid neighbor distance (not 0 or Inf)
                        sumX += xFocal[i]
                        sumY += yFocal[i]
                        sumH += hFocal[i]
                    end
                end
                xAvg = sumX / nNeighbors
                yAvg = sumY / nNeighbors
                meanH = sumH / nNeighbors ## Note: This is an arithmetic mean, not an angular mean
                ϕ += cohereStrength * alignStrength * meanH
                ## Calculate cohesion
                headingDiff = atan(yAvg - a.y, xAvg - a.x) - a.h
                angle = atan(sin(headingDiff), cos(headingDiff))
                ϕ += cohereStrength * angle
                ## Avoid collision without temporary arrays
                repulsionSumEta = 0.0
                hasRepulsionNeighbor = false
                for i in eachindex(ΔFocal)
                    if ΔFocal[i] > 0 && ΔFocal[i] < repulsionZone ## If it's a neighbor and within repulsion zone
                        repulsionSumEta += ηFocal[i]
                        hasRepulsionNeighbor = true
                    end
                end
                if hasRepulsionNeighbor
                    ϕ -= cohereStrength * repulseStrength * repulsionSumEta
                end
            end
        end
        ## Flee if predator is in strike range
        if a.t == 1
            angleAttack = torusAngles(a.x, a.y, a.h, [a.tx], [a.ty], habitatSize)[1]
            fleeAngle = angleAttack
            ϕ -= evadeStrength * fleeAngle
        end
        ## Keep the angle in bounds
        ϕ = wrapAngle(ϕ)
        ## Take a step
        dx = w * cos(ϕ) ## * ifelse(a.t == 1, 1, 1e-1)
        dy = w * sin(ϕ) ## * ifelse(a.t == 1, 1, 1e-1)
        xStep = a.x + dx
        yStep = a.y + dy
        ## Keep agents inside the habitat
        xStep, yStep = torusReflect(xStep, yStep, habitatSize)
        ## Update prey object
        return Prey(xStep, yStep, atan(dy, dx), max(0.0, a.e - cost), a.t, a.tx, a.ty)
    ## If out of energy, sit still and wait for death
    else 
        return Prey(a.x, a.y, a.h, a.e, a.t, a.tx, a.ty)
    end
end

"""
moveAllPrey! : Apply movePrey to all Prey
"""
function moveAllPrey!(preys::Vector{Prey}, w::Float64, u::Float64, habitatSize::Int, ΔPrey::Array{Float64}, ηPrey::Array{Float64}, cohereStrength::Float64)
    ## Extract coordinates once for all prey (these arrays are currently necessary for movePrey's API)
    xPrey = (p->p.x).(preys)
    yPrey = (p->p.y).(preys)
    hPrey = (p->p.h).(preys)
    for i in eachindex(preys) ## Use eachindex to avoid collect(enumerate) allocation
        agent = preys[i] ## Access the agent by index
        ## Pass views to movePrey to avoid column slicing allocations
        preys[i] = movePrey(a = agent, w = w, u = u, habitatSize = habitatSize, 
                                ΔFocal = view(ΔPrey, :, i), ηFocal = view(ηPrey, :, i),
                                xFocal = view(xPrey, :), yFocal = view(yPrey, :), hFocal = view(hPrey, :), ## These views are mostly syntactic for 1D arrays
                                cohereStrength = cohereStrength)
    end
end

"""
updateAllPrey! : Apply movePrey to all Prey
"""
function updateAllPrey!(landscape::Array{Float64}, preys::Vector{Prey}, minEat = 0.05, maxEat = 0.28, maxE = 1.0)
    ## Avoid collect(enumerate) allocation
    for i ∈ eachindex(preys)
        prey = preys[i] ## Access prey by index
        ## Can toggle here: only eat if you're not chased, prey.t == 0
        if true
            xTile = Int(ceil(prey.x))
            yTile = Int(ceil(prey.y))
            hunger = 1 - (prey.e / maxE)
            ## Use direct min/max calls to avoid temporary array allocations
            consume = min(landscape[xTile, yTile], max(minEat, hunger * maxEat))
            preys[i] = Prey(prey.x, prey.y, prey.h, min(1.0, prey.e + consume), prey.t, prey.tx, prey.ty)
            landscape[xTile, yTile] = landscape[xTile, yTile] - consume
        end
    end
end

"""
reproducePrey : Have one baby if you get enough energy
"""
function reproducePrey!(; agents::Vector{Prey}, habitatSize::Int, carryingCapacity::Int, ρ = preyReproductiveRate, reproductionEnergy = 0.5, reproductionCost = 0.30, babyE = 0.25)
    nPrey = length(agents)
    popRatio = nPrey / carryingCapacity
    ## New immigration when prey are very rare
    if nPrey ≤ 3
        arrival = Prey(rand(Uniform(0.5, habitatSize - 0.5)), rand(Uniform(0.5, habitatSize - 0.5)), rand(Uniform(-π, π)), babyE, 0, 0.0, 0.0)
        push!(agents, arrival)
    end
    ## Collect new babies in a temporary vector to avoid modifying `agents` during iteration
    newBabies = Prey[]
    ## Apply births
    for i in eachindex(agents) ## Iterate by index
        a = agents[i] ## Access agent by index
        if a.e > reproductionEnergy ## Check if gravid directly, avoiding findallF and 'in' check
            if rand() < (ρ * max(1.0 - popRatio, 0))
                ## Reduce energy of parent
                agents[i] = Prey(a.x, a.y, a.h, a.e - reproductionCost, a.t, a.tx, a.ty)
                ## Create new baby
                babyX = a.x
                babyY = a.y
                newAgent = Prey(babyX, babyY, rand(Uniform(-π, π)), babyE, 0, a.tx, a.ty) ## Removed redundant Float64.()
                push!(newBabies, newAgent) ## Add baby to the temporary list
            end
        end
    end
    ## Add all new babies to the main agents vector after the loop
    append!(agents, newBabies)
end

"""
Calculate distance and angles for agents within sensory radius.
"""
function preyDistances(prey::Vector{Prey}, habitatSize::Int, sensoryRange = preySensoryRange)
    ## Generate empty arrays to fill in
    nPrey = length(prey)
    ΔPrey = zeros(nPrey, nPrey)
    ηPrey = zeros(nPrey, nPrey)
    ## Create a lock for thread-safe writing to the arrays
    arrayLock = ReentrantLock()
    ## Extract agent coordinates from agent objects
    coordPrey = transpose([(p->p.x).(prey) (p->p.y).(prey)])
    heads = (p->p.h).(prey)
    ## Pre-calculate squared sensory range for efficiency
    sensoryRangeSq = sensoryRange^2
    Threads.@threads for i ∈ 1:nPrey
        pointX = coordPrey[1, i]
        pointY = coordPrey[2, i]
        headI = heads[i]
        idxsPreyFound = Int[]
        for jCandidate ∈ 1:nPrey
            if i == jCandidate
                continue
            end
            neighborXj = coordPrey[1, jCandidate]
            neighborYj = coordPrey[2, jCandidate]
            dx = abs(pointX - neighborXj)
            dy = abs(pointY - neighborYj)
            dx = min(dx, habitatSize - dx)
            dy = min(dy, habitatSize - dy)
            distSq = dx^2 + dy^2
            if distSq <= sensoryRangeSq
                push!(idxsPreyFound, jCandidate)
            end
        end
        idxsPrey = idxsPreyFound
        if !isempty(idxsPrey)
            # lock this section to prevent race conditions on ηPrey
            lock(arrayLock) do
                ηPrey[idxsPrey, i] = torusAngles(pointX, pointY, headI, coordPrey[1, idxsPrey], coordPrey[2, idxsPrey], habitatSize)
            end
        end
        for j ∈ idxsPrey
            if j > i
                neighborXj = coordPrey[1, j]
                neighborYj = coordPrey[2, j]
                dx = abs(pointX - neighborXj)
                dy = abs(pointY - neighborYj)
                dx = min(dx, habitatSize - dx)
                dy = min(dy, habitatSize - dy)
                dist = sqrt(dx^2 + dy^2)
                # lock this section to prevent race conditions on ΔPrey
                lock(arrayLock) do
                    ΔPrey[i, j] = dist
                    ΔPrey[j, i] = dist
                end
            end
        end
    end
    return ΔPrey, ηPrey
end

"""
Calculate distance and angles for agents within sensory radius.
"""
function preyDistancesSerial(prey::Vector{Prey}, habitatSize::Int, sensoryRange = preySensoryRange)
    ## Generate empty arrays to fill in
    nPrey = length(prey)
    ΔPrey = zeros(nPrey, nPrey)
    ηPrey = zeros(nPrey, nPrey)
    ## Extract agent coordinates from agent objects
    coordPrey = transpose([(p->p.x).(prey) (p->p.y).(prey)])
    heads = (p->p.h).(prey)
    ## Pre-calculate squared sensory range for efficiency
    sensoryRangeSq = sensoryRange^2
    ## Loop through prey and compute neighbor features (Serial version - no Threads.@threads)
    for i ∈ 1:nPrey
        ## Prep to get distances to other agents
        pointX = coordPrey[1, i]
        pointY = coordPrey[2, i]
        headI = heads[i]
        ## Manually find idxsPrey (replacing BallTree's inrange)
        idxsPreyFound = Int[]
        for jCandidate ∈ 1:nPrey
            if i == jCandidate ## Skip self
                continue
            end
            neighborXj = coordPrey[1, jCandidate]
            neighborYj = coordPrey[2, jCandidate]
            ## Calculate periodic Euclidean distance squared
            dx = abs(pointX - neighborXj)
            dy = abs(pointY - neighborYj)
            dx = min(dx, habitatSize - dx) ## Periodic wrap
            dy = min(dy, habitatSize - dy) ## Periodic wrap
            distSq = dx^2 + dy^2
            if distSq <= sensoryRangeSq
                push!(idxsPreyFound, jCandidate)
            end
        end
        idxsPrey = idxsPreyFound ## Assign to the variable name used in original function
        ## Angles to other agents (uses your original vectorized torusAngles)
        if !isempty(idxsPrey)
            ηPrey[idxsPrey, i] = torusAngles(pointX, pointY, headI, coordPrey[1, idxsPrey], coordPrey[2, idxsPrey], habitatSize)
        end
        ## Only populate the upper triangle for distances (replaces evaluate)
        for j ∈ idxsPrey
            if j > i ## This condition from your original code is preserved
                neighborXj = coordPrey[1, j]
                neighborYj = coordPrey[2, j]
                dx = abs(pointX - neighborXj)
                dy = abs(pointY - neighborYj)
                dx = min(dx, habitatSize - dx)
                dy = min(dy, habitatSize - dy)
                dist = sqrt(dx^2 + dy^2) ## Take sqrt here as we need the distance itself
                ΔPrey[i, j] = dist
                ΔPrey[j, i] = dist ## Symmetric fill as in original code
            end
        end
    end
    return ΔPrey, ηPrey
end

;