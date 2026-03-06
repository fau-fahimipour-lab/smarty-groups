## ---
## Producer functions
## ---
"""
initProducer : Initialize the prey
"""
function initProducer(; habitatSize::Int, coverage, K)
    landscape = zeros(habitatSize, habitatSize)
    for i ∈ 1:habitatSize
        for j in 1:habitatSize
            if rand(Bernoulli(coverage))
                landscape[i, j] = K
            end
        end
    end
    return landscape
end

"""
growProducer : Update producer biomass one time step
"""
function growProducer(landscape::Array{Float64}, habitatSize::Int, regrowth::Float64, carryingCapacity::Float64)
    ## Blank Matrix
    newLandscape = zeros(habitatSize, habitatSize)
    ## Note: This function is not internally threaded, so using rand() directly is fine.
    ## If growProducer is called concurrently from multiple threads, Julia's global RNG is thread-safe.
    for i ∈ 1:habitatSize
        iUp = ifelse(i == 1, habitatSize, i - 1)
        iDown = ifelse(i == habitatSize, 1, i + 1)
        ## Move across columns
        for j ∈ 1:habitatSize
            jLeft = ifelse(j == 1, habitatSize, j - 1)
            jRight = ifelse(j == habitatSize, 1, j + 1)
            ## Directly sum the 3x3 neighborhood to avoid temporary array allocation
            sumSubMatrix = landscape[iUp, jLeft] + landscape[iUp, j] + landscape[iUp, jRight] +
                           landscape[i, jLeft]   + landscape[i, j]   + landscape[i, jRight] +
                           landscape[iDown, jLeft] + landscape[iDown, j] + landscape[iDown, jRight]
            randVal = rand(Bernoulli(0.01)) ## Use global RNG
            ## Use min() directly to avoid temporary array allocation
            newLandscape[i, j] = min(carryingCapacity, landscape[i, j] + regrowth * sumSubMatrix + (randVal * 0.05 * carryingCapacity))
        end
    end
    return newLandscape
end

"""
punchOut : punch out a patch neighborhood given coordinates of an agent
"""
function punchOut(x::Float64, habitatSize::Int)
    xTile = Int(ceil(x))
    xUp   = Int(ifelse(xTile == 1, habitatSize, xTile - 1))
    xDown = Int(ifelse(xTile == habitatSize, 1, xTile + 1))
    return [xUp, xTile, xDown]
end

"""
biomassCentroid : given a neighborhood, find the centroid of producer biomass
"""
function angleToProducerCentroid(punch::Array{Float64}, ax::Float64, ay::Float64, ah::Float64)
    ## If there are producers in the neighborhood
    ζ = sum(punch)
    ## Compute producer biomass centroid
    xCentroid = sum([punch[1, :] (2 .* punch[2, :]) (3 .* punch[3, :])]) / ζ
    yCentroid = sum([punch[:, 1] (2 .* punch[:, 2]) (3 .* punch[:, 3])]) / ζ
    ## Compute relative angle to agent
    hx = ax + cos(ah)                  
    hy = ay + sin(ah)
    headerDiff = atan(hy - ay, hx - ax)
    preΘ       = atan(yCentroid - ay, xCentroid - ax) - headerDiff
    Θ          = atan(sin(preΘ), cos(preΘ))
    return Θ
end

;