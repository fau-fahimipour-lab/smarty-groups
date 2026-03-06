## ---
## Load our packages
## ---
using IterTools, CSV, ColorSchemes, Plots, Crayons, DataFrames

## IMPORTANT: Set GR environment variables to force headless mode
## This prevents GR from creating visible windows and dock icons
ENV["GKSwstype"] = "100"

## Load source files
cd(@__DIR__)
include("src01_util.jl")
include("src02_resource.jl")
include("src03_prey.jl")
include("src04_predator.jl")

## ---
## Simulation functions
## ---
"""
Main workhorse that performs one simulation.

Input
----------
All parameters are described above.

Returns
-------
{Float} array with the relevant information.

Debug:
nPrey = 20; nPredators = 5; w = 0.02; u = 0.10; productivity = 0.0004; maxTime = 100; recordTime = 1; habitatSize = 3; cohereStrength = 0.001; carryingCapacity = 10.0; learnRate = 3e-4; burnTime = 100; preyCarryingCapacity  = 32; predatorTurnFactor = 0.95; predatorSensoryRange  = 0.50; predatorSensingFactor = 10.0; initialEnergy = 10.0; supervisedPhase = 1000; hybridPhase = 0; episodeLength = 10
"""
function sim(;
    nPredators::Int,
    w::Float64,
    u::Float64,
    productivity::Float64,
    maxTime::Int,
    habitatSize::Int,
    cohereStrength::Float64,
    carryingCapacity::Float64,
    learnRate::Float64,
    initialEnergy::Float64 = 1.0,   # Starting energy for predators
    simNumber,
    store,
    burnTime              = 400,
    nPrey                 = 50,
    preyCarryingCapacity  = 100,
    predatorTurnFactor    = 0.95,
    predatorSensoryRange  = 0.95, 
    predatorSensingFactor = 8.0,
    recordTimeBuffer      = 1000,
    episodeLength         = 300,
    logInterval           = 500)
    ## -
    ## Initiate the run
    ## -
    t = 0
    turnMaxima = (predatorTurnFactor * u) * sqrt(π / 2)
    land = initProducer(habitatSize = habitatSize, coverage = 1 / 3, K = carryingCapacity)
    preyVec = initPrey(n = nPrey, habitatSize = habitatSize)
    ## Initialize predators with REFLEx network structure
    predatorVec = initPredator(n = nPredators, habitatSize = habitatSize, initialE = initialEnergy)
    ## Compute initial distances
    ΔPreyPrey, ηPreyPrey = preyDistances(preyVec, habitatSize)
    ΔPredators, ΔPrey, ηPredators, ηPrey = neighborDistances(predatorVec, preyVec, habitatSize, predatorSensoryRange, predatorSensingFactor)
    ## Burn-in grass growth
    for _ in 1:burnTime
        land = growProducer(land, habitatSize, 2 * productivity, carryingCapacity)
    end
    ## Statistics tracking
    meanEnergyHistory     = Float64[]
    semEnergyHistory      = Float64[]
    minmaxEnergyHistory   = Tuple{Float64, Float64}[]
    preyCountHistory      = Int[]
    stepHistory           = Int[]
    captureHistory        = Int[]
    catchHistory          = 0
    lastRecordTime        = 0
    anim                  = Animation()
    ## Start simulation loop
    while t < maxTime
        ## Grow producers
        land = growProducer(land, habitatSize, productivity, carryingCapacity)
        ## Move prey
        moveAllPrey!(preyVec, w, u, habitatSize, ΔPreyPrey, ηPreyPrey, cohereStrength)
        ## Update prey status
        updateAllPrey!(land, preyVec)
        ## Move predators with hybrid learning
        moveAllPredators!(predatorVec, w, w, turnMaxima, habitatSize, ΔPredators, ηPredators, land, carryingCapacity,
            predatorSensoryRange, predatorSensingFactor, learnRate, (p->p.h).(predatorVec), (p->p.e).(predatorVec),
            preyVec, productivity, cohereStrength, 
            preyCarryingCapacity, ΔPrey, ηPrey, episodeLength, globalStep = t)
        ## Update predators status
        updateAllPredators!(ΔPrey, predatorVec, preyVec)
        ## Have prey babies
        reproducePrey!(agents = preyVec, habitatSize = habitatSize, carryingCapacity = preyCarryingCapacity)
        ## Kill prey that are out of energy
        killPrey!(agents = preyVec)
        ## Compute distances and angles for the next time step
        ΔPreyPrey, ηPreyPrey = preyDistances(preyVec, habitatSize)
        ΔPredators, ΔPrey, ηPredators, ηPrey = neighborDistances(predatorVec, preyVec, habitatSize, predatorSensoryRange, predatorSensingFactor)
        ## Update policy network by ES
        updateAllNetworks!(predatorVec, ghostState(deepcopy(land), deepcopy(preyVec), deepcopy(predatorVec)),
                            episodeLength, habitatSize, predatorSensoryRange, predatorSensingFactor, productivity, 
                            carryingCapacity, w, u, cohereStrength, turnMaxima, learnRate, preyCarryingCapacity)
        ## Collect statistics at intervals
        if t % logInterval == 1
            ePredator  = (p->p.e).(predatorVec)
            meanEnergy = mean(ePredator)
            semEnergy  = std(ePredator) / sqrt(nPredators)
            mmEnergy   = (minimum(ePredator), maximum(ePredator))
            push!(meanEnergyHistory, meanEnergy)
            push!(semEnergyHistory, semEnergy)
            push!(minmaxEnergyHistory, mmEnergy)
            push!(preyCountHistory, length(preyVec))
            push!(stepHistory, t)
            push!(captureHistory, catchHistory)
            lastRecordTime = t
            println(Crayon(foreground = (165, 93, 97)), "Time = $(t), Mean Energy = $(round(meanEnergy, digits = 2)), " * "Prey = $(length(preyVec)), Catches = $(catchHistory)")
        end
        ## Update total catches
        catches   = sum((p->p.age).(predatorVec) .== 0)
        catchHistory += catches
        ## Update ticker
        t += 1
        ## -
        ## Make movie for final k timesteps
        ## -
        if t ≥ maxTime - recordTimeBuffer
            ## Extract data
            xPrey = (p->p.x).(preyVec)
            yPrey = (p->p.y).(preyVec)
            ePrey = (p->p.e).(preyVec)
            xPredator = (p->p.x).(predatorVec)
            yPredator = (p->p.y).(predatorVec)
            ePredator = (p->p.e).(predatorVec)
            ## Render plots - Using original code structure
            p1 = heatmap(transpose(land), alpha = 0.5,
                        c = palette(["#8CC0D1", "#E8D9C0", "#C6AE90", "#9B8877", "#7B7D6E", "#5D7A5C", "#3A5E3B", "#253C26"], 24),
                        lims = (0.5, habitatSize + 0.5), clim = (0, carryingCapacity), legend = :none, showaxis = false, ticks = false, widen = false)
            p1 = scatter!(xPrey, yPrey, ms = 3, alpha = 0.75, mc = "#f7f7f7", 
                        markerstrokecolor = "#252525", markerstrokealpha = 0.75, legend = :none)
            p1 = scatter!(xPredator, yPredator, ms = 5, markercolor = "#000000", 
                        markerstrokecolor = "#959595", markerstrokealpha = 0.75, legend = :none)                    
            ## Text values
            annotate!(xPrey, yPrey, text.(round.(ePrey, digits = 2), "#000000", :right, 4))
            annotate!(xPredator, yPredator, text.(round.(ePredator, digits = 1), "#132229", :left, 4))
            ## Add a more informative title but keep everything else identical
            title!(p1, "T = $(t), P = $(length(preyVec)), E = $(round(mean(ePredator), digits = 1))", titlefont = font(7, "Computer Modern"))
            ## Push to frame vector
            frame(anim)
        end
    end
    ## -
    ## Create and save additional statistics plots
    ## -
    ## Write movie
    writeInd = lpad(simNumber, 6, "0")
    mPath = joinpath(store, "movieBehavior_$writeInd.mp4")
    gr(size = (360, 270), dpi = 240)
    gif(anim, mPath, fps = 30)
    ## Write figures
    if !isempty(stepHistory)
        ## 1. Energy history plot
        p_energy = plot(stepHistory, meanEnergyHistory, 
                       title = "Predator Energy vs Learning Phase", 
                       xlabel = "Timestep", 
                       ylabel = "Mean Energy",
                       linewidth = 2, legend = false)
        p_energy = plot!(stepHistory, meanEnergyHistory .+ semEnergyHistory, lw = 1, lc = "#737373", linestyle = :dash)
        p_energy = plot!(stepHistory, meanEnergyHistory .- semEnergyHistory, lw = 1, lc = "#737373", linestyle = :dash)
        p_energy = plot!(stepHistory, getindex.(minmaxEnergyHistory, 1), lw = 2, lc = "#167B68", linestyle = :dot)
        p_energy = plot!(stepHistory, getindex.(minmaxEnergyHistory, 2), lw = 2, lc = "#167B68", linestyle = :dot)
        ## Output figure
        pPath    = joinpath(store, "plotEnergy_$writeInd.png")
        savefig(p_energy, pPath)
        ## 2. Prey population plot
        p_prey = plot(stepHistory, preyCountHistory, 
                     linewidth = 2,
                     title = "Prey Population vs Learning Phase", 
                     xlabel = "Timestep", 
                     ylabel = "Prey Count",
                     legend = false)
        ## Add vertical lines for phase transitions
        pPath    = joinpath(store, "plotPrey_$writeInd.png")
        savefig(p_prey, pPath)
        ## 3. Catch history plot
        p_prey = plot(stepHistory, captureHistory, 
                     linewidth = 2,
                     title = "Cumulative captures vs Learning Phase", 
                     xlabel = "Timestep", 
                     ylabel = "Captures",
                     legend = false)
        ## Add vertical lines for phase transitions
        pPath    = joinpath(store, "plotCapture_$writeInd.png")
        savefig(p_prey, pPath)
    end
    ## Output final predator structs
    return predatorVec
end


"""
Function to run a batch of simulations and save the output for statistics
"""
function batch(nSimsPerLevel::Int, preyCoherences::Array{Float64}, groupSizes::Array{Int}, times; store = "./results")
    ## Check if results directory exists and find the last completed simulation
    if isdir(store)
        files       = readdir(store, join = true)
        trait_files = filter(x -> occursin("plotEnergy_", x), files)
        last_index  = maximum(parse(Int, match(r"(?<=plotEnergy_)(\d+)", x).captures[1]) for x ∈ trait_files)
        start_index = last_index + 1
        println("Resuming from simulation number $start_index")
    ## Make it if it's missing
    else
        mkdir(store)
        start_index = 1
    end
    ## Total simulations
    preyCoherencesR = repeat(preyCoherences, nSimsPerLevel)
    sweepArray      = collect(product(preyCoherencesR, groupSizes))
    total           = length(sweepArray) * nSimsPerLevel
    metadata        = reduce(vcat, [collect(t)' for t ∈ sweepArray])
    metadf          = DataFrame(metadata, [:flocking, :groupSize])
    outputMetaPath  = joinpath(store, "batchMetadata.csv")
    CSV.write(outputMetaPath, metadf)
    ## Simulation loop starting from the next index
    for i ∈ start_index:total
        idx = (i - 1) % length(sweepArray) + 1
        co = sweepArray[idx][1]
        gs = sweepArray[idx][2]
        ## Run one sim...
        writeInd = lpad(i, 6, "0")
        out = sim(
            ## Core parameters
            nPredators = gs,           # Number of predators
            w = 0.04,                  # Base movement speed
            u = 0.16,                  # Turn rate
            productivity = 6.5e-4,     # Resource growth rate
            maxTime = times,           # Run long enough to see all phases
            habitatSize = 8,           # Size of the environment
            ## Environment parameters
            cohereStrength = co,       # Prey cohesion strength
            carryingCapacity = 10.0,   # Resource carrying capacity
            preyCarryingCapacity = 32, # Maximum prey population
            nPrey = 20,                # Starting prey count
            ## Predator parameters
            predatorSensoryRange  = 0.64,  # How far predators can sense a prey
            predatorSensingFactor = 7.0,   # Predator sensing multiplier
            predatorTurnFactor    = 0.95,  # Predator turn rate factor (relative to prey's)
            ## Learning parameters
            learnRate = 0.72,          # Learning rate for ES
            initialEnergy = 10.0,      # Higher starting energy
            ## Misc
            simNumber = i,
            store = store)
        ## Write networks and performance stats to disc
        allAgentEnergyData = DataFrame(agent_num = Int[], simulation_num = Int[], energy = Float64[])
        for (j, a) ∈ collect(enumerate(out))
            ## Construct file paths
            predIndex      = lpad(j, 3, "0")
            outputNetPath  = joinpath(store, "networkForAgent_$(predIndex)_inSimulation_$(writeInd).jld2")
            ## Write the networks
            modelState     = Flux.state(a.network)
            jldsave(outputNetPath; modelState)
            ## Cumulative agent energy
            push!(allAgentEnergyData, (agent_num = j, simulation_num = i, energy = a.e))
        end
        ## Write energy file after each simulation
        outputAllEnergyPath = joinpath(store, "simulation_$(writeInd)_finalAgentEnergies.csv")
        CSV.write(outputAllEnergyPath, allAgentEnergyData)
        ## Drop a line
        println(round(100 * i / total, digits = 1), "% of simulations done")
    end
end

## ---
## JIT batch run
## ---
if isdir("./jitTest/")
    rm("./jitTest/", recursive = true)
end
@time batch(1, [0.0], [3], 10, store = "./jitTest")
rm("./jitTest/", recursive = true)

## ---
## This is the actual main batch run
## ---
if isdir("./results/")
    rm("./results/", recursive = true)
end
println("Using [", Threads.nthreads(), "] threads...")
@time batch(32,                   ## Reps per level 
            [0.0],                ## Flocking strength
            [8],                  ## Group sizes
            101000);              ## Max time



