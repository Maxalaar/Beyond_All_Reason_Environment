"""All mobile units of Beyond All Reason, organised by faction → type → tier.

Use for IDE autocomplete in scenario configurations::

    from beyond_all_reason_environment.units import units

    scenario = ScenarioConfiguration(
        allyteams=[
            AllyteamConfiguration(controller=Agent(), units={units.armada.bot.t1.pawn: 5}),
            AllyteamConfiguration(controller=HoldAndFire(), side="Cortex", units={units.cortex.bot.t1.grunt: 5}),
        ],
    )
"""


class _ArmadaBotT1:
    construction_bot = "armck"  # Construction Bot
    tick = "armflea"  # Tick
    mace = "armham"  # Mace
    crossbow = "armjeth"  # Crossbow
    pawn = "armpw"  # Pawn
    lazarus = "armrectr"  # Lazarus
    rocketeer = "armrock"  # Rocketeer
    centurion = "armwar"  # Centurion


class _ArmadaBotT2:
    archangel = "armaak"  # Archangel
    advanced_construction_bot = "armack"  # Advanced Construction Bot
    platypus = "armamph"  # Platypus
    smuggler = "armaser"  # Smuggler
    commander = "armdecom"  # Commander
    butler = "armfark"  # Butler
    sprinter = "armfast"  # Sprinter
    fatboy = "armfboy"  # Fatboy
    hound = "armfido"  # Hound
    butler_hack = "armhack"  # Butler
    compass = "armmark"  # Compass
    gunslinger = "armmav"  # Gunslinger
    umbrella = "armscab"  # Umbrella
    sharpshooter = "armsnipe"  # Sharpshooter
    webber = "armspid"  # Webber
    recluse = "armsptk"  # Recluse
    ghost = "armspy"  # Ghost
    tumbleweed = "armvader"  # Tumbleweed
    welder = "armzeus"  # Welder


class _ArmadaVehicleT1:
    shellshocker = "armart"  # Shellshocker
    beaver = "armbeaver"  # Beaver
    construction_vehicle = "armcv"  # Construction Vehicle
    rover = "armfav"  # Rover
    blitz = "armflash"  # Blitz
    janus = "armjanus"  # Janus
    groundhog = "armmlv"  # Groundhog
    pincer = "armpincer"  # Pincer
    whistler = "armsam"  # Whistler
    stout = "armstump"  # Stout


class _ArmadaVehicleT2:
    advanced_construction_vehicle = "armacv"  # Advanced Construction Vehicle
    bull = "armbull"  # Bull
    consul = "armconsul"  # Consul
    turtle = "armcroc"  # Turtle
    gremlin = "armgremlin"  # Gremlin
    consul_hacv = "armhacv"  # Consul
    umbra = "armjam"  # Umbra
    jaguar = "armlatnk"  # Jaguar
    starlight = "armmanni"  # Starlight
    mauser = "armmart"  # Mauser
    ambassador = "armmerl"  # Ambassador
    prophet = "armseer"  # Prophet
    shredder = "armyork"  # Shredder


class _ArmadaAircraftT1:
    stork = "armatlas"  # Stork
    construction_aircraft = "armca"  # Construction Aircraft
    attack_drone = "armdrone"  # Attack Drone
    falcon = "armfig"  # Falcon
    osprey = "armhvytrans"  # Osprey
    banshee = "armkam"  # Banshee
    blink = "armpeep"  # Blink
    stormbringer = "armthund"  # Stormbringer


class _ArmadaAircraftT2:
    advanced_construction_aircraft = "armaca"  # Advanced Construction Aircraft
    oracle = "armawac"  # Oracle
    hornet = "armblade"  # Hornet
    roughneck = "armbrawl"  # Roughneck
    abductor = "armdfly"  # Abductor
    highwind = "armhawk"  # Highwind
    cormorant = "armlance"  # Cormorant
    liche = "armliche"  # Liche
    blizzard = "armpnix"  # Blizzard
    stiletto = "armstil"  # Stiletto


class _ArmadaShipT1:
    construction_ship = "armcs"  # Construction Ship
    dolphin = "armdecade"  # Dolphin
    ellysaw = "armpship"  # Ellysaw
    skater = "armpt"  # Skater
    grim_reaper = "armrecl"  # Grim Reaper
    corsair = "armroy"  # Corsair
    eel = "armsub"  # Eel
    torpedo_ship = "armtorps"  # Torpedo Ship


class _ArmadaShipT2:
    dragonslayer = "armaas"  # Dragonslayer
    advanced_construction_sub = "armacsub"  # Advanced Construction Sub
    haven = "armantiship"  # Haven
    dreadnought = "armbats"  # Dreadnought
    haven_carry = "armcarry"  # Haven
    paladin = "armcrus"  # Paladin
    nexus = "armdronecarry"  # Nexus
    epoch = "armepoch"  # Epoch
    excalibur = "armexcalibur"  # Excalibur
    voyager = "armhacs"  # Voyager
    maelstrom = "armlship"  # Maelstrom
    voyager_mls = "armmls"  # Voyager
    longbow = "armmship"  # Longbow
    seadragon = "armseadragon"  # Seadragon
    serpent = "armserp"  # Serpent
    bermuda = "armsjam"  # Bermuda
    barracuda = "armsubk"  # Barracuda
    depth_charge_drone = "armtdrone"  # Depth Charge Drone
    trident = "armtrident"  # Trident


class _ArmadaHovercraftT1:
    sweeper = "armah"  # Sweeper
    crocodile = "armanac"  # Crocodile
    construction_hovercraft = "armch"  # Construction Hovercraft
    possum = "armmh"  # Possum
    seeker = "armsh"  # Seeker


class _ArmadaSeaplaneT1:
    construction_seaplane = "armcsa"  # Construction Seaplane
    experimental_construction_aircraft = "armhaca"  # Experimental Construction Aircraft
    sabre = "armsaber"  # Sabre
    tsunami = "armsb"  # Tsunami
    puffin = "armseap"  # Puffin
    horizon = "armsehak"  # Horizon
    cyclone = "armsfig"  # Cyclone
    cyclone_sfig2 = "armsfig2"  # Cyclone


class _CortexBotT1:
    grunt = "corak"  # Grunt
    construction_bot = "corck"  # Construction Bot
    trasher = "corcrash"  # Trasher
    graverobber = "cornecro"  # Graverobber
    aggravator = "corstorm"  # Aggravator
    thug = "corthud"  # Thug


class _CortexBotT2:
    manticore = "coraak"  # Manticore
    advanced_construction_bot = "corack"  # Advanced Construction Bot
    duck = "coramph"  # Duck
    sumo = "corcan"  # Sumo
    commander = "cordecom"  # Commander
    twitcher = "corfast"  # Twitcher
    twitcher_hack = "corhack"  # Twitcher
    arbiter = "corhrk"  # Arbiter
    commando = "cormando"  # Commando
    sheldon = "cormort"  # Sheldon
    fiend = "corpyro"  # Fiend
    bedbug = "corroach"  # Bedbug
    skuttle = "corsktl"  # Skuttle
    deceiver = "corspec"  # Deceiver
    spectre = "corspy"  # Spectre
    mammoth = "corsumo"  # Mammoth
    termite = "cortermite"  # Termite
    augur = "corvoyr"  # Augur


class _CortexVehicleT1:
    construction_vehicle = "corcv"  # Construction Vehicle
    rascal = "corfav"  # Rascal
    garpike = "corgarp"  # Garpike
    incisor = "corgator"  # Incisor
    pounder = "corlevlr"  # Pounder
    lasher = "cormist"  # Lasher
    trapper = "cormlv"  # Trapper
    muskrat = "cormuskrat"  # Muskrat
    brute = "corraid"  # Brute
    wolverine = "corwolv"  # Wolverine


class _CortexVehicleT2:
    advanced_construction_vehicle = "coracv"  # Advanced Construction Vehicle
    banisher = "corban"  # Banisher
    obscurer = "coreter"  # Obscurer
    tzar = "corgol"  # Tzar
    printer = "corhacv"  # Printer
    saviour = "cormabm"  # Saviour
    quaker = "cormart"  # Quaker
    poison_arrow = "corparrow"  # Poison Arrow
    phantom = "corphantom"  # Phantom
    printer_printer = "corprinter"  # Printer
    tiger = "correap"  # Tiger
    salamander = "corsala"  # Salamander
    alligator = "corseal"  # Alligator
    fury = "corsent"  # Fury
    siegebreaker = "corsiegebreaker"  # Siegebreaker
    tremor = "cortrem"  # Tremor
    printer_vac = "corvac"  # Printer
    corvacct = "corvacct"  # corvacct
    omen = "corvrad"  # Omen
    negotiator = "corvroc"  # Negotiator


class _CortexAircraftT1:
    shuriken = "corbw"  # Shuriken
    construction_aircraft = "corca"  # Construction Aircraft
    attack_drone = "cordrone"  # Attack Drone
    finch = "corfink"  # Finch
    hephaestus = "corhvytrans"  # Hephaestus
    whirlwind = "corshad"  # Whirlwind
    hercules = "corvalk"  # Hercules
    valiant = "corveng"  # Valiant


class _CortexAircraftT2:
    advanced_construction_aircraft = "coraca"  # Advanced Construction Aircraft
    wasp = "corape"  # Wasp
    condor = "corawac"  # Condor
    dragon = "corcrwh"  # Dragon
    hailstorm = "corhurc"  # Hailstorm
    skyhook = "corseah"  # Skyhook
    angler = "cortitan"  # Angler
    nighthawk = "corvamp"  # Nighthawk


class _CortexShipT1:
    construction_ship = "corcs"  # Construction Ship
    supporter = "coresupp"  # Supporter
    riptide = "corpship"  # Riptide
    herring = "corpt"  # Herring
    death_cavalry = "correcl"  # Death Cavalry
    oppressor = "corroy"  # Oppressor
    orca = "corsub"  # Orca


class _CortexShipT2:
    advanced_construction_sub = "coracsub"  # Advanced Construction Sub
    oasis = "corantiship"  # Oasis
    arrow_storm = "corarch"  # Arrow Storm
    despot = "corbats"  # Despot
    black_hydra = "corblackhy"  # Black Hydra
    oasis_carry = "corcarry"  # Oasis
    buccaneer = "corcrus"  # Buccaneer
    desolator = "cordesolator"  # Desolator
    disperser = "cordronecarry"  # Disperser
    brimstone = "corfship"  # Brimstone
    pathfinder = "corhacs"  # Pathfinder
    pathfinder_mls = "cormls"  # Pathfinder
    messenger = "cormship"  # Messenger
    onager = "coronager"  # Onager
    black_prince = "corprince"  # Black Prince
    sentinel = "corsentinel"  # Sentinel
    predator = "corshark"  # Predator
    phantasm = "corsjam"  # Phantasm
    kraken = "corssub"  # Kraken
    depth_charge_drone = "cortdrone"  # Depth Charge Drone


class _CortexHovercraftT1:
    birdeater = "corah"  # Birdeater
    construction_hovercraft = "corch"  # Construction Hovercraft
    halberd = "corhal"  # Halberd
    mangonel = "cormh"  # Mangonel
    goon = "corsh"  # Goon
    cayman = "corsnap"  # Cayman


class _CortexSeaplaneT1:
    construction_seaplane = "corcsa"  # Construction Seaplane
    cutlass = "corcut"  # Cutlass
    experimental_construction_aircraft = "corhaca"  # Experimental Construction Aircraft
    watcher = "corhunt"  # Watcher
    dam_buster = "corsb"  # Dam Buster
    monsoon = "corseap"  # Monsoon
    bat = "corsfig"  # Bat
    bat_sfig2 = "corsfig2"  # Bat


class _LegionBotT1:
    toxotai = "legaabot"  # Toxotai
    ballista = "legbal"  # Ballista
    phobos = "legcen"  # Phobos
    goblin = "leggob"  # Goblin
    karkinos = "legkark"  # Karkinos
    satyr = "leglob"  # Satyr
    zagreus = "legrezbot"  # Zagreus


class _LegionBotT2:
    aquilon = "legadvaabot"  # Aquilon
    tiresias = "legajamk"  # Tiresias
    telchine = "legamph"  # Telchine
    euclid = "legaradk"  # Euclid
    eidolon = "legaspy"  # Eidolon
    belcher = "legbart"  # Belcher
    legion_commander = "legdecom"  # Legion Commander
    thanatos = "leghrk"  # Thanatos
    incinerator = "leginc"  # Incinerator
    infestor = "leginfestor"  # Infestor
    phalanx = "legshot"  # Phalanx
    snapper = "legsnapper"  # Snapper
    arquebus = "legsrail"  # Arquebus
    hoplite = "legstr"  # Hoplite


class _LegionVehicleT1:
    cetus = "legamphtank"  # Cetus
    barrage = "legbar"  # Barrage
    decurion = "leggat"  # Decurion
    alaris = "leghades"  # Alaris
    helios = "leghelios"  # Helios
    sapper = "legmlv"  # Sapper
    lance = "legrail"  # Lance
    wheelie = "legscout"  # Wheelie


class _LegionVehicleT2:
    prometheus = "legaheattank"  # Prometheus
    cleaver = "legamcluster"  # Cleaver
    gladiator = "legaskirmtank"  # Gladiator
    hera = "legavantinuke"  # Hera
    cicero = "legavjam"  # Cicero
    pheme = "legavrad"  # Pheme
    boreas = "legavroc"  # Boreas
    triton = "legfloat"  # Triton
    inferno = "leginf"  # Inferno
    medusa = "legmed"  # Medusa
    quickshot = "legmrv"  # Quickshot
    mantis = "legvcarry"  # Mantis
    charon = "legvflak"  # Charon


class _LegionAircraftT1:
    hippotes = "legatrans"  # Hippotes
    blindfold = "legcib"  # Blindfold
    legion_drone = "legdrone"  # Legion Drone
    noctua = "legfig"  # Noctua
    martyr = "legkam"  # Martyr
    aeolus = "leglts"  # Aeolus
    mosquito = "legmos"  # Mosquito


class _LegionAircraftT2:
    ajax = "legafigdef"  # Ajax
    aesacus = "legatorpbomber"  # Aesacus
    tyrannus = "legfort"  # Tyrannus
    legion_heavy_drone = "legheavydrone"  # Legion Heavy Drone
    legion_heavy_drone_heavydronesmall = "legheavydronesmall"  # Legion Heavy Drone
    legionnaire = "legionnaire"  # Legionnaire
    harbinger = "legmineb"  # Harbinger
    wildfire = "legnap"  # Wildfire
    phoenix = "legphoenix"  # Phoenix
    venator = "legvenator"  # Venator
    whisper = "legwhisper"  # Whisper


class _LegionShipT1:
    iapetus = "legnavyaaship"  # Iapetus
    octeres = "legnavyartyship"  # Octeres
    syracusia = "legnavydestro"  # Syracusia
    argonaut = "legnavyfrigate"  # Argonaut
    dionysus = "legnavyrezsub"  # Dionysus
    hippocampus = "legnavyscout"  # Hippocampus
    ketea = "legnavysub"  # Ketea


class _LegionShipT2:
    notus = "leganavyaaship"  # Notus
    hecate = "leganavyantinukecarrier"  # Hecate
    leocampus = "leganavyantiswarm"  # Leocampus
    corinth = "leganavyartyship"  # Corinth
    scylla = "leganavybattleship"  # Scylla
    architeuthis = "leganavybattlesub"  # Architeuthis
    thalassa = "leganavycruiser"  # Thalassa
    neptune = "leganavyflagship"  # Neptune
    sphyrna = "leganavyheavysub"  # Sphyrna
    ultor = "leganavymissileship"  # Ultor
    dolus = "leganavyradjamship"  # Dolus


class _LegionHovercraftT1:
    alpheus = "legah"  # Alpheus
    cardea = "legcar"  # Cardea
    salacia = "legmh"  # Salacia
    nereus = "legner"  # Nereus
    glaucus = "legsh"  # Glaucus


class _LegionSeaplaneT1:
    pyrphoros = "legspbomber"  # Pyrphoros
    hecatoncheir = "legspcarrier"  # Hecatoncheir
    astrapios = "legspfighter"  # Astrapios
    okeanos = "legspradarsonarplane"  # Okeanos
    enyo = "legspsurfacegunship"  # Enyo
    ladon = "legsptorpgunship"  # Ladon


class _ArmadaBot:
    t1 = _ArmadaBotT1
    t2 = _ArmadaBotT2


class _ArmadaVehicle:
    t1 = _ArmadaVehicleT1
    t2 = _ArmadaVehicleT2


class _ArmadaAircraft:
    t1 = _ArmadaAircraftT1
    t2 = _ArmadaAircraftT2


class _ArmadaShip:
    t1 = _ArmadaShipT1
    t2 = _ArmadaShipT2


class _ArmadaHovercraft:
    t1 = _ArmadaHovercraftT1


class _ArmadaSeaplane:
    t1 = _ArmadaSeaplaneT1


class _CortexBot:
    t1 = _CortexBotT1
    t2 = _CortexBotT2


class _CortexVehicle:
    t1 = _CortexVehicleT1
    t2 = _CortexVehicleT2


class _CortexAircraft:
    t1 = _CortexAircraftT1
    t2 = _CortexAircraftT2


class _CortexShip:
    t1 = _CortexShipT1
    t2 = _CortexShipT2


class _CortexHovercraft:
    t1 = _CortexHovercraftT1


class _CortexSeaplane:
    t1 = _CortexSeaplaneT1


class _LegionBot:
    t1 = _LegionBotT1
    t2 = _LegionBotT2


class _LegionVehicle:
    t1 = _LegionVehicleT1
    t2 = _LegionVehicleT2


class _LegionAircraft:
    t1 = _LegionAircraftT1
    t2 = _LegionAircraftT2


class _LegionShip:
    t1 = _LegionShipT1
    t2 = _LegionShipT2


class _LegionHovercraft:
    t1 = _LegionHovercraftT1


class _LegionSeaplane:
    t1 = _LegionSeaplaneT1


class _Armada:
    bot = _ArmadaBot
    vehicle = _ArmadaVehicle
    aircraft = _ArmadaAircraft
    ship = _ArmadaShip
    hovercraft = _ArmadaHovercraft
    seaplane = _ArmadaSeaplane


class _Cortex:
    bot = _CortexBot
    vehicle = _CortexVehicle
    aircraft = _CortexAircraft
    ship = _CortexShip
    hovercraft = _CortexHovercraft
    seaplane = _CortexSeaplane


class _Legion:
    bot = _LegionBot
    vehicle = _LegionVehicle
    aircraft = _LegionAircraft
    ship = _LegionShip
    hovercraft = _LegionHovercraft
    seaplane = _LegionSeaplane


class units:
    armada = _Armada
    cortex = _Cortex
    legion = _Legion
