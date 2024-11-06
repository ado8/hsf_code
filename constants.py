import numpy as np
import os

HSF_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_DIR = f"{HSF_DIR}/media"
STATIC_DIR = f"{HSF_DIR}/static"
ROOT_DIR = os.path.dirname(HSF_DIR)
SUBARU_HSF_DIR = f"{ROOT_DIR}/Subaru/projects/flows"
UKIRT_HSF_DIR = f"{ROOT_DIR}/UKIRT/projects/flows"
DATA_DIR = f"{HSF_DIR}/reduced_data"
SCAT_DIR = "/data/poohbah/1/assassin/SCAT"
ISIS_DIR = f"{ROOT_DIR}/UKIRT/isis"

C = 299792.458
DEFAULT_Z = 0.04
UKIRT_INTERVAL = 3  # get observations every 3 days
SUB_TYPES = ("nosub", "refsub", "rotsub")
CALSPEC_PM = {
    "C26202": (7.44, -1.331),
    "P330E": (-8.882, -38.705),
    "SNAP-2": (-2.929, -10.887),
    "VB8": (-813.038, -870.609),
    "2M0036+18": (901.56, 124.02),
    "2M055914": (571.34, -338.169),
    "G191B2B": (12.701, -93.416),
    "GD71": (76.728, -172.96),
    "GD153": (206.591, -202.990),
    "HS2027+0651": (9.24, 1.53),
    "NGC2506G31": (-2.515, 4.057),
    "SDSS132811": (-130.926, -30.747),
    "SDSSJ151421": (4.35, -26.855),
    "SF1615+001A": (2.554, -10.932),
    "wd2341_322": (-215.905, -59.871),
    "WD1657+343": (8.77, -31.23),
    "WD0947+857": (-28.13 - 27.27),
    "WD1026+453": (-90.473, 1.925),
    "GD50": (75.1, -155.9),
    "Vega": (200.94, 286.23),
}
REF_PM = {
    "AP Sgr": (0.886, -3.729),
    "BF Oph": (0.390, -0.289),
    "NGC4242": (-0.353, -0.566),
    "RV Sco": (1.792, -6.294),
    "RX Cam": (-0.205, -0.428),
    "RY Cma": (-2.668, 1.798),
    "SS Sct": (1.907, 1.263),
    "TX Cyg": (-1.084, -2.194),
    "U Aql": (0.969, -11.962),
    "U Sgr": (-1.795, -6.127),
    "V0386 Cyg": (-0.990, -2.213),
    "V482 Sco": (0.256, -2.840),
    "W Gem": (-0.940, -2.371),
}
NO_COLOR_CORRECTION = {
    "J": ("J", 0, "J", "H", 0),
    "H": ("H", 0, "J", "H", 0),
    "K": ("K", 0, "J", "K", 0),
}
HEWETT_COLOR_CORRECTION = {  # 2MASS color terms from Hewett et al 2006, final table
    # Luminosity classes III and V computed separately
    "JIII": ("J", -0.003, "J", "H", -0.01),
    "JV": ("J", -0.067, "J", "H", 0.01),
    "HIII": ("H", 0.065, "H", "K", 0.01),
    "HV": ("H", 0.08, "H", "K", 0),
    "KIII": ("K", 0.075, "H", "K", 0),
    "KV": ("K", -0.081, "H", "K", 0),
}
UKIDSS_COLOR_CORRECTION = {  # 2MASS color terms from Dye et al 2006, eqn 3
    "Z": ("J", 0.95, "J", "H", 0),
    "Y": ("J", 0.5, "J", "H", 0.08),
    "J": ("J", -0.075, "J", "H", 0),
    "H": ("H", 0.04, "J", "H", -0.04),
    "K": ("K", 0.015, "J", "K", 0),
}
HODGKIN_COLOR_CORRECTION = {  # 2MASS color terms from Hodgkin et al 2009, eqns 4-8
    "Z": ("J", 0.95, "J", "H", 0),
    "Y": ("J", 0.5, "J", "H", 0.08),
    "J": ("J", -0.065, "J", "H", 0),
    "H": ("H", 0.07, "J", "H", -0.03),
    "K": ("K", 0.01, "J", "K", 0),
}
STAN_COLOR_CORRECTION = {
    "Y": ("Y", 0.3679534057327641, "J", "H", 0.14878461158997566),
    "Ymef2": ("Y", 0.370779550990509, "J", "H", 0.1484323836838301),
    "Ymef3": ("Y", 0.36566044132865605, "J", "H", 0.14896067143646943),
    "J": ("J", 0, "J", "H", 0),
    "Jmef2": ("J", -0.10611486633258567, "J", "H", 0.025368799386560538),
    "Jmef3": ("J", -0.10500081887580834, "J", "H", 0.02338804275714695),
    "H": ("H", 0, "J", "H", 0),
    "Hmef2": ("H", 0.164835170226681, "J", "H", -0.06748510411944995),
    "Hmef3": ("H", 0.18352699094722907, "J", "H", -0.07649818982238152),
}
LC_PARAMS = ("mjd", "mag", "dmag", "ujy", "dujy")
MODEL_NAMES = (
    "snpy_max_model",
    "snpy_ebv_model2",
    "salt3-nir",
    "spline",
)
MAG_TO_FLUX = {  # from Hewett et al. 2006
    "c": 0,
    "o": 0,
    "asg": 0,
    "ztfg": 0,
    "ztfr": 0,
    "ztfi": 0,
    "Y": 0.634,
    "J": 0.938,
    "H": 1.379,
    "K": 1.900,
}
SIGMA_MU_Z = 250 / C
SIGMA_LENS = 0.055
SIGMA_INT = 0.1
ALPHA = {  # salt nuisance parameter
    "salt3": 0.133,
    "salt2-extended": 0.133,
    "salt3-nir": 0.12378421,
}
BETA = {  # salt nuisance parameter
    "salt3": 2.846,
    "salt2-extended": 2.846,
    "salt3-nir": 3.52849593,
}
FIDUCIAL_M = -19.0  # salt nuisance parameter
PRE_SN = 20  # lcs start this many days before detection
POST_SN = 60  # lcs continue this many days after detection
ATLAS_CORES = 4  # Use this many cores in parallel when pulling ATLAS lcs
ATLAS_SIGMA_FOR_DETECTION = 3
ATLAS_SKY_LIM = 16
ATLAS_AXIS_RATIO_LIM = 1.5
ATLAS_CHIP_SIZE = 10560
IM_PAD = 80
EDGE_PAD = 100
CMB_L = 263.85  # CMB dipole w.r.t. heliocentric frame (Bennett+2003)
CMB_B = 48.25
CMB_V = 371
MK_LAT = 19.828333
MK_LON = -155.478333
MK_HEIGHT = 4160
PSF_ECCENTRICITY_QUANTILE = 0.8
PSF_SIZE_QUANTILE = (0.1, 0.7)
PSF_FLUX_QUANTILE = 0.95
ZP_SIGMA_FROM_MEDIAN = 3
FOCAS_LACOSMIC_CONTRAST = 1
PLOTTING_DICT = {
    "linestyle": "",
    "capsize": 5,
    "elinewidth": 2,
    "label_size": 15,
    "tick_size": 14,
    "left_pad": 5,
    "right_pad": 20,
    "today_pad": [(-1, 0.3), (0.1, 0.3)],
    "detect_pad": (0.1, -0.1),
    "detect_size": 18,
    "text_pad": (-15, 0.4),
    "today_size": 18,
    "today_color": "white",
    "figsize": (12, 10),
    "ylim_cap": (7, 20),
    "legend_size": 15,
    "legend_loc": "upper left",
    "legend_bbox": (1, 1.05),
    "line_alpha": 0.8,
    "subplots_adjust": [0.08, 0.05, 0.78, 0.98],
}
FILT_BAYESN_NAMES = {
    "asg": "g_DES",  # not really, but for now close enough
    "o": "o_ATLAS",
    "c": "c_ATLAS",
    "ztfg": "p48g",
    "ztfr": "p48r",
    "ztfi": "p48i",
    "Z": "z_WFCAM",
    "Y": "Y_WFCAM",
    "J": "J_WFCAM",
    "H": "H_WFCAM",
    "K": "K_WFCAM",
}
FILT_SNPY_NAMES = {
    "o": "ATri",
    "c": "ATgr",
    "ztfg": "ztfg",
    "ztfr": "ztfr",
    "ztfi": "ztfi",
    "Z": "WFCAMz",
    "Y": "WFCAMY",
    "J": "WFCAMJ",
    "H": "WFCAMH",
    "K": "WFCAMK",
    "asg": "g",
}
BPS_COMBO_DICT = {
    "K": 1024,
    "H": 512,
    "J": 256,
    "Y": 128,
    "Z": 64,
    "asg": 32,
    "ztfi": 16,
    "ztfr": 8,
    "ztfg": 4,
    "o": 2,
    "c": 1,
    "": 0,
}
VARIANT_PREFERENCE = ("2D", "0D", "1D3")
VARIANT_COMBO_DICT = {
    "tphot": 1024,
    "2D": 512,
    "1D4": 256,
    "1D3": 128,
    "1D2": 64,
    "0D": 32,
    "ref": 16,
    "rot": 8,
    "magap_big": 4,
    "magap": 2,
    "magpsf": 1,
    "none": 0,
    # "": 0,
}
COLOR_DICT = {
    "c": "cyan",
    "o": "orange",
    "ztfg": "yellowgreen",
    "ztfr": "red",
    "ztfi": "brown",
    "asg": "lime",
    "Z": "grey",
    "Y": "violet",
    "J": "white",
    "H": "pink",
    "K": "blue",
}
EXP_TIMES_OLD = np.array(
    [[18.06, 18.47, 18.99, 19.37, 19.59, 19.75], [1, 2, 5, 10, 15, 20]]
)
EXP_TIMES = {
    "Z": [18.74, 19.25, 19.86, 20.28, 20.51, 20.68],
    "Y": [18.44, 18.91, 19.47, 19.87, 20.10, 20.26],
    "J": [18.06, 18.47, 18.99, 19.37, 19.59, 19.75],
    "H": [17.22, 17.61, 18.11, 18.49, 18.71, 18.86],
    "K": [16.63, 17.02, 17.52, 17.90, 18.12, 18.27],
}
BRIGHT_PAD = 90
TELLURICS = {
    "[OI] 1": (5570, 5584),
    "[OI] 2": (6298, 6302),
    "O2_A": (6867, 6884),
    "O2_B": (7594, 7621),
}
SKYLINE_CAL = [  # intensities guesstimated from https://www.naoj.hawaii.edu/Observing/Instruments/FOCAS/Detail/UsersGuide/Observing/SkyLines/OH_Spectrum.htm
    [5577.34, "[OI] AN", 2400],
    # [5889.95, 'Na I', 900],
    # [5891.5, 'Na I', 900],
    # [5895.92, 'Na I', 900],
    [6300.3, "[OI] AN", 1100],
    [6363.8, "[OI] AN", 400],
    [6533.044, "6-1 P1", 200],
    # [6553.616999999999, '6-1 P1', 200],
    [6863.955, "7-2 Q1", 500],
    [6948.935, "7-2 P1", 400],
    [7316.281999999999, "8-3 P1", 800],
    [7340.885, "8-3 P1", 1000],
    [7369.2488, "8-3 P1", 800],
    [7524.1145, "8-3 P1", 650],
    [7794.12, "9-4 P1", 800],
    [7821.503000000001, "9-4 P1", 900],
    [7964.65, "5-1 P1", 900],
    [7993.331999999999, "5-1 P1", 1000],
    [8025.6662, "5-1 P1", 800],
    [8399.17, "6-2 P1", 1000],
    [8430.174, "6-2 P1", 1300],
    [8465.3585, "6-2 P1", 900],
]
SNIFS_DICHROIC = (5100, 5300)
SCAT_CLASSIFICATION_NUMBER = 10
SCAT_HOST_Z_NUMBER = 10
# Nikola Knezevic's script from https://www.wis-tns.org/content/tns-getting-started #API-Search/Get objects
TNS = "www.wis-tns.org"
# TNS = "sandbox.wis-tns.org"
TNS_URL_API = "https://" + TNS + "/api/get"
TNS_BOT_ID = "121906"
TNS_BOT_NAME = "SCAT_Bot1"
TNS_EXT_HTTP_ERRORS = (403, 500, 503)
TNS_ERR_MSG = (
    "Forbidden",
    "Internal Server Error: Something is broken",
    "Service Unavailable",
)
REFCAT_PADDING = 1.1
REFCAT_SHORT_COLUMNS = ("ra", "dec", "g", "r", "i", "z", "j", "c", "o")
REFCAT_ALL_COLUMNS = (
    "ra",
    "dec",
    "plx",
    "dplx",
    "pmra",
    "dpmra",
    "pmdec",
    "dpmdec",
    "gaia",
    "dgaia",
    "bp",
    "dbp",
    "rp",
    "drp",
    "teff",
    "agaia",
    "dupvar",
    "ag",
    "rp1",
    "r1",
    "r10",
    "g",
    "dg",
    "gchi",
    "gcontrib",
    "r",
    "dr",
    "rchi",
    "rcontrib",
    "i",
    "di",
    "ichi",
    "icontrib",
    "z",
    "dz",
    "zchi",
    "zcontrib",
    "nstat",
    "j",
    "dj",
    "h",
    "dh",
    "k",
    "dk",
)
FORCE_COLUMNS = (
    "mjd",
    "mag",
    "dmag",
    "ujy",
    "dujy",
    "bandpass",
    "err",
    "chin",
    "ra",
    "dec",
    "x",
    "y",
    "major",
    "minor",
    "phi",
    "apfit",
    "mag5sig",
    "sky",
    "obs",
)
TPHOT_COLUMNS = (
    "x",
    "y",
    "peakval",
    "skyval",
    "peakfit",
    "dpeak",
    "skyfit",
    "flux",
    "dflux",
    "major",
    "minor",
    "phi",
    "err",
    "chin",
)
PHOT_COLUMNS = (
    "x",
    "y",
    "raref",
    "decref",
    "m",
    "g",
    "r",
    "i",
    "zp",
    "flux",
    "dflux",
    "err",
    "dx",
    "dy",
)
DDC_COLUMNS = (
    "ra",
    "dec",
    "mag",
    "dmag",
    "x",
    "y",
    "major",
    "minor",
    "phi",
    "det",
    "chin",
    "pvr",
    "ptr",
    "pmv",
    "pkn",
    "pno",
    "pbn",
    "pcr",
    "pxt",
    "psc",
    "dup",
    "wpflx",
    "dflx",
)
TPHOT_RAD = 5

SIMBAD_TYPE_CHOICES = [
    ("?", "Object of unknown nature"),
    ("ev", "transient event"),
    ("Rad", "Radio-source"),
    ("mR", "metric Radio-source"),
    ("cm", "centimetric Radio-source"),
    ("mm", "millimetric Radio-source"),
    ("smm", "sub-millimetric source"),
    ("HI", "HI (21cm) source"),
    ("rB", "radio Burst"),
    ("Mas", "Maser"),
    ("IR", "Infra-Red source"),
    ("FIR", "Far-Infrared source"),
    ("MIR", "Mid-Infrared source"),
    ("NIR", "Near-Infrared source"),
    ("blu", "Blue object"),
    ("UV", "UV-emission source"),
    ("X", "X-ray source"),
    ("UX?", "Ultra-luminous X-ray candidate"),
    ("ULX", "Ultra-luminous X-ray source"),
    ("gam", "gamma-ray source"),
    ("gB", "gamma-ray Burst"),
    ("err", "Not an object (error, artefact, ...)"),
    ("grv", "Gravitational Source"),
    ("Lev", "(Micro)Lensing Event"),
    ("LS?", "Possible gravitational lens System"),
    ("Le?", "Possible gravitational lens"),
    ("LI?", "Possible gravitationally lensed image"),
    ("gLe", "Gravitational Lens"),
    ("gLS", "Gravitational Lens System (lens+images)"),
    ("GWE", "Gravitational Wave Event"),
    ("..?", "Candidate objects"),
    ("G? ", "Possible Galaxy"),
    ("SC?", "Possible Supercluster of Galaxies"),
    ("C?G", "Possible Cluster of Galaxies"),
    ("Gr?", "Possible Group of Galaxies"),
    ("As?", "[sic]"),
    ("**?", "Physical Binary Candidate"),
    ("EB?", "Eclipsing Binary Candidate"),
    ("Sy?", "Symbiotic Star Candidate"),
    ("CV?", "Cataclysmic Binary Candidate"),
    ("No?", "Nova Candidate"),
    ("XB?", "X-ray binary Candidate"),
    ("LX?", "Low-Mass X-ray binary Candidate"),
    ("HX?", "High-Mass X-ray binary Candidate"),
    ("Pec", "Possible Peculiar Star"),
    ("Y*?", "Young Stellar Object Candidate"),
    ("TT?", "T Tau star Candidate"),
    ("C*?", "Possible Carbon Star"),
    ("S*?", "Possible S Star"),
    ("OH?", "Possible Star with envelope of OH/IR type"),
    ("WR?", "Possible Wolf-Rayet Star"),
    ("Be?", "Possible Be Star"),
    ("Ae?", "Possible Herbig Ae/Be Star"),
    ("HB?", "Possible Horizontal Branch Star"),
    ("RR?", "Possible Star of RR Lyr type"),
    ("Ce?", "Possible Cepheid"),
    ("WV?", "Possible Variable Star of W Vir type"),
    ("RB?", "Possible Red Giant Branch star"),
    ("sg?", "Possible Supergiant star"),
    ("s?r", "Possible Red supergiant star"),
    ("s?y", "Possible Yellow supergiant star"),
    ("s?b", "Possible Blue supergiant star"),
    ("AB?", "Asymptotic Giant Branch Star candidate"),
    ("LP?", "Long Period Variable candidate"),
    ("Mi?", "Mira candidate"),
    ("pA?", "Post-AGB Star Candidate"),
    ("BS?", "Candidate blue Straggler Star"),
    ("HS?", "Hot subdwarf candidate"),
    ("WD?", "White Dwarf Candidate"),
    ("N*?", "Neutron Star Candidate"),
    ("BH?", "Black Hole Candidate"),
    ("SN?", "SuperNova Candidate"),
    ("LM?", "Low-mass star candidate"),
    ("BD?", "Brown Dwarf Candidate"),
    ("mul", "Composite object"),
    ("reg", "Region defined in the sky"),
    ("vid", "Underdense region of the Universe"),
    ("SCG", "Supercluster of Galaxies"),
    ("ClG", "Cluster of Galaxies"),
    ("GrG", "Group of Galaxies"),
    ("CGG", "Compact Group of Galaxies"),
    ("PaG", "Pair of Galaxies"),
    ("IG", "Interacting Galaxies"),
    ("C?*", "Possible (open) star cluster"),
    ("Gl?", "Possible Globular Cluster"),
    ("Cl*", "Cluster of Stars"),
    ("GlC", "Globular Cluster"),
    ("OpC", "Open (galactic) Cluster"),
    ("As*", "Association of Stars"),
    ("St*", "Stellar Stream"),
    ("MGr", "Moving Group"),
    ("**", "Double or multiple star"),
    ("EB*", "Eclipsing binary"),
    ("SB*", "Spectroscopic binary"),
    ("El*", "Ellipsoidal variable Star"),
    ("Sy*", "Symbiotic Star"),
    ("CV*", "Cataclysmic Variable Star"),
    ("No*", "Nova"),
    ("XB*", "X-ray Binary"),
    ("LXB", "Low Mass X-ray Binary"),
    ("HXB", "High Mass X-ray Binary"),
    ("ISM", "Interstellar matter"),
    ("PoC", "Part of Cloud"),
    ("PN?", "Possible Planetary Nebula"),
    ("CGb", "Cometary Globule"),
    ("bub", "Bubble"),
    ("EmO", "Emission Object"),
    ("Cld", "Cloud"),
    ("GNe", "Galactic Nebula"),
    ("DNe", "Dark Cloud (nebula)"),
    ("RNe", "Reflection Nebula"),
    ("MoC", "Molecular Cloud"),
    ("glb", "Globule (low-mass dark cloud)"),
    ("cor", "Dense core"),
    ("SFR", "Star forming region"),
    ("HVC", "High-velocity Cloud"),
    ("HII", "HII (ionized) region"),
    ("PN ", "Planetary Nebula"),
    ("sh ", "HI shell"),
    ("SR?", "SuperNova Remnant Candidate"),
    ("SNR", "SuperNova Remnant"),
    ("of?", "Outflow candidate"),
    ("out", "Outflow"),
    ("HH ", "Herbig-Haro Object"),
    ("*  ", "Star"),
    ("V*?", "Star suspected of Variability"),
    ("Pe*", "Peculiar Star"),
    ("HB*", "Horizontal Branch Star"),
    ("Y*O", "Young Stellar Object"),
    ("Ae*", "Herbig Ae/Be star"),
    ("Em*", "Emission-line Star"),
    ("Be*", "Be Star"),
    ("BS*", "Blue Straggler Star"),
    ("RG*", "Red Giant Branch star"),
    ("AB*", "Asymptotic Giant Branch Star (He-burning)"),
    ("C*", "Carbon Star"),
    ("S*", "S Star"),
    ("sg*", "Evolved supergiant star"),
    ("s*r", "Red supergiant star"),
    ("s*y", "Yellow supergiant star"),
    ("s*b", "Blue supergiant star"),
    ("HS*", "Hot subdwarf"),
    ("pA*", "Post-AGB Star (proto-PN)"),
    ("WD*", "White Dwarf"),
    ("LM*", "Low-mass star (M<1solMass)"),
    ("BD*", "Brown Dwarf (M<0.08solMass)"),
    ("N*", "Confirmed Neutron Star"),
    ("OH*", "OH/IR star"),
    ("TT*", "T Tau-type Star"),
    ("WR*", "Wolf-Rayet Star"),
    ("PM*", "High proper-motion Star"),
    ("HV*", "High-velocity Star"),
    ("V* ", "Variable Star"),
    ("Ir*", "Variable Star of irregular type"),
    ("Or*", "Variable Star of Orion Type"),
    ("Er*", "Eruptive variable Star"),
    ("RC*", "Variable Star of R CrB type"),
    ("RC?", "Variable Star of R CrB type candiate"),
    ("Ro*", "Rotationally variable Star"),
    ("a2*", "Variable Star of alpha2 CVn type"),
    ("Psr", "Pulsar"),
    ("BY*", "Variable of BY Dra type"),
    ("RS*", "Variable of RS CVn type"),
    ("Pu*", "Pulsating variable Star"),
    ("RR*", "Variable Star of RR Lyr type"),
    ("Ce*", "Cepheid variable Star"),
    ("dS*", "Variable Star of delta Sct type"),
    ("RV*", "Variable Star of RV Tau type"),
    ("WV*", "Variable Star of W Vir type"),
    ("bC*", "Variable Star of beta Cep type"),
    ("cC*", "Classical Cepheid (delta Cep type)"),
    ("gD*", "Variable Star of gamma Dor type"),
    ("SX*", "Variable Star of SX Phe type (subdwarf)"),
    ("LP*", "Long-period variable star"),
    ("Mi*", "Variable Star of Mira Cet type"),
    ("SN*", "SuperNova"),
    ("su*", "Sub-stellar object"),
    ("Pl?", "Extra-solar Planet Candidate"),
    ("Pl", "Extra-solar Confirmed Planet"),
    ("G", "Galaxy"),
    ("PoG", "Part of a Galaxy"),
    ("GiC", "Galaxy in Cluster of Galaxies"),
    ("BiC", "Brightest galaxy in a Cluster (BCG)"),
    ("GiG", "Galaxy in Group of Galaxies"),
    ("GiP", "Galaxy in Pair of Galaxies"),
    ("rG", "Radio Galaxy"),
    ("H2G", "HII Galaxy"),
    ("LSB", "Low Surface Brightness Galaxy"),
    ("AG?", "Possible Active Galaxy Nucleus"),
    ("Q?", "Possible Quasar"),
    ("Bz?", "Possible Blazar"),
    ("BL?", "Possible BL Lac"),
    ("EmG", "Emission-line galaxy"),
    ("SBG", "Starburst Galaxy"),
    ("bCG", "Blue compact Galaxy"),
    ("LeI", "Gravitationally Lensed Image"),
    ("LeG", "Gravitationally Lensed Image of a Galaxy"),
    ("LeQ", "Gravitationally Lensed Image of a Quasar"),
    ("AGN", "Active Galaxy Nucleus"),
    ("LIN", "LINER-type Active Galaxy Nucleus"),
    ("SyG", "Seyfert Galaxy"),
    ("Sy1", "Seyfert 1 Galaxy"),
    ("Sy2", "Seyfert 2 Galaxy"),
    ("Bla", "Blazar"),
    ("BLL", "BL Lac - type object"),
    ("OVV", "Optically Violently Variable object"),
    ("QSO", "Quasar"),
]
NED_TYPE_CHOICES = [
    ("*", "Star or Point Source"),
    ("**", "Double star"),
    ("*Ass", "Stellar association"),
    ("*Cl", "Star cluster"),
    ("AbLS", "Absorption line system"),
    ("Blue*", "Blue star"),
    ("C*", "Carbon star"),
    ("EmLS", "Emission line source"),
    ("EmObj", "Emission object"),
    ("exG*", "Extragalactic star (not a member of an identified galaxy)"),
    ("Flare*", "Flare star"),
    ("G", "Galaxy"),
    ("GammaS", "Gamma ray source"),
    ("GClstr", "Cluster of galaxies"),
    ("GGroup", "Group of galaxies"),
    ("GPair", "Galaxy pair"),
    ("GTrpl", "Galaxy triple"),
    ("G_Lens", "Lensed image of a galaxy"),
    ("HII", "HII region"),
    ("IrS", "Infrared source"),
    ("MCld", "Molecular cloud"),
    ("Neb", "Nebula"),
    ("Nova", "Nova"),
    ("Other", "Other classification (e.g. comet; plate defect)"),
    ("PN", "Planetary nebula"),
    ("PofG", "Part of galaxy"),
    ("Psr", "Pulsar"),
    ("QGroup", "Group of QSOs"),
    ("QSO", "Quasi-stellar object"),
    ("Q_Lens", "Lensed image of a QSO"),
    ("RadioS", "Radio source"),
    ("Red*", "Red star"),
    ("RfN", "Reflection nebula"),
    ("SN", "Supernova"),
    ("SNR", "Supernova remnant"),
    ("UvES", "Ultraviolet excess source"),
    ("UvS", "Ultraviolet source"),
    ("V*", "Variable star"),
    ("VisS", "Visual source"),
    ("WD*", "White dwarf"),
    ("WR*", "Wolf-Rayet star"),
    ("XrayS", "X-ray source"),
    ("!*", "Galactic star"),
    ("!**", "Galactic double star"),
    ("!*Ass", "Galactic star association"),
    ("!*Cl", "Galactic Star cluster"),
    ("!Blue*", "Galactic blue star"),
    ("!C*", "Galactic carbon star"),
    ("!EmObj", "Galactic emission line object"),
    ("!Flar*", "Galactic flare star"),
    ("!HII", "Galactic HII region"),
    ("!MCld", "Galactic molecular cloud"),
    ("!Neb", "Galactic nebula"),
    ("!Nova", "Galactic nova"),
    ("!PN", "Galactic planetary nebula"),
    ("!Psr", "Galactic pulsar"),
    ("!RfN", "Galactic reflection nebula"),
    ("!Red*", "Galactic red star"),
    ("!SN", "Galactic supernova"),
    ("!SNR", "Galactic supernova remnant"),
    ("!V*", "Galactic variable star"),
    ("!WD*", "Galactic white dwarf"),
    ("!WR*", "Galactic Wolf-Rayet star"),
]
NED_Z_CHOICES = [
    ("-", "usually a reliable spectroscopic value"),
    (":", "an uncertain value"),
    ("::", "a highly uncertain value"),
    ("?", "a very uncertain (questionable) value"),
    ("1LIN", "a spectroscopic value from a single line, assuming the line is known"),
    ("AVGnn", "average, based on nn measurements"),
    (
        "CONT",
        "continuum, based on Balmer/4000A break (e.g., Kriek et al. 2008ApJ...677..219K)",
    ),
    ("EST", "an estimated value"),
    ("FoF", "Friends-of-Friends (velocity of near neighbor)"),
    (
        "LUM",
        "estimated from assumed luminosity for a brightest cluster galaxy (Nelson et al. ApJ 563, 629, 2001)",
    ),
    (
        "MFA",
        "a value from a matched filter algorithm (see Postman et al. AJ 111, 615, 1996)",
    ),
    ("MOD", "a modelled value"),
    ("PAH", "redshift determined from PAH features"),
    ("PHOT", "estimated using photometry"),
    (
        "PEAK",
        "determined from peak of Gaussian distribution (e.g., Krick et al. 2009ApJ...700..123K)",
    ),
    ("PRED", "a predicted value"),
    ("SED", "a value from a spectral energy distribution"),
    ("SPEC", "an explicitly declared spectroscopic value"),
    ("TENT", "a tentative value"),
    (
        "TOMO",
        "a tomographic redshift for a lensing object (see e.g. Hennawi and Spergel ApJ 624, 59, 2005)",
    ),
    (
        "SN",
        "the redshift of a host galaxy determined from the expansion velocity of a supernova",
    ),
]

# List of photometric redshift sources since NED doesn't categorize them well.
# NedEntry.objects.filter(zflag='PHOT').values_list('redshifts__refcode', flat=True).distinct()
PHOTO_Z_REFCODES = [
    "2002PASJ...54..661T",  # Properties of Spiral-Peculiar Type of Kiso Ultraviolet-Excess Galaxies
    "2009ApJS..180...67R",  # Efficient Photometric Selection of Quasars from the Sloan Digital Sky Survey. II. ~1,000,000 Quasars from Data Release 6
    "2004A&A...421..913W",  # A catalogue of the Chandra Deep Field South with multi-colour classification and photometric redshifts from COMBO-17
    "2010ApJS..187..272W",  # Erratum: "Galaxy Clusters Identified from the Sloan Digital Sky Survey DR6 and their Properties" (2009, ApJS, 183, 197)
    "2012ApJS..199...34W",  # A Catalog of 132,684 Clusters of Galaxies Identified from Sloan Digital Sky Survey III
    "2013MNRAS.436..275W",  # Substructure and dynamical state of 2092 rich clusters of galaxies derived from photometric data
    "2014ApJS..210....9B",  # Two Micron All Sky Survey Photometric Redshift Catalog: A Comprehensive Three-dimensional Census of the Whole Sky
    "2015MNRAS.453...38R",  # redMaPPer - IV. Photometric membership identification of red cluster galaxies with 1 per cent precision
    "2015ApJ...807..178W",  # Calibration of the Optical Mass Proxy for Clusters of Galaxies and an Update of the WHL12 Cluster Catalog
]
SALT_FAIL_LIMITS = {"x0_mag": 20, "x1": 10, "c": 10}
OT_SKIP = [
    "23ixf",
]
DERIV_TOLS = {
    "dmu/dP": (np.e ** (-0.2), np.e**0.2),
    "dmB/dP": (np.e ** (-0.2), np.e**0.2),
    "ds/dP": (-0.2, 0.2),
    "dc/dP": (-0.2, 0.2),
}
DERIV_TOLS_STRICT = {
    "dmu/dP": (0.99, 1.015),
    "dmB/dP": (0.99, 1.01),
    "ds/dP": (-0.05, 0.05),
    "dc/dP": (-0.007, 0.002),
}
STANDARD_SALT_CUTS = {
    "e_x1_floor": 0.1,
    # "x1+err": 5,
    "x1": 3,
    "e_x1": 1.5,
    "c": 0.3,
    "e_c": 0.2,
}
# Latest first epoch, Earliest last epoch, Minimum phase coverage
ACCEPTABLE_PHASE_COVERAGES = ((2, 8, 10), (6, 8, 15))
STANDARD_CHI2_CUTS = {  # based on comparison to DEHVILS sample.
    "salt3-nir": 1.304,
    "snpy_ebv_model2": 4.136,
    "snpy_max_model": 4.474,
    "other": 2.0,
}
STANDARD_CUTS = {
    "mwebv": 0.3,
    # "david-salt3-nir-derivs": "standard",
    "min_bandpasses": 2,
    "min_obs_num": 5,
    "salt": "standard",
    "EBVhost": 0.3,
    # "st": (0.75, 1.18),  # "tight" cut from jones 19
    "st": (0.6, 1.3),  # "loose" cut from jones 19
    "e_st": 0.2,  # jones 19
    "phase": "standard",
    "chi2_reduced_data_model": "standard",
}
SCOLNIC18_CUTS = {  # 2018ApJ...859..101S
    "x1": (-3, 3),
    "c": (-0.3, 0.3),
    # and more
}
JONES_CUTS = {  # 2022ApJ...933..172J
    "EBVhost": (-0.5, 0.3),
    "st": (0.75, 1.18),
    "e_st": 0.2,
}
AVELINO_CUTS = {  # 2019ApJ...887..106A
    "dm15": (0.8, 1.6),
    "ebvhost": (-0.15, 0.4),
    "mwebv": 1,
    "z": (0.0, 0.04),
    "min_obs_num": 3,
}
FOUNDATION_CUTS = {  # not Iax, 91bg, 00cx, 06gz
    "mwebv": 0.25,
    "min_obs_num": 11,  # in PS1 gri
    "e_t0": (0, 1),
    "e_Tmax": (0, 1),
    "e_x1": (0, 1),
    "c": (-0.3, 0.3),
    "x1": (-3, 3),
    # Chauvenet's criterion applied to pulls (rather than residuals)
}
PANTHEON_CUTS = {
    "e_x1": (0, 1.5),
    "e_t0": (0, 2),
    "e_Tmax": (0, 2),
    "c": (-0.3, 0.3),
    "x1": (-3, 3),
    "mwebv": 0.2,
    "phase": (5, 5),  # second 5 implies no min req for last phase.
}
REVERSE_CUT_ORDER = [
    "chi2_reduced",
    "chi2_reduced_model",
    "phase",
    "e_c",
    "c",
    "x1+err",
    "e_EBVhost",
    "EBVhost",
    "e_st",
    "st",
    "e_Tmax",
    "Tmax",
    "e_t0",
    "t0",
    "min_obs_num",
    "mwebv",
]
DARK_DICT = {
    "lines.color": "white",
    "patch.edgecolor": "white",
    "text.color": "white",
    "axes.facecolor": "black",
    "axes.edgecolor": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "white",
    "figure.facecolor": "black",
    "figure.edgecolor": "black",
    "savefig.facecolor": "black",
    "savefig.edgecolor": "black",
}
EFF_WL_DICT = {
    "asg": 4750.823082638873,
    "ztfg": 4862.08059712661,
    "c": 5487.402789822219,
    "ztfr": 6462.52471576389,
    "o": 6943.7318479288815,
    "ztfi": 7916.761996909543,
    "Y": 10327.936846349896,
    "J": 12528.070400046909,
    "H": 16422.882780242744,
}
