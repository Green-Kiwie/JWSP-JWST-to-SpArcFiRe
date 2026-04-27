#!/usr/bin/env python3
"""Test current filters against 1-galaxies and 1-not-galaxies test sets.

Loads FITS thumbnails and runs multi_scale_structure_score() and
thumbnail_has_galaxy_profile() on each, reporting pass/fail and diagnostics.
"""
import sys
import os
import numpy as np
from astropy.io import fits

# Add modules path
sys.path.insert(0, '/extra/wayne2/preserve/nntran5/JWSP-JWST-to-SpArcFiRe/src/modules')
import sep_helpers

THUMB_DIR = '/extra/wayne2/preserve/nntran5/JWSP-JWST-to-SpArcFiRe/outputs/JWST_galaxy_thumbnails'

# Galaxy names from predict.csv mapping (101 galaxies)
GALAXIES = [
    '104_616260_-55_949604_f150w_v2',
    '110_818138_-73_451785_f200w_v1',
    '110_840510_-73_440256_f150w_v1',
    '11_422268_38_022426_f200w_v1',
    '11_654237_-25_484035_f150w_v3',
    '124_083173_19_259218_f200w_v1',
    '124_117257_19_222105_f200w_v1',
    '124_119094_19_212782_f380m_v1',
    '12_884522_29_678550_f200w_v1',
    '149_354466_2_555925_f200w_v1',
    '149_982285_-22_812877_f200w_v1',
    '152_671828_-4_768257_f200w_v1',
    '159_185206_-2_465198_f150w_v1',
    '15_024157_-33_722791_f200w_v1',
    '15_024288_-33_722973_f200w_v1',
    '15_078888_-33_723005_f200w_v1',
    '162_071090_-1_142393_f356w_v1',
    '162_092220_-1_153939_f356w_v1',
    '166_077106_21_587467_f356w_v1',
    '167_645920_-13_485037_f444w_v1',
    '172_934687_-12_346196_f150w_v1',
    '177_251337_22_438931_f200w_v1',
    '177_254392_22_429935_f200w_v1',
    '177_255082_22_431738_f200w_v1',
    '177_371068_22_400098_f480m_v1',
    '177_399552_22_398072_f115w_v2',
    '177_408594_22_380119_f430m_v1',
    '177_462886_22_299616_f200w_v1',
    '183_115098_27_474327_f200w_v1',
    '186_698228_21_715675_f200w_v1',
    '187_451287_27_125852_f277w_v4',
    '188_959957_4_933565_f380m_v1',
    '188_964963_4_931690_f380m_v1',
    '188_978811_4_927904_f480m_v2',
    '188_991180_4_932818_f200w_v2',
    '190_718467_13_266453_f115w_v1',
    '191_441038_3_552940_f200w_v1',
    '216_072410_24_226635_f150w_v1',
    '216_584686_35_170436_f200w_v1',
    '216_623743_35_181265_f200w_v1',
    '216_648320_35_164363_f200w_v1',
    '218_144041_-44_192128_f115w_v1',
    '218_164540_-44_185065_f115w_v1',
    '247_886457_30_154626_f090w_v2',
    '247_910040_30_154704_f090w_v2',
    '250_875430_31_143473_f150w_v1',
    '255_519434_30_382071_f150w_v1',
    '260_597319_65_814639_f200w_v1',
    '260_638783_65_876858_f200w_v1',
    '260_669117_65_887558_f200w_v1',
    '300_667176_-30_209561_f444w_v1',
    '323_069988_0_265862_f380m_v3',
    '338_224726_29_525693_f356w_v1',
    '338_227217_29_527861_f444w_v1',
    '39_959267_-1_524123_f115w_v1',
    '39_967119_-1_577046_f090w_v4',
    '39_967119_-1_577046_f150w_v2',
    '39_967119_-1_577046_f200w_v3',
    '39_967120_-1_518222_f115w_v1',
    '39_967120_-1_518222_f200w_v3',
    '39_967265_-1_577101_f115w_v2',
    '39_979490_-1_567521_f115w_v1',
    '39_983206_-1_531043_f115w_v1',
    '39_983206_-1_531043_f200w_v3',
    '39_983206_-1_589849_f150w_v2',
    '39_983206_-1_589849_f200w_v3',
    '3_533239_-30_324814_f277w_v1',
    '3_535175_-30_388682_f277w_v1',
    '3_536087_-30_324833_f090w_v1',
    '3_538429_-30_325653_f277w_v1',
    '3_540621_-30_332301_f277w_v1',
    '3_541129_-30_325672_f090w_v1',
    '3_543321_-30_332302_f090w_v1',
    '3_559632_-30_330029_f277w_v1',
    '3_571422_-30_379850_f277w_v5',
    '3_586685_-30_361353_f277w_v1',
    '3_589063_-30_408681_f115w_v1',
    '3_611493_-30_407242_f115w_v3',
    '40_989300_-50_146519_f356w_v1',
    '40_992711_-50_144352_f444w_v1',
    '46_328820_-31_857922_f356w_v1',
    '46_331393_-31_855791_f444w_v1',
    '53_140946_-27_780941_f115w_v8',
    '54_729199_-35_591641_f090w_v1',
    '56_153401_32_187583_f277w_v2',
    '64_038563_-24_021311_f200w_v2',
    '64_328246_-11_908666_f200w_v3',
    '64_346658_-11_900000_f115w_v2',
    '64_346658_-11_900000_f150w_v1',
    '64_382662_-11_846235_f150w_v2',
    '64_401981_-11_900004_f115w_v1',
    '64_401981_-11_900004_f150w_v2',
    '64_404494_-11_904904_f115w_v1',
    '64_412087_-11_867181_f200w_v3',
    '80_455658_-69_517573_f480m_v5',
    '80_476672_-69_506460_f430m_v7',
    '80_477527_-69_504552_f200w_v1',
    '80_477761_-69_481396_f480m_v8',
    '80_520120_-69_498430_f356w_v2',
    '8_900734_36_516855_f200w_v1',
    '9_284636_44_303928_f200w_v1',
]

NOT_GALAXIES = [
    '124_109895_19_241581_f200w_v1',
    '172_939927_-12_322756_f150w_v1',
    '177_257411_22_377413_f200w_v1',
    '177_372070_22_398671_f200w_v1',
    '177_399052_22_379459_f200w_v1',
    '187_436716_27_137415_f430m_v2',
    '187_454274_27_138363_f277w_v1',
    '187_459656_27_111975_f277w_v1',
    '187_464118_27_134211_f380m_v1',
    '187_466818_27_140529_f430m_v1',
    '187_475185_27_125177_f430m_v1',
    '188_960263_4_934021_f200w_v2',
    '188_978811_4_927904_f200w_v2',
    '188_988942_4_924429_f200w_v2',
    '188_989001_4_930048_f200w_v2',
    '239_807068_47_603338_f115w_v1',
    '239_807985_47_597863_f140m_v1',
    '245_949868_-26_442991_f150w_v1',
    '252_979883_-46_154449_f200w_v1',
    '252_980047_-46_145470_f356w_v3',
    '252_981802_-46_154505_f200w_v2',
    '253_009385_-46_153327_f356w_v3',
    '266_144893_-29_483359_f356w_v1',
    '266_164350_-29_449835_f480m_v1',
    '266_180185_-29_484745_f158m_v1',
    '266_184054_-29_473053_f480m_v2',
    '300_669176_-30_200573_f200w_v1',
    '323_070152_0_254260_f150w_v5',
    '334_420119_-16_495169_f277w_v1',
    '334_431680_-16_483229_f277w_v1',
    '334_448271_-16_493568_f277w_v1',
    '345_880450_-62_907378_f200w_v1',
    '345_893248_-62_933814_f158m_v1',
    '345_926035_-62_889299_f200w_v1',
    '345_945125_-62_900798_f158m_v1',
    '345_977432_-62_892698_f158m_v1',
    '39_959267_-1_524123_f200w_v3',
    '39_967265_-1_577101_f200w_v3',
    '39_979490_-1_567521_f200w_v3',
    '3_531318_-30_373875_f277w_v1',
    '3_559632_-30_330029_f090w_v1',
    '52_150388_31_428894_f200w_v1',
    '52_233507_31_369178_f150w_v1',
    '53_140946_-27_780941_f200w_v8',
    '64_328246_-11_908666_f115w_v2',
    '64_382662_-11_846235_f200w_v3',
    '64_404494_-11_904904_f200w_v3',
    '80_392693_-69_473251_f356w_v7',
    '80_399205_-69_501003_f090w_v2',
    '80_404693_-69_478843_f158m_v3',
    '80_408790_-69_481494_f356w_v3',
    '80_409167_-69_464574_f158m_v7',
    '80_419722_-69_492063_f277w_v3',
    '80_435791_-69_481614_f200w_v2',
    '80_435799_-69_477363_f115w_v2',
    '80_436752_-69_516877_f200w_v1',
    '80_445052_-69_487185_f158m_v4',
    '80_445224_-69_503437_f150w_v1',
    '80_445664_-69_492922_f200w_v2',
    '80_448707_-69_477556_f277w_v5',
    '80_452758_-69_492665_f150w_v6',
    '80_455143_-69_505546_f480m_v4',
    '80_455421_-69_502580_f115w_v13',
    '80_457940_-69_480709_f200w_v5',
    '80_458419_-69_515970_f200w_v6',
    '80_462685_-69_485541_f277w_v8',
    '80_464374_-69_497375_f430m_v7',
    '80_466266_-69_506389_f150w_v2',
    '80_466862_-69_504297_f150w_v1',
    '80_466932_-69_487431_f150w_v3',
    '80_467008_-69_464336_f277w_v5',
    '80_467120_-69_507011_f200w_v3',
    '80_467920_-69_484384_f115w_v4',
    '80_468639_-69_496246_f200w_v1',
    '80_473466_-69_484349_f158m_v6',
    '80_476430_-69_495583_f140m_v4',
    '80_476579_-69_493014_f090w_v2',
    '80_477761_-69_481396_f158m_v5',
    '80_479457_-69_481834_f380m_v7',
    '80_480859_-69_495756_f150w_v3',
    '80_481527_-69_498994_f277w_v8',
    '80_481534_-69_497395_f150w_v1',
    '80_482315_-69_495719_f150w_v1',
    '80_484168_-69_493879_f150w_v4',
    '80_484273_-69_484482_f200w_v1',
    '80_486891_-69_488088_f140m_v3',
    '80_488489_-69_512683_f150w_v3',
    '80_489231_-69_514479_f200w_v7',
    '80_490510_-69_488216_f200w_v1',
    '80_493657_-69_507762_f158m_v5',
    '80_494325_-69_490802_f277w_v1',
    '80_494436_-69_515699_f150w_v3',
    '80_495673_-69_481240_f200w_v1',
    '80_501138_-69_495755_f140m_v1',
    '80_502629_-69_492685_f140m_v3',
    '80_504924_-69_482678_f150w_v4',
    '80_506388_-69_506031_f200w_v3',
    '80_507908_-69_513631_f200w_v8',
    '80_510314_-69_491597_f158m_v4',
    '80_510593_-69_488305_f200w_v5',
    '80_511406_-69_481243_f200w_v7',
    '80_515333_-69_496573_f380m_v3',
    '80_518841_-69_513493_f200w_v1',
    '80_519067_-69_489032_f150w_v4',
    '80_519389_-69_507756_f200w_v1',
    '80_520679_-69_498353_f200w_v2',
    '80_528927_-69_480440_f200w_v2',
    '80_531752_-69_497459_f158m_v4',
    '80_532477_-69_462857_f200w_v4',
    '80_538027_-69_491797_f200w_v1',
    '80_542522_-69_514867_f150w_v4',
    '80_542989_-69_471920_f380m_v8',
    '80_997577_-70_065538_f200w_v1',
    '81_043872_-70_100762_f200w_v4',
    '81_062330_-70_081816_f090w_v3',
    '81_072599_-70_068528_f115w_v4',
    '81_077793_-70_094280_f150w_v4',
    '81_085026_-70_082100_f115w_v1',
    '81_106901_-70_092988_f200w_v1',
    '81_118399_-70_091111_f158m_v2',
    '81_140519_-70_061691_f115w_v2',
    '82_783841_-68_796817_f430m_v1',
    '82_798141_-68_782591_f200w_v1',
    '9_329921_1_415021_f480m_v1',
    '110_818138_-73_451785_f150w_v1',
    '110_840510_-73_440256_f200w_v1',
    '124_117257_19_222105_f150w_v1',
    '149_354466_2_555925_f150w_v1',
    '149_354466_2_555925_f380m_v1',
    '149_982285_-22_812877_f150w_v1',
    '152_671828_-4_768257_f150w_v1',
    '159_185206_-2_465198_f200w_v1',
    '166_077106_21_587467_f200w_v1',
    '172_934687_-12_346196_f200w_v1',
    '177_254392_22_429935_f150w_v1',
    '177_371068_22_400098_f200w_v1',
    '183_115098_27_474327_f150w_v1',
    '186_698228_21_715675_f150w_v1',
    '188_959957_4_933565_f200w_v1',
    '191_441038_3_552940_f150w_v1',
    '216_072410_24_226635_f200w_v1',
    '216_584686_35_170436_f150w_v1',
    '216_623743_35_181265_f150w_v1',
    '216_648320_35_164363_f150w_v1',
    '218_144041_-44_192128_f200w_v1',
    '218_164540_-44_185065_f200w_v1',
    '247_886457_30_154626_f150w_v2',
    '247_910040_30_154704_f150w_v2',
    '250_875430_31_143473_f200w_v1',
    '255_519434_30_382071_f200w_v1',
    '260_597319_65_814639_f150w_v1',
    '260_638783_65_876858_f150w_v1',
    '260_669117_65_887558_f150w_v1',
    '300_667176_-30_209561_f200w_v1',
    '338_224726_29_525693_f200w_v1',
    '338_227217_29_527861_f200w_v1',
    '39_959267_-1_524123_f200w_v1',
    '39_967119_-1_577046_f115w_v2',
    '39_967120_-1_518222_f090w_v4',
    '39_967120_-1_518222_f150w_v2',
    '39_979490_-1_567521_f200w_v1',
    '39_983206_-1_531043_f090w_v4',
    '39_983206_-1_531043_f150w_v2',
    '39_983206_-1_589849_f115w_v1',
    '3_533239_-30_324814_f090w_v1',
    '3_535175_-30_388682_f090w_v1',
    '3_536087_-30_324833_f277w_v1',
    '3_538429_-30_325653_f090w_v1',
    '3_540621_-30_332301_f090w_v1',
    '3_541129_-30_325672_f277w_v1',
    '3_543321_-30_332302_f277w_v1',
    '3_559632_-30_330029_f200w_v1',
    '3_571422_-30_379850_f090w_v5',
    '3_586685_-30_361353_f090w_v1',
    '3_589063_-30_408681_f200w_v1',
    '3_611493_-30_407242_f200w_v3',
    '40_989300_-50_146519_f200w_v1',
    '40_992711_-50_144352_f200w_v1',
    '46_328820_-31_857922_f200w_v1',
    '46_331393_-31_855791_f200w_v1',
    '53_140946_-27_780941_f090w_v8',
    '54_729199_-35_591641_f200w_v1',
    '56_153401_32_187583_f090w_v2',
    '64_038563_-24_021311_f150w_v2',
    '64_328246_-11_908666_f150w_v3',
    '64_346658_-11_900000_f200w_v1',
    '64_401981_-11_900004_f200w_v1',
    '64_412087_-11_867181_f150w_v3',
    '80_455658_-69_517573_f200w_v5',
    '80_476672_-69_506460_f200w_v7',
    '80_477527_-69_504552_f150w_v1',
    '80_520120_-69_498430_f200w_v2',
    '8_900734_36_516855_f150w_v1',
    '9_284636_44_303928_f150w_v1',
    '11_422268_38_022426_f150w_v1',
    '11_654237_-25_484035_f200w_v3',
    '12_884522_29_678550_f150w_v1',
    '15_024157_-33_722791_f150w_v1',
    '15_024288_-33_722973_f150w_v1',
    '15_078888_-33_723005_f150w_v1',
]


def name_to_fits_path(name):
    """Convert underscore galaxy name to FITS path.
    e.g. 218_144041_-44_192128_f115w_v1 -> 218.144041_-44.192128_f115w_v1.fits
    """
    parts = name.split('_')
    # Find the filter part (starts with 'f')
    filter_idx = None
    for i, p in enumerate(parts):
        if p.startswith('f') and any(c.isdigit() for c in p):
            filter_idx = i
            break
    if filter_idx is None:
        raise ValueError(f"Cannot parse filter from name: {name}")
    
    # Coordinate parts are before the filter
    coord_parts = parts[:filter_idx]
    filter_parts = parts[filter_idx:]
    
    # Group coordinate parts into RA and DEC
    # Pattern: ra_radec_dec_decdec or with negatives
    # e.g. ['218', '144041', '-44', '192128']
    # -> RA = 218.144041, DEC = -44.192128
    # Find where DEC starts (first negative or third element)
    dec_start = None
    for i in range(1, len(coord_parts)):
        if coord_parts[i].startswith('-'):
            dec_start = i
            break
    if dec_start is None:
        # No negative - DEC starts at index 2
        dec_start = 2
    
    ra_str = '.'.join(coord_parts[:dec_start])
    dec_str = '.'.join(coord_parts[dec_start:])
    # Fix double negative: if dec has -.XX format
    if dec_str.startswith('-.'):
        dec_str = '-' + dec_str[2:]
    
    fits_name = f"{ra_str}_{dec_str}_{'_'.join(filter_parts)}.fits"
    return os.path.join(THUMB_DIR, fits_name)


def load_thumbnail(fits_path):
    """Load FITS thumbnail as numpy array."""
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(np.float64)
    data = np.nan_to_num(data, nan=0.0)
    return data


def analyze_thumbnail(name, data):
    """Run all filter checks and return diagnostics."""
    result = {}
    result['shape'] = data.shape
    result['min'] = float(np.min(data))
    result['max'] = float(np.max(data))
    result['mean'] = float(np.mean(data))
    
    # Multi-scale structure score
    score = sep_helpers.multi_scale_structure_score(data)
    result['structure'] = score
    
    # Galaxy profile check
    profile_pass = sep_helpers.thumbnail_has_galaxy_profile(data)
    result['profile_pass'] = profile_pass
    
    # Get more detailed profile diagnostics
    d = data.astype(np.float64)
    h, w = d.shape
    cy, cx = h / 2.0, w / 2.0
    total_flux = np.sum(np.abs(d))
    
    y_idx, x_idx = np.ogrid[:h, :w]
    r = np.sqrt((x_idx - cx)**2 + (y_idx - cy)**2)
    r_max = np.sqrt(cx**2 + cy**2)
    
    # Interior zero fraction
    inner_mask = r < r_max * 0.5
    n_inner = int(np.sum(inner_mask))
    if n_inner > 0:
        result['interior_zero_frac'] = float(np.sum(d[inner_mask] == 0)) / n_inner
    else:
        result['interior_zero_frac'] = 0.0
    
    # Quadrant fluxes
    icy, icx = int(cy), int(cx)
    q_flux = [
        np.sum(np.abs(d[:icy, :icx])),
        np.sum(np.abs(d[:icy, icx:])),
        np.sum(np.abs(d[icy:, :icx])),
        np.sum(np.abs(d[icy:, icx:])),
    ]
    min_q = min(q_flux)
    max_q = max(q_flux)
    result['quadrant_ratio'] = max_q / min_q if min_q > 1e-12 else float('inf')
    result['max_quadrant_frac'] = max_q / total_flux if total_flux > 1e-12 else 0.0
    
    # Radial profile bins
    n_bins = 8
    bin_edges = np.linspace(0, r_max, n_bins + 1)
    bin_flux = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        if np.any(mask):
            bin_flux[i] = np.sum(np.abs(d[mask]))
    
    result['center_fraction'] = bin_flux[0] / total_flux if total_flux > 1e-12 else 0.0
    result['outer_fraction'] = float(np.sum(bin_flux[4:])) / total_flux if total_flux > 1e-12 else 0.0
    
    # Additional diagnostics inspired by COSMOS2025
    # Concentration: flux in r<3px / flux in r<8px 
    mask_r3 = r < 3
    mask_r8 = r < 8
    flux_r3 = np.sum(np.abs(d[mask_r3]))
    flux_r8 = np.sum(np.abs(d[mask_r8]))
    result['concentration_r3_r8'] = flux_r3 / flux_r8 if flux_r8 > 1e-12 else 0.0
    
    # Half-light radius estimate
    sorted_r = np.sort(r.ravel())
    sorted_flux = np.abs(d).ravel()[np.argsort(r.ravel())]
    cumflux = np.cumsum(sorted_flux)
    if cumflux[-1] > 1e-12:
        half_idx = np.searchsorted(cumflux, 0.5 * cumflux[-1])
        result['half_light_radius'] = float(sorted_r[min(half_idx, len(sorted_r)-1)])
    else:
        result['half_light_radius'] = 0.0
    
    # Number of bright peaks (sources)
    from scipy import ndimage
    threshold = np.mean(d) + 3 * np.std(d)
    binary = d > threshold
    labeled, n_sources = ndimage.label(binary)
    result['n_bright_sources'] = n_sources
    
    # Peak offset from center
    peak_y, peak_x = np.unravel_index(np.argmax(d), d.shape)
    result['peak_offset'] = np.sqrt((peak_x - cx)**2 + (peak_y - cy)**2)
    result['peak_offset_frac'] = result['peak_offset'] / r_max
    
    # Clipped core check details
    cs = 2
    if icy >= cs and icx >= cs and icy + cs < h and icx + cs < w:
        center_patch = d[icy - cs:icy + cs + 1, icx - cs:icx + cs + 1]
        img_min = np.min(d)
        img_range = np.max(d) - img_min
        if img_range > 1e-12:
            floor_thresh = img_min + 0.05 * img_range
            n_floor_center = int(np.sum(center_patch <= floor_thresh))
            result['n_floor_center'] = n_floor_center
            ring_mask = (r >= 3) & (r < 7)
            if np.any(ring_mask):
                ring_mean = float(np.mean(d[ring_mask]))
                result['ring_mean'] = ring_mean
                result['floor_thresh'] = floor_thresh
                result['ring_to_floor'] = ring_mean / floor_thresh if floor_thresh > 0 else 0.0
    
    # Overall pass/fail
    result['structure_pass'] = score['is_galaxy_like']
    result['overall_pass'] = score['is_galaxy_like'] and profile_pass
    
    return result


def print_results(name, result, expected_pass):
    """Print formatted results."""
    actual = 'PASS' if result['overall_pass'] else 'FAIL'
    expected = 'PASS' if expected_pass else 'FAIL'
    correct = '✓' if actual == expected else '✗ WRONG'
    
    struct_pass = 'PASS' if result['structure_pass'] else 'FAIL'
    prof_pass = 'PASS' if result['profile_pass'] else 'FAIL'
    
    s = result['structure']
    print(f"\n{'='*80}")
    print(f"  {name}  [{actual}]  expected={expected}  {correct}")
    print(f"  shape={result['shape']}  struct={struct_pass}  profile={prof_pass}")
    print(f"  e1={s['e1']:.4f}  e4={s['e4']:.4f}  e16={s['e16']:.4f}  "
          f"c/f={s['coarse_to_fine']:.4f}  dir={s['directional_ratio']:.2f}  "
          f"sat={s['saturated_fraction']:.4f}")
    print(f"  center_frac={result['center_fraction']:.4f}  outer_frac={result['outer_fraction']:.4f}  "
          f"quad_ratio={result['quadrant_ratio']:.2f}  max_q_frac={result['max_quadrant_frac']:.4f}")
    print(f"  conc_r3r8={result['concentration_r3_r8']:.4f}  hlr={result['half_light_radius']:.2f}  "
          f"n_sources={result['n_bright_sources']}  peak_off={result['peak_offset']:.2f} ({result['peak_offset_frac']:.3f})")
    print(f"  int_zero={result['interior_zero_frac']:.4f}", end='')
    if 'n_floor_center' in result:
        print(f"  floor_center={result['n_floor_center']}  ring/floor={result.get('ring_to_floor', 0):.2f}", end='')
    print()


def main():
    print("="*80)
    print("  FILTER TEST: 1-galaxies (should PASS) and 1-not-galaxies (should FAIL)")
    print("="*80)
    
    galaxy_results = []
    not_galaxy_results = []
    
    print("\n" + "~"*80)
    print("  GALAXIES (expected: PASS)")
    print("~"*80)
    
    for name in GALAXIES:
        fits_path = name_to_fits_path(name)
        if not os.path.exists(fits_path):
            print(f"\n  MISSING: {name} -> {fits_path}")
            continue
        data = load_thumbnail(fits_path)
        result = analyze_thumbnail(name, data)
        galaxy_results.append((name, result))
        print_results(name, result, expected_pass=True)
    
    print("\n" + "~"*80)
    print("  NOT-GALAXIES (expected: FAIL)")
    print("~"*80)
    
    for name in NOT_GALAXIES:
        fits_path = name_to_fits_path(name)
        if not os.path.exists(fits_path):
            print(f"\n  MISSING: {name} -> {fits_path}")
            continue
        data = load_thumbnail(fits_path)
        result = analyze_thumbnail(name, data)
        not_galaxy_results.append((name, result))
        print_results(name, result, expected_pass=False)
    
    # Summary
    print("\n" + "="*80)
    print("  SUMMARY")
    print("="*80)
    
    gal_pass = sum(1 for _, r in galaxy_results if r['overall_pass'])
    gal_total = len(galaxy_results)
    ng_fail = sum(1 for _, r in not_galaxy_results if not r['overall_pass'])
    ng_total = len(not_galaxy_results)
    
    print(f"\n  Galaxies:     {gal_pass}/{gal_total} correctly PASS")
    print(f"  Not-galaxies: {ng_fail}/{ng_total} correctly FAIL")
    
    # List failures
    wrong_gal = [(n, r) for n, r in galaxy_results if not r['overall_pass']]
    wrong_ng = [(n, r) for n, r in not_galaxy_results if r['overall_pass']]
    
    if wrong_gal:
        print(f"\n  FALSE NEGATIVES (galaxies wrongly blocked):")
        for n, r in wrong_gal:
            print(f"    {n}  struct={r['structure_pass']}  profile={r['profile_pass']}")
    
    if wrong_ng:
        print(f"\n  FALSE POSITIVES (non-galaxies wrongly passing):")
        for n, r in wrong_ng:
            s = r['structure']
            print(f"    {n}")
            print(f"      e1={s['e1']:.4f} e16={s['e16']:.4f} c/f={s['coarse_to_fine']:.4f} "
                  f"dir={s['directional_ratio']:.2f} sat={s['saturated_fraction']:.4f}")
            print(f"      center={r['center_fraction']:.4f} outer={r['outer_fraction']:.4f} "
                  f"quad={r['quadrant_ratio']:.2f} maxq={r['max_quadrant_frac']:.4f}")
            print(f"      conc={r['concentration_r3_r8']:.4f} hlr={r['half_light_radius']:.2f} "
                  f"nsrc={r['n_bright_sources']} poff={r['peak_offset_frac']:.3f}")
    
    if not wrong_gal and not wrong_ng:
        print("\n  PERFECT: All classifications correct!")


if __name__ == '__main__':
    main()
