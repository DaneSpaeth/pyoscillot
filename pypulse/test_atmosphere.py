from astropy.io import fits

with fits.open("PHOENIX-ACES-AGSS-COND-2011_AtmosFITS_Z-0.0/lte02300-0.00-0.0.PHOENIX-ACES-AGSS-COND-2011.ATMOS.fits") as hdul:
    print(hdul.info())
    header = hdul[1].header
    data = hdul[1].data

print(len(data))
