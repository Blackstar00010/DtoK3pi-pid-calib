from src.py.utils.config import cut_method


class LegacyStats:
    def __init__(self, dmmean, dmwidtho, dmwidthi, deltammean, deltamwidtho, deltamwidthi, dmmin, dmmax, deltammin, deltammax):
        self.dmmean = dmmean
        self.dmwidtho = dmwidtho
        self.dmwidthi = dmwidthi
        self.deltammean = deltammean
        self.deltamwidtho = deltamwidtho
        self.deltamwidthi = deltamwidthi
        self.dmmin = dmmin
        self.dmmax = dmmax
        self.deltammin = deltammin


class Stats:
    def __init__(self, dmmean, deltammean, dmstd, deltamstd, default_factor):
        self.dmmean = dmmean
        self.deltammean = deltammean
        self.dmstd = dmstd
        self.deltamstd = deltamstd
        self.std_factor = default_factor
        self.dmradius = self.dmstd * self.std_factor
        self.deltamradius = self.deltamstd * self.std_factor

    def dmmin(self, factor=None):
        factor = self.std_factor if factor is None else factor
        return self.dmmean - factor * self.dmstd

    def dmmax(self, factor=None):
        factor = self.std_factor if factor is None else factor
        return self.dmmean + factor * self.dmstd

    def deltammin(self, factor=None):
        factor = self.std_factor if factor is None else factor
        return self.deltammean - factor * self.deltamstd

    def deltammax(self, factor=None):
        factor = self.std_factor if factor is None else factor
        return self.deltammean + factor * self.deltamstd

    def get(self, name):
        """Get the value of the attribute with the given name. NOT RECOMMENDED"""
        name = name.lower().strip()
        if name == 'dmmean':
            return self.dmmean
        elif name == 'deltammean':
            return self.deltammean
        elif name == 'dmstd':
            return self.dmstd
        elif name == 'deltamstd':
            return self.deltamstd
        elif name == 'dmradius':
            return self.dmradius
        elif name == 'deltamradius':
            return self.deltamradius
        elif name == 'std_factor':
            return self.std_factor
        elif name == 'dmmin':
            return self.dmmin()
        elif name == 'dmmax':
            return self.dmmax()
        elif name == 'deltammin':
            return self.deltammin()
        elif name == 'deltammax':
            return self.deltammax()
        else:
            raise ValueError(f"Unknown name: {name}")


# used for intro part
prototype = LegacyStats(1862, 40, 10, 145.35, 3.5, 1, 1800, 1915, 139, 156)
# prototype = LegacyStats(1862, 60, 10, 145.35, 5, 1, 1800, 1920, 138, 155)

# used for the mvp of this project
sneha_stats = Stats((1820 + 1900) / 2, (139 + 156) / 2, (1900 - 1820) / 2, (156 - 139) / 2, 1)

# calculated from the MC file; see `concat_mc_bkg` in `dataprep.py`
cut_std = 6 if cut_method == 'ellipse' else 5
mc_stats = Stats(1865.4285, 145.4904, 7.5276, 0.7585, cut_std)

train_bkgsig_ratios = [1, 2, 5, 10, 20, 50, 100, 'all']  # ratios of signal to background in training

if __name__ == "__main__":
    print('consts.py is not supposed to be executed')
    raise SystemExit(0)
