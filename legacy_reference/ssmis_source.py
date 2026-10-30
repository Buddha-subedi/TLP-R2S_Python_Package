"""Match original F16 SSMIS XCAL NetCDF granules to X_ERA5 MATLAB orbits.

The X_ERA5 date may differ from the source granule's START date near UTC
midnight. Match the numeric six-digit granule identifier, never a substring.
"""
from pathlib import Path
import re

SOURCE_RE = re.compile(
    r'^1C\.F16\.SSMIS\.XCAL2021-V\.(?P<date>\d{8})-S(?P<start>\d{6})-E(?P<end>\d{6})\.(?P<orbit>\d{6})\.V\w+\.nc$',
    re.IGNORECASE,
)
INPUT_RE = re.compile(r'^X_ERA5_(\d{8})_(\d+)$', re.IGNORECASE)

class SourceIndex:
    def __init__(self, root):
        self.root = Path(root)
        self.cache = {}

    def _year(self, year):
        year = str(year)
        if year not in self.cache:
            folder = self.root / year
            if not folder.is_dir():
                self.cache[year] = []
                return []
            matches = []
            for p in folder.rglob('*'):
                if p.is_file() and p.suffix.lower() == '.nc':
                    m = SOURCE_RE.fullmatch(p.name)
                    if m:
                        matches.append((p, m.groupdict()))
            self.cache[year] = matches
            print(f'Indexed {len(matches)} original SSMIS NetCDF files in {folder}', flush=True)
        return self.cache[year]

    def find(self, year, date, orbit):
        year = int(year)
        orbit_num = int(orbit)
        # A granule can cross midnight/year boundaries; source filename date
        # may be the preceding day/year relative to X_ERA5 filename date.
        candidates = [entry for y in (year - 1, year, year + 1)
                      for entry in self._year(y) if int(entry[1]['orbit']) == orbit_num]
        if len(candidates) != 1:
            raise FileNotFoundError(
                f'Expected exactly one F16 original SSMIS .nc granule for '
                f'X_ERA5 date={date}, orbit={orbit} (searched {year-1}-{year+1}); '
                f'found {len(candidates)}: {[p.name for p, _ in candidates[:8]]}'
            )
        return candidates[0][0]

def identify(path):
    m = INPUT_RE.fullmatch(Path(path).stem)
    if not m:
        raise ValueError(f'Cannot derive date and orbit from {path}')
    return m.groups()

def output_paths(row, out, index):
    date, orbit = identify(row['path'])
    source = index.find(row['year'], date, orbit)
    stem = 'TLP-R2S_V01_' + source.stem
    base = Path(out) / row['year']
    return (source, base / 'netcdf' / (stem + '.nc'),
            base / 'plots' / (Path(row['path']).stem + '.png'),
            base / 'contributions' / (Path(row['path']).stem + '.npz'))
