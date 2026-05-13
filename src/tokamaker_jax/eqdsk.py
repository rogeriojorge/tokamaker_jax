"""Reusable EQDSK/gEQDSK fixed-boundary input importer."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Eqdsk:
    """Parsed fixed-boundary EQDSK/gEQDSK input."""

    path: str
    header: str
    nr: int
    nz: int
    rdim: float
    zdim: float
    rcentr: float
    rleft: float
    zmid: float
    rmaxis: float
    zmaxis: float
    simag: float
    sibry: float
    bcentr: float
    current: float
    simag2: float
    xdum: float
    rmaxis2: float
    xdum2: float
    zmaxis2: float
    xdum3: float
    sibry2: float
    xdum4: float
    xdum5: float
    fpol: np.ndarray
    pres: np.ndarray
    ffprim: np.ndarray
    pprime: np.ndarray
    psi_grid: np.ndarray
    qpsi: np.ndarray
    r_grid: np.ndarray
    z_grid: np.ndarray
    nbbbs: int
    limitr: int
    boundary: np.ndarray
    limiter: np.ndarray
    numeric_value_count: int

    def diagnostics(self) -> dict[str, Any]:
        """Return JSON-friendly parsing diagnostics and shape metadata."""

        return {
            "path": self.path,
            "valid": True,
            "header": self.header,
            "nr": self.nr,
            "nz": self.nz,
            "profile_length": int(self.fpol.shape[0]),
            "psi_shape": [int(value) for value in self.psi_grid.shape],
            "r_grid": {
                "min": float(self.r_grid[0]) if self.r_grid.size else None,
                "max": float(self.r_grid[-1]) if self.r_grid.size else None,
                "count": int(self.r_grid.size),
            },
            "z_grid": {
                "min": float(self.z_grid[0]) if self.z_grid.size else None,
                "max": float(self.z_grid[-1]) if self.z_grid.size else None,
                "count": int(self.z_grid.size),
            },
            "nbbbs": self.nbbbs,
            "limitr": self.limitr,
            "boundary_shape": [int(value) for value in self.boundary.shape],
            "limiter_shape": [int(value) for value in self.limiter.shape],
            "numeric_value_count": self.numeric_value_count,
            "warnings": self._diagnostic_warnings(),
        }

    def to_json_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable record of the parsed input."""

        return {
            "path": self.path,
            "header": self.header,
            "nr": self.nr,
            "nz": self.nz,
            "rdim": self.rdim,
            "zdim": self.zdim,
            "rcentr": self.rcentr,
            "rleft": self.rleft,
            "zmid": self.zmid,
            "rmaxis": self.rmaxis,
            "zmaxis": self.zmaxis,
            "simag": self.simag,
            "sibry": self.sibry,
            "bcentr": self.bcentr,
            "current": self.current,
            "fpol": self.fpol.tolist(),
            "pres": self.pres.tolist(),
            "ffprim": self.ffprim.tolist(),
            "pprime": self.pprime.tolist(),
            "psi_grid": self.psi_grid.tolist(),
            "qpsi": self.qpsi.tolist(),
            "r_grid": self.r_grid.tolist(),
            "z_grid": self.z_grid.tolist(),
            "nbbbs": self.nbbbs,
            "limitr": self.limitr,
            "boundary": self.boundary.tolist(),
            "limiter": self.limiter.tolist(),
            "diagnostics": self.diagnostics(),
        }

    def _diagnostic_warnings(self) -> list[str]:
        warnings = []
        if self.nbbbs == 0:
            warnings.append("no boundary coordinates present")
        if self.limitr == 0:
            warnings.append("no limiter coordinates present")
        if self.rmaxis != self.rmaxis2 or self.zmaxis != self.zmaxis2:
            warnings.append("duplicate magnetic-axis header fields differ")
        if self.simag != self.simag2 or self.sibry != self.sibry2:
            warnings.append("duplicate flux-boundary header fields differ")
        return warnings


def parse_eqdsk(path: str | Path) -> Eqdsk:
    """Parse a standard EQDSK/gEQDSK file for exact fixed-boundary ingestion."""

    eqdsk_path = Path(path)
    try:
        lines = eqdsk_path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"EQDSK file does not exist: {eqdsk_path}") from exc
    if not lines:
        raise ValueError(f"empty EQDSK file: {eqdsk_path}")

    header = lines[0].rstrip()
    dims = [int(value) for value in re.findall(r"[-+]?\d+", header)]
    if len(dims) < 2:
        raise ValueError(f"could not parse EQDSK dimensions from first line: {eqdsk_path}")
    nr, nz = dims[-2], dims[-1]
    if nr <= 0 or nz <= 0:
        raise ValueError(f"EQDSK grid dimensions must be positive, got nr={nr}, nz={nz}")

    values = np.asarray(_floating_values("\n".join(lines[1:])), dtype=np.float64)
    minimum = 20 + 5 * nr + nr * nz
    if values.size < minimum:
        raise ValueError(
            f"EQDSK file has {values.size} numeric values after header, expected at least {minimum}"
        )

    rdim, zdim, rcentr, rleft, zmid = _float_tuple(values[0:5])
    rmaxis, zmaxis, simag, sibry, bcentr = _float_tuple(values[5:10])
    current, simag2, xdum, rmaxis2, xdum2 = _float_tuple(values[10:15])
    zmaxis2, xdum3, sibry2, xdum4, xdum5 = _float_tuple(values[15:20])

    offset = 20
    fpol = values[offset : offset + nr].copy()
    offset += nr
    pres = values[offset : offset + nr].copy()
    offset += nr
    ffprim = values[offset : offset + nr].copy()
    offset += nr
    pprime = values[offset : offset + nr].copy()
    offset += nr
    psi_grid = values[offset : offset + nr * nz].reshape((nz, nr)).copy()
    offset += nr * nz
    qpsi = values[offset : offset + nr].copy()
    offset += nr

    nbbbs, limitr, boundary, limiter = _parse_optional_coordinates(values, offset, eqdsk_path)
    r_grid = np.linspace(rleft, rleft + rdim, nr)
    z_grid = np.linspace(zmid - 0.5 * zdim, zmid + 0.5 * zdim, nz)

    return Eqdsk(
        path=str(eqdsk_path),
        header=header,
        nr=int(nr),
        nz=int(nz),
        rdim=rdim,
        zdim=zdim,
        rcentr=rcentr,
        rleft=rleft,
        zmid=zmid,
        rmaxis=rmaxis,
        zmaxis=zmaxis,
        simag=simag,
        sibry=sibry,
        bcentr=bcentr,
        current=current,
        simag2=simag2,
        xdum=xdum,
        rmaxis2=rmaxis2,
        xdum2=xdum2,
        zmaxis2=zmaxis2,
        xdum3=xdum3,
        sibry2=sibry2,
        xdum4=xdum4,
        xdum5=xdum5,
        fpol=fpol,
        pres=pres,
        ffprim=ffprim,
        pprime=pprime,
        psi_grid=psi_grid,
        qpsi=qpsi,
        r_grid=r_grid,
        z_grid=z_grid,
        nbbbs=nbbbs,
        limitr=limitr,
        boundary=boundary,
        limiter=limiter,
        numeric_value_count=int(values.size),
    )


def diagnose_eqdsk(path: str | Path) -> dict[str, Any]:
    """Return JSON-friendly parse diagnostics without raising parse errors."""

    eqdsk_path = Path(path)
    try:
        return parse_eqdsk(eqdsk_path).diagnostics()
    except (FileNotFoundError, OSError, ValueError) as exc:
        return {
            "path": str(eqdsk_path),
            "valid": False,
            "exists": eqdsk_path.exists(),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _parse_optional_coordinates(
    values: np.ndarray,
    offset: int,
    path: Path,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    if values.size == offset:
        empty = np.empty((0, 2), dtype=np.float64)
        return 0, 0, empty, empty
    if values.size < offset + 2:
        raise ValueError(f"EQDSK file has incomplete boundary/limiter counts: {path}")

    nbbbs = _integer_count(values[offset], "nbbbs", path)
    limitr = _integer_count(values[offset + 1], "limitr", path)
    offset += 2
    expected = 2 * nbbbs + 2 * limitr
    if values.size < offset + expected:
        raise ValueError(
            "EQDSK file has incomplete boundary/limiter coordinates: "
            f"expected {expected} values after counts, found {values.size - offset}"
        )

    boundary = values[offset : offset + 2 * nbbbs].reshape((nbbbs, 2)).copy()
    offset += 2 * nbbbs
    limiter = values[offset : offset + 2 * limitr].reshape((limitr, 2)).copy()
    return nbbbs, limitr, boundary, limiter


def _integer_count(value: float, name: str, path: Path) -> int:
    rounded = int(round(float(value)))
    if rounded < 0 or not np.isclose(value, rounded, rtol=0.0, atol=1e-9):
        raise ValueError(f"EQDSK {name} count must be a nonnegative integer in {path}")
    return rounded


def _float_tuple(values: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _floating_values(text: str) -> list[float]:
    return [
        float(value.replace("D", "E").replace("d", "E"))
        for value in re.findall(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?", text)
    ]
