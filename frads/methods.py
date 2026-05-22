"""Typical Radiance matrix-based simulation workflows"""

import copy
import hashlib
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from shutil import rmtree

from frads.geom import Polygon, parse_polygon
from frads.matrix import (
    load_matrix,
    SensorSender,
    SurfaceSender,
    SurfaceReceiver,
    ViewSender,
    SunReceiver,
    SkyReceiver,
    load_binary_matrix,
    BASIS_DIMENSION,
    parse_rad_header,
    Matrix,
    matrix_multiply_rgb,
    SunMatrix,
    to_sparse_matrix3,
    sparse_matrix_multiply_rgb_vtds,
)
from frads.sky import parse_epw, parse_wea, WeaMetaData, WeaData, gen_perez_sky, gendaymtx_peak
from frads.utils import random_string
from frads._absdf_cache import cache_get as _absdf_cache_get
import numpy as np
import pyradiance as pr
from pyradiance import parse_view
from scipy.sparse import csr_matrix


logger = logging.getLogger("frads.methods")


def _absdf_enabled() -> bool:
    """True iff FRADS_USE_ABSDF=1 (Task-61 aBSDF + 5PM path active)."""
    return os.environ.get("FRADS_USE_ABSDF") == "1"


def _absdf_xml_dir() -> Path:
    """Directory containing <state_name>_klems.xml files for aBSDF mode."""
    raw = os.environ.get("FRADS_ABSDF_XML_DIR")
    if not raw:
        raise ValueError(
            "FRADS_USE_ABSDF=1 requires FRADS_ABSDF_XML_DIR to point to "
            "the directory containing <state>_klems.xml files."
        )
    p = Path(raw)
    if not p.is_dir():
        raise FileNotFoundError(f"FRADS_ABSDF_XML_DIR {p} is not a directory")
    return p


def _absdf_cache_dir() -> Path:
    """Disk cache for per-pane Cds matrices."""
    raw = os.environ.get(
        "FRADS_ABSDF_CACHE_DIR",
        str(Path.cwd() / "simulation" / "cache" / "cds"),
    )
    p = Path(raw)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _tt_enabled() -> bool:
    """True iff FRADS_USE_TT_T=1 (tensor-tree T-matrix path active).

    When enabled, the V*T*D multi-bounce path is evaluated via ``dctimestep``
    against the tensor-tree BSDF XML (continuous Shirley-Chiu sampling)
    instead of dense ``np.dot()`` with a Klems-145 T-matrix. Solves the
    multi-bounce direct-sun patch quantization that drives the front-WPI
    spike under Variante-B of the Task-61 high-resolution BSDF strategy.
    """
    return os.environ.get("FRADS_USE_TT_T") == "1"


def _tt_xml_dir() -> Path:
    """Directory containing <state_name>_tt.xml tensor-tree files."""
    raw = os.environ.get("FRADS_TT_XML_DIR")
    if not raw:
        raise ValueError(
            "FRADS_USE_TT_T=1 requires FRADS_TT_XML_DIR to point to "
            "the directory containing <state>_tt.xml files."
        )
    p = Path(raw)
    if not p.is_dir():
        raise FileNotFoundError(f"FRADS_TT_XML_DIR {p} is not a directory")
    return p


def _tt_xml_for_state(state_key) -> Path:
    """Resolve the tensor-tree XML path for a given state identifier.

    state_key can be an int (e.g. 60) or a string ('60', 'state-60', etc).
    Tries common naming patterns used in the FTG mockup pipeline.
    """
    raw = str(state_key)
    digits = "".join(c for c in raw if c.isdigit()) or raw
    xml_dir = _tt_xml_dir()
    for name in (
        f"TGU-eyriseS350_{digits}_tt.xml",
        f"{digits}_tt.xml",
        f"state-{digits}_tt.xml",
        f"{raw}_tt.xml",
    ):
        candidate = xml_dir / name
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No tensor-tree XML for state '{state_key}' found in {xml_dir}"
    )


def _tt_cache_dir() -> Path:
    """Disk cache for combined V*T*D matrices baked via dctimestep."""
    raw = os.environ.get(
        "FRADS_TT_CACHE_DIR",
        str(Path.cwd() / "simulation" / "cache" / "tt_vtd"),
    )
    p = Path(raw)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _rgb_to_rgbe(rgb: np.ndarray) -> np.ndarray:
    """Encode (H, W, 3) float RGB as (H, W, 4) uint8 RGBE for Radiance HDR.

    Mirrors the encoder originally written for
    ``01_FRADS/rl/evaluation/render_dgp_5pm.py``. We carry it inside
    frads-fork because pyradiance's ``pvalue -r -df`` pipeline silently
    emits zero pixels on this build, which kills any in-process 5PM
    image -> HDR -> evalglare round trip.
    """
    H, W, _ = rgb.shape
    m = rgb.max(axis=2)
    rgbe = np.zeros((H, W, 4), dtype=np.uint8)
    valid = m > 1e-32
    mant, exp = np.frexp(m[valid])
    scale = 256.0 * mant / m[valid]
    rgbe[valid, 0] = np.clip(rgb[valid, 0] * scale, 0, 255).astype(np.uint8)
    rgbe[valid, 1] = np.clip(rgb[valid, 1] * scale, 0, 255).astype(np.uint8)
    rgbe[valid, 2] = np.clip(rgb[valid, 2] * scale, 0, 255).astype(np.uint8)
    rgbe[valid, 3] = (exp + 128).astype(np.uint8)
    return rgbe


def _write_radiance_hdr(
    rgb: np.ndarray, path: "str | Path", extra_header: bytes = b""
) -> None:
    """Write a (H, W, 3) float RGB array as a Radiance .hdr file."""
    H, W, _ = rgb.shape
    rgbe = _rgb_to_rgbe(rgb.astype(np.float32))
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\n")
        f.write(b"FORMAT=32-bit_rle_rgbe\n")
        if extra_header:
            f.write(extra_header)
        f.write(b"\n")
        f.write(f"-Y {H} +X {W}\n".encode())
        f.write(rgbe.tobytes())


def _parse_evalglare_dgp(out: "bytes | str") -> float:
    """Parse evalglare's stdout to extract the DGP value.

    evalglare's default summary format is a single line with a CSV
    header followed by colon-separated values, e.g.::

        dgp,dgi,ugr,vcp,cgi,Lveil: 0.159000 0.000000 0.000000 100.000000 0.000000 0.000000

    Some invocations also emit progress/diagnostic lines before that.
    Strategy: locate the first colon-bearing summary line (header lists
    the requested glare metrics in order, DGP first), then take the
    first floating-point token after the colon. Falls back to scanning
    all whitespace tokens line-by-line for a parseable float -- which
    still finds the DGP because it is the first numeric value emitted.
    """
    text = out.decode() if isinstance(out, bytes) else out

    for line in text.splitlines():
        if ":" not in line:
            continue
        head, _, tail = line.partition(":")
        if "dgp" not in head.lower():
            continue
        for tok in tail.split():
            try:
                return float(tok)
            except ValueError:
                continue

    for line in text.splitlines():
        for tok in line.strip().split():
            try:
                return float(tok)
            except ValueError:
                continue

    raise RuntimeError(f"evalglare produced no parseable DGP. stdout: {text!r}")


def _gendaymtx_direct_sky(
    wea_bytes: bytes, sky_mfactor: int, absdf_mode: bool,
) -> np.ndarray:
    """Build the per-timestep direct-only sky vector S_ds.

    NOTE: This vector is consumed as ``S_ds`` in ``V_d*T*D_d*S_ds`` which is
    subtracted from ``V*T*D*S`` (where ``S`` comes from the regular
    ``get_sky_matrix`` using ``pr.gendaymtx`` without ``-5``).  S_ds must use
    the SAME sun-distribution as S, otherwise the subtraction produces
    negative WPI values.  In an earlier aBSDF-patch revision we used the
    ``-5`` flag here, which broke this invariant -- it has been removed.
    The ``-5``/``-O0`` peak-extraction sky is only used downstream in
    ``_gendaymtx_direct_sun`` for the high-resolution ``Ssun`` vector.
    """
    smx = pr.gendaymtx(
        wea_bytes, outform="d", mfactor=sky_mfactor,
        header=False, sun_only=True,
    )
    nrows = BASIS_DIMENSION.get(f"r{sky_mfactor}", 145) + 1
    return load_binary_matrix(smx, nrows=nrows, ncols=1, ncomp=3, dtype="d")


def _gendaymtx_direct_sun(
    wea_bytes: bytes, sun_mfactor: int, absdf_mode: bool,
) -> np.ndarray:
    """Build the per-timestep one-sun matrix S_sun (5185-patch for r6).

    In aBSDF mode: uses ``-5`` flag plus ``onesun=True`` (``-O 0``) so each
    Reinhart patch gets unit sun radiance scaled to the 0.533° solar disk
    solid angle. This is McNeil's high-resolution sun-coefficient input.
    """
    if absdf_mode:
        smx = gendaymtx_peak(
            wea_bytes, mfactor=sun_mfactor, direct_only=True, onesun=True,
        )
    else:
        smx = pr.gendaymtx(
            wea_bytes, outform="d", mfactor=sun_mfactor,
            header=False, sun_only=True, onesun=True,
        )
    nrows = BASIS_DIMENSION.get(f"r{sun_mfactor}", 5165) + 1
    return load_binary_matrix(smx, nrows=nrows, ncols=1, ncomp=3, dtype="d")


@dataclass(slots=True)
class SceneConfig:
    """Radiance scene configuration object.

    It can be initialized with either a raw data bytes or a list
    of files. If a list of files is provided, they will be concatenated in
    the order they are provided.

    Attributes:
        files: A list of files to be concatenated to form the scene.
        for name, material in self.model.material.items():
            materials[name] = parse_material(name, material)
        for name, material in  self.model.material_no_mass.items():
            materials[name] = parse_material_no_mass(name, material)
        for name, material in self.model.window_material_simple_glazing_system.items():
            materials[name] = parse_window_material_simple_glazing_system(name, material)
        for name, material in self.model.window_material_glazing.items():
            materials[name] = parse_window_material_glazing(name, material)
        for name, material in self.model.window_material_blind.items():
            materials.update(parse_window_material_blind(material))
        bytes: A raw data string to be used as the scene.
        files_mtime: Files last modification time.
    """

    files: list[Path] = field(default_factory=list)
    bytes: bytes = b""
    files_mtime: list[float] = field(init=False, default_factory=list)

    def __post_init__(self):
        if len(self.files) > 0:
            for fpath in self.files:
                self.files_mtime.append(os.path.getmtime(fpath))


@dataclass(slots=True)
class MatrixConfig:
    matrix_file: str | Path = ""
    matrix_data: None | np.ndarray = None

    def __post_init__(self):
        if self.matrix_data is None:
            self.matrix_data = load_matrix(self.matrix_file)
        elif isinstance(self.matrix_data, list):
            self.matrix_data = np.array(self.matrix_data)


@dataclass(slots=True)
class ShadingGeometryConfig:
    shading_geometry_file: str | Path = ""
    shading_geometry_bytes: None | np.ndarray = None

    def __post_init__(self): ...


@dataclass(slots=True)
class MaterialConfig:
    """Material file/data configuration object.

    It can be initialized with either a raw data bytes or a list
    of files. If a list of files is provided, they will be concatenated in
    the order they are provided.

    Attributes:
        file: A file to be used as the material.
        bytes: A raw data string to be used as the material.
        matrices: A dictionary of matrix files/data.
        glazing_materials: A dictionary of glazing materials used for edgps calculations.
        file_mtime: File last modification time.

    Raises:
        ValueError: If no file, bytes, or matrices are provided.
    """

    files: list[Path] = field(default_factory=list)
    bytes: bytes = b""
    matrices: dict[str, MatrixConfig] = field(default_factory=dict)
    matrices_mlnp: dict[str, MatrixConfig] = field(default_factory=dict)
    glazing_materials: dict[str, pr.Primitive] = field(default_factory=dict)
    files_mtime: list[float] = field(init=False, default_factory=list)

    def __post_init__(self):
        if len(self.files) > 0:
            for fpath in self.files:
                self.files_mtime.append(os.path.getmtime(fpath))
        for k, v in self.matrices.items():
            if isinstance(v, dict):
                self.matrices[k] = MatrixConfig(**v)
        if self.bytes == b"" and len(self.files) == 0 and len(self.matrices) == 0:
            raise ValueError("MaterialConfig must have either file, bytes or matrices")


@dataclass(slots=True)
class WindowConfig:
    """Window file/data configuration object.

    Each WindowConfig instance coresponds to a window group, which
    can be initialized with either a file path or byte strings.
    In addition, the BSDF matrix files/data, high resolution tensor
    tree files, and shading geometry files or bytestring data can
    be initialized as well.

    Attributes:
        file: A file to be used as the window group.
        bytes: A raw data string to be used as the window group.
        matrix_name: A matrix name to be used for the window group.
        proxy_geometry: A raw data string to be used as the shading geometry.
        files_mtime: Files last modification time.

    Raises:
        ValueError: If neither file nor bytes are provided.
    """

    file: str | Path = ""
    bytes: bytes = b""
    matrix_name: str = ""
    polygon: None | Polygon = None
    proxy_geometry: dict[str, bytes] = field(default_factory=dict)
    files_mtime: list[float] = field(init=False, default_factory=list)

    def __post_init__(self):
        if os.path.exists(self.file):
            self.files_mtime.append(os.path.getmtime(self.file))
            if not isinstance(self.file, Path):
                self.file = Path(self.file)
        if self.bytes == b"":
            if self.file != "":
                with open(self.file, "rb") as f:
                    self.bytes = f.read()
            else:
                raise ValueError("WindowConfig must have either file or bytes")


@dataclass(slots=True)
class SensorConfig:
    """
    A configuration class for sensors that includes information on the file,
    data, and file modification time.

    Attributes:
        file: Path to the file containing sensor data. Default is an empty string.
        data: list of lists containing float data.
            Default is an empty list.
        file_mtime: Modification time of the file. This attribute is
            automatically initialized based on the 'file' attribute.

    Raises:
        ValueError: If neither file nor data are provided.
    """

    file: str = ""
    data: list[list[float]] = field(default_factory=list)
    file_mtime: float = field(init=False, default=0.0)

    def __post_init__(self):
        """
        Post-initialization method to set the file modification time and load data
        from the file if necessary.
        """
        if self.file != "":
            self.file_mtime = os.path.getmtime(self.file)
        if len(self.data) == 0:
            if self.file != "":
                with open(self.file) as f:
                    self.data = [
                        [float(i) for i in line.split()] for line in f.readlines()
                    ]
            else:
                raise ValueError("SensorConfig must have either file or data")


@dataclass(slots=True)
class ViewConfig:
    """
    A configuration class for views that includes information on the file,
    data, x/y resoluation, and file modification time.

    Attributes:
        file: Path to the file containing view data. Default is an empty string.
        view: A View object. Default is an empty string.
        xres: X resolution of the view. Default is 512.
        yres: Y resolution of the view. Default is 512.
        file_mtime: Modification time of the file. This attribute is
            automatically initialized based on the 'file' attribute.

    Raises:
        ValueError: If neither file nor view are provided.
    """

    file: str | Path = ""
    view: pr.View | str = field(default_factory=str)
    xres: int = 512
    yres: int = 512
    file_mtime: float = field(init=False, default=0.0)

    def __post_init__(self):
        if self.file != "":
            self.file_mtime = os.path.getmtime(self.file)
            if not isinstance(self.file, Path):
                self.file = Path(self.file)
        if os.path.exists(self.file) and self.view == "":
            self.view = pr.viewfile(str(self.file))
        elif self.view != "":
            if not isinstance(self.view, pr.View):
                self.view = parse_view(self.view)
        else:
            raise ValueError("ViewConfig must have either file or view")


@dataclass(slots=True)
class SurfaceConfig:
    """
    A configuration class for surfaces that includes information on the file,
    data, basis, and file modification time.

    Attributes:
        file: Path to the file containing surface data. Default is an empty string.
        primitives: A list of primitives. Default is an empty list.
        basis: A string representing the basis. Default is 'u'.
        file_mtime: Modification time of the file. This attribute is
            automatically initialized based on the 'file' attribute.

    Raises:
        ValueError: If neither file nor primitives are provided.
    """

    file: str | Path = ""
    primitives: list[pr.Primitive] = field(default_factory=list)
    basis: str = "u"
    file_mtime: float = field(init=False, default=0.0)

    def __post_init__(self):
        if self.file != "":
            self.file_mtime = os.path.getmtime(self.file)
        if not isinstance(self.file, Path):
            self.file = Path(self.file)
        if self.file.exists() and len(self.primitives) == 0:
            self.primitives = pr.parse_primitive(self.file.read_text())
        elif len(self.primitives) == 0:
            raise ValueError("SurfaceConfig must have either file or primitives")


@dataclass(slots=True)
class Settings:
    """Settings is a dataclass that holds the settings for a Radiance simulation.

    Attributes:
        name: The name of the simulation.
        num_processors: The number of processors to use for the simulation.
        method: The Radiance method to use for the simulation.
        overwrite: Whether to overwrite existing files.
        save_matrices: Whether to save the matrices generated by the simulation.
        sky_basis: The sky basis to use for the simulation.
        window_basis: The window basis to use for the simulation.
        non_coplanar_basis: The non-coplanar basis to use for the simulation.
        sun_basis: The sun basis to use for the simulation.
        sun_culling: Whether to cull suns.
        separate_direct: Whether to separate direct and indirect contributions.
        epw_file: The path to the EPW file to use for the simulation.
        wea_file: The path to the WEA file to use for the simulation.
        start_hour: The start hour for the simulation.
        end_hour: The end hour for the simulation.
        daylight_hours_only: Whether to simulate only daylight hours.
        latitude: The latitude for the simulation.
        longitude: The longitude for the simulation.
        timezone: The timezone for the simulation.
        orientation: sky rotation.
        site_elevation: The elevation for the simulation.
        sensor_sky_matrix: The sky matrix sampling parameters
        view_sky_matrix: View sky matrix sampling parameters
        sensor_sun_matrix: Sensor sun matrix sampling parameters
        view_sun_matrix: View sun matrix sampling parameters
        sensor_window_matrix: Sensor window matrix sampling parameters
        view_window_matrix: View window matrix sampling parameters
        daylight_matrix: Daylight matrix sampling parameters
        output_directory: The directory to save the output files.
    """

    name: str = field(default="")
    num_processors: int = 4
    method: str = field(default="3phase")
    overwrite: bool = False
    save_matrices: bool = False
    matrix_dir: str = field(default="Matrices")
    sky_basis: str = field(default="r4")
    window_basis: str = field(default="kf")
    non_coplanar_basis: str = field(default="kf")
    sun_basis: str = field(default="r6")
    sun_culling: bool = field(default=True)
    separate_direct: bool = field(default=False)
    epw_file: str = field(default="")
    wea_file: str = field(default="")
    start_hour: float = field(default=8)
    end_hour: float = field(default=18)
    daylight_hours_only: bool = True
    latitude: float = field(default=37)
    longitude: float = field(default=122)
    time_zone: int = field(default=120)
    orientation: float = field(default=0)
    site_elevation: float = field(default=100)
    sensor_sky_matrix: list[str] = field(
        default_factory=lambda: ["-ab", "6", "-ad", "8192", "-lw", "5e-5"]
    )
    sensor_sun_matrix: list[str] = field(
        default_factory=lambda: [
            "-ab",
            "1",
            "-ad",
            "256",
            "-lw",
            "1e-3",
            "-dj",
            "0",
            "-st",
            "0",
        ]
    )
    view_sun_matrix: list[str] = field(
        default_factory=lambda: [
            "-ab",
            "1",
            "-ad",
            "256",
            "-lw",
            "1e-3",
            "-dj",
            "0",
            "-st",
            "0",
        ]
    )
    view_sky_matrix: list[str] = field(
        default_factory=lambda: ["-ab", "6", "-ad", "8192", "-lw", "5e-5"]
    )
    daylight_matrix: list[str] = field(
        default_factory=lambda: ["-ab", "2", "-c", "5000"]
    )
    sensor_window_matrix: list[str] = field(
        default_factory=lambda: ["-ab", "5", "-ad", "8192", "-lw", "5e-5"]
    )
    surface_window_matrix: list[str] = field(
        default_factory=lambda: [
            "-ab",
            "5",
            "-ad",
            "8192",
            "-lw",
            "5e-5",
            "-c",
            "10000",
        ]
    )
    view_window_matrix: list[str] = field(
        default_factory=lambda: ["-ab", "5", "-ad", "8192", "-lw", "5e-5"]
    )
    files_mtime: list[float] = field(init=False, default_factory=list)
    output_directory: str = field(default="./")

    def __post_init__(self):
        if self.wea_file != "":
            self.files_mtime.append(os.path.getmtime(self.wea_file))
        if self.epw_file != "":
            self.files_mtime.append(os.path.getmtime(self.epw_file))


@dataclass
class Model:
    """Model dataclass.

    Attributes:
        scene: SceneConfig object
        windows: A dictionary of WindowConfig
        materials: MaterialConfig object
        sensors: A dictionary of SensorConfig
        views: A dictionary of ViewConfig
    """

    materials: "MaterialConfig"
    scene: "SceneConfig" = field(default_factory=SceneConfig)
    windows: dict[str, "WindowConfig"] = field(default_factory=dict)
    sensors: dict[str, "SensorConfig"] = field(default_factory=dict)
    views: dict[str, "ViewConfig"] = field(default_factory=dict)
    surfaces: dict[str, "SurfaceConfig"] = field(default_factory=dict)

    # Make Path() out of all path strings
    def __post_init__(self):
        if isinstance(self.scene, dict):
            self.scene = SceneConfig(**self.scene)
        if isinstance(self.materials, dict):
            self.materials = MaterialConfig(**self.materials)
        for k, v in self.windows.items():
            if isinstance(v, dict):
                self.windows[k] = WindowConfig(**v)
        for k, v in self.sensors.items():
            if isinstance(v, dict):
                self.sensors[k] = SensorConfig(**v)
        for k, v in self.views.items():
            if isinstance(v, dict):
                self.views[k] = ViewConfig(**v)
        for k, v in self.surfaces.items():
            if isinstance(v, dict):
                self.surfaces[k] = SurfaceConfig(**v)

        self.scene_cfg = True
        self.windows_cfg = True
        self.sensors_cfg = True
        self.views_cfg = True
        self.surfaces_cfg = True

        if not self.scene.files and not self.scene.bytes:
            self.scene_cfg = False
        if self.windows == {}:
            self.windows_cfg = False
        if self.sensors == {}:
            self.sensors_cfg = False
        if self.views == {}:
            self.views_cfg = False
        if self.surfaces == {}:
            self.surfaces_cfg = False

        # add view to sensors if not already there
        for k, v in self.views.items():
            if k in self.sensors:
                if [tuple(self.sensors[k].data[0])] == [
                    self.views[k].view.vp + self.views[k].view.vdir
                ]:
                    continue
                else:
                    raise ValueError(f"Sensor {k} data does not match view {k} data")
            else:
                self.sensors[k] = SensorConfig(
                    data=[self.views[k].view.vp + self.views[k].view.vdir]
                )

        for k, v in self.windows.items():
            if v.matrix_name != "":
                if v.matrix_name not in self.materials.matrices:
                    raise ValueError(
                        f"{k} matrix name {v.matrix_name} not found in materials"
                    )


@dataclass
class WorkflowConfig:
    """Workflow configuration dataclass.

    Workflow configuration is initialized with the Settings
    and Model dataclasses. A hash string is generated from
    the config content.

    Attributes:
        settings: A Settings object.
        model: A Model object.
        hash_str: A hash string of the config content.
    """

    settings: "Settings"
    model: "Model"
    hash_str: str = field(init=False)

    def __post_init__(self):
        if isinstance(self.settings, dict):
            self.settings = Settings(**self.settings)
        if isinstance(self.model, dict):
            self.model = Model(**self.model)
        if (
            not self.model.sensors_cfg
            and not self.model.views_cfg
            and not self.model.surfaces_cfg
        ):
            raise ValueError(
                f"Sensors, views, or surfaces must be specified for {self.settings.method} method"
            )
        if (
            self.settings.method == "3phase" or self.settings.method == "5phase"
        ) and not self.model.windows_cfg:
            raise ValueError(
                f"Windows must be specified in Model for the {self.settings.method} method"
            )
        tmp_dict = copy.copy(self.__dict__)
        tmp_settings = copy.copy(self.__dict__["settings"])
        tmp_settings.output_directory = ""
        tmp_dict["settings"] = tmp_settings
        self.hash_str = hashlib.md5(str(tmp_dict).encode()).hexdigest()[:16]

    @staticmethod
    def from_dict(obj: dict) -> "WorkflowConfig":
        """Generate a WorkflowConfig object from a dictionary.
        Args:
            obj: A dictionary of workflow configuration.
        Returns:
            A WorkflowConfig object.
        """
        settings = Settings(**obj["settings"])
        model = Model(**obj["model"])
        return WorkflowConfig(settings, model)


class PhaseMethod:
    """Base class for phase methods.

    This class is not meant to be used by itself.
    Use one of the subclasses instead.
    The base class instantiate a set of common attributes,
    along with a host of methods that are shared by all phase methods.

    Attributes:
        config: A WorkflowConfig object.
        view_senders: A dictionary of ViewSender objects.
        sensor_senders: A dictionary of SensorSender objects.
        sky_receiver: A SkyReceiver object.
        wea_header: Weather file header string
        wea_metadata: Weather file metadata object
        wea_str: Weather data string
        tmpdir: A temporary directory for storing intermediate files.
        octdir: A directory for storing octree files.
        mtxdir: A directory for storing matrix files.
        mfile: A matrix file path.
    """

    def __init__(self, config: WorkflowConfig):
        """
        Initialize a phase method.

        Args:
            config: A WorkflowConfig object.
        """
        self.config = config

        # Setup the view and sensor senders
        self.view_senders = {}
        self.sensor_senders = {}
        self.surface_senders = {}
        for name, sensors in self.config.model.sensors.items():
            self.sensor_senders[name] = SensorSender(sensors.data)
        for name, view in self.config.model.views.items():
            self.view_senders[name] = ViewSender(
                view.view, xres=view.xres, yres=view.yres
            )
        for name, surface in self.config.model.surfaces.items():
            self.surface_senders[name] = SurfaceSender(
                surfaces=surface.primitives,
                basis=surface.basis,
            )

        # Setup the sky receiver object
        self.sky_receiver = SkyReceiver(self.config.settings.sky_basis)

        # Figure out the weather related stuff
        if self.config.settings.epw_file != "":
            with open(self.config.settings.epw_file) as f:
                self.wea_metadata, self.wea_data = parse_epw(f.read())
            self.wea_header = self.wea_metadata.wea_header()
            self.wea_str = self.wea_header + "\n".join(str(d) for d in self.wea_data)
        elif self.config.settings.wea_file != "":
            with open(self.config.settings.wea_file) as f:
                self.wea_metadata, self.wea_data = parse_wea(f.read())
            self.wea_header = self.wea_metadata.wea_header()
            self.wea_str = self.wea_header + "\n".join(str(d) for d in self.wea_data)
        else:
            if (
                self.config.settings.latitude is None
                or self.config.settings.longitude is None
            ):
                raise ValueError(
                    "Latitude and longitude must be specified if no weather file is given"
                )
            self.wea_header = self.get_wea_header()
            self.wea_metadata = WeaMetaData(
                "city",
                "country",
                self.config.settings.latitude,
                self.config.settings.longitude,
                self.config.settings.time_zone,
                self.config.settings.site_elevation,
            )
            self.wea_data = None

        # Setup Temp and Octrees directory
        self.outdir = Path(self.config.settings.output_directory)
        self.outdir.mkdir(exist_ok=True)
        self.octdir = self.outdir / "Octrees"
        self.octdir.mkdir(exist_ok=True)
        self.mtxdir = self.outdir / "Matrices"
        self.mtxdir.mkdir(exist_ok=True)
        self.mfile = (self.mtxdir / self.config.hash_str).with_suffix(".npz")

        # Generate a base octree
        self.octree = self.octdir / f"{random_string(5)}.oct"

    def __enter__(self):
        """
        Context manager enter method. This method is called when
        the class is used as a context manager. Anything happens
        after the with statement is run.
        """
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """
        Context manager exit method. This method is called when
        the class is used as a context manager. Cleans up the
        Temp and Octrees directory.
        """
        rmtree(self.octdir, ignore_errors=True)
        rmtree(self.mtxdir, ignore_errors=True)

    def get_wea_header(self, unit: int = 1):
        meta_data = WeaMetaData(
            city=str(self.config.settings.latitude),
            country=str(self.config.settings.longitude),
            latitude=self.config.settings.latitude,
            longitude=self.config.settings.longitude,
            timezone=self.config.settings.time_zone,
            elevation=self.config.settings.site_elevation,
            unit=unit,
        )
        return meta_data.wea_header()

    def generate_matrices(self):
        raise NotImplementedError

    def calculate_view(self, view, time, dni, dhi):
        raise NotImplementedError

    def calculate_sensor(self, sensor, time, dni, dhi):
        raise NotImplementedError

    def get_sky_matrix(
        self,
        time: datetime | list[datetime],
        dni: float | list[float],
        dhi: float | list[float],
        solar_spectrum: bool = False,
        orient: float = 0.0,
    ) -> np.ndarray:
        """Generates a sky matrix based on the time, Direct Normal Irradiance (DNI), and
        Diffuse Horizontal Irradiance (DHI).

        Args:
            time: The specific time for the matrix.
            dni: The Direct Normal Irradiance value.
            dhi: The Diffuse Horizontal Irradiance value.
            solar_spectrum: Whether to use the solar spectrum or not.
            orient: The orientation of the matrix.

        Returns:
            numpy.ndarray: The generated sky matrix, with dimensions based on the
                BASIS_DIMENSION setting for the current sky_basis configuration.
        """
        _wea = self.wea_header
        _ncols = 1
        if (
            isinstance(time, datetime)
            and isinstance(dni, (float, int))
            and isinstance(dhi, (float, int))
        ):
            _wea += str(WeaData(time, dni, dhi))
        elif isinstance(time, list) and isinstance(dni, list) and isinstance(dhi, list):
            rows = [str(WeaData(t, n, d)) for t, n, d in zip(time, dni, dhi)]
            _wea += "\n".join(rows)
            _ncols = len(time)
        else:
            raise ValueError(
                "Time, DNI, and DHI must be either single values or lists of values"
            )
        smx = pr.gendaymtx(
            _wea.encode(),
            outform="d",
            mfactor=int(self.config.settings.sky_basis[-1]),
            header=False,
            solar_radiance=solar_spectrum,
            rotate=orient,
        )
        return load_binary_matrix(
            smx,
            nrows=BASIS_DIMENSION[self.config.settings.sky_basis] + 1,
            ncols=_ncols,
            ncomp=3,
            dtype="d",
        )

    def get_melanopic_sky_matrix(
        self,
        time: datetime | list[datetime],
        dni: float | list[float],
        dhi: float | list[float],
        aod: float | list[float] = 0.112,
        sky_cover: float | list[float] = 0.0,
        outdir: str = "./",
    ) -> np.ndarray:
        """Generates a sky matrix based on the time, Direct Normal Irradiance (DNI), and
        Diffuse Horizontal Irradiance (DHI).

        Args:
            time: The specific time for the matrix.
            dni: The Direct Normal Irradiance value.
            dhi: The Diffuse Horizontal Irradiance value.

        Returns:
            numpy.ndarray: The generated sky matrix, with dimensions based on the
                BASIS_DIMENSION setting for the current sky_basis configuration.
        """
        _wea = self.get_wea_header(unit=3)
        _ncols = 1
        if (
            isinstance(time, datetime)
            and isinstance(dni, (float, int))
            and isinstance(dhi, (float, int))
            and isinstance(aod, (float, int))
            and isinstance(sky_cover, (float, int))
        ):
            _wea += str(WeaData(time, dni, dhi, aod, sky_cover))
        elif (
            isinstance(time, list)
            and isinstance(dni, list)
            and isinstance(dhi, list)
            and isinstance(aod, list)
            and isinstance(sky_cover, list)
        ):
            rows = [
                str(WeaData(t, n, d, a, s))
                for t, n, d, a, s in zip(time, dni, dhi, aod, sky_cover)
            ]
            _wea += "\n".join(rows)
            _ncols = len(time)
        smx = pr.gensdaymtx(
            _wea.encode(),
            outform="f",
            mfactor=int(self.config.settings.sky_basis[-1]),
            header=True,
            nthreads=self.config.settings.num_processors,
            out_dir=outdir,
        )
        smx = pr.getinfo(
            pr.Rcomb(transform="M", header=False, outform="f").add_input(smx)(),
            strip_header=True,
        )
        return load_binary_matrix(
            smx,
            nrows=BASIS_DIMENSION[self.config.settings.sky_basis] + 1,
            ncols=_ncols,
            ncomp=1,
            dtype="f",
        )

    def get_sky_matrix_from_wea(self, mfactor: int, sun_only=False, onesun=False):
        if self.wea_str is None:
            raise ValueError("No weather string available")
        _sun_str = pr.gendaymtx(
            self.wea_str.encode(),
            sun_file="-",
            dryrun=True,
            daylight_hours_only=True,
        )
        prims = pr.parse_primitive(_sun_str.decode())
        _datetimes = [
            datetime(2023, 1, 1) + timedelta(int(p.identifier.lstrip("solar")))
            for p in prims
            if p.ptype == "light"
        ]
        _matrix = pr.gendaymtx(
            self.wea_str.encode(),
            sun_only=sun_only,
            onesun=onesun,
            outform="d",
            daylight_hours_only=True,
            mfactor=mfactor,
        )
        _nrows, _ncols, _ncomp, _dtype = parse_rad_header(pr.getinfo(_matrix).decode())
        return load_binary_matrix(
            _matrix,
            nrows=_nrows,
            ncols=_ncols,
            ncomp=_ncomp,
            dtype=_dtype,
            header=True,
        )

    def save_matrices(self):
        raise NotImplementedError


class TwoPhaseMethod(PhaseMethod):
    """Implements two phase method."""

    def __init__(self, config: WorkflowConfig):
        """Initializes the two phase method

        Args:
            config: A WorkflowConfig object
        """
        super().__init__(config)
        oct_stdin = config.model.materials.bytes + config.model.scene.bytes
        for window in config.model.windows.values():
            oct_stdin += window.bytes
        with open(self.octree, "wb") as f:
            f.write(
                pr.oconv(
                    *config.model.materials.files,
                    *config.model.scene.files,
                    stdin=oct_stdin,
                )
            )
        self.view_sky_matrices = {}
        self.sensor_sky_matrices = {}
        for vs in self.view_senders:
            self.view_sky_matrices[vs] = Matrix(
                self.view_senders[vs], [self.sky_receiver], self.octree
            )
        for ss in self.sensor_senders:
            self.sensor_sky_matrices[ss] = Matrix(
                self.sensor_senders[ss], [self.sky_receiver], self.octree
            )

    def generate_matrices(self) -> None:
        """Generate matrices for all view and sensor points."""
        # First check if matrices files already exist
        if self.mfile.exists() and (not self.config.settings.overwrite):
            self.load_matrices()
            return
        # Then check if overwrite is set to True
        for _, mtx in self.view_sky_matrices.items():
            mtx.generate(
                self.config.settings.view_sky_matrix,
                nproc=self.config.settings.num_processors,
            )
        for _, mtx in self.sensor_sky_matrices.items():
            mtx.generate(
                self.config.settings.sensor_sky_matrix,
                nproc=self.config.settings.num_processors,
            )
        if self.config.settings.save_matrices:
            self.save_matrices()

    def load_matrices(self):
        """
        Load matrices from a .npz file
        """
        logger.info(f"Loading matrices from {self.mfile}")
        if not self.mfile.exists():
            raise FileNotFoundError("Matrices file not found")
        mdata = np.load(self.mfile)
        for view, mtx in self.view_sky_matrices.items():
            mtx.array = mdata[f"{view}_sky_matrix"]
        for sensor, mtx in self.sensor_sky_matrices.items():
            mtx.array = mdata[f"{sensor}_sky_matrix"]

    def calculate_view(
        self, view: str, time: datetime, dni: float, dhi: float
    ) -> np.ndarray:
        """Calculate (render) a view.
        Args:
            view: A view name, must bed loaded during configuration.
            time: A datetime object
            dni: Direct normal irradiance
            dhi: Diffuse horizontal irradiance
        Returns:
            A image as a numpy array
        """
        sky_matrix = self.get_sky_matrix(time, dni, dhi)
        return matrix_multiply_rgb(self.view_sky_matrices[view].array, sky_matrix)

    def calculate_sensor(
        self, sensor: str, time: datetime, dni: float, dhi: float
    ) -> np.ndarray:
        """Calculate a sensor view.

        Args:
            sensor: A sensor name, must be loaded during configuration.
            time: A datetime object
            dni: Direct normal irradiance
            dhi: Diffuse horizontal irradiance
        Returns:
            Sensor illuminance value
        """
        sky_matrix = self.get_sky_matrix(time, dni, dhi)
        return matrix_multiply_rgb(
            self.sensor_sky_matrices[sensor].array,
            sky_matrix,
            weights=[47.4, 119.9, 11.6],
        )

    def calculate_view_from_wea(self, view: str) -> np.ndarray:
        """Render a series of images for a view. Rendering using
        the weather file loaded during configuration.

        Args:
            view: View name, must be loaded during configuration.
        Returns:
            A numpy array containing a series of images
        """
        if self.wea_data is None:
            raise ValueError("No wea data available")
        sky_matrix = self.get_sky_matrix_from_wea(
            int(self.config.settings.sky_basis[-1])
        )
        # arbitrary chunksize
        chunksize = 300
        shape = (
            self.view_sky_matrices[view].nrows,
            sky_matrix.shape[1],
            3,
        )
        final = np.memmap(
            f"{view}_2ph.dat",
            shape=shape,
            dtype=np.float64,
            mode="w+",
            order="F",
        )
        for idx in range(0, sky_matrix.shape[1], chunksize):
            end = min(idx + chunksize, sky_matrix.shape[1])
            res = matrix_multiply_rgb(
                self.view_sky_matrices[view].array,
                sky_matrix[:, idx:end, :],
            )
            final[:, idx:end, 0] = res[:, :, 0]
            final[:, idx:end, 1] = res[:, :, 1]
            final[:, idx:end, 2] = res[:, :, 2]
            final.flush()
        return final

    def calculate_sensor_from_wea(self, sensor: str) -> np.ndarray:
        """Calculate a sensor for tue duration of the weather file
        that's loaded during configuration.

        Args:
            sensor: sensor name, must be loaded during configuration.
        Returns:
            A numpy array containing a series of illuminance values
        """
        if self.wea_data is None:
            raise ValueError("No wea data available")
        return matrix_multiply_rgb(
            self.sensor_sky_matrices[sensor].array,
            self.get_sky_matrix_from_wea(int(self.config.settings.sky_basis[-1])),
            weights=[47.4, 119.9, 11.6],
        )

    def save_matrices(self):
        """Save matrices to a .npz file in the Matrices directory.
        File name is the hash string of the configuration.
        """
        matrices = {}
        for view, mtx in self.view_sky_matrices.items():
            matrices[f"{view}_sky_matrix"] = mtx.array
        for sensor, mtx in self.sensor_sky_matrices.items():
            matrices[f"{sensor}_sky_matrix"] = mtx.array
        np.savez(self.mtxdir / self.config.hash_str, **matrices)


class ThreePhaseMethod(PhaseMethod):
    """Three phase method implementation.

    Attributes:
        config: A WorkflowConfig object
        octree: A path to the octree file
        window_senders: A dictionary of window sender matrices
        window_receivers: A dictionary of window receiver matrices
        window_bsdfs: A dictionary of window BSDF matrices
        daylight_matrices: A dictionary of daylight matrices
        view_window_matrices: A dictionary of view window matrices
        sensor_window_matrices: A dictionary of sensor window matrices
    """

    def __init__(self, config):
        super().__init__(config)
        with open(self.octree, "wb") as f:
            f.write(
                pr.oconv(
                    *config.model.materials.files,
                    *config.model.scene.files,
                    stdin=config.model.materials.bytes + config.model.scene.bytes,
                )
            )
        self.window_senders: dict[str, SurfaceSender] = {}
        self.window_receivers = {}
        self.window_bsdfs = {}
        self.daylight_matrices = {}
        for _name, window in self.config.model.windows.items():
            _prims = pr.parse_primitive(window.bytes.decode())
            if window.matrix_name != "":
                self.window_bsdfs[_name] = self.config.model.materials.matrices[
                    window.matrix_name
                ].matrix_data
            else:
                # raise ValueError("No matrix data or file available", _name)
                logger.info(f"No matrix data or file available: {_name}")
            if _name in self.window_bsdfs:
                window_basis = [
                    k
                    for k, v in BASIS_DIMENSION.items()
                    if v == self.window_bsdfs[_name].shape[0]
                ][0]
            else:
                window_basis = self.config.settings.window_basis
            self.window_receivers[_name] = SurfaceReceiver(
                _prims,
                window_basis,
            )
            self.window_senders[_name] = SurfaceSender(_prims, window_basis)
            self.daylight_matrices[_name] = Matrix(
                self.window_senders[_name],
                [self.sky_receiver],
                self.octree,
            )
        self.view_window_matrices = {}
        self.sensor_window_matrices = {}
        self.surface_window_matrices = {}
        for _v, sender in self.view_senders.items():
            self.view_window_matrices[_v] = Matrix(
                sender, list(self.window_receivers.values()), self.octree
            )
        for _s, sender in self.sensor_senders.items():
            self.sensor_window_matrices[_s] = Matrix(
                sender, list(self.window_receivers.values()), self.octree
            )
        for _s, sender in self.surface_senders.items():
            self.surface_window_matrices[_s] = Matrix(
                sender, list(self.window_receivers.values()), self.octree
            )

    def generate_matrices(self, view_matrices: bool = True):
        """Generate all required matrices

        Args:
            view_matrices: Toggle to generate view matrices. Toggle it off can be
                useful for not needing the view matrices but still need the view data
                for things like edgps calculation.
        """
        if self.mfile.exists() and (not self.config.settings.overwrite):
            self.load_matrices()
            return
        if view_matrices:
            for _, mtx in self.view_window_matrices.items():
                mtx.generate(
                    self.config.settings.view_window_matrix,
                    nproc=self.config.settings.num_processors,
                )
        for _, mtx in self.sensor_window_matrices.items():
            mtx.generate(
                self.config.settings.sensor_window_matrix,
                nproc=self.config.settings.num_processors,
            )
        for _, mtx in self.surface_window_matrices.items():
            mtx.generate(
                self.config.settings.surface_window_matrix,
                nproc=self.config.settings.num_processors,
            )
        for _, mtx in self.daylight_matrices.items():
            mtx.generate(
                self.config.settings.daylight_matrix,
                nproc=self.config.settings.num_processors,
            )
        if self.config.settings.save_matrices:
            self.save_matrices()

    def load_matrices(self):
        """Load matrices from a .npz file in the Matrices directory."""
        logger.info(f"Loading matrices from {self.mfile}")
        mdata = np.load(self.mfile)
        for view, mtx in self.view_window_matrices.items():
            if (key := f"{view}_window_matrix") in mdata:
                mtx.array = mdata[key]
        for sensor, mtx in self.sensor_window_matrices.items():
            mtx.array = mdata[f"{sensor}_window_matrix"]
        for surface, mtx in self.surface_window_matrices.items():
            mtx.array = mdata[f"{surface}_window_matrix"]
        for name, mtx in self.daylight_matrices.items():
            mtx.array = mdata[f"{name}_daylight_matrix"]

    def calculate_view(
        self,
        view: str,
        bsdf: np.ndarray,
        time: datetime,
        dni: float,
        dhi: float,
    ) -> np.ndarray:
        """Calculate (render) a view.

        Args:
            view: The view name
            bsdf: The BSDF matrix
            time: The datetime object
            dni: The direct normal irradiance
            dhi: The diffuse horizontal irradiance
        Returns:
            A image as numpy array
        """
        sky_matrix = self.get_sky_matrix(time, dni, dhi)
        res = []
        if isinstance(bsdf, list):
            if len(bsdf) != len(self.config.model.windows):
                raise ValueError("Number of BSDF should match number of windows.")
        for idx, _name in enumerate(self.config.model.windows):
            _bsdf = bsdf[idx] if isinstance(bsdf, list) else bsdf
            res.append(
                matrix_multiply_rgb(
                    self.view_window_matrices[view].array[idx],
                    _bsdf,
                    self.daylight_matrices[_name].array,
                    sky_matrix,
                )
            )
        return np.sum(res, axis=0)

    def calculate_sensor(
        self,
        sensor: str,
        bsdf: dict[str, str],
        time: datetime,
        dni: float,
        dhi: float,
    ) -> np.ndarray:
        """Calculate illuminance for a sensor.

        Args:
            sensor: The sensor name
            bsdf: A dictionary of window name as key and bsdf matrix or matrix name as value
            time: The datetime object
            dni: The direct normal irradiance
            dhi: The diffuse horizontal irradiance
        Returns:
            A float value of illuminance
        """
        # DEBUG: marker for ThreePhase calculate_sensor (with caller type)
        print(f"[THREEPHASE_CALC_SENSOR_ENTRY] sensor={sensor} time={time} self_type={type(self).__name__}", flush=True)
        sky_matrix = self.get_sky_matrix(time, dni, dhi)
        res = []
        if isinstance(bsdf, list):
            if len(bsdf) != len(self.config.model.windows):
                raise ValueError("Number of BSDF should match number of windows.")
        for idx, _name in enumerate(self.config.model.windows):
            _bsdf_key = bsdf[idx] if isinstance(bsdf, list) else bsdf[_name]
            _bsdf = self.config.model.materials.matrices[_bsdf_key].matrix_data
            res.append(
                matrix_multiply_rgb(
                    self.sensor_window_matrices[sensor].array[idx],
                    _bsdf,
                    self.daylight_matrices[_name].array,
                    sky_matrix,
                    weights=[47.4, 119.9, 11.6],
                )
            )
        return np.sum(res, axis=0)

    def calculate_view_from_wea(self, view: str) -> np.ndarray:
        """Calculate(render) view from wea data.

        Args:
            view: The view name
        Returns:
            A series of HDR images as a numpy array
        """
        if self.wea_data is None:
            raise ValueError("No wea data available")
        sky_matrix = self.get_sky_matrix_from_wea(
            int(self.config.settings.sky_basis[-1])
        )
        # arbitrary chunksize
        chunksize = 300
        shape = (
            self.view_senders[view].xres * self.view_senders[view].yres,
            sky_matrix.shape[1],
            3,
        )
        final = np.memmap(
            f"{view}_3ph.dat",
            shape=shape,
            dtype=np.float64,
            mode="w+",
            order="F",
        )
        for idx in range(0, sky_matrix.shape[1], chunksize):
            end = min(idx + chunksize, sky_matrix.shape[1])
            _chunksize = end - idx
            res = np.zeros(
                (
                    self.view_senders[view].xres * self.view_senders[view].yres,
                    _chunksize,
                    3,
                )
            )
            for widx, _name in enumerate(self.config.model.windows):
                res += matrix_multiply_rgb(
                    self.view_window_matrices[view].array[widx],
                    self.window_bsdfs[_name],
                    self.daylight_matrices[_name].array,
                    sky_matrix[:, idx:end, :],
                )
            final[:, idx:end, 0] = res[:, :, 0]
            final[:, idx:end, 1] = res[:, :, 1]
            final[:, idx:end, 2] = res[:, :, 2]
            final.flush()
        return final

    def calculate_sensor_from_wea(self, sensor: str) -> np.ndarray:
        """Calculates the sensor values from wea data.

        Args:
            sensor: The specific sensor for which the calculation is to be
                performed.

        Returns:
            numpy.ndarray: A matrix containing the calculated sensor values based on
                the Weather Attribute data, sensor configuration, and various matrices
                related to windows, daylight, and sky.

        Raises:
            ValueError: If no wea data is available.

        Examples:
            sensor_values = sensor_config.calculate_sensor_from_wea("sensor_name")
        """
        if self.wea_data is None:
            raise ValueError("No wea data available")
        sky_matrix = self.get_sky_matrix_from_wea(
            int(self.config.settings.sky_basis[-1])
        )
        res = np.zeros((self.sensor_senders[sensor].yres, sky_matrix.shape[1]))
        for idx, _name in enumerate(self.config.model.windows):
            res += matrix_multiply_rgb(
                self.sensor_window_matrices[sensor].array[idx],
                self.window_bsdfs[_name],
                self.daylight_matrices[_name].array,
                sky_matrix,
                weights=[47.4, 119.9, 11.6],
            )
        return res

    def calculate_surface(
        self,
        surface: str,
        bsdf: dict[str, str],
        time: datetime,
        dni: float,
        dhi: float,
        solar_spectrum: bool = False,
        sky_matrix: None | np.ndarray = None,
    ) -> np.ndarray:
        weights = [47.4, 119.9, 11.6]
        if solar_spectrum:
            weights = [1.0, 1.0, 1.0]
        if sky_matrix is None:
            sky_matrix = self.get_sky_matrix(
                time, dni, dhi, solar_spectrum=solar_spectrum
            )
        res = np.zeros((self.surface_senders[surface].yres, sky_matrix.shape[1]))
        for idx, _name in enumerate(self.config.model.windows):
            _bsdf = self.config.model.materials.matrices[bsdf[_name]].matrix_data
            res += matrix_multiply_rgb(
                self.surface_window_matrices[surface].array[idx],
                _bsdf,
                self.daylight_matrices[_name].array,
                sky_matrix,
                weights=weights,
            )
        return res

    def calculate_mev(
        self,
        sensor: str,
        bsdf: dict[str, str],
        time: datetime,
        dni: float,
        dhi: float,
        sky_cover: float,
    ) -> np.ndarray:
        """Calculate menalonpic vertical illuminance.

        Args:
            sensor: sensor name, must be in config.model.sensors
            bsdf: a dictionary of window name as key and bsdf matrix or matrix name as value
            time: datetime object
            dni: direct normal irradiance
            dhi: diffuse horizontal irradiance
            sky_cover: sky cover fraction
        Returns:
            Menalonpic vertical illuminance
        """
        outdir = self.config.settings.output_directory
        sky_matrix = self.get_melanopic_sky_matrix(
            time, dni, dhi, sky_cover=sky_cover, outdir=outdir
        )
        res = []
        if isinstance(bsdf, list):
            if len(bsdf) != len(self.config.model.windows):
                raise ValueError("Number of BSDF should match number of windows.")
        for idx, _name in enumerate(self.config.model.windows):
            matrix_name = bsdf[_name]
            _bsdf = self.config.model.materials.matrices_mlnp[matrix_name].matrix_data
            _vmx = self.sensor_window_matrices[sensor].array[idx][:, :, 0]
            _dmx = self.daylight_matrices[_name].array[:, :, 0]
            _smx = sky_matrix[:, :, 0]
            res.append(np.linalg.multi_dot([_vmx, _bsdf, _dmx, _smx]))
        return np.sum(res, axis=0)

    def calculate_edgps(
        self,
        view: str,
        bsdf: dict[str, str],
        time: datetime,
        dni: float,
        dhi: float,
        ambient_bounce: int = 0,
        save_hdr: None | str | Path = None,
    ) -> tuple[float, float]:
        """Calculate enhanced simplified daylight glare probability (EDGPs) for a view.

        Args:
            view: view name, must be in config.model.views
            bsdf: a dictionary of window name as key and bsdf matrix or matrix name as value
            time: datetime object
            dni: direct normal irradiance
            dhi: diffuse horizontal irradiance
            ambient_bounce: ambient bounce, default to 1. Could be set to zero for
                macroscopic non-scattering systems.
        Returns:
            EDGPs
        """
        # generate octree with bsdf
        stdins = []
        stdins.append(
            gen_perez_sky(
                time,
                self.wea_metadata.latitude,
                self.wea_metadata.longitude,
                self.wea_metadata.timezone,
                dirnorm=dni,
                diffhor=dhi,
            )
        )
        for wname, sname in bsdf.items():
            if (_pgs := self.config.model.windows[wname].proxy_geometry) != {}:
                stdins.append(_pgs[sname])

        octree = self.octdir / f"{random_string(5)}.oct"
        with open(octree, "wb") as f:
            f.write(pr.oconv(stdin=b"".join(stdins), octree=self.octree))

        # render image with -ab 1
        params = ["-ab", str(ambient_bounce)]
        hdr = pr.rpict(
            pr.get_view_args(self.view_senders[view].view),
            octree,
            xres=800,
            yres=800,
            params=params,
        )
        ev = self.calculate_sensor(
            view,
            bsdf,
            time,
            dni,
            dhi,
        )
        if save_hdr is not None:
            with open(save_hdr, "wb") as f:
                f.write(hdr)
        res = pr.evalglare(hdr, fast=1, correction_mode="l-", ev=ev.item())
        edgps = float(res)
        os.remove(octree)
        return edgps, ev.item()

    def save_matrices(self):
        """Saves the view window matrices, sensor window matrices, and daylight matrices
        to a NumPy `.npz` file.

        The matrices are saved with keys formed by concatenating the corresponding
        view, sensor, or window name with '_window_matrix' or '_daylight_matrix'.
        """
        matrices = {}
        for view, mtx in self.view_window_matrices.items():
            matrices[f"{view}_window_matrix"] = mtx.array
        for sensor, mtx in self.sensor_window_matrices.items():
            matrices[f"{sensor}_window_matrix"] = mtx.array
        for surface, mtx in self.surface_window_matrices.items():
            matrices[f"{surface}_window_matrix"] = mtx.array
        for window, mtx in self.daylight_matrices.items():
            matrices[f"{window}_daylight_matrix"] = mtx.array
        np.savez(self.mfile, **matrices)


class FivePhaseMethod(PhaseMethod):
    """
    A class representing the Five-Phase Method, which is an extension of the
    three-phase method, allowing for more complex simulations. It includes
    various matrices, octrees, and other attributes used in the simulation.

    Attributes:
        blacked_out_octree: Path to the octree with blacked-out surfaces.
        vmap_oct: Path to the vmap octree.
        cdmap_oct: Path to the cdmap octree.
        window_senders: dictionary of window sender objects.
        window_receivers: dictionary of window receiver objects.
        window_bsdfs: dictionary of window BSDFs.
        view_window_matrices: dictionary of view window matrices.
        sensor_window_matrices: dictionary of sensor window matrices.
        daylight_matrices: dictionary of daylight matrices.
        direct_sun_matrix: Direct sun matrix.
    """

    def __init__(self, config: WorkflowConfig):
        """
        Initializes the FivePhaseMethod object by setting up octrees, matrices,
        and other necessary attributes.

        Reads materials and scene files, constructs necessary octrees, and
        prepares window objects, sun receivers, and various mapping matrices
        based on the provided configuration.

        Args:
            config: WorkflowConfig object containing all necessary information for
                initializing the five-phase method.
        """
        super().__init__(config)
        with open(self.octree, "wb") as f:
            f.write(
                pr.oconv(
                    *config.model.materials.files,
                    *config.model.scene.files,
                    stdin=(config.model.materials.bytes + config.model.scene.bytes),
                )
            )
        self.blacked_out_octree: Path = self.octdir / f"{random_string(5)}.oct"
        # aBSDF mode (FRADS_USE_ABSDF=1): one octree per (pane_idx, state_key);
        # default mode keeps the single blacked_out_octree above.
        self.blacked_out_octrees: dict[tuple[int, str], Path] = {}
        self._absdf = _absdf_enabled()
        self.vmap_oct: Path = self.octdir / f"vmap_{random_string(5)}.oct"
        self.cdmap_oct: Path = self.octdir / f"cdmap_{random_string(5)}.oct"
        self.window_senders: dict[str, SurfaceSender] = {}
        self.window_receivers: dict[str, SurfaceReceiver] = {}
        self.window_bsdfs: dict[str, np.ndarray] = {}
        self.view_window_matrices: dict[str, Matrix] = {}
        self.sensor_window_matrices: dict[str, Matrix] = {}
        self.daylight_matrices: dict[str, Matrix] = {}
        self.view_window_direct_matrices: dict[str, Matrix] = {}
        self.sensor_window_direct_matrices: dict[str, Matrix] = {}
        self.daylight_direct_matrices: dict[str, Matrix] = {}
        # Default mode: flat dict[sensor_name, SunMatrix];
        # aBSDF mode: nested dict[sensor_name, dict[(pane_idx, state_key), SunMatrix]]
        # The runtime aggregation in calculate_sensor/view chooses based on _absdf.
        self.sensor_sun_direct_matrices: dict = {}
        self.view_sun_direct_matrices: dict = {}
        self.view_sun_direct_illuminance_matrices: dict = {}
        self.vmap: dict[str, np.ndarray] = {}
        self.cdmap: dict[str, np.ndarray] = {}
        self.direct_sun_matrix: np.ndarray = self.get_sky_matrix_from_wea(
            mfactor=int(self.config.settings.sun_basis[-1]),
            onesun=True,
            sun_only=True,
        )
        self._prepare_window_objects()
        self._prepare_sun_receivers()
        self.direct_sun_matrix = to_sparse_matrix3(self.direct_sun_matrix)
        self._gen_blacked_out_octree()
        self._prepare_mapping_octrees()
        self._prepare_view_sender_objects()
        self._prepare_sensor_sender_objects()

    def _gen_blacked_out_octree(self):
        black_scene = b"\n".join(
            pr.Xform(s, modifier="black")() for s in self.config.model.scene.files
        )
        if self.config.model.scene.bytes != b"":
            black_scene += pr.Xform(self.config.model.scene.bytes, modifier="black")()
        black = pr.Primitive("void", "plastic", "black", [], [0, 0, 0, 0, 0])
        glow = pr.Primitive("void", "glow", "glowing", [], [1, 1, 1, 0])
        with open(self.blacked_out_octree, "wb") as f:
            f.write(
                pr.oconv(
                    *self.config.model.materials.files,
                    # *self.config.model.windows,
                    stdin=self.config.model.materials.bytes
                    + glow.bytes
                    + black.bytes
                    + black_scene,
                )
            )
        if self._absdf:
            self._gen_per_pane_state_octrees(
                black_scene=black_scene,
                black_bytes=black.bytes,
                glow_bytes=glow.bytes,
            )

    def _gen_per_pane_state_octrees(
        self,
        *,
        black_scene: bytes,
        black_bytes: bytes,
        glow_bytes: bytes,
    ) -> None:
        """Build one ``blacked_out_octree`` per (pane_idx, state_key) for the
        Task-61-conformant aBSDF / Peak-Extraction path.

        Each octree contains:

        - all original materials and ``glow`` for sky sampling
        - the target pane's window polygon **with an aBSDF modifier** pointing
          at the per-state Klems XML (`<state>_klems.xml` in
          ``FRADS_ABSDF_XML_DIR``)
        - the other panes' window polygons forced to ``black`` modifier so
          they absorb instead of transmitting (isolates the target pane's
          direct-sun contribution; required for linear superposition over
          spatially distinct panes)
        - the rest of the scene blackened (no inter-reflections, matching
          McNeil 2013 §3.2-§3.3)

        ``self.blacked_out_octrees[(pane_idx, state_key)]`` holds the
        resulting octree path. The set of state keys comes from
        ``self.config.model.materials.matrices`` (BSDF construction names).
        """
        xml_dir = _absdf_xml_dir()
        pane_names = list(self.config.model.windows.keys())
        state_keys = sorted(self.config.model.materials.matrices.keys())
        if not state_keys:
            raise RuntimeError(
                "FRADS_USE_ABSDF=1 but no Complex Fenestration State matrices "
                "found in materials.matrices."
            )
        # Verify each state has an XML; otherwise raise early.
        missing: list[str] = []
        for sk in state_keys:
            if not (xml_dir / f"{sk}_klems.xml").is_file():
                missing.append(f"{sk}_klems.xml")
        if missing:
            raise FileNotFoundError(
                "FRADS_USE_ABSDF=1 requires per-state Klems XMLs in "
                f"{xml_dir}; missing: {missing}"
            )

        for pane_idx, pane_name in enumerate(pane_names):
            window_bytes = self.config.model.windows[pane_name].bytes
            # Other panes -> blackened polygons
            other_black = b""
            for j, other_name in enumerate(pane_names):
                if j == pane_idx:
                    continue
                other_black += pr.Xform(
                    self.config.model.windows[other_name].bytes,
                    modifier="black",
                )()
            for state_key in state_keys:
                xml_path = xml_dir / f"{state_key}_klems.xml"
                mat_name = f"absdf_p{pane_idx}_{state_key}".replace("-", "_")
                absdf_prim = pr.Primitive(
                    "void",
                    "aBSDF",
                    mat_name,
                    [str(xml_path), "0", "0", "1", "."],
                    [],
                )
                # Window polygon under the aBSDF modifier
                target_pane = pr.Xform(window_bytes, modifier=mat_name)()
                octree_path = (
                    self.octdir / f"absdf_p{pane_idx}_{state_key}_{random_string(4)}.oct"
                )
                with open(octree_path, "wb") as f:
                    f.write(
                        pr.oconv(
                            *self.config.model.materials.files,
                            stdin=(
                                self.config.model.materials.bytes
                                + glow_bytes
                                + black_bytes
                                + absdf_prim.bytes
                                + target_pane
                                + other_black
                                + black_scene
                            ),
                        )
                    )
                self.blacked_out_octrees[(pane_idx, state_key)] = octree_path

    def _prepare_window_objects(self):
        for _name, window in self.config.model.windows.items():
            _prims = pr.parse_primitive(window.bytes)
            self.window_receivers[_name] = SurfaceReceiver(
                _prims, self.config.settings.window_basis
            )
            self.window_senders[_name] = SurfaceSender(
                _prims, self.config.settings.window_basis
            )
            if window.matrix_name != "":
                self.window_bsdfs[_name] = self.config.model.materials.matrices[
                    window.matrix_name
                ].matrix_data
            elif window.matrix_data != []:
                self.window_bsdfs[_name] = np.array(window.matrix_data)
            else:
                raise ValueError("No matrix data or file available", _name)
            self.daylight_matrices[_name] = Matrix(
                self.window_senders[_name],
                [self.sky_receiver],
                self.octree,
            )
            self.daylight_direct_matrices[_name] = Matrix(
                self.window_senders[_name],
                [self.sky_receiver],
                self.blacked_out_octree,
            )

    def _prepare_view_sender_objects(self):
        for _v, sender in self.view_senders.items():
            self.vmap[_v] = load_binary_matrix(
                pr.rtrace(
                    sender.content,
                    params=["-ffd", "-av", ".31831", ".31831", ".31831"],
                    octree=self.vmap_oct,
                ),
                nrows=sender.xres * sender.yres,
                ncols=1,
                ncomp=3,
                dtype="d",
                header=True,
            )
            self.cdmap[_v] = load_binary_matrix(
                pr.rtrace(
                    sender.content,
                    params=["-ffd", "-av", ".31831", ".31831", ".31831"],
                    octree=self.cdmap_oct,
                ),
                nrows=sender.xres * sender.yres,
                ncols=1,
                ncomp=3,
                dtype="d",
                header=True,
            )
            self.view_window_matrices[_v] = Matrix(
                sender, list(self.window_receivers.values()), self.octree
            )
            self.view_window_direct_matrices[_v] = Matrix(
                sender,
                list(self.window_receivers.values()),
                self.blacked_out_octree,
            )
            if self._absdf:
                self.view_sun_direct_matrices[_v] = self._build_per_pane_sunmatrix_dict(
                    sender, self.view_sun_receiver,
                )
                self.view_sun_direct_illuminance_matrices[_v] = self._build_per_pane_sunmatrix_dict(
                    sender, self.view_sun_receiver,
                )
            else:
                self.view_sun_direct_matrices[_v] = SunMatrix(
                    sender, self.view_sun_receiver, self.blacked_out_octree
                )
                self.view_sun_direct_illuminance_matrices[_v] = SunMatrix(
                    sender, self.view_sun_receiver, self.blacked_out_octree
                )

    def _prepare_sensor_sender_objects(self):
        for _s, sender in self.sensor_senders.items():
            self.sensor_window_matrices[_s] = Matrix(
                sender, list(self.window_receivers.values()), self.octree
            )
            self.sensor_window_direct_matrices[_s] = Matrix(
                sender,
                list(self.window_receivers.values()),
                self.blacked_out_octree,
            )
            if self._absdf:
                self.sensor_sun_direct_matrices[_s] = self._build_per_pane_sunmatrix_dict(
                    sender, self.sensor_sun_receiver,
                )
            else:
                self.sensor_sun_direct_matrices[_s] = SunMatrix(
                    sender, self.sensor_sun_receiver, self.blacked_out_octree
                )

    def _build_per_pane_sunmatrix_dict(self, sender, sun_receiver):
        """Build ``dict[(pane_idx, state_key), SunMatrix]`` for one sender.

        Each entry uses the per-pane-state octree from
        ``self.blacked_out_octrees`` so that ``rcontrib`` traces direct-sun
        rays through one pane's aBSDF material while the other panes absorb.
        """
        out: dict[tuple[int, str], SunMatrix] = {}
        for key, octree in self.blacked_out_octrees.items():
            out[key] = SunMatrix(sender, sun_receiver, octree)
        return out

    def _prepare_sun_receivers(self):
        if self.config.settings.sun_culling:
            window_normals = [
                parse_polygon(r.surfaces[0]).normal.tobytes()
                for r in self.window_receivers.values()
            ]
            unique_window_normals = [np.frombuffer(arr) for arr in set(window_normals)]
            self.sensor_sun_receiver = SunReceiver(
                self.config.settings.sun_basis,
                sun_matrix=self.direct_sun_matrix,
                full_mod=True,
            )
            self.view_sun_receiver = SunReceiver(
                self.config.settings.sun_basis,
                sun_matrix=self.direct_sun_matrix,
                window_normals=unique_window_normals,
                full_mod=False,
            )
        else:
            self.sensor_sun_receiver = SunReceiver(
                self.config.settings.sun_basis, full_mod=True
            )
            self.view_sun_receiver = SunReceiver(
                self.config.settings.sun_basis, full_mod=False
            )

    def _prepare_mapping_octrees(self):
        blacked_out_windows = []
        glowing_windows = []
        for _, sender in self.window_senders.items():
            for window in sender.surfaces:
                blacked_out_windows.append(
                    str(
                        pr.Primitive(
                            "black",
                            window.ptype,
                            window.identifier,
                            window.sargs,
                            window.fargs,
                        )
                    )
                )
                glowing_windows.append(
                    str(
                        pr.Primitive(
                            "glowing",
                            window.ptype,
                            window.identifier,
                            window.sargs,
                            window.fargs,
                        )
                    )
                )
        black = pr.Primitive("void", "plastic", "black", [], [0, 0, 0, 0, 0])
        glow = pr.Primitive("void", "glow", "glowing", [], [1, 1, 1, 0])
        blacked_out_windows = str(black) + " ".join(blacked_out_windows)
        glowing_windows = str(glow) + " ".join(glowing_windows)
        with open(self.vmap_oct, "wb") as wtr:
            wtr.write(pr.oconv(stdin=glowing_windows.encode(), octree=self.octree))
        logger.info("Generating view matrix material map octree")
        with open(self.cdmap_oct, "wb") as wtr:
            wtr.write(pr.oconv(stdin=blacked_out_windows.encode(), octree=self.octree))

    def generate_matrices(self, view_matrices: bool = True):
        """Generate all matrices required for the Five-Phase Method.

        Args:
            view_matrices: When False, skip the view-related matrices
                (``view_window_matrices``, ``view_window_direct_matrices``,
                ``view_sun_direct_matrices``,
                ``view_sun_direct_illuminance_matrices``) and only build the
                sensor and daylight matrices.  This matches the existing
                ThreePhaseMethod.generate_matrices signature so callers like
                ``EnergyPlusSetup.initialize_radiance`` (which is sensor-only
                in EnergyPlus integration) can avoid the heavy view-matrix
                rcontrib step.
        """
        if self.mfile.exists():
            if not self.config.settings.overwrite:
                self.load_matrices()
                return
        logger.info("Generating matrices (view_matrices=%s)...", view_matrices)
        logger.info("Step 1/5: Generating window matrices...")
        if view_matrices:
            for mtx in self.view_window_matrices.values():
                mtx.generate(
                    self.config.settings.view_window_matrix,
                    memmap=True,
                    nproc=self.config.settings.num_processors,
                )
        for mtx in self.sensor_window_matrices.values():
            mtx.generate(
                self.config.settings.sensor_window_matrix,
                nproc=self.config.settings.num_processors,
            )
        logger.info("Step 2/5: Generating daylight matrices...")
        for mtx in self.daylight_matrices.values():
            mtx.generate(
                self.config.settings.daylight_matrix,
                nproc=self.config.settings.num_processors,
            )
        logger.info("Step 3/5: Generating direct window matrices...")
        if view_matrices:
            for _, mtx in self.view_window_direct_matrices.items():
                mtx.generate(["-ab", "1"], sparse=True)
        for _, mtx in self.sensor_window_direct_matrices.items():
            mtx.generate(["-ab", "1"], sparse=True)
        logger.info("Step 4/5: Generating direct daylight matrices...")
        for _, mtx in self.daylight_direct_matrices.items():
            mtx.generate(["-ab", "0"], sparse=True)
        logger.info("Step 5/5: Generating direct sun matrices...")
        if self._absdf:
            self._generate_absdf_sun_matrices(view_matrices=view_matrices)
        else:
            for _, mtx in self.sensor_sun_direct_matrices.items():
                mtx.generate(["-ab", "0"])
            if view_matrices:
                for _, mtx in self.view_sun_direct_matrices.items():
                    mtx.generate(["-ab", "0"])
                for _, mtx in self.view_sun_direct_illuminance_matrices.items():
                    mtx.generate(["-ab", "0", "-i+"])
        logger.info("Done!")
        if self.config.settings.save_matrices:
            self.save_matrices()

    def _generate_absdf_sun_matrices(self, view_matrices: bool) -> None:
        """Generate per-pane-state Cds matrices, caching each on disk.

        For the aBSDF / Peak-Extraction path we trace through octrees that
        contain the window aBSDF primitive. rcontrib needs at least one
        ambient bounce (``-ab 1``) so the BSDF material can be sampled --
        ``-ab 0`` would skip the BSDF entirely (see McNeil 2013 §4.1).
        """
        cache_dir = _absdf_cache_dir()
        nproc = self.config.settings.num_processors

        def _key_parts(
            *, role: str, sender_name: str, pane_idx: int, state_key: str
        ) -> list[bytes]:
            xml_bytes = (
                _absdf_xml_dir() / f"{state_key}_klems.xml"
            ).read_bytes()
            return [
                b"absdf-v1",
                role.encode(),
                sender_name.encode(),
                str(pane_idx).encode(),
                state_key.encode(),
                self.config.settings.sun_basis.encode(),
                xml_bytes,
            ]

        def _generate_and_cache(
            role: str, sender_name: str, mtx: SunMatrix, params: list[str],
            pane_idx: int, state_key: str,
        ) -> None:
            def compute() -> np.ndarray:
                mtx.generate(params, nproc=nproc, sparse=False)
                return np.asarray(mtx.array)
            arr = _absdf_cache_get(
                _key_parts(
                    role=role, sender_name=sender_name,
                    pane_idx=pane_idx, state_key=state_key,
                ),
                compute,
                cache_dir,
            )
            mtx.array = arr

        for sensor_name, pane_dict in self.sensor_sun_direct_matrices.items():
            for (pane_idx, state_key), mtx in pane_dict.items():
                _generate_and_cache(
                    "sensor_sun", sensor_name, mtx, ["-ab", "1"],
                    pane_idx, state_key,
                )

        if view_matrices:
            for view_name, pane_dict in self.view_sun_direct_matrices.items():
                for (pane_idx, state_key), mtx in pane_dict.items():
                    _generate_and_cache(
                        "view_sun", view_name, mtx, ["-ab", "1"],
                        pane_idx, state_key,
                    )
            for view_name, pane_dict in self.view_sun_direct_illuminance_matrices.items():
                for (pane_idx, state_key), mtx in pane_dict.items():
                    _generate_and_cache(
                        "view_sun_ill", view_name, mtx, ["-ab", "1", "-i+"],
                        pane_idx, state_key,
                    )

    def load_matrices(self):
        """ """
        logger.info(f"Loading matrices from {self.mfile}")
        mdata = np.load(self.mfile, allow_pickle=True)
        for view, mtx in self.view_window_matrices.items():
            mtx.array = mdata[f"{view}_window_matrix"]
        for sensor, mtx in self.sensor_window_matrices.items():
            mtx.array = mdata[f"{sensor}_window_matrix"]
        for window, mtx in self.daylight_matrices.items():
            mtx.array = mdata[f"{window}_daylight_matrix"]
        for view, mtx in self.view_window_direct_matrices.items():
            mtx.array = mdata[f"{view}_window_direct_matrix"]
        for sensor, mtx in self.sensor_window_direct_matrices.items():
            mtx.array = mdata[f"{sensor}_window_direct_matrix"]
        for window, mtx in self.daylight_direct_matrices.items():
            mtx.array = mdata[f"{window}_daylight_direct_matrix"]
        if self._absdf:
            # aBSDF sun matrices live in their own disk cache (per pane*state);
            # the bulk .npz only carries the diffuse + window matrices above.
            self._generate_absdf_sun_matrices(view_matrices=bool(self.view_sun_direct_matrices))
        else:
            for sensor, mtx in self.sensor_sun_direct_matrices.items():
                mtx.array = mdata[f"{sensor}_sun_direct_matrix"]
            for view, mtx in self.view_sun_direct_matrices.items():
                mtx.array = mdata[f"{view}_sun_direct_matrix"]
            for view, mtx in self.view_sun_direct_illuminance_matrices.items():
                mtx.array = mdata[f"{view}_sun_direct_illuminance_matrix"]

    def calculate_view(
        self,
        view: str,
        bsdf,
        time: datetime,
        dni: float,
        dhi: float,
        sky_scale: float = 1.0,
        sun_scale: float = 1.0,
        cdf_scale: float = 1.0,
    ) -> np.ndarray:
        """Per-timestep Five-Phase Method image calculation.

        Implements the canonical 5PM pixel formula::

            image = sky_scale * (V*T*D*S - Vd*T*Dd*Sd)   # Klems-diffuse
                  + sun_scale * (Cds*Ssun)                # direct sun specular
                  + cdf_scale * (Cdf*Ssun .* cdmap)       # direct sun via aperture

        Args:
            view: View name (key into ``self.view_window_matrices`` etc.).
            bsdf: dict ``{window_name: matrix_key}`` or list of matrix keys
                (one per window, same order as ``config.model.windows``).
            time: datetime for this timestep.
            dni: Direct normal irradiance [W/m^2].
            dhi: Diffuse horizontal irradiance [W/m^2].
            sky_scale: Multiplier on the Klems-diffuse contribution.
                Defaults to 1.0 (canonical 5PM).
            sun_scale: Multiplier on the direct-sun specular contribution.
            cdf_scale: Multiplier on the direct-sun aperture contribution.

        Returns:
            ndarray of shape ``(npixels, 1, 3)`` with per-pixel RGB radiance.
        """
        sky_mfactor = int(self.config.settings.sky_basis[-1])
        sun_mfactor = int(self.config.settings.sun_basis[-1])

        sky_matrix = self.get_sky_matrix(time, dni, dhi)
        _wea_str = self.wea_header + str(WeaData(time, dni, dhi))

        direct_sky = _gendaymtx_direct_sky(
            _wea_str.encode(), sky_mfactor, self._absdf,
        )
        direct_sky_sparse = to_sparse_matrix3(direct_sky)

        direct_sun = _gendaymtx_direct_sun(
            _wea_str.encode(), sun_mfactor, self._absdf,
        )
        direct_sun_sparse = to_sparse_matrix3(direct_sun)

        npix = self.view_window_matrices[view].nrows
        diag_diffuse = np.zeros((npix, 1, 3), dtype=np.float64)
        diag_cdr = np.zeros_like(diag_diffuse)
        diag_cdf = np.zeros_like(diag_diffuse)

        for c in range(3):
            for widx, _name in enumerate(self.config.model.windows):
                _key = bsdf[widx] if isinstance(bsdf, list) else bsdf[_name]
                _T = self.config.model.materials.matrices[_key].matrix_data

                tdmx = np.dot(_T[:, :, c], self.daylight_matrices[_name].array[:, :, c])
                tdsmx = np.dot(tdmx, sky_matrix[:, :1, c])
                vtdsmx = np.dot(
                    self.view_window_matrices[view].array[widx][:, :, c], tdsmx
                )

                tdmx_d = np.dot(
                    csr_matrix(_T[:, :, c]),
                    self.daylight_direct_matrices[_name].array[c],
                )
                tdsmx_d = tdmx_d.dot(direct_sky_sparse[c][:, :1])
                vtdsmx_d = self.view_window_direct_matrices[view].array[widx][c].dot(
                    tdsmx_d
                )
                if hasattr(vtdsmx_d, "toarray"):
                    vtdsmx_d = vtdsmx_d.toarray()

                diag_diffuse[:, :, c] += vtdsmx - vtdsmx_d

        if self._absdf:
            cdr_pane_dict = self.view_sun_direct_matrices[view]
            cdf_pane_dict = self.view_sun_direct_illuminance_matrices[view]
            for pane_idx, _name in enumerate(self.config.model.windows):
                state_key = bsdf[pane_idx] if isinstance(bsdf, list) else bsdf[_name]
                cdr_mtx = cdr_pane_dict[(pane_idx, state_key)]
                cdf_mtx = cdf_pane_dict[(pane_idx, state_key)]
                for c in range(3):
                    cdr = cdr_mtx.array[c].dot(direct_sun_sparse[c][:, :1])
                    cdf_raw = cdf_mtx.array[c].dot(direct_sun_sparse[c][:, :1])
                    cdf = cdf_raw.multiply(csr_matrix(self.cdmap[view][:, :, c]))
                    cdr_d = cdr.toarray() if hasattr(cdr, "toarray") else cdr
                    cdf_d = cdf.toarray() if hasattr(cdf, "toarray") else cdf
                    diag_cdr[:, :, c] += cdr_d
                    diag_cdf[:, :, c] += cdf_d
        else:
            for c in range(3):
                cdr = self.view_sun_direct_matrices[view].array[c].dot(
                    direct_sun_sparse[c][:, :1]
                )
                cdf_raw = self.view_sun_direct_illuminance_matrices[view].array[c].dot(
                    direct_sun_sparse[c][:, :1]
                )
                cdf = cdf_raw.multiply(csr_matrix(self.cdmap[view][:, :, c]))

                cdr_d = cdr.toarray() if hasattr(cdr, "toarray") else cdr
                cdf_d = cdf.toarray() if hasattr(cdf, "toarray") else cdf

                diag_cdr[:, :, c] = cdr_d
                diag_cdf[:, :, c] = cdf_d

        return (sky_scale * diag_diffuse
                + sun_scale * diag_cdr
                + cdf_scale * diag_cdf)

    def calculate_sensor(
        self,
        sensor: str,
        bsdf,
        time: datetime,
        dni: float,
        dhi: float,
        sky_scale: float = 1.0,
        sun_scale: float = 1.0,
    ) -> np.ndarray:
        """Per-timestep Five-Phase Method sensor illuminance calculation.

        Implements the canonical 5PM sensor formula::

            result = sky_scale * (V*T*D*S - Vd*T*Dd*Sd)  # Klems-diffuse
                   + sun_scale * (Cds*Ssun)               # direct sun specular

        (There is no Cdf term for sensors -- the window-visibility-blob path
        is only meaningful per-pixel in an image, not at a point.)

        Args:
            sensor: Sensor name (key into ``self.sensor_window_matrices`` etc.).
            bsdf: dict ``{window_name: matrix_key}`` or list of matrix keys
                (one per window, same order as ``config.model.windows``).
            time: datetime for this timestep.
            dni: Direct normal irradiance [W/m^2].
            dhi: Diffuse horizontal irradiance [W/m^2].
            sky_scale: Multiplier on the Klems-diffuse contribution.
            sun_scale: Multiplier on the direct-sun specular contribution.

        Returns:
            ndarray of illuminance values [lux] for the sensor points.
        """
        # DEBUG: visible marker at function entry
        print(f"[FIVEPHASE_CALC_SENSOR_ENTRY] sensor={sensor} time={time}", flush=True)
        weights = [47.4, 119.9, 11.6]

        sky_mfactor = int(self.config.settings.sky_basis[-1])
        sun_mfactor = int(self.config.settings.sun_basis[-1])

        sky_matrix = self.get_sky_matrix(time, dni, dhi)
        _wea_str = self.wea_header + str(WeaData(time, dni, dhi))

        direct_sky_matrix = _gendaymtx_direct_sky(
            _wea_str.encode(), sky_mfactor, self._absdf,
        )
        direct_sky_matrix_sparse = to_sparse_matrix3(direct_sky_matrix)

        direct_sun_sky = _gendaymtx_direct_sun(
            _wea_str.encode(), sun_mfactor, self._absdf,
        )
        direct_sun_sky_sparse = to_sparse_matrix3(direct_sun_sky)

        # res3: standard 3PM (V * T * D * S) summed over windows
        res3 = np.zeros((self.sensor_senders[sensor].yres, 1))
        for idx, _name in enumerate(self.config.model.windows):
            _bsdf_key = bsdf[idx] if isinstance(bsdf, list) else bsdf[_name]
            _bsdf = self.config.model.materials.matrices[_bsdf_key].matrix_data
            res3 += matrix_multiply_rgb(
                self.sensor_window_matrices[sensor].array[idx],
                _bsdf,
                self.daylight_matrices[_name].array,
                sky_matrix,
                weights=weights,
            )

        # res3d: direct component via sparse Vd*T*Dd*Sd
        res3d = np.zeros((self.sensor_senders[sensor].yres, 1))
        for idx, _name in enumerate(self.config.model.windows):
            _bsdf_key = bsdf[idx] if isinstance(bsdf, list) else bsdf[_name]
            _bsdf = self.config.model.materials.matrices[_bsdf_key].matrix_data
            _res_d = np.zeros((self.sensor_senders[sensor].yres, 1))
            for c, w in enumerate(weights):
                td = np.dot(
                    csr_matrix(_bsdf[:, :, c]),
                    self.daylight_direct_matrices[_name].array[c],
                )
                tds = td.dot(direct_sky_matrix_sparse[c][:, :1])
                vtds = self.sensor_window_direct_matrices[sensor].array[idx][c].dot(tds)
                if hasattr(vtds, "toarray"):
                    vtds = vtds.toarray()
                _res_d += w * vtds
            res3d += _res_d

        # rescd: direct sun component via high-res sun coefficients.
        # aBSDF path: sum over panes, each with its current state's Cds matrix.
        rescd = np.zeros((self.sensor_senders[sensor].yres, 1))
        if self._absdf:
            pane_dict = self.sensor_sun_direct_matrices[sensor]
            for pane_idx, _name in enumerate(self.config.model.windows):
                state_key = bsdf[pane_idx] if isinstance(bsdf, list) else bsdf[_name]
                mtx = pane_dict[(pane_idx, state_key)]
                for c, w in enumerate(weights):
                    cds = mtx.array[c].dot(direct_sun_sky_sparse[c][:, :1])
                    if hasattr(cds, "toarray"):
                        cds = cds.toarray()
                    rescd += w * cds
        else:
            for c, w in enumerate(weights):
                cds = self.sensor_sun_direct_matrices[sensor].array[c].dot(
                    direct_sun_sky_sparse[c][:, :1]
                )
                if hasattr(cds, "toarray"):
                    cds = cds.toarray()
                rescd += w * cds

        result = sky_scale * (res3 - res3d) + sun_scale * rescd

        # DEBUG: visible marker that this code path runs
        print(f"[CALC_SENSOR_5PM] sensor={sensor} time={time} env_breakdown={os.environ.get('FRADS_DEBUG_SENSOR_BREAKDOWN', 'NONE')}", flush=True)
        # Optional sensor-component breakdown for 5PM validation tests.
        # Enabled by FRADS_DEBUG_SENSOR_BREAKDOWN=<csv-path>. One line appended per
        # call: time,sensor,res3,res3d,rescd,result,sky_scale,sun_scale,dni,dhi
        _breakdown_csv = os.environ.get("FRADS_DEBUG_SENSOR_BREAKDOWN")
        if _breakdown_csv:
            try:
                t_str = time.strftime("%Y-%m-%d %H:%M:%S") if hasattr(time, "strftime") else str(time)
                line = (
                    f"{t_str},{sensor},"
                    f"{float(res3.sum()):.4f},"
                    f"{float(res3d.sum()):.4f},"
                    f"{float(rescd.sum()):.4f},"
                    f"{float(result.sum()):.4f},"
                    f"{sky_scale},{sun_scale},{dni},{dhi}\n"
                )
                with open(_breakdown_csv, "a") as _f:
                    _f.write(line)
            except Exception:
                pass

        return result.flatten()

    def calculate_view_from_wea(self, view: str):
        logger.info("Step 1/2: Generating sky matrix from wea")
        sky_matrix = self.get_sky_matrix_from_wea(
            int(self.config.settings.sky_basis[-1])
        )
        direct_sky_matrix = self.get_sky_matrix_from_wea(
            int(self.config.settings.sky_basis[-1]), sun_only=True
        )
        direct_sky_matrix = to_sparse_matrix3(direct_sky_matrix)
        logger.info("Step 2/2: Multiplying matrices...")
        chunksize = 300
        shape = (
            self.view_window_matrices[view].nrows,
            sky_matrix.shape[1],
            3,
        )
        res = np.memmap(
            f"{view}_5ph.dat",
            shape=shape,
            dtype=np.float64,
            mode="w+",
            order="F",
        )
        for idx in range(0, sky_matrix.shape[1], chunksize):
            end = min(idx + chunksize, sky_matrix.shape[1])
            _res = [[], [], []]
            for widx, _name in enumerate(self.config.model.windows):
                for c in range(3):
                    tdmx = np.dot(
                        self.window_bsdfs[_name][:, :, c],
                        self.daylight_matrices[_name].array[:, :, c],
                    )
                    tdsmx = np.dot(tdmx, sky_matrix[:, idx:end, c])
                    vtdsmx = np.dot(
                        self.view_window_matrices[view].array[widx][:, :, c],
                        tdsmx,
                    )
                    tdmx = np.dot(
                        csr_matrix(self.window_bsdfs[_name][:, :, c]),
                        self.daylight_direct_matrices[_name].array[c],
                    )
                    tdsmx = np.dot(tdmx, direct_sky_matrix[c][:, idx:end])
                    vtdsmx_d = np.dot(
                        self.view_window_direct_matrices[view].array[widx][c],
                        tdsmx,
                    )
                    _res[c].append(vtdsmx - vtdsmx_d.toarray())
            for c in range(3):
                cdr = np.dot(
                    self.view_sun_direct_matrices[view].array[c],
                    self.direct_sun_matrix[c][:, idx:end],
                )
                cdf = np.dot(
                    self.view_sun_direct_illuminance_matrices[view].array[c],
                    self.direct_sun_matrix[c][:, idx:end],
                ).multiply(csr_matrix(self.cdmap[view][:, :, c]))
                res[:, idx:end, c] = (
                    np.sum(_res[c], axis=0) + cdr.toarray() + cdf.toarray()
                )
            res.flush()
        return res

    def calculate_sensor_from_wea(self, sensor):
        sky_matrix = self.get_sky_matrix_from_wea(
            int(self.config.settings.sky_basis[-1])
        )
        direct_sky_matrix = self.get_sky_matrix_from_wea(
            int(self.config.settings.sky_basis[-1]), sun_only=True
        )
        direct_sky_matrix = to_sparse_matrix3(direct_sky_matrix)
        res3 = np.zeros((self.sensor_senders[sensor].yres, sky_matrix.shape[1]))
        res3d = np.zeros((self.sensor_senders[sensor].yres, sky_matrix.shape[1]))
        for idx, _name in enumerate(self.config.model.windows):
            res3 += matrix_multiply_rgb(
                self.sensor_window_matrices[sensor].array[idx],
                self.window_bsdfs[_name],
                self.daylight_matrices[_name].array,
                sky_matrix,
                weights=[47.4, 119.9, 11.6],
            )
            res3d += sparse_matrix_multiply_rgb_vtds(
                self.sensor_window_direct_matrices[sensor].array[idx],
                self.window_bsdfs[_name],
                self.daylight_direct_matrices[_name].array,
                direct_sky_matrix,
                weights=[47.4, 119.9, 11.6],
            )
        rescd = np.zeros((self.sensor_senders[sensor].yres, sky_matrix.shape[1]))
        for c, w in enumerate([47.4, 119.9, 11.6]):
            rescd += w * np.dot(
                self.sensor_sun_direct_matrices[sensor].array[c],
                self.direct_sun_matrix[c],
            )
        return res3 - res3d + rescd

    def calculate_dgp(
        self,
        view: str,
        bsdf: dict[str, str],
        time: datetime,
        dni: float,
        dhi: float,
        ev_sensor: str | None = None,
        save_hdr: None | str | Path = None,
    ) -> tuple[float, float]:
        """Daylight Glare Probability via 5PM matrix-image + evalglare.

        Canonical scientific path for BSDF-modeled glare evaluation:

          1. Render fisheye HDR from V*T*D*S - Vd*T*Dd*Sd + Cds*Ssun + Cdf*Ssun
             (McNeil 2013 IEA SHC Task 50; aBSDF peak-extraction Task 61).
          2. Compute Ev via the 5PM sensor matrix at a co-located sensor.
          3. Hand the HDR + external Ev to evalglare (Wienold 2006).

        This replaces the rpict-based calculate_edgps (Wienold 2009
        eDGPs) when the workflow is FivePhaseMethod: rpict cannot
        resolve Klems-BSDF transmission per patch and renders facade
        glass as opaque polygons -- yielding the saturated DGP plateau
        observed for FTG (max 0.018 vs real ~0.29). The 5PM-image path
        uses the SAME calibrated matrices as the sensor pipeline, so
        the DGP and WPI metrics share one numerical model.

        Requires the view matrices (V, Vd, Cds, Cdf) to be populated.
        ``EnergyPlusSetup.initialize_radiance`` calls
        ``generate_matrices(view_matrices=False)`` by default; pass
        ``view_matrices=True`` (the override added in this commit) when
        a calculate_dgp consumer is configured.

        Args:
            view: View name (key into ``self.view_window_matrices``).
            bsdf: dict ``{window_name: matrix_key}`` for current CFS state.
            time, dni, dhi: per-step weather.
            ev_sensor: Sensor name in ``self.sensor_window_matrices``
                co-located with the view. Used for evalglare's external
                Ev calibration. If missing or absent from the matrices,
                evalglare derives Ev from the HDR.
            save_hdr: Optional path to keep the rendered HDR (debug).

        Returns:
            Tuple ``(dgp, ev)``. ``ev`` is the 5PM-sensor Ev in lux
            when ``ev_sensor`` resolves, else 0.0.
        """
        if view not in self.view_window_matrices:
            raise RuntimeError(
                f"calculate_dgp: view '{view}' has no view_window_matrices. "
                "Call initialize_radiance with view_matrices=True (or run "
                "generate_matrices(view_matrices=True) once) before using "
                "this method."
            )
        if self.view_window_matrices[view].array is None:
            raise RuntimeError(
                f"calculate_dgp: view '{view}' has an unpopulated "
                "view_window_matrices entry (Matrix.array is None). "
                "initialize_radiance was called with view_matrices=False; "
                "rerun with view_matrices=True."
            )

        # 1. 5PM image
        img = self.calculate_view(view, bsdf, time, dni, dhi)
        vs = self.view_senders[view]
        yres, xres = vs.yres, vs.xres
        rgb = img.reshape(yres, xres, 3).astype(np.float32)

        # 2. Optional Ev from co-located sensor
        ev_value = 0.0
        if ev_sensor is not None and ev_sensor in self.sensor_window_matrices:
            ev_array = self.calculate_sensor(ev_sensor, bsdf, time, dni, dhi)
            ev_value = float(
                ev_array.item() if ev_array.size == 1 else ev_array.mean()
            )

        # 3. Write HDR to disk (evalglare needs a file path for view header)
        hdr_path = self.octdir / f"dgp_{random_string(5)}.hdr"
        view_args = pr.get_view_args(vs.view)
        view_header = b"VIEW= " + " ".join(view_args).encode() + b"\n"
        _write_radiance_hdr(rgb, hdr_path, extra_header=view_header)
        if save_hdr is not None:
            from shutil import copyfile
            copyfile(hdr_path, save_hdr)

        # 4. evalglare with external Ev when available
        try:
            if ev_value > 0:
                out = pr.evalglare(
                    str(hdr_path), correction_mode="l-", ev=ev_value
                )
            else:
                out = pr.evalglare(str(hdr_path), correction_mode="l-")
            dgp = _parse_evalglare_dgp(out)
        finally:
            try:
                os.remove(hdr_path)
            except OSError:
                pass

        return dgp, ev_value

    def calculate_edgps(
        self,
        view: str,
        bsdf: dict[str, str],
        time: datetime,
        dni: float,
        dhi: float,
        ambient_bounce: int = 0,
        save_hdr: None | str | Path = None,
        ev_sensor: str | None = None,
    ) -> tuple[float, float]:
        """Five-Phase Method eDGP: rpict fisheye + evalglare.

        Mirrors :meth:`ThreePhaseMethod.calculate_edgps` but adapts to the
        5PM architecture where views and sensors are tracked separately.
        Without this override, FivePhaseMethod inherits no eDGP method
        (the 3PM one lives on ``ThreePhaseMethod``, not ``PhaseMethod``),
        so EnergyPlus callbacks calling ``calculate_edgps`` raise
        ``AttributeError`` silently inside pyenergyplus' ctypes layer and
        deadlock the frads-gym main thread waiting on ``obs_data_queue``.

        The vertical eye-illuminance (Ev) used by evalglare for
        calibration is computed from a co-located sensor when
        ``ev_sensor`` names a key of ``self.sensor_window_matrices``.
        Otherwise evalglare derives Ev from the HDR — acceptable for
        unblocking the pipeline; pass ``ev_sensor`` for proper 5PM Ev.
        """
        stdins = [
            gen_perez_sky(
                time,
                self.wea_metadata.latitude,
                self.wea_metadata.longitude,
                self.wea_metadata.timezone,
                dirnorm=dni,
                diffhor=dhi,
            )
        ]
        for wname, sname in bsdf.items():
            if (_pgs := self.config.model.windows[wname].proxy_geometry) != {}:
                stdins.append(_pgs[sname])

        octree = self.octdir / f"edgps_{random_string(5)}.oct"
        with open(octree, "wb") as f:
            f.write(pr.oconv(stdin=b"".join(stdins), octree=self.octree))

        hdr = pr.rpict(
            pr.get_view_args(self.view_senders[view].view),
            octree,
            xres=800,
            yres=800,
            params=["-ab", str(ambient_bounce)],
        )
        if save_hdr is not None:
            with open(save_hdr, "wb") as f:
                f.write(hdr)

        ev_value = 0.0
        if ev_sensor is not None and ev_sensor in self.sensor_window_matrices:
            ev_array = self.calculate_sensor(ev_sensor, bsdf, time, dni, dhi)
            ev_value = float(
                ev_array.item() if ev_array.size == 1 else ev_array.mean()
            )

        if ev_value > 0:
            res = pr.evalglare(hdr, fast=1, correction_mode="l-", ev=ev_value)
        else:
            res = pr.evalglare(hdr, fast=1, correction_mode="l-")
        edgps = float(res)
        os.remove(octree)
        return edgps, ev_value

    def save_matrices(self):
        matrices = {}
        for view, mtx in self.view_window_matrices.items():
            matrices[f"{view}_window_matrix"] = mtx.array
        for sensor, mtx in self.sensor_window_matrices.items():
            matrices[f"{sensor}_window_matrix"] = mtx.array
        for window, mtx in self.daylight_matrices.items():
            matrices[f"{window}_daylight_matrix"] = mtx.array
        for view, mtx in self.view_window_direct_matrices.items():
            matrices[f"{view}_window_direct_matrix"] = mtx.array
        for sensor, mtx in self.sensor_window_direct_matrices.items():
            matrices[f"{sensor}_window_direct_matrix"] = mtx.array
        for window, mtx in self.daylight_direct_matrices.items():
            matrices[f"{window}_daylight_direct_matrix"] = mtx.array
        if not self._absdf:
            # aBSDF sun matrices are persisted via the per-pane disk cache
            # (see _generate_absdf_sun_matrices); keep them out of the bulk
            # .npz so the file stays comparable to upstream Frads.
            for sensor, mtx in self.sensor_sun_direct_matrices.items():
                matrices[f"{sensor}_sun_direct_matrix"] = mtx.array
            for view, mtx in self.view_sun_direct_matrices.items():
                matrices[f"{view}_sun_direct_matrix"] = mtx.array
            for view, mtx in self.view_sun_direct_illuminance_matrices.items():
                matrices[f"{view}_sun_direct_illuminance_matrix"] = mtx.array
        np.savez_compressed(self.mfile, **matrices)


def get_workflow(config):
    workflow = None
    if config.settings.method.lower().startswith(("2", "two")):
        workflow = TwoPhaseMethod(config)
    elif config.settings.method.lower().startswith(("3", "three")):
        workflow = ThreePhaseMethod(config)
    elif config.settings.method.lower().startswith(("5", "five")):
        workflow = FivePhaseMethod(config)
    else:
        raise NotImplementedError("Method not implemented")
    return workflow
