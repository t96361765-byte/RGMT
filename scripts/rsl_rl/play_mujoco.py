"""Deploy an exported Extreme-RGMT policy (TorchScript or ONNX) in MuJoCo.

This entry point is intentionally independent of Isaac Lab.  It reproduces the
policy-facing interface used by ``play.py``:

* MuJoCo physics at 200 Hz and policy inference at 50 Hz.
* A 1728-D observation: 21 x command(38), 10 x proprio(64), 10 x action(29).
* A 29-D joint-position residual added to the current reference pose.
* Isaac-Lab-equivalent joint PD gains and effort limits.
* The original G1 fist/fist-pan MuJoCo XML or compatible fist-pan URDF loader.

The input motion must be a processed 50 Hz RGMT/BeyondMimic NPZ, not a raw
OmniRetarget ``qpos`` file.
"""

from __future__ import annotations

import argparse
import math
import tempfile
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError as error:
    raise ImportError(
        "MuJoCo is required. Install it in the active environment with: pip install mujoco"
    ) from error


POLICY_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)

MOTION_FIELDS = (
    "fps",
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)

COMMAND_STEPS = 21
HISTORY_STEPS = 10
COMMAND_DIM = 38
PROPRIO_DIM = 64
ACTION_DIM = 29
OBSERVATION_DIM = COMMAND_STEPS * COMMAND_DIM + HISTORY_STEPS * (PROPRIO_DIM + ACTION_DIM)


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_policy_path() -> Path:
    return _repository_root() / "logs" / "exported" / "policy.pt"


def _default_model_path() -> Path:
    project_fist_pan = (
        _repository_root()
        / "source"
        / "RGMT"
        / "data"
        / "Robots"
        / "G1"
        / "g1_29dof_fist_pan"
        / "g1_29dof_fist_pan.urdf"
    )
    if project_fist_pan.is_file():
        return project_fist_pan

    # Portable fallbacks retained for deployments copied from the original machine.
    desktop_scene = (
        Path.home()
        / "Desktop"
        / "江苏省前沿技术研发计划"
        / "相关模型"
        / "g1_29dof_fist"
        / "scene_g1_29dof_fist_plane.xml"
    )
    if desktop_scene.is_file():
        return desktop_scene
    return (
        _repository_root().parent
        / "holosoma"
        / "src"
        / "holosoma_retargeting"
        / "holosoma_retargeting"
        / "models"
        / "g1"
        / "g1_29dof_fist.xml"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policy",
        type=Path,
        default=_default_policy_path(),
        help=(
            "Exported policy.pt (TorchScript) or policy.onnx. The backend is selected "
            "from the file suffix (default: logs/exported/policy.pt)."
        ),
    )
    parser.add_argument("--motion-file", type=Path, required=True, help="Processed 50 Hz RGMT NPZ.")
    parser.add_argument("--mushroom", action="store_true", help="Add the stationary training mushroom.")
    parser.add_argument("--mushroom-scale", type=float, default=0.85,
                        help="Uniform mushroom scale; default 0.85 matches TheShy training. Base stays on ground.")
    parser.add_argument(
        "--model",
        "--robot-model",
        dest="model",
        type=Path,
        default=_default_model_path(),
        help=(
            "G1 29-DoF MuJoCo scene XML, or a compatible hand-variant URDF. For a URDF, "
            "the script preserves the default fist MJCF body/contact model and replaces only "
            "the left/right hand mesh, collision, mass, and inertia from the URDF."
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help=(
            "Inference device. PT uses PyTorch; ONNX uses the corresponding ONNX Runtime "
            "execution provider. CPU is the portable default."
        ),
    )
    parser.add_argument("--physics-dt", type=float, default=0.005, help="MuJoCo physics time step.")
    parser.add_argument(
        "--control-decimation",
        type=int,
        default=4,
        help="Physics steps per policy action; 0.005 x 4 gives the trained 50 Hz controller.",
    )
    parser.add_argument("--start-time", type=float, default=0.0, help="Motion start time in seconds.")
    parser.add_argument(
        "--action-scale",
        type=float,
        default=1.0,
        help="Scale applied to the 29-D policy residual before adding it to the reference pose.",
    )
    parser.add_argument(
        "--torque-scale",
        type=float,
        default=1.0,
        help="Uniform scale applied after selecting the effort-limit profile.",
    )
    parser.add_argument(
        "--effort-profile",
        choices=("model", "isaac"),
        default="model",
        help=(
            "Joint effort limits: 'model' uses the limits authored in the G1 MJCF/URDF and "
            "is the higher-confidence sim-to-real default; 'isaac' reproduces the larger "
            "limits used while training this policy in Isaac Lab."
        ),
    )
    parser.add_argument(
        "--shoulder-gain-scale",
        type=float,
        default=1.0,
        help=(
            "Scale shoulder stiffness and damping together. Keep 1.0 for physical validation; "
            "0.9 is an optional, bounded sim-to-sim diagnostic value."
        ),
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="Sleep when MuJoCo is faster than wall time. It cannot compensate for a slow machine.",
    )
    parser.add_argument(
        "--loop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset to --start-time after the clip ends (default: true).",
    )
    parser.add_argument("--headless", action="store_true", help="Run without the MuJoCo viewer.")
    parser.add_argument(
        "--show-collision-geoms",
        action="store_true",
        help="Also render collision primitives. They remain active for physics when hidden.",
    )
    parser.add_argument(
        "--max-control-steps",
        type=int,
        default=None,
        help="Stop after this many 50 Hz policy steps; useful for smoke tests.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="Print tracking/timing diagnostics every N policy steps; use 0 to disable.",
    )
    return parser


def _space_separated(values: str, expected: int, field: str) -> str:
    parsed = values.split()
    if len(parsed) != expected:
        raise ValueError(f"{field} must contain {expected} values, got {values!r}.")
    return " ".join(parsed)


def _included_robot_xml(model_path: Path) -> Path:
    """Resolve the robot XML included by a scene, or return a robot XML unchanged."""
    root = ET.parse(model_path).getroot()
    include = root.find("include")
    if include is None:
        return model_path
    include_file = include.get("file")
    if not include_file:
        raise ValueError(f"MuJoCo include in {model_path} has no file attribute.")
    robot_path = (model_path.parent / include_file).resolve()
    if not robot_path.is_file():
        raise FileNotFoundError(f"Included MuJoCo robot XML not found: {robot_path}")
    return robot_path


def _find_fist_template_for_variant(urdf_path: Path) -> Path:
    """Find the fist MJCF next to a relocated hand-variant URDF when possible."""
    base_name = urdf_path.stem.removesuffix("_pan")
    sibling_model_dir = urdf_path.parent.parent / base_name
    candidates = [
        sibling_model_dir / f"scene_{base_name}_plane.xml",
        sibling_model_dir / f"{base_name}.xml",
        _default_model_path().expanduser().resolve(),
    ]
    checked: list[Path] = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in checked:
            continue
        checked.append(candidate)
        if candidate.is_file() and candidate.suffix.lower() == ".xml":
            return candidate
    locations = "\n  - ".join(str(path) for path in checked)
    raise FileNotFoundError(
        "A compatible fist MJCF XML is required when --robot-model is a URDF. "
        f"Checked:\n  - {locations}"
    )


def _load_hand_variant_urdf(urdf_path: Path) -> tuple[mujoco.MjModel, str]:
    """Build a floating-base MuJoCo model while changing only URDF hand properties.

    Loading a G1 URDF directly in MuJoCo produces a fixed-base model without the
    deployment scene or authored actuator-force ranges.  Instead, use the validated
    fist MJCF as the body/contact template and copy the two hand definitions from the
    requested URDF.  This isolates the intended hand-shape change from unrelated URDF
    collision edits.
    """
    template_scene = _find_fist_template_for_variant(urdf_path)
    template_robot = _included_robot_xml(template_scene)
    mjcf_root = ET.parse(template_robot).getroot()
    urdf_root = ET.parse(urdf_path).getroot()

    mesh_dir = urdf_path.parent / "meshes"
    if not mesh_dir.is_dir():
        raise FileNotFoundError(f"URDF mesh directory not found: {mesh_dir}")
    compiler = mjcf_root.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(mjcf_root, "compiler")
    compiler.set("angle", "radian")
    compiler.set("meshdir", mesh_dir.resolve().as_posix())

    asset = mjcf_root.find("asset")
    worldbody = mjcf_root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError(f"Template MJCF is missing asset/worldbody sections: {template_robot}")

    # Every non-hand mesh is expected to be an unchanged copy of the fist model.
    missing_meshes = []
    for mesh in asset.findall("mesh"):
        mesh_file = mesh.get("file")
        is_replaced_hand = mesh.get("name") in {"left_rubber_hand", "right_rubber_hand"}
        if mesh_file and not is_replaced_hand and not (mesh_dir / Path(mesh_file).name).is_file():
            missing_meshes.append(Path(mesh_file).name)
        elif mesh_file:
            mesh.set("file", Path(mesh_file).name)
    if missing_meshes:
        raise FileNotFoundError(
            f"Hand-variant directory is missing template meshes: {sorted(missing_meshes)}"
        )

    for side in ("left", "right"):
        link_name = f"{side}_rubber_hand"
        urdf_link = urdf_root.find(f"./link[@name='{link_name}']")
        mjcf_body = mjcf_root.find(f".//body[@name='{link_name}']")
        mesh_asset = asset.find(f"./mesh[@name='{link_name}']")
        if urdf_link is None or mjcf_body is None or mesh_asset is None:
            raise ValueError(f"Compatible model must define {link_name} in both URDF and MJCF.")

        urdf_visual = urdf_link.find("visual")
        urdf_collision = urdf_link.find("collision")
        urdf_inertial = urdf_link.find("inertial")
        if urdf_visual is None or urdf_collision is None or urdf_inertial is None:
            raise ValueError(f"URDF link {link_name} must contain visual, collision, and inertial.")

        urdf_mesh = urdf_visual.find("geometry/mesh")
        cylinder = urdf_collision.find("geometry/cylinder")
        mass = urdf_inertial.find("mass")
        inertia = urdf_inertial.find("inertia")
        if urdf_mesh is None or cylinder is None or mass is None or inertia is None:
            raise ValueError(
                f"URDF link {link_name} must use a mesh visual, cylinder collision, mass, and inertia."
            )
        mesh_file = Path(urdf_mesh.get("filename", "")).name
        if not mesh_file or not (mesh_dir / mesh_file).is_file():
            raise FileNotFoundError(f"URDF hand mesh not found for {link_name}: {mesh_dir / mesh_file}")
        mesh_asset.set("file", mesh_file)
        mesh_asset.set(
            "scale",
            _space_separated(urdf_mesh.get("scale", "1 1 1"), 3, f"{link_name} mesh scale"),
        )

        visual_origin = urdf_visual.find("origin")
        collision_origin = urdf_collision.find("origin")
        inertial_origin = urdf_inertial.find("origin")
        visual_geom = mjcf_body.find("./geom[@type='mesh']")
        collision_geom = next(
            (geom for geom in mjcf_body.findall("geom") if geom.get("type") != "mesh"), None
        )
        mjcf_inertial = mjcf_body.find("inertial")
        if visual_geom is None or collision_geom is None or mjcf_inertial is None:
            raise ValueError(f"Template MJCF hand body is incomplete: {link_name}")

        visual_geom.set("mesh", link_name)
        visual_geom.set("pos", _space_separated(visual_origin.get("xyz", "0 0 0"), 3, "visual xyz"))
        visual_geom.set("euler", _space_separated(visual_origin.get("rpy", "0 0 0"), 3, "visual rpy"))
        visual_geom.attrib.pop("quat", None)

        radius = float(cylinder.get("radius", "nan"))
        length = float(cylinder.get("length", "nan"))
        if not math.isfinite(radius) or not math.isfinite(length) or radius <= 0.0 or length <= 0.0:
            raise ValueError(f"Invalid hand collision cylinder for {link_name}.")
        collision_geom.set("type", "cylinder")
        collision_geom.set("size", f"{radius:.12g} {0.5 * length:.12g}")
        collision_geom.set("pos", _space_separated(collision_origin.get("xyz", "0 0 0"), 3, "collision xyz"))
        collision_geom.set("euler", _space_separated(collision_origin.get("rpy", "0 0 0"), 3, "collision rpy"))
        collision_geom.attrib.pop("quat", None)
        collision_geom.attrib.pop("mesh", None)

        inertial_rpy = _space_separated(inertial_origin.get("rpy", "0 0 0"), 3, "inertial rpy")
        if any(abs(float(value)) > 1.0e-12 for value in inertial_rpy.split()):
            raise ValueError(f"Rotated hand inertials are not supported for {link_name}.")
        inertia_values = [
            inertia.get("ixx"), inertia.get("iyy"), inertia.get("izz"),
            inertia.get("ixy"), inertia.get("ixz"), inertia.get("iyz"),
        ]
        if any(value is None for value in inertia_values):
            raise ValueError(f"Incomplete inertia tensor for {link_name}.")
        mjcf_inertial.set("pos", _space_separated(inertial_origin.get("xyz", "0 0 0"), 3, "inertial xyz"))
        mjcf_inertial.set("mass", mass.get("value", ""))
        mjcf_inertial.set("fullinertia", " ".join(inertia_values))
        mjcf_inertial.attrib.pop("diaginertia", None)
        mjcf_inertial.attrib.pop("quat", None)

    # Restore the deployment scene's checkerboard material as well as its collision plane.
    if asset.find("./texture[@name='groundplane']") is None:
        ET.SubElement(
            asset,
            "texture",
            {
                "name": "groundplane",
                "type": "2d",
                "builtin": "checker",
                "mark": "edge",
                "rgb1": "0.2 0.3 0.4",
                "rgb2": "0.1 0.2 0.3",
                "markrgb": "0.8 0.8 0.8",
                "width": "300",
                "height": "300",
            },
        )
    if asset.find("./material[@name='groundplane']") is None:
        ET.SubElement(
            asset,
            "material",
            {
                "name": "groundplane",
                "texture": "groundplane",
                "texuniform": "true",
                "texrepeat": "5 5",
                "reflectance": "0.2",
            },
        )

    ground_geom = worldbody.find("./geom[@name='ground']")
    if ground_geom is None:
        ground_geom = ET.SubElement(
            worldbody,
            "geom",
            {
                "name": "ground",
                "type": "plane",
                "size": "0 0 0.01",
                "friction": "1.0 0.005 0.0001",
                "condim": "3",
                "group": "2",
            },
        )
    ground_geom.set("material", "groundplane")
    ground_geom.attrib.pop("rgba", None)
    if worldbody.find("./light") is None:
        ET.SubElement(
            worldbody,
            "light",
            {"pos": "0 0 1.5", "dir": "0 0 -1", "directional": "true"},
        )

    mjcf_root.set("model", f"{urdf_path.stem}_deployment")
    model = mujoco.MjModel.from_xml_string(ET.tostring(mjcf_root, encoding="unicode"))
    note = f"hand variant from {urdf_path}; body/contact template {template_robot}"
    return model, note


def _load_floating_urdf(urdf_path: Path, mushroom_scale: float | None = None) -> tuple[mujoco.MjModel, str]:
    """Compile the project G1 URDF as a free-base MuJoCo scene.

    MuJoCo fixes a normal URDF root to the world and fuses the pelvis away.  Insert
    an explicit floating joint before compilation, save the compiled representation
    as MJCF, and add the deployment ground plane and light to that representation.
    """
    urdf_root = ET.parse(urdf_path).getroot()
    if urdf_root.tag != "robot":
        raise ValueError(f"Expected a URDF <robot> root: {urdf_path}")
    if urdf_root.find("./link[@name='world']") is not None:
        raise ValueError("URDF already defines a link named 'world'; cannot add floating base.")
    if urdf_root.find("./joint[@name='floating_base']") is not None:
        raise ValueError("URDF already defines a joint named 'floating_base'.")
    if urdf_root.find("./link[@name='pelvis']") is None:
        raise ValueError(f"G1 URDF is missing its pelvis root link: {urdf_path}")

    mujoco_extension = urdf_root.find("mujoco")
    if mujoco_extension is None:
        mujoco_extension = ET.Element("mujoco")
        urdf_root.insert(0, mujoco_extension)
    compiler = mujoco_extension.find("compiler")
    if compiler is None:
        compiler = ET.SubElement(mujoco_extension, "compiler")
    # Mesh filenames in the URDF already begin with ``meshes/``.
    compiler.set("meshdir", str(urdf_path.parent.resolve()))
    compiler.set("discardvisual", "false")

    world_link = ET.Element("link", {"name": "world"})
    floating_joint = ET.Element("joint", {"name": "floating_base", "type": "floating"})
    ET.SubElement(floating_joint, "parent", {"link": "world"})
    ET.SubElement(floating_joint, "child", {"link": "pelvis"})
    urdf_root.insert(1, world_link)
    urdf_root.insert(2, floating_joint)

    urdf_xml = ET.tostring(urdf_root, encoding="unicode")
    urdf_model = mujoco.MjModel.from_xml_string(urdf_xml)

    # MuJoCo exposes its URDF-to-MJCF result only through mj_saveLastXML.
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as temporary_file:
        compiled_path = Path(temporary_file.name)
    try:
        mujoco.mj_saveLastXML(str(compiled_path), urdf_model)
        mjcf_root = ET.parse(compiled_path).getroot()
    finally:
        compiled_path.unlink(missing_ok=True)

    asset = mjcf_root.find("asset")
    if asset is None:
        asset = ET.SubElement(mjcf_root, "asset")
    if asset.find("./texture[@name='groundplane']") is None:
        ET.SubElement(
            asset,
            "texture",
            {
                "name": "groundplane",
                "type": "2d",
                "builtin": "checker",
                "mark": "edge",
                "rgb1": "0.2 0.3 0.4",
                "rgb2": "0.1 0.2 0.3",
                "markrgb": "0.8 0.8 0.8",
                "width": "300",
                "height": "300",
            },
        )
    if asset.find("./material[@name='groundplane']") is None:
        ET.SubElement(
            asset,
            "material",
            {
                "name": "groundplane",
                "texture": "groundplane",
                "texuniform": "true",
                "texrepeat": "5 5",
                "reflectance": "0.2",
            },
        )

    worldbody = mjcf_root.find("worldbody")
    if worldbody is None:
        raise ValueError("MuJoCo's compiled URDF is missing its worldbody.")
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "ground",
            "type": "plane",
            "size": "0 0 0.01",
            "friction": "1.0 0.005 0.0001",
            "condim": "3",
            "group": "2",
            "material": "groundplane",
        },
    )
    ET.SubElement(
        worldbody,
        "light",
        {"pos": "0 0 1.5", "dir": "0 0 -1", "directional": "true"},
    )

    if mushroom_scale is not None:
        _add_mushroom(mjcf_root, mushroom_scale)
    model = mujoco.MjModel.from_xml_string(ET.tostring(mjcf_root, encoding="unicode"))
    note = f"floating-base G1 scene compiled from {urdf_path}"
    return model, note


def _add_mushroom(root: ET.Element, scale: float) -> None:
    """Add a fixed apparatus, not the freejoint in the standalone example scene."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("--mushroom-scale must be finite and positive")
    mesh = _repository_root() / "source/RGMT/data/Objects/mushroom/mujoco_xml/mushroom_visual.obj"
    if not mesh.is_file():
        raise FileNotFoundError(f"Mushroom mesh missing: {mesh}")
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    rgba = "0.8 0.8 0.8 1"
    material_args = {"name": "rgmt_mushroom_material"}
    mtl = mesh.with_suffix(".mtl")
    if mtl.is_file():
        for line in mtl.read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if fields and fields[0] == "Kd" and len(fields) == 4:
                rgba = " ".join(fields[1:]) + " 1"
            elif fields and fields[0] == "map_Kd":
                texture = mtl.parent / line.strip()[7:].strip()
                if texture.is_file():
                    ET.SubElement(asset, "texture", name="rgmt_mushroom_texture", type="2d", file=str(texture.resolve()))
                    material_args["texture"] = "rgmt_mushroom_texture"
    ET.SubElement(asset, "material", rgba=rgba, **material_args)
    ET.SubElement(asset, "mesh", name="rgmt_mushroom_mesh", file=str(mesh.resolve()),
                  scale=f"{scale} {scale} {scale}")
    world = root.find("worldbody")
    if world is None:
        raise ValueError("Scene lacks worldbody")
    height = 0.276493043 * scale
    body = ET.SubElement(world, "body", name="rgmt_mushroom", pos=f"0 0 {height:.12g}")
    # MuJoCo uses a convex hull for mesh contacts; this is not an exact PhysX
    # triangle-mesh contact equivalent. The original mesh is retained for display.
    ET.SubElement(body, "geom", name="rgmt_mushroom_collision", type="mesh", mesh="rgmt_mushroom_mesh",
                  material="rgmt_mushroom_material", group="0", contype="1", conaffinity="1",
                  friction="2 0.005 0.0001", condim="3", priority="1")
    # Viewer hides collision group 0 by default. Render the original mesh in
    # group 1 independently, without adding contacts or inertial contribution.
    ET.SubElement(body, "geom", name="rgmt_mushroom_visual", type="mesh", mesh="rgmt_mushroom_mesh",
                  material="rgmt_mushroom_material", group="1", contype="0", conaffinity="0",
                  density="0")
    print(f"[INFO]: Fixed mushroom: scale={scale:g}, position=(0, 0, {height:.12g}); "
          "MTL appearance loaded when available; MuJoCo convex-hull mesh collision.")


def _load_mujoco_model(model_path: Path, mushroom_scale: float | None = None) -> tuple[mujoco.MjModel, str]:
    suffix = model_path.suffix.lower()
    if suffix == ".xml":
        if mushroom_scale is not None:
            raise ValueError("--mushroom currently requires --robot-model pointing to the G1 URDF")
        return mujoco.MjModel.from_xml_path(str(model_path)), "full MuJoCo XML"
    if suffix == ".urdf":
        return _load_floating_urdf(model_path, mushroom_scale)
    raise ValueError(f"--robot-model must be a MuJoCo .xml or compatible G1 .urdf: {model_path}")


def _normalize_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if np.any(~np.isfinite(norm)) or np.any(norm < 1.0e-8):
        raise ValueError("Encountered a non-finite or zero quaternion.")
    return quaternion / norm


def _quaternion_matrix_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = _normalize_quaternion_wxyz(np.asarray(quaternion, dtype=np.float64))
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _rotate_inverse_wxyz(quaternion: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """Rotate world-frame vector(s) into the frame represented by ``quaternion``."""
    quaternion = np.asarray(quaternion, dtype=np.float64)
    vectors = np.asarray(vectors, dtype=np.float64)
    if quaternion.ndim == 1:
        return vectors @ _quaternion_matrix_wxyz(quaternion)
    matrices = np.stack([_quaternion_matrix_wxyz(value) for value in quaternion], axis=0)
    return np.einsum("...ji,...j->...i", matrices, vectors)


@dataclass(frozen=True)
class MotionClip:
    path: Path
    fps: float
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_lin_vel_w: np.ndarray
    body_ang_vel_w: np.ndarray

    @property
    def frame_count(self) -> int:
        return int(self.joint_pos.shape[0])

    @property
    def duration(self) -> float:
        return (self.frame_count - 1) / self.fps

    @classmethod
    def load(cls, path: Path) -> MotionClip:
        path = path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Motion NPZ not found: {path}")
        with np.load(path, allow_pickle=False) as source:
            missing = [name for name in MOTION_FIELDS if name not in source]
            if missing:
                raise KeyError(
                    f"{path} is not a processed RGMT NPZ; missing fields: {missing}. "
                    "Convert the raw OmniRetarget qpos file first."
                )
            fps_values = np.asarray(source["fps"]).reshape(-1)
            if fps_values.size != 1:
                raise ValueError(f"{path} must contain exactly one fps value.")
            fps = float(fps_values[0])
            arrays = {name: np.asarray(source[name], dtype=np.float64) for name in MOTION_FIELDS[1:]}

        if not np.isclose(fps, 50.0, rtol=0.0, atol=1.0e-6):
            raise ValueError(f"{path} has fps={fps:g}; Extreme-RGMT deployment requires 50 Hz.")
        joint_pos = arrays["joint_pos"]
        if joint_pos.ndim != 2 or joint_pos.shape[1] != ACTION_DIM:
            raise ValueError(f"{path} joint_pos must have shape [T, 29], got {joint_pos.shape}.")
        expected_frames = joint_pos.shape[0]
        expected_shapes = {
            "joint_vel": (expected_frames, ACTION_DIM),
            "body_pos_w": (expected_frames, 30, 3),
            "body_quat_w": (expected_frames, 30, 4),
            "body_lin_vel_w": (expected_frames, 30, 3),
            "body_ang_vel_w": (expected_frames, 30, 3),
        }
        for name, expected in expected_shapes.items():
            if arrays[name].shape != expected:
                raise ValueError(f"{path} {name} must have shape {expected}, got {arrays[name].shape}.")
        for name, value in arrays.items():
            if not np.isfinite(value).all():
                raise ValueError(f"{path} contains non-finite values in {name}.")
        if expected_frames < 2:
            raise ValueError(f"{path} must contain at least two frames.")

        arrays["body_quat_w"] = _normalize_quaternion_wxyz(arrays["body_quat_w"])
        return cls(path=path, fps=fps, **arrays)

    def sample(self, field: str, times: float | np.ndarray) -> np.ndarray:
        values = getattr(self, field)
        query = np.clip(np.asarray(times, dtype=np.float64), 0.0, self.duration)
        frame = query * self.fps
        lower_index = np.floor(frame).astype(np.int64)
        upper_index = np.minimum(lower_index + 1, self.frame_count - 1)
        alpha = frame - lower_index
        value0 = values[lower_index]
        value1 = values[upper_index]
        blend = alpha.reshape(alpha.shape + (1,) * (value0.ndim - alpha.ndim))
        if field == "body_quat_w":
            sign = np.where(np.sum(value0 * value1, axis=-1, keepdims=True) < 0.0, -1.0, 1.0)
            return _normalize_quaternion_wxyz(value0 + blend * (value1 * sign - value0))
        return value0 + blend * (value1 - value0)

    def root_state(self, motion_time: float) -> tuple[np.ndarray, ...]:
        # The processed schema always stores pelvis at body index zero.
        return (
            self.sample("body_pos_w", motion_time)[0],
            self.sample("body_quat_w", motion_time)[0],
            self.sample("body_lin_vel_w", motion_time)[0],
            self.sample("body_ang_vel_w", motion_time)[0],
        )


@dataclass(frozen=True)
class ModelBindings:
    qpos_addresses: np.ndarray
    dof_addresses: np.ndarray
    default_joint_positions: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    effort_limits: np.ndarray
    pelvis_body_id: int

    @classmethod
    def create(
        cls,
        model: mujoco.MjModel,
        torque_scale: float,
        effort_profile: str,
        shoulder_gain_scale: float,
    ) -> ModelBindings:
        qpos_addresses: list[int] = []
        dof_addresses: list[int] = []
        for name in POLICY_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"MuJoCo model is missing required joint: {name}")
            if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError(f"Required joint is not a hinge: {name}")
            qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
            dof_addresses.append(int(model.jnt_dofadr[joint_id]))

        pelvis_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if pelvis_body_id < 0:
            raise ValueError("MuJoCo model is missing the pelvis body.")

        default_positions = np.zeros(ACTION_DIM, dtype=np.float64)
        stiffness = np.empty(ACTION_DIM, dtype=np.float64)
        damping = np.empty(ACTION_DIM, dtype=np.float64)
        effort_limits = np.empty(ACTION_DIM, dtype=np.float64)
        for index, name in enumerate(POLICY_JOINT_NAMES):
            if "hip_" in name:
                stiffness[index], damping[index], effort_limits[index] = 100.0, 2.5, 100.0
                if "hip_pitch" in name:
                    default_positions[index] = -0.20
            elif "knee" in name:
                stiffness[index], damping[index], effort_limits[index] = 150.0, 4.0, 150.0
                default_positions[index] = 0.40
            elif "ankle" in name:
                stiffness[index], damping[index], effort_limits[index] = 40.0, 2.0, 40.0
                if "ankle_pitch" in name:
                    default_positions[index] = -0.20
            elif "waist" in name:
                stiffness[index], damping[index], effort_limits[index] = 150.0, 4.0, 150.0
            elif "wrist" in name:
                stiffness[index], damping[index], effort_limits[index] = 20.0, 1.0, 20.0
            else:
                stiffness[index], damping[index], effort_limits[index] = 40.0, 5.0, 40.0
                if "shoulder_roll" in name:
                    default_positions[index] = 0.30 if name.startswith("left") else -0.30
                elif "elbow" in name:
                    default_positions[index] = 1.00

            if "shoulder" in name:
                stiffness[index] *= shoulder_gain_scale
                damping[index] *= shoulder_gain_scale

            if effort_profile == "model":
                joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                force_range = np.asarray(model.jnt_actfrcrange[joint_id], dtype=np.float64)
                authored_limit = min(abs(float(force_range[0])), abs(float(force_range[1])))
                if authored_limit <= 0.0:
                    raise ValueError(
                        f"MuJoCo joint {name} has no positive actuator-force range; "
                        "use --effort-profile isaac or fix the MJCF."
                    )
                effort_limits[index] = authored_limit

        effort_limits *= torque_scale
        return cls(
            qpos_addresses=np.asarray(qpos_addresses, dtype=np.int32),
            dof_addresses=np.asarray(dof_addresses, dtype=np.int32),
            default_joint_positions=default_positions,
            stiffness=stiffness,
            damping=damping,
            effort_limits=effort_limits,
            pelvis_body_id=pelvis_body_id,
        )


class TorchScriptPolicy:
    """Inference adapter for an exported ``policy.pt`` TorchScript module."""

    backend_name = "TorchScript"

    def __init__(self, path: Path, device_name: str):
        try:
            import torch
        except ImportError as error:
            raise ImportError(
                "TorchScript policy selected, but PyTorch is not installed."
            ) from error
        self.torch = torch
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available to PyTorch.")
        self.device = torch.device(device_name)
        try:
            self.module = torch.jit.load(str(path), map_location=self.device)
        except (RuntimeError, ValueError) as error:
            raise RuntimeError(
                f"Failed to load {path} as an exported TorchScript policy.pt. "
                "Use the exported/policy.pt produced by scripts/rsl_rl/play.py; "
                "RSL-RL model_*.pt training checkpoints are not supported."
            ) from error
        self.module.eval()

    @property
    def device_description(self) -> str:
        return str(self.device)

    def infer(self, observation: np.ndarray) -> np.ndarray:
        tensor = self.torch.from_numpy(observation).to(self.device)
        with self.torch.inference_mode():
            actions = self.module(tensor)
        return actions.detach().cpu().numpy()

    def reset(self) -> None:
        if hasattr(self.module, "reset"):
            self.module.reset()


class OnnxPolicy:
    """Inference adapter for an exported ``policy.onnx`` model."""

    backend_name = "ONNX Runtime"

    def __init__(self, path: Path, device_name: str):
        try:
            import onnxruntime as ort
        except ImportError as error:
            raise ImportError(
                "ONNX policy selected, but onnxruntime is not installed. Install "
                "onnxruntime for CPU inference or onnxruntime-gpu for CUDA inference."
            ) from error

        available_providers = ort.get_available_providers()
        if device_name == "cuda":
            provider = "CUDAExecutionProvider"
            if provider not in available_providers:
                raise RuntimeError(
                    "--device cuda requested for ONNX, but CUDAExecutionProvider is unavailable. "
                    f"Available providers: {available_providers}. Install onnxruntime-gpu or use "
                    "--device cpu."
                )
            # Prefer the CUDA/cuDNN DLLs bundled with the active PyTorch installation.
            # This avoids relying on a separately configured system CUDA PATH.
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
            providers = [provider, "CPUExecutionProvider"]
        else:
            provider = "CPUExecutionProvider"
            if provider not in available_providers:
                raise RuntimeError(
                    f"CPUExecutionProvider is unavailable; available providers: {available_providers}."
                )
            providers = [provider]

        try:
            self.session = ort.InferenceSession(str(path), providers=providers)
        except Exception as error:
            raise RuntimeError(f"Failed to load exported ONNX policy: {path}") from error
        active_providers = self.session.get_providers()
        if provider not in active_providers:
            raise RuntimeError(
                f"Requested {provider}, but ONNX Runtime activated {active_providers}. "
                "Refusing to silently fall back to another inference device."
            )
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                f"Policy ONNX must have one input and one output, got {len(inputs)} and {len(outputs)}."
            )
        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.device_description = provider

    def infer(self, observation: np.ndarray) -> np.ndarray:
        inputs = np.ascontiguousarray(observation, dtype=np.float32)
        return np.asarray(
            self.session.run([self.output_name], {self.input_name: inputs})[0]
        )

    def reset(self) -> None:
        # The exported Extreme-RGMT actor is feed-forward; history lives in the observation.
        return None


def _load_policy(path: Path, device_name: str) -> TorchScriptPolicy | OnnxPolicy:
    suffix = path.suffix.lower()
    if suffix == ".pt":
        return TorchScriptPolicy(path, device_name)
    if suffix == ".onnx":
        return OnnxPolicy(path, device_name)
    raise ValueError(
        f"Unsupported policy file extension {path.suffix!r}. Use exported policy.pt or policy.onnx."
    )


class ExtremeRGMTMuJoCoDeployment:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.policy_path = args.policy.expanduser().resolve()
        self.model_path = args.model.expanduser().resolve()
        if not self.policy_path.is_file():
            raise FileNotFoundError(f"Exported policy not found: {self.policy_path}")
        if not self.model_path.is_file():
            raise FileNotFoundError(f"MuJoCo XML/URDF not found: {self.model_path}")
        if args.physics_dt <= 0.0:
            raise ValueError("--physics-dt must be positive.")
        if args.control_decimation <= 0:
            raise ValueError("--control-decimation must be positive.")
        if (
            args.action_scale <= 0.0
            or args.torque_scale <= 0.0
            or args.shoulder_gain_scale <= 0.0
        ):
            raise ValueError(
                "--action-scale, --torque-scale, and --shoulder-gain-scale must be positive."
            )
        if args.max_control_steps is not None and args.max_control_steps <= 0:
            raise ValueError("--max-control-steps must be positive.")
        if args.log_interval < 0:
            raise ValueError("--log-interval cannot be negative.")

        self.control_dt = args.physics_dt * args.control_decimation
        if not math.isclose(self.control_dt, 0.02, rel_tol=0.0, abs_tol=1.0e-9):
            raise ValueError(
                f"Extreme-RGMT was trained at 50 Hz, but physics_dt * decimation = {self.control_dt:g}s. "
                "Use --physics-dt 0.005 --control-decimation 4."
            )

        self.motion = MotionClip.load(args.motion_file)
        if not 0.0 <= args.start_time < self.motion.duration:
            raise ValueError(
                f"--start-time must be in [0, {self.motion.duration:.6f}), got {args.start_time}."
            )

        self.model, self.model_load_note = _load_mujoco_model(
            self.model_path, args.mushroom_scale if args.mushroom else None
        )
        self.model.opt.timestep = args.physics_dt
        self.bindings = ModelBindings.create(
            self.model,
            args.torque_scale,
            args.effort_profile,
            args.shoulder_gain_scale,
        )
        # Isaac Lab uses 0.01 armature for every actuated joint.
        self.model.dof_armature[self.bindings.dof_addresses] = 0.01
        self.data = mujoco.MjData(self.model)

        self.policy = _load_policy(self.policy_path, args.device)
        test_actions = self.policy.infer(np.zeros((1, OBSERVATION_DIM), dtype=np.float32))
        if test_actions.shape != (1, ACTION_DIM):
            raise ValueError(
                f"Policy output must have shape [1, {ACTION_DIM}], got {test_actions.shape}."
            )
        if not np.isfinite(test_actions).all():
            raise ValueError("Policy produced non-finite actions for the zero-input validation.")

        self.proprio_history = np.zeros((HISTORY_STEPS, PROPRIO_DIM), dtype=np.float32)
        self.action_history = np.zeros((HISTORY_STEPS, ACTION_DIM), dtype=np.float32)
        self.actions = np.zeros(ACTION_DIM, dtype=np.float64)
        self.motion_time = float(args.start_time)
        self.control_steps = 0
        self.wall_start = 0.0
        self._reset(self.motion_time)

    def _joint_positions(self) -> np.ndarray:
        return np.asarray(self.data.qpos[self.bindings.qpos_addresses], dtype=np.float64)

    def _joint_velocities(self) -> np.ndarray:
        return np.asarray(self.data.qvel[self.bindings.dof_addresses], dtype=np.float64)

    def _build_proprioception(self) -> np.ndarray:
        root_quaternion = np.asarray(self.data.qpos[3:7], dtype=np.float64)
        projected_gravity = _rotate_inverse_wxyz(root_quaternion, np.asarray([0.0, 0.0, -1.0]))
        # MuJoCo freejoint angular qvel is already expressed in the body-local frame.
        angular_velocity_body = np.asarray(self.data.qvel[3:6], dtype=np.float64)
        joint_error = self._joint_positions() - self.bindings.default_joint_positions
        proprioception = np.concatenate(
            (projected_gravity, angular_velocity_body, joint_error, self._joint_velocities())
        )
        if proprioception.shape != (PROPRIO_DIM,):
            raise RuntimeError(f"Proprioception has unexpected shape: {proprioception.shape}")
        return proprioception.astype(np.float32)

    def _build_command_window(self) -> np.ndarray:
        offsets = np.arange(-(COMMAND_STEPS // 2), COMMAND_STEPS // 2 + 1, dtype=np.float64)
        query_times = np.clip(self.motion_time + offsets * self.control_dt, 0.0, self.motion.duration)
        root_quaternion = self.motion.sample("body_quat_w", query_times)[:, 0]
        linear_velocity_world = self.motion.sample("body_lin_vel_w", query_times)[:, 0]
        angular_velocity_world = self.motion.sample("body_ang_vel_w", query_times)[:, 0]
        linear_velocity_body = _rotate_inverse_wxyz(root_quaternion, linear_velocity_world)
        angular_velocity_body = _rotate_inverse_wxyz(root_quaternion, angular_velocity_world)
        gravity_world = np.zeros_like(linear_velocity_world)
        gravity_world[:, 2] = -1.0
        gravity_body = _rotate_inverse_wxyz(root_quaternion, gravity_world)
        joint_positions = self.motion.sample("joint_pos", query_times)
        command = np.concatenate(
            (linear_velocity_body, angular_velocity_body, gravity_body, joint_positions), axis=-1
        )
        if command.shape != (COMMAND_STEPS, COMMAND_DIM):
            raise RuntimeError(f"Command window has unexpected shape: {command.shape}")
        return command.astype(np.float32)

    def _build_observation(self) -> np.ndarray:
        observation = np.concatenate(
            (
                self._build_command_window().reshape(-1),
                self.proprio_history.reshape(-1),
                self.action_history.reshape(-1),
            )
        ).astype(np.float32)
        if observation.shape != (OBSERVATION_DIM,):
            raise RuntimeError(f"Policy observation has unexpected shape: {observation.shape}")
        return observation

    def _reset(self, motion_time: float) -> None:
        mujoco.mj_resetData(self.model, self.data)
        root_position, root_quaternion, root_linear_velocity, root_angular_velocity_world = (
            self.motion.root_state(motion_time)
        )
        self.data.qpos[:3] = root_position
        self.data.qpos[3:7] = root_quaternion
        self.data.qvel[:3] = root_linear_velocity
        self.data.qvel[3:6] = _rotate_inverse_wxyz(root_quaternion, root_angular_velocity_world)
        self.data.qpos[self.bindings.qpos_addresses] = self.motion.sample("joint_pos", motion_time)
        self.data.qvel[self.bindings.dof_addresses] = self.motion.sample("joint_vel", motion_time)
        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.motion_time = float(motion_time)
        self.actions.fill(0.0)
        initial_proprioception = self._build_proprioception()
        self.proprio_history[:] = initial_proprioception
        self.action_history.fill(0.0)
        self.policy.reset()

    def _infer_actions(self) -> np.ndarray:
        observation = self._build_observation()[None, :]
        result = self.policy.infer(observation).squeeze(0).astype(np.float64)
        if result.shape != (ACTION_DIM,) or not np.isfinite(result).all():
            raise RuntimeError(f"Policy produced invalid actions: shape={result.shape}")
        return result

    def _apply_pd_step(self, target_positions: np.ndarray) -> None:
        position_error = target_positions - self._joint_positions()
        torques = self.bindings.stiffness * position_error - self.bindings.damping * self._joint_velocities()
        torques = np.clip(torques, -self.bindings.effort_limits, self.bindings.effort_limits)
        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[self.bindings.dof_addresses] = torques
        mujoco.mj_step(self.model, self.data)

    def _control_step(self) -> None:
        self.actions = self._infer_actions()
        reference_joint_positions = self.motion.sample("joint_pos", self.motion_time)
        target_positions = reference_joint_positions + self.args.action_scale * self.actions
        for _ in range(self.args.control_decimation):
            self._apply_pd_step(target_positions)

        self.motion_time += self.control_dt
        self.control_steps += 1
        self.proprio_history[:-1] = self.proprio_history[1:]
        self.proprio_history[-1] = self._build_proprioception()
        self.action_history[:-1] = self.action_history[1:]
        self.action_history[-1] = self.actions.astype(np.float32)

    def _print_diagnostics(self) -> None:
        reference_joint_positions = self.motion.sample(
            "joint_pos", min(self.motion_time, self.motion.duration)
        )
        joint_rmse = float(
            np.sqrt(np.mean((self._joint_positions() - reference_joint_positions) ** 2))
        )
        root_reference = self.motion.sample("body_pos_w", min(self.motion_time, self.motion.duration))[0]
        root_error = float(np.linalg.norm(np.asarray(self.data.qpos[:3]) - root_reference))
        elapsed = max(time.perf_counter() - self.wall_start, 1.0e-9)
        step_rate = self.control_steps / elapsed
        print(
            f"[MUJOCO] step={self.control_steps} motion_time={self.motion_time:.3f}s "
            f"rate={step_rate:.1f}Hz rtf={step_rate * self.control_dt:.2f} "
            f"joint_rmse={joint_rmse:.4f}rad root_error={root_error:.4f}m",
            flush=True,
        )

    def _should_stop(self) -> bool:
        return (
            self.args.max_control_steps is not None
            and self.control_steps >= self.args.max_control_steps
        )

    def _handle_motion_end(self) -> bool:
        if self.motion_time < self.motion.duration:
            return False
        if not self.args.loop:
            return True
        self._reset(float(self.args.start_time))
        return False

    def run(self) -> None:
        print(f"[INFO] {self.policy.backend_name} policy: {self.policy_path}")
        print(f"[INFO] MuJoCo model: {self.model_path}")
        print(f"[INFO] Model loading: {self.model_load_note}")
        print(
            f"[INFO] Motion: {self.motion.path} ({self.motion.frame_count} frames, "
            f"{self.motion.fps:g} Hz, {self.motion.duration:.3f} s)"
        )
        print(
            f"[INFO] Physics: {1.0 / self.args.physics_dt:.0f} Hz; "
            f"policy: {1.0 / self.control_dt:.0f} Hz; "
            f"inference device: {self.policy.device_description}"
        )
        print(
            f"[INFO] Effort profile: {self.args.effort_profile}; "
            f"torque scale: {self.args.torque_scale:g}; "
            f"shoulder gain scale: {self.args.shoulder_gain_scale:g}"
        )

        if self.args.headless:
            self.wall_start = time.perf_counter()
            next_deadline = self.wall_start
            if self.args.max_control_steps is None and self.args.loop:
                self.args.max_control_steps = math.ceil(
                    (self.motion.duration - self.args.start_time) / self.control_dt
                )
            while not self._should_stop():
                self._control_step()
                if self.args.log_interval and self.control_steps % self.args.log_interval == 0:
                    self._print_diagnostics()
                if self._handle_motion_end():
                    break
                next_deadline = self._sleep_until_deadline(next_deadline)
            return

        import mujoco.viewer

        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Viewer initialization can take several seconds; exclude it from
            # controller throughput and real-time-factor diagnostics.
            self.wall_start = time.perf_counter()
            next_deadline = self.wall_start
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = self.bindings.pelvis_body_id
            viewer.cam.distance = 4.0
            viewer.cam.azimuth = 140.0
            viewer.cam.elevation = -15.0
            # URDF visual meshes are compiled into geom group 1, collision-only
            # primitives into group 0, and the generated ground into group 2.
            # Hide collision primitives by default without changing contact physics.
            viewer.opt.geomgroup[0] = int(self.args.show_collision_geoms)
            viewer.opt.geomgroup[1] = 1
            viewer.opt.geomgroup[2] = 1
            while viewer.is_running() and not self._should_stop():
                self._control_step()
                viewer.sync()
                if self.args.log_interval and self.control_steps % self.args.log_interval == 0:
                    self._print_diagnostics()
                if self._handle_motion_end():
                    break
                next_deadline = self._sleep_until_deadline(next_deadline)

    def _sleep_until_deadline(self, previous_deadline: float) -> float:
        next_deadline = previous_deadline + self.control_dt
        if not self.args.real_time:
            return next_deadline
        remaining = next_deadline - time.perf_counter()
        if remaining > 0.0:
            time.sleep(remaining)
        elif remaining < -self.control_dt:
            # Do not accumulate an ever-growing lag when the machine cannot keep up.
            next_deadline = time.perf_counter()
        return next_deadline


def main() -> None:
    args = _build_parser().parse_args()
    deployment = ExtremeRGMTMuJoCoDeployment(args)
    deployment.run()


if __name__ == "__main__":
    main()
