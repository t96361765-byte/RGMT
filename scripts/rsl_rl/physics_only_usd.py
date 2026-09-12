"""Build disposable physics-only USD assets; never modify the source assets."""

from pathlib import Path


def build_physics_only_usd(source: str, destination: str) -> dict:
    from pxr import Usd, UsdGeom, UsdPhysics

    stage = Usd.Stage.Open(str(Path(source).resolve()))
    if stage is None:
        raise ValueError(f"Cannot open USD asset: {source}")
    # Make instance contents editable and keep their geometry in the flattened file.
    while True:
        instances = [p for p in stage.Traverse() if p.IsInstance()]
        if not instances:
            break
        stage.SetEditTarget(stage.GetSessionLayer())
        for prim in instances:
            prim.SetInstanceable(False)

    def render_property(name):
        return name in ('normals', 'primvars:displayColor', 'primvars:displayOpacity') or (
            name.startswith('primvars:st')
        )

    def physics_signature(s):
        result = {}
        for prim in s.Traverse():
            schemas = [v for v in prim.GetAppliedSchemas() if 'physics' in v.lower()]
            if not schemas and not prim.GetTypeName().startswith('Physics'):
                continue
            values = {}
            for attr in prim.GetAttributes():
                name = attr.GetName()
                if name.startswith(('physics:', 'physx', 'drive:', 'xformOp')) or (
                    prim.HasAPI(UsdPhysics.CollisionAPI)
                    and name != 'visibility' and not render_property(name)
                    and not name.startswith(('outputs:', 'inputs:'))
                ):
                    values[name] = repr(attr.Get())
            relationships = {
                r.GetName(): tuple(map(str, r.GetTargets())) for r in prim.GetRelationships()
                if not r.GetName().startswith('material:binding') or 'physics' in r.GetName()
                or any(s.GetPrimAtPath(t).HasAPI(UsdPhysics.MaterialAPI)
                       for t in r.GetTargets() if s.GetPrimAtPath(t))
            }
            # Ancestor transforms affect collider placement even without physics APIs.
            values['world_transform'] = repr(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
            result[str(prim.GetPath())] = (prim.GetTypeName(), tuple(schemas), values, relationships)
        return result

    before = physics_signature(stage)
    removal = []
    for prim in stage.Traverse():
        if prim.GetName() == 'visuals':
            if any(str(path).startswith(str(prim.GetPath()) + '/') or str(path) == str(prim.GetPath())
                   for path in before):
                raise ValueError(f"Visual subtree contains physics: {prim.GetPath()}")
            removal.append(prim.GetPath())
        elif prim.GetTypeName() in ('Material', 'Shader'):
            # Keep materials with friction/restitution and their children.
            ancestor = prim
            physical = False
            while ancestor and not ancestor.IsPseudoRoot():
                if str(ancestor.GetPath()) in before:
                    physical = True
                ancestor = ancestor.GetParent()
            if not physical:
                removal.append(prim.GetPath())
    stage.SetEditTarget(stage.GetSessionLayer())
    removal = [path for path in removal if not any(
        path != parent and path.HasPrefix(parent) for parent in removal
    )]
    for path in removal:
        stage.OverridePrim(path).SetActive(False)
    # Flatten drops inactive visual references, including the broken helper-link references.
    cleaned = Usd.Stage.Open(stage.Flatten())
    for prim in cleaned.Traverse():
        for prop in list(prim.GetProperties()):
            name = prop.GetName()
            if render_property(name) or (
                name.startswith('material:binding') and 'physics' not in name
                and not any(cleaned.GetPrimAtPath(t).HasAPI(UsdPhysics.MaterialAPI)
                            for t in prop.GetTargets() if cleaned.GetPrimAtPath(t))
            ):
                prim.RemoveProperty(name)
    after = physics_signature(cleaned)
    if before != after:
        raise ValueError('Physics-only conversion changed physical properties; refusing to export')
    target = Path(destination).resolve()
    if target == Path(source).resolve():
        raise ValueError('Output must differ from source')
    target.parent.mkdir(parents=True, exist_ok=True)
    cleaned.GetRootLayer().Export(str(target))
    reopened = Usd.Stage.Open(str(target))
    if physics_signature(reopened) != before:
        raise ValueError('Exported USD failed physical equivalence check')
    return {'removed_subtrees': len(removal), 'physical_prims': len(before), 'output': str(target)}


def configure_headless_training_assets(env_cfg, log_dir: str) -> None:
    """Use stripped assets only for the TheShy training robot and its apparatus."""
    robot = getattr(env_cfg, 'robot', None)
    spawn = getattr(robot, 'spawn', None)
    if Path(getattr(spawn, 'usd_path', '')).stem != 'g1_theshy':
        return
    for name, cfg in [('robot', spawn), ('mushroom', getattr(env_cfg, 'mushroom_spawn', None))]:
        if cfg is None:
            continue
        target = Path(log_dir).resolve() / 'physics_assets' / f'{name}.usdc'
        info = build_physics_only_usd(cfg.usd_path, str(target))
        cfg.usd_path = str(target)
        print(f'[INFO]: Physics-only training asset: {info}')
