import re
import tempfile
from pathlib import Path

import mujoco
import yaml


def align_source(model, action, env_path):
    training = yaml.load(Path(env_path).read_text(), Loader=yaml.BaseLoader)
    groups = training["scene"]["robot"]["actuators"]
    spec_path = Path(action.params)
    spec = yaml.safe_load(spec_path.read_text())
    params_path = (spec_path.parent / spec["robot"]["params"]).resolve()
    params = yaml.safe_load(params_path.read_text())
    names = params["robot"]["policy_joint_order"]

    def value(raw, name):
        if isinstance(raw, dict):
            return float(next(entry for pattern, entry in raw.items() if re.fullmatch(pattern, name)))
        return float(raw)

    fields = {name: [] for name in ("kp", "kd", "torque_limits", "action_scale")}
    for name in names:
        group = next(group for group in groups.values()
                     if any(re.fullmatch(pattern, name) for pattern in group["joint_names_expr"]))
        stiffness = value(group["stiffness"], name)
        damping = value(group["damping"], name)
        effort = value(group["effort_limit_sim"], name)
        fields["kp"].append(stiffness)
        fields["kd"].append(damping)
        fields["torque_limits"].append(effort)
        fields["action_scale"].append(value(training["actions"]["joint_pos"]["scale"], name))
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        dof = model.jnt_dofadr[joint]
        model.dof_armature[dof] = value(group["armature"], name)
        model.dof_damping[dof] = 0
        model.dof_frictionloss[dof] = 0
        for actuator in range(model.nu):
            if model.actuator_trnid[actuator, 0] == joint:
                model.actuator_ctrlrange[actuator] = [-effort, effort]
                model.actuator_ctrllimited[actuator] = True
    params["robot"].update(fields)
    params.update(fields)
    output = Path(tempfile.mkdtemp(prefix="actiongo-training-aligned-"))
    (output / "params.yaml").write_text(yaml.safe_dump(params))
    spec["robot"]["params"] = str(output / "params.yaml")
    spec["runtime"].update(target_interp_sec=0.0, warmup_steps=0)
    spec["action"]["target_interp_sec"] = 0.0
    spec["mimic_time"].update(policy_step_source="lowstate_tick", warmup_steps=0)
    spec["handoff"]["activation_blend_sec"] = 0.0
    spec["safety"] = {"profile": "simulation_parity"}
    (output / "spec.yaml").write_text(yaml.safe_dump(spec))
    action.params = str(output / "spec.yaml")
    print(f"[training_alignment] source={env_path} spec={action.params}; "
          "50Hz simulation clock; exact gains/scales/effort/armature; no target interpolation/clamp", flush=True)
