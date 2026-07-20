from pyexpat import model
from flax import struct
from typing import Any, Union, Tuple
from types import ModuleType

import jax 
import jax.numpy as jnp
import numpy as np

from loco_mujoco.core.domain_randomizer import DomainRandomizer

from mujoco.mjx import Data, Model
from mujoco import MjData, MjModel

from loco_mujoco.core.utils.backend import assert_backend_is_supported

import mujoco

from scipy.spatial.transform import Rotation as R
from jax.scipy.spatial.transform import Rotation as jaxR

@struct.dataclass
class ProsthesisRandomizerState:
    """
    Represents the state of the prosthesis randomizer.
    """
    joint_stiffness: Union[np.ndarray, jax.Array]
    dof_damping: Union[np.ndarray, jax.Array]
    body_position: Union[np.ndarray, jax.Array]
    body_orientation: Union[np.ndarray, jax.Array]



class ProsthesisRandomizer(DomainRandomizer): 
    """
    Randomize prosthesis properties in environment: 

    Gives options to randomize: 
        - joint stiffness
        - dof damping
        - body position and orientation
        - initial body velocity perturbations
    """

    def __init__(self, env, **kwargs):
        self._init_joint_stiffness = None
        self._init_dof_damping = None
        self._init_body_position = None
        self._init_body_orientation = None



        self._body_pos_indices = {}
        self._body_quat_indices = {}
        self._joint_indices = {}
        self._dof_indices = {}

        super().__init__(env, **kwargs)


        # Get indices for prosthesis bodies, joints, and dofs

        # Joint stiffness
        valid_stiffness_dict = {}
        if "prosthesis_joint_stiffness_range" in self.rand_conf:
            self.stiffness_dict = self.rand_conf["prosthesis_joint_stiffness_range"]
            joint_names = list(self.stiffness_dict.keys())
        for side in env.prosthesis_side: 
            full_joint_names = [name + side for name in joint_names]
        for full_joint_name in full_joint_names:
            idx = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, full_joint_name)
            if idx == -1:
                raise ValueError(f"Joint name {full_joint_name} not found in model.")
            else: 
                self._joint_indices[full_joint_name] = idx
                joint_name = full_joint_name.replace(side,'')
                valid_stiffness_dict[full_joint_name] = self.stiffness_dict[joint_name]
        self.stiffness_dict = valid_stiffness_dict


        # DOF damping
        valid_damping_dict = {}
        if "prosthesis_dof_damping_range" in self.rand_conf: 
            self.damping_dict = self.rand_conf["prosthesis_dof_damping_range"]
            dof_names = list(self.damping_dict.keys())
        for side in env.prosthesis_side: 
            full_dof_names = [name + side for name in dof_names]
        for full_dof_name in full_dof_names:
            idx = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, full_dof_name)
            if idx == -1:
                raise ValueError(f"DOF name {full_dof_name} not found in model.")
            else: 
                idx =  env.model.jnt_dofadr[idx]
                self._dof_indices[full_dof_name] = idx
                dof_name = full_dof_name.replace(side,'')
                valid_damping_dict[full_dof_name] = self.damping_dict[dof_name]
        self.damping_dict = valid_damping_dict


        # Body position 
        valid_position_dict = {}
        if "prosthesis_body_position_range" in self.rand_conf: 
            self.body_position_dict = self.rand_conf["prosthesis_body_position_range"]
            pos_body_names = list(self.body_position_dict.keys())
        for side in env.prosthesis_side: 
            full_pos_body_names = [name + side for name in pos_body_names]
        for full_body_name in full_pos_body_names:
            idx = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, full_body_name)
            if idx == -1:
                raise ValueError(f"Body name {body_name} not found in model.")
            else: 
                body_name = full_body_name.replace(side,'')
                self._body_pos_indices[full_body_name] = idx
                valid_position_dict[full_body_name] = self.body_position_dict[body_name]
        self.body_position_dict = valid_position_dict


        # Body orientation
        valid_orientation_dict = {}
        if "prosthesis_body_orientation_range" in self.rand_conf: 
            self.body_orientation_dict = self.rand_conf["prosthesis_body_orientation_range"]
            quat_body_names = list(self.body_orientation_dict.keys())
        for side in env.prosthesis_side: 
            full_quat_body_names = [name + side for name in quat_body_names]
        for full_body_name in full_quat_body_names:
            idx = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, full_body_name)
            if idx == -1:
                raise ValueError(f"Body name {body_name} not found in model.")
            else: 
                body_name = full_body_name.replace(side,'')
                self._body_quat_indices[full_body_name] = idx
                valid_orientation_dict[full_body_name] = self.body_orientation_dict[body_name]
        self.body_orientation_dict = valid_orientation_dict

        self.visualize_randomization = kwargs.pop("visualize_randomization", True)


    def init_state(self, 
                   env: Any, 
                   key: Any, 
                   model: Union[MjModel, Model], 
                   data: Union[MjData, Data], 
                   backend: ModuleType) -> ProsthesisRandomizerState:
        """
        Initialize the prosthesis randomizer state.

        Args:
            env (Any): The environment instance.
            key (Any): Random seed key.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            DefaultRandomizerState: The initialized randomizer state.

        """

        assert_backend_is_supported(backend)
        return ProsthesisRandomizerState(
            joint_stiffness=backend.array(model.jnt_stiffness.copy()),
            dof_damping=backend.array(model.dof_damping.copy()),
            body_position=backend.array(model.body_pos.copy()),
            body_orientation=backend.array(model.body_quat.copy()),
            # body_vel_perturb=self._init_body_vel_perturb,
        )
    

    def reset(self,
              env: Any,
              model: Union[MjModel, Model],
              data: Union[MjData, Data],
              carry: Any,
              backend: ModuleType) -> Tuple[Union[MjData, Data], Any]:
        """
        Reset the randomizer, applying domain randomization.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[MjData, Data], Any]: The updated simulation data and carry.

        """
        assert_backend_is_supported(backend)
        domain_randomizer_state = carry.domain_randomizer_state

        if backend == np and self._init_joint_stiffness is None:
            self._init_joint_stiffness = model.jnt_stiffness.copy()
            self._init_dof_damping = model.dof_damping.copy()
            self._init_body_position = model.body_pos.copy()
            self._init_body_orientation = model.body_quat.copy()

        prosthesis_joint_stiffness, carry = self._sample_joint_stiffness(model, carry, backend)
        prosthesis_dof_damping, carry = self._sample_dof_damping(model, carry, backend)
        prosthesis_body_position, carry = self._sample_body_position(model, carry, backend)
        prosthesis_body_orientation, carry = self._sample_body_orientation(model, carry, backend)


        # Joint stiffness 
        # Joint stiffness
        if backend == jnp:
            joint_names = list(prosthesis_joint_stiffness.keys())
            if len(joint_names) > 0:
                joint_values  = jnp.array(list(prosthesis_joint_stiffness.values()))
                joint_indices = jnp.array([self._joint_indices[name] for name in joint_names], dtype=jnp.int32)
                joint_stiffness = model.jnt_stiffness.at[joint_indices].set(joint_values)
            else:
                joint_stiffness = model.jnt_stiffness
        else:
            joint_stiffness = self._init_joint_stiffness.copy()
            for joint_name, value in prosthesis_joint_stiffness.items():
                idx = self._joint_indices[joint_name]
                joint_stiffness[idx] = value
        # if backend == jnp: 
        #     joint_names = list(prosthesis_joint_stiffness.keys())
        #     joint_values = jnp.array(list(prosthesis_joint_stiffness.values()))
        #     joint_indices = jnp.array([self._joint_indices[name] for name in joint_names])
        #     joint_stiffness = model.jnt_stiffness.at[joint_indices].set(joint_values)
        # else:
        #     joint_stiffness = self._init_joint_stiffness.copy()
        #     for joint_name, value in prosthesis_joint_stiffness.items():
        #         idx = self._joint_indices[joint_name]
        #         joint_stiffness[idx] = value

        # DOF damping
        # DOF damping
        if backend == jnp:
            dof_names = list(prosthesis_dof_damping.keys())
            if len(dof_names) > 0:
                dof_values  = jnp.array(list(prosthesis_dof_damping.values()))
                dof_indices = jnp.array([self._dof_indices[name] for name in dof_names], dtype=jnp.int32)
                dof_damping = model.dof_damping.at[dof_indices].set(dof_values)
            else:
                dof_damping = model.dof_damping
        else:
            dof_damping = self._init_dof_damping.copy()
            for dof_name, value in prosthesis_dof_damping.items():
                idx = self._dof_indices[dof_name]
                dof_damping[idx] = value
        # if backend == jnp:
        #     dof_names = list(prosthesis_dof_damping.keys())
        #     dof_values = jnp.array(list(prosthesis_dof_damping.values()))
        #     dof_indices = jnp.array([self._dof_indices[name] for name in dof_names])
        #     dof_damping = model.dof_damping.at[dof_indices].set(dof_values)
        # else:
        #     dof_damping = self._init_dof_damping.copy()
        #     for dof_name, value in prosthesis_dof_damping.items():
        #         idx = self._dof_indices[dof_name]
        #         dof_damping[idx] = value

        
        # Body position
        if backend == jnp:
            body_names = list(prosthesis_body_position.keys())
            pos_values = jnp.array(list([prosthesis_body_position[name] for name in body_names]))
            pos_indices = jnp.array([self._body_pos_indices[name] for name in body_names])
            body_position = model.body_pos.at[pos_indices].set(pos_values)
        else: 
            body_position = self._init_body_position.copy()
            for body_name, value in prosthesis_body_position.items():
                idx = self._body_pos_indices[body_name]
                body_position[idx] = value


        # Body orientation
        if backend == jnp:
            body_names = list(prosthesis_body_orientation.keys())
            ori_values = jnp.array(list([prosthesis_body_orientation[name] for name in body_names]))
            ori_indices = jnp.array([self._body_quat_indices[name] for name in body_names])
            body_orientation = model.body_quat.at[ori_indices].set(ori_values)
        else: 
            body_orientation = self._init_body_orientation.copy()
            for body_name, value in prosthesis_body_orientation.items():
                idx = self._body_quat_indices[body_name]
                body_orientation[idx] = value


        carry = carry.replace(domain_randomizer_state=domain_randomizer_state.replace(
            joint_stiffness=joint_stiffness,
            dof_damping=dof_damping,
            body_position=body_position,
            body_orientation=body_orientation,
        ))

        return data, carry



    

    def update(self,
               env: Any,
               model: Union[MjModel, Model],
               data: Union[MjData, Data],
               carry: Any,
               backend: ModuleType) -> Tuple[Union[MjModel, Model], Union[MjData, Data], Any]:
        
        """
        Update the randomizer by applying the state changes to the model.

        Args:
            env (Any): The environment instance.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[MjModel, Model], Union[MjData, Data], Any]: The updated simulation model, data, and carry.

        """

        if self.rand_conf["randomize_prosthesis_joint_stiffness"]:
            joint_stiffness = carry.domain_randomizer_state.joint_stiffness
            model = self._set_attribute_in_model(model, "jnt_stiffness", joint_stiffness, backend)
        if self.rand_conf["randomize_prosthesis_dof_damping"]:
            dof_damping = carry.domain_randomizer_state.dof_damping
            model = self._set_attribute_in_model(model, "dof_damping", dof_damping, backend)
        if self.rand_conf["randomize_prosthesis_body_position"]:
            body_position = carry.domain_randomizer_state.body_position
            model = self._set_attribute_in_model(model, "body_pos", body_position, backend)
        if self.rand_conf["randomize_prosthesis_body_orientation"]:
            body_orientation = carry.domain_randomizer_state.body_orientation
            model = self._set_attribute_in_model(model, "body_quat", body_orientation, backend)


        return model, data, carry     
    


    def update_observation(self,
                           env: Any,
                           obs: Union[np.ndarray, jnp.ndarray],
                           model: Union[MjModel, Model],
                           data: Union[MjData, Data],
                           carry: Any,
                           backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:

        """
            Update the observation with randomization effects.

            Args:
                env (Any): The environment instance.
                obs (Union[np.ndarray, jnp.ndarray]): The observation to be updated.
                model (Union[MjModel, Model]): The simulation model.
                data (Union[MjData, Data]): The simulation data.
                carry (Any): Carry instance with additional state information.
                backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

            Returns:
                Tuple[Union[np.ndarray, jnp.ndarray], Any]: The updated observation and carry.

            """
        
        assert_backend_is_supported(backend)
        return obs, carry
    

    def update_action(self,
                      env: Any,
                      action: Union[np.ndarray, jnp.ndarray],
                      model: Union[MjModel, Model],
                      data: Union[MjData, Data],
                      carry: Any,
                      backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Update the action with randomization effects.

        Args:
            env (Any): The environment instance.
            action (Union[np.ndarray, jnp.ndarray]): The action to be updated.
            model (Union[MjModel, Model]): The simulation model.
            data (Union[MjData, Data]): The simulation data.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).

        Returns:
            Tuple[Union[np.ndarray, jnp.ndarray], Any]: The updated action and carry.

        """

        assert_backend_is_supported(backend)
        return action, carry
    


    def _sample_joint_stiffness(self, model: Union[MjModel, Model],
                                carry: Any, backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Sample joint stiffness values for the prosthesis joints.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).
        """

        assert_backend_is_supported(backend)
        sampled_stiffness = {
            joint_name: 0.0 
            for joint_name in self.stiffness_dict.keys() 
            #for side in carry.env.prosthesis_side
        }

        # if self.rand_conf["randomize_prosthesis_joint_stiffness"]:
        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            rand_values = jax.random.uniform(_k, shape=(len(self.stiffness_dict),))
            carry = carry.replace(key=key)
        elif backend == np:
            rand_values = np.random.uniform(size=(len(self.stiffness_dict),))
        

        for i, (joint_name, (low, high)) in enumerate(self.stiffness_dict.items()):

            if self.rand_conf["randomize_prosthesis_joint_stiffness"]:
                stiffness_val = low + (high - low) * rand_values[i]
            else: 
                if joint_name not in self._joint_indices:
                    raise KeyError(f"Joint '{joint_name}' not found in joint indices.")
                idx = self._joint_indices[joint_name]
                if backend == jnp:
                    stiffness_val = model.jnt_stiffness.at[idx].get()
                elif backend == np:
                    stiffness_val = model.jnt_stiffness[idx].copy()
                
            sampled_stiffness[joint_name] = stiffness_val

        return sampled_stiffness, carry
    


    def _sample_dof_damping(self, model: Union[MjModel, Model], 
                            carry: Any, backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Sample DOF damping values for the prosthesis DOFs.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).
        """

        assert_backend_is_supported(backend)
        sampled_damping = {}

        sampled_damping = {dof_name:0.0 for dof_name in self.damping_dict.keys()}

        
        if backend == jnp:
            key = carry.key
            key, _k = jax.random.split(key)
            rand_values = jax.random.uniform(_k, shape=(len(self.damping_dict),))
            carry = carry.replace(key=key)
        elif backend == np:
            rand_values = np.random.uniform(size=(len(self.damping_dict),))

        for i, (dof_name, (low, high)) in enumerate(self.damping_dict.items()):
            if self.rand_conf["randomize_prosthesis_dof_damping"]:  
                damping_val = low + (high - low) * rand_values[i]
            else: 
                if dof_name not in self._dof_indices:
                    raise KeyError(f"DOF '{dof_name}' not found in DOF indices.")
                idx = self._dof_indices[dof_name]
                if backend == np:
                    damping_val = model.dof_damping[idx].copy()
                else:
                    damping_val = model.dof_damping.at[idx].get()
            sampled_damping[dof_name] = damping_val


        return sampled_damping, carry
    


    def _sample_body_position(self, model: Union[MjModel, Model], 
                              carry: Any, backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Sample body position values for the prosthesis bodies.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).
        """

        assert_backend_is_supported(backend)
        sampled_positions = {}

        sampled_positions_offset = {body_name: backend.zeros(3) for body_name in self.body_position_dict.keys()}

        for i, (body_name, axis_ranges) in enumerate(self.body_position_dict.items()):
            offset = backend.zeros(3)
            for j, axis in enumerate(['x','y','z']):
                if axis in axis_ranges:
                    low, high = axis_ranges[axis]
                    if self.rand_conf["randomize_prosthesis_body_position"]:
                        if backend == jnp:
                            key = carry.key
                            key, _k = jax.random.split(key)
                            rand_value = jax.random.uniform(_k, shape=())
                            carry = carry.replace(key=key)
                            offset = offset.at[j].set(low + (high - low) * rand_value)
                        elif backend == np:
                            rand_value = np.random.uniform()
                            offset[j] = low + (high - low) * rand_value
                    else:
                        if backend == jnp:
                            offset = offset.at[j].set(0.0)  # No offset if not randomizing
                        elif backend == np:
                            offset[j] = 0.0  # No offset if not randomizing

            sampled_positions_offset[body_name] = offset

            body_id = self._body_pos_indices[body_name]
            if backend == jnp: 
                original_position = model.body_pos.at[body_id].get() 
            elif backend == np:
                original_position = model.body_pos[body_id].copy()  

            sampled_positions[body_name] = original_position + sampled_positions_offset[body_name]

        return sampled_positions, carry
    


    def _sample_body_orientation(self, model: Union[MjModel, Model], 
                                 carry: Any, backend: ModuleType) -> Tuple[Union[np.ndarray, jnp.ndarray], Any]:
        """
        Sample body orientation values for the prosthesis bodies.

        Args:
            model (Union[MjModel, Model]): The simulation model.
            carry (Any): Carry instance with additional state information.
            backend (ModuleType): Backend module used for calculation (e.g., numpy or jax.numpy).
        """

        assert_backend_is_supported(backend)
        sampled_orientation = {} 
  

        for i, (body_name, axis_ranges) in enumerate(self.body_orientation_dict.items()):
            euler_angles = backend.zeros(3)
            for j, axis in enumerate(['x','y','z']):
                if axis in axis_ranges:
                    low, high = axis_ranges[axis]
                    if self.rand_conf["randomize_prosthesis_body_orientation"]:
                        if backend == jnp:
                            key = carry.key
                            key, _k = jax.random.split(key)
                            rand_value = jax.random.uniform(_k, shape=())
                            carry = carry.replace(key=key)
                            euler_angles = euler_angles.at[j].set(low + (high - low) * rand_value)
                        elif backend == np:
                            rand_value = np.random.uniform()
                            euler_angles[j] = low + (high - low) * rand_value
                    else:
                        if backend == jnp: 
                            euler_angles = euler_angles.at[j].set(0.0)  # No offset if not randomizing
                        elif backend == np:
                            euler_angles[j] = 0.0  # No offset if not randomizing


            body_id = self._body_quat_indices[body_name]
            if backend == jnp: 
                original_orientation_wxyz = model.body_quat.at[body_id].get()
                # Convert (w, x, y, z) to (x, y, z, w) for rotation calculations
                original_orientation_xyzw = jnp.concatenate([original_orientation_wxyz[...,1:4], original_orientation_wxyz[...,0:1]], axis=0)
                # Convert to scipy Rotation
                init_rotation = jaxR.from_quat(original_orientation_xyzw)
            elif backend == np:
                original_orientation_wxyz = model.body_quat[body_id].copy()
                # Convert (w, x, y, z) to (x, y, z, w) for rotation calculations
                original_orientation_xyzw = np.concatenate([original_orientation_wxyz[1:4], original_orientation_wxyz[0:1]], axis=0)
                # Convert to scipy Rotation
                init_rotation = R.from_quat(original_orientation_xyzw)

            # Convert sampled euler angles to new orientation
            orientation_euler = init_rotation.as_euler('xyz', degrees=False)
            for j, axis in enumerate(['x','y','z']):
                if backend == jnp:
                    orientation_euler = orientation_euler.at[j].set(orientation_euler[j] + euler_angles[j])
                elif backend == np:
                    orientation_euler[j] += euler_angles[j]

            if backend == np:
                sampled_rotation = R.from_euler('xyz', orientation_euler, degrees=False)
                sampled_orientation_unsorted = sampled_rotation.as_quat() 
                sampled_orientation_sort = np.concatenate(
                    [sampled_orientation_unsorted[3:4], sampled_orientation_unsorted[0:3]], axis=0
                )
            elif backend == jnp:
                sampled_rotation = jaxR.from_euler('xyz', orientation_euler, degrees=False)
                sampled_orientation_unsorted = sampled_rotation.as_quat() 
                sampled_orientation_sort = jnp.concatenate(
                    [sampled_orientation_unsorted[3:4], sampled_orientation_unsorted[0:3]], axis=0
                )
            
            sampled_orientation[body_name] = sampled_orientation_sort

        return sampled_orientation, carry
    


    @staticmethod
    def _select_env_value(value, env_idx=None):
        value_np = np.array(value)
        if env_idx is None:
            return value_np[0] if value_np.ndim > 1 else value_np
        return value_np[env_idx] if value_np.ndim > 1 else value_np

    def _update_mjM_from_domain_randomization(self, model, carry, env_idx=None):
        """
        Updates the Mujoco model like the MJX model after randomization (based on randomization params).

        Args:
            carry: The current state of the environment including domain randomization state.
            env_idx: Optional environment index to visualize when randomization state is batched.
        """
        domain_randomizer_state = carry.domain_randomizer_state 

        if self.rand_conf["randomize_prosthesis_joint_stiffness"]:
            model.jnt_stiffness = self._select_env_value(domain_randomizer_state.joint_stiffness, env_idx)
        if self.rand_conf["randomize_prosthesis_dof_damping"]:
            model.dof_damping = self._select_env_value(domain_randomizer_state.dof_damping, env_idx)
        if self.rand_conf["randomize_prosthesis_body_position"]:
            model.body_pos = self._select_env_value(domain_randomizer_state.body_position, env_idx)
        if self.rand_conf["randomize_prosthesis_body_orientation"]:
            model.body_quat = self._select_env_value(domain_randomizer_state.body_orientation, env_idx)

        return model
        
    




    






        


