from typing import Union, List, Tuple
import mujoco 
from loco_mujoco.core import ObservationType
from loco_mujoco.environments.humanoids.skeletons import MjxSkeletonMuscle
import numpy as np
from loco_mujoco.core.observations.goals import GoalRandomRootVelocity
from flax import struct
from loco_mujoco.environments.base import  LocoCarry
import jax
import jax.numpy as jnp
from loco_mujoco.core.utils import info_property
from collections.abc import Mapping
from scipy.spatial.transform import Rotation



class MjxSkeletonMuscleProsthesis(MjxSkeletonMuscle):
    """
    Mjx version of SkeletonMuscle with specs for adding a prosthesis.
    """

    mjx_enabled = True

    # Define valid options as class constants
    VALID_PROSTHESIS_SIDES = {"left", "right", "bilateral"}
    VALID_PROSTHESIS_TYPES = {"None", "transtibial", "transfemoral"}
    SIDE_SUFFIX_MAP = {
        "left": ["_l"],
        "right": ["_r"],
        "bilateral": ["_l", "_r"]  # Define this if needed
    }
    VALID_PROSTHESIS_SUBTYPES = {"SACH", "ESR"} # Define valid prosthesis subtypes for transtibial prosthesis
    VALID_ESR_MODEL_TYPES = {"linear", "nonlinear"} # Define valid model types of ESR prosthesis -> linear or nonlinear fit of loading-displacement curves by Lecomte

    def __init__(self, 
                 scaling: float =  1.0,
                 alpha_box_feet: float = 0.5,
                 timestep: float = 0.002, n_substeps: int = 5, **kwargs):
        """
        Constructor for MjxSkeletonMuscleProsthesis.
        Args:
            timestep (float): The time step for the simulation.
            n_substeps (int): The number of substeps for the simulation.
            **kwargs: Additional keyword arguments for configuration.
        Raises:
            ValueError: If required arguments are missing.
        """

        # Validate prosthesis_side
        if "prosthesis_side" not in kwargs:
            raise ValueError("Missing required argument: 'prosthesis_side'")
        prosthesis_side = kwargs.pop("prosthesis_side")
        if prosthesis_side not in self.VALID_PROSTHESIS_SIDES:
            raise ValueError(
            f"Invalid value for 'prosthesis_side': '{prosthesis_side}'. "
            f"Must be one of {self.VALID_PROSTHESIS_SIDES}"
            )
        self.prosthesis_side = self.SIDE_SUFFIX_MAP[prosthesis_side]

        # Validate prosthesis_type
        if "prosthesis_type" not in kwargs:
            raise ValueError("Missing required argument: 'prosthesis_type'")
        self.prosthesis_type = kwargs.pop("prosthesis_type")
        if self.prosthesis_type not in self.VALID_PROSTHESIS_TYPES:
            raise ValueError(
            f"Invalid value for 'prosthesis_type': '{self.prosthesis_type}'. "
            f"Must be one of {self.VALID_PROSTHESIS_TYPES}"
            )

        # Check if all required arguments are defined
        if "prosthesis_subtype" not in kwargs:
            raise ValueError("Missing required argument: 'prosthesis_subtype'")
        self.prosthesis_subtype = kwargs.pop("prosthesis_subtype")
        if self.prosthesis_subtype not in self.VALID_PROSTHESIS_SUBTYPES:
            raise ValueError(
                f"Invalid prosthesis_subtype: '{self.prosthesis_subtype}'. "
                f"Must be one of {self.VALID_PROSTHESIS_SUBTYPES}")

        if "amputated_tibia_length" not in kwargs:
            raise ValueError("Missing required argument: 'amputated_tibia_length'")
        self.amputated_tibia_length = kwargs.pop("amputated_tibia_length")
        
        if "tibia_socket_overlap" not in kwargs:
            raise ValueError("Missing required argument: 'tibia_socket_overlap'")
        self.tibia_socket_overlap = kwargs.pop("tibia_socket_overlap")

        self.tibia_socket_offset = kwargs.pop("tibia_socket_offset", [0.0, 0.0, 0.0])
        
        # Define parameters for SACH prosthesis, standard values for SACH
        if hasattr(self, "prosthesis_subtype") and self.prosthesis_subtype == "SACH":
            self.SACH_total_mass = kwargs.pop("SACH_total_mass", 0.575)  # kg 
            # From literature for specific foot size (based on amputee height)
            self.foot_total_mass = self.SACH_total_mass

        # Define parameters for ESR prosthesis
        if hasattr(self, "prosthesis_subtype") and self.prosthesis_subtype == "ESR":
            # Define base stiffness and damping in esr_hinge_joint
            self.ESR_hinge_base_stiffness = kwargs.pop("ESR_hinge_base_stiffness", 250.0) 
            self.ESR_hinge_base_damping = kwargs.pop("ESR_hinge_base_damping", 7.0) 

            # Define default ESR-model type as linear
            self.ESR_model_type = kwargs.pop("ESR_model_type", "linear")
            # Check if valid ESR-Model-Type is defined
            if self.ESR_model_type not in self.VALID_ESR_MODEL_TYPES:
                raise ValueError(f"Invalid ESR_model_type. Must be one of {self.VALID_ESR_MODEL_TYPES}")
            
            # Load ESR Parameter
            if self.ESR_model_type == "linear":
                self.linear_params = kwargs.pop("linear_params", {
                    "k_heel": 57507.3,  # N/m (from Lecomte curve fit, Variflex XC heel loading)
                    "k_keel": 24103.6   # N/m (from Lecomte curve fit, Variflex XC keel loading)
                })  
            else:
                self.nonlinear_params = kwargs.pop("nonlinear_params", {
                    "heel": {"a": 1781838.2, "b": 32644.4}, # a: N/m², b: N/m (from Lecomte curve fit, Variflex XC heel loading)
                    "keel": {"a":  172171.9, "b": 18212.8}  # a: N/m², b: N/m (from Lecomte curve fit, Variflex XC keel loading)
                })

            # Load mass
            self.ESR_total_mass = kwargs.pop("ESR_total_mass", 0.594)   # kg # From Vari-Flex based on Vari-Flex Catalog page (weight with pyramid and foot cover)
            self.ESR_rearfoot_mass = 0.6 * self.ESR_total_mass          # 0.356 kg
            self.ESR_forefoot_mass = 0.4 * self.ESR_total_mass          # 0.238 kg
            self.foot_total_mass = self.ESR_total_mass
            
            # lever arm (FJC)
            self.ESR_lever_arm = kwargs.pop("ESR_lever_arm", 0.128)     # m, Lecomte

            # Pylon Correction compared to SACH pylon
            self.ESR_pylon_correction = kwargs.pop("ESR_pylon_correction",0.008)
            self.ESR_visualization_scaling = kwargs.pop("ESR_visualization_scaling", 0.000604)  

        # Socket parameters estimated from models and papers -> similar for both prosthesis types
        self.original_socket_mass = kwargs.pop("socket_mass", 0.3)  # kg
        self.original_socket_inertia = kwargs.pop("socket_inertia", [0.0136, 0.0021, 0.0136, 0, 0, 0])  # kg*m^2
        self.original_socket_relative_center_of_mass = kwargs.pop("socket_relative_center_of_mass", np.array([0, 0.0491, 0])) # meters

        # Handling joints
        self.joint_stiffness = kwargs.pop("joint_stiffness", None) # Dictionary with joint name and stiffness value
        self.joint_damping = kwargs.pop("joint_damping", None) # Dictionary with joint name and damping value
        self.remove_joint_names = kwargs.pop("remove_joint_names", None) # List of joint names to remove
        self.adapt_joint_range = kwargs.pop("adapt_joint_range", None) # Dict with joint name and new limits 

        # Preparation of Muscle Reattachment
        self.reattach_muscles = kwargs.pop("reattach_muscles", None) # Dictionary with muscle name and reattachment amputation offset
        self.reattach_muscle_names = self.reattach_muscles.keys() if self.reattach_muscles is not None else []

        # Observation for RL
        self.add_pos_ori_to_observation = kwargs.pop("add_pos_ori_to_observation", False) 
        if self.add_pos_ori_to_observation:
            domain_randomization_params = kwargs.get("domain_randomization_params", {})
            if domain_randomization_params.get("randomize_prosthesis_body_position"):
                self.prosthesis_body_position_range = domain_randomization_params.get("prosthesis_body_position_range")
            if domain_randomization_params.get("randomize_prosthesis_body_orientation"):
                self.prosthesis_body_orientation_range = domain_randomization_params.get("prosthesis_body_orientation_range")
            
        # Visualization of Prosthesis
        self.visualize_prosthesis = kwargs.pop("visualize_prosthesis", True)

        # NOTE: For both prosthesis types the same socket values
        # socket joint DOF
        if "socket_joint_dofs" in kwargs:
            self.socket_joint_dofs = kwargs.pop("socket_joint_dofs")
        else: 
            self.socket_joint_dofs = ['socket_tx', 'socket_ty', 'socket_tz', 'socket_flexion', 'socket_adduction', 'socket_rotation']

        # socket joint stiffness
        self.default_socket_joint_stiffnesses = {
            "socket_tx": 43500,
            "socket_ty": 43500,
            "socket_tz": 20000,
            "socket_flexion": 997,
            "socket_adduction": 623,
            "socket_rotation": 10
        }
        user_stiffnesses = kwargs.pop("socket_joint_stiffnesses", {}) # If provided should be dict like defult_socket_joint_stiffnesses
        self.socket_joint_stiffnesses = {**self.default_socket_joint_stiffnesses, **user_stiffnesses}

        # socket damping
        self.default_socket_joint_dampings = {
            "socket_tx": 40,
            "socket_ty": 4,
            "socket_tz": 40,
            "socket_flexion": 10,
            "socket_adduction": 6,
            "socket_rotation": 2
        }
        user_dampings = kwargs.pop("socket_joint_dampings", {}) # If provided should be dict like defult_socket_joint_dampings
        self.socket_joint_dampings = {**self.default_socket_joint_dampings, **user_dampings}

        # socket joint ranges
        self.default_socket_joint_ranges = {
            "socket_tx": [-0.01, 0.01],
            "socket_ty": [-0.02, 0.02],
            "socket_tz": [-0.01, 0.01],
            "socket_flexion": [-0.174, 0.087],
            "socket_adduction": [-0.1, 0.1],
            "socket_rotation": [-0.35, 0.35]
        }
        user_ranges = kwargs.pop("socket_joint_ranges", {}) # If provided should be dict like defult_socket_joint_ranges
        self.socket_joint_ranges = {**self.default_socket_joint_ranges, **user_ranges}

        # For ESR: disable native MuJoCo spring in tx/ty, ESR force law takes over via qfrc_applied
        if self.prosthesis_subtype == "ESR":
            self.socket_joint_stiffnesses["socket_ty"] = 0.0
            self.socket_joint_stiffnesses["socket_tx"] = 0.0
            # Larger range to allow blade deflection
            self.socket_joint_ranges["socket_ty"] = [-0.05, 0.05]
            self.socket_joint_ranges["socket_tx"] = [-0.03, 0.03]

        # NOTE: think about it, whether changed or not -> in reality slackness exists
        # Slackness of the socket joint
        self.delta_shift_slack = kwargs.pop("delta_shift_slack", 0.0) # Amount of slack in the socket joint that allows for shifting before applying forces to the body (to prevent large forces from small position changes within the socket)
        self.socket_ty_slack = kwargs.pop("socket_ty_slack", False) # Whether to add slack to the socket_ty joint to allow for vertical movement within the socket before forces are applied (to prevent large forces from small position changes within the socket in the vertical direction)

        # For evaluation add sensors 
        self.add_sensors = kwargs.pop("add_sensors", False) # Whether to add sensors to the prosthesis bodies for evaluation and visualization purposes

        # Handle multi-contact geom options and solref
        if "multi_contact_geom_type" in kwargs:
            self.multi_contact_geom_type = kwargs.pop("multi_contact_geom_type") # Only 2boxes implemented for now
            kwargs["keep_feet_joints"] = True
        self.contact_geom_solref = kwargs.pop("contact_geom_solref", [0.02, 1.0])

        self.actuators_removed = []
        self.amputated_body_names = []

        # Load XML-File data into spec and modify leg 
        spec = mujoco.MjSpec.from_file(self.get_default_xml_file_path())
        spec = self.replace_leg_level(spec)

        # Replace data with values in dictionaries
        if self.adapt_joint_range is not None: 
            for joint_name, limit_range in self.adapt_joint_range.items():
                self.limit_joint_range(spec, joint_name, limit_range)


        # Add sensors for evaluation 
        if self.add_sensors: 
            print("Adding force and torque sensors for evaluation...")
            for side in ["left_", "right_"]: 
                for sensor_name in ["hip_mimic", "knee_mimic", "foot_mimic"]:
                    self.add_force_sensor(spec, f"{side}{sensor_name}")
                    self.add_torque_sensor(spec, f"{side}{sensor_name}")
            if hasattr(self, "prosthesis_type") and self.prosthesis_type == "transtibial":
                for side in self.prosthesis_side:
                    pylon_sensor_name = f"pylon_mimic{side}"
                    self.add_force_sensor(spec, pylon_sensor_name)
                    self.add_torque_sensor(spec, pylon_sensor_name)
            
            # ESR: add hinge joint position and velocity sensors
            if self.prosthesis_subtype == "ESR":
                for side in self.prosthesis_side:
                    spec.add_sensor(
                        name=f"esr_hinge_pos{side}",
                        type=mujoco.mjtSensor.mjSENS_JOINTPOS,
                        objtype=mujoco.mjtObj.mjOBJ_JOINT,
                        objname=f"esr_hinge{side}"
                    )
                    spec.add_sensor(
                        name=f"esr_hinge_vel{side}",
                        type=mujoco.mjtSensor.mjSENS_JOINTVEL,
                        objtype=mujoco.mjtObj.mjOBJ_JOINT,
                        objname=f"esr_hinge{side}"
                    )
                    spec.add_sensor(
                        name=f"esr_hinge_torque{side}",
                        type=mujoco.mjtSensor.mjSENS_JOINTACTFRC,  # actual joint force/torque
                        objtype=mujoco.mjtObj.mjOBJ_JOINT,
                        objname=f"esr_hinge{side}"
                    )
        
        super().__init__(timestep=timestep, n_substeps=n_substeps,
                         spec=spec,
                         **kwargs)

        if self.prosthesis_subtype == "ESR":
            self.esr_hinge_info = {}
            for side in self.prosthesis_side:
                hinge_id = mujoco.mj_name2id(
                    self._model, mujoco.mjtObj.mjOBJ_JOINT, f"esr_hinge{side}"
                )
                if hinge_id == -1:
                    raise RuntimeError(
                        f"esr_hinge{side} not found in compiled model. "
                        f"Make sure visualize_prosthesis=True so the hinge joint is created."
                    )
                self.esr_hinge_info[side] = {
                    "hinge_qpos": self._model.jnt_qposadr[hinge_id],
                    "hinge_dof":  self._model.jnt_dofadr[hinge_id],
                }
                print(f"  Cached esr_hinge{side}: qpos={self._model.jnt_qposadr[hinge_id]}, dof={self._model.jnt_dofadr[hinge_id]}")

        # Cache hinge sensor IDs for evaluation
        if self.prosthesis_subtype == "ESR" and self.add_sensors:
            self.esr_hinge_sensor_ids = {}
            for side in self.prosthesis_side:
                pos_id = mujoco.mj_name2id(
                    self._model, mujoco.mjtObj.mjOBJ_SENSOR, f"esr_hinge_pos{side}"
                )
                vel_id = mujoco.mj_name2id(
                    self._model, mujoco.mjtObj.mjOBJ_SENSOR, f"esr_hinge_vel{side}"
                )
                torque_id = mujoco.mj_name2id(
                    self._model, mujoco.mjtObj.mjOBJ_SENSOR, f"esr_hinge_torque{side}"
                )
                self.esr_hinge_sensor_ids[side] = {
                    "pos":    self._model.sensor_adr[pos_id],
                    "vel":    self._model.sensor_adr[vel_id],
                    "torque": self._model.sensor_adr[torque_id]
                }  

    def add_force_sensor(self, spec, site_name):
        """
        Adds a force sensor to the specified body in the mjcf model specification.
        
        Args:
            spec (mjcf.RootElement): The MJCF root model object.
            site_name (str): The name of the site to attach the force sensor to.
        """
        sensor_site = spec.find_site(site_name)
        if sensor_site is None:
            raise ValueError(f"Site '{site_name}' not found in the model specification.")
        
        # Add a force sensor
        force_sensor = spec.add_sensor(
            name=f"{sensor_site.name}_force_sensor",
            type=mujoco.mjtSensor.mjSENS_FORCE,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=sensor_site.name
        )


    def add_torque_sensor(self, spec, site_name):
        """
        Adds a torque sensor to the specified body in the mjcf model specification.
        
        Args:
            spec (mjcf.RootElement): The MJCF root model object.
            site_name (str): The name of the site to attach the torque sensor to.
        """
        sensor_site = spec.find_site(site_name)
        if sensor_site is None:
            raise ValueError(f"Site '{site_name}' not found in the model specification.")
        
        # Add a torque sensor
        torque_sensor = spec.add_sensor(
            name=f"{sensor_site.name}_torque_sensor",
            type=mujoco.mjtSensor.mjSENS_TORQUE,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname=sensor_site.name
        )

        
    
    def limit_joint_range(self, spec, joint_name, joint_limit):
        """
        Iterates through the MuJoCo model specification (spec) to find a specific joint
        and overrides its movement boundaries (joint limits).

        Note: The input values in 'joint_limit' must be provided in DEGREES, 
        as they are automatically converted to RADIANS inside this method.
        
        Args:
            spec: The MjSpec object containing the current MuJoCo model blueprint.
            joint_name (str): The name of the joint to modify (e.g., "prosthetic_ankle").
            joint_limit (tuple/list): A pair of [min_angle, max_angle] in degrees.
        """
        for j in spec.joints:
            if j.name == joint_name: 
                j.range = [np.deg2rad(joint_limit[0]), np.deg2rad(joint_limit[1])]
    

    def replace_leg_level(self, spec): 
        """
        replaces the leg level in the model specification based on the prosthesis type.
        Args: 
            spec (MjSpec): The model specification object to modify.
        Returns:
            MjSpec: The modified model specification with the prosthesis leg level.
        """

        if self.prosthesis_type == "None": 
            return spec
        elif self.prosthesis_type == "transtibial":
            print("Replacing leg with transtibial prosthesis for sides: ", self.prosthesis_side)
            self.amputated_body_names = [
                name
                for side in self.prosthesis_side
                for name in [
                    f"calcn{side}",
                    f"toes{side}",
                    f"talus{side}"
                ]
            ]
            # SACH Subtype
            if hasattr(self, "prosthesis_subtype") and self.prosthesis_subtype == "SACH": 
                print("Using SACH prosthesis subtype. Scaling foot and adapting body properties accordingly.")
                spec = self.adapt_spec_with_prosthesis_adapter(spec)
                spec = self.add_prosthesis_properties(spec)
                if self.reattach_muscles is not None:
                    spec = self.reattach_muscles_above_amputation(spec)

            # ESR Subtype
             # NOTE: Adapted for ESR subtype
            if hasattr(self, "prosthesis_subtype") and self.prosthesis_subtype == "ESR": 
                print("Using ESR prosthesis subtype. Scaling foot and adapting body properties accordingly.")
                spec = self.adapt_spec_with_prosthesis_adapter(spec)
                spec = self.add_prosthesis_properties(spec)
                if self.reattach_muscles is not None:
                    spec = self.reattach_muscles_above_amputation(spec)

            return spec 

        elif self.prosthesis_type == "transfemoral": 
            raise NotImplementedError("Transfemoral prosthesis not implemented yet.")
        else:
            raise ValueError(f"Invalid prosthesis type: {self.prosthesis_type}")
        
    
    def calculate_tibia_socket_parameters(self, spec: mujoco.MjSpec, original_talus_pos, side): 
        """
        Calculates geometric, mass, and inertial properties for the amputated 
        residual limb (tibia) and its matching prosthetic socket.
        
        Scales down the biological tibia based on amputation length, adjusts 
        its inertia tensor, and determines the layout dimensions for the socket.
        
        Args:
            spec (mujoco.MjSpec): The active MuJoCo model blueprint.
            original_talus_pos (array): 3D position of the original ankle bone.
            side (str): Side modifier (e.g., "_l" or "_r").
            
        Returns:
            dict: Socket dimensions, mass, center of mass, position, and radius.
        """
        tibia_name = f"tibia{side}"
        tibia_body = spec.find_body(tibia_name)
        original_tibia_mass = tibia_body.mass
        original_tibia_fullinertia = tibia_body.fullinertia.copy()
        original_tibia_length = abs(original_talus_pos[1])
        original_tibia_center_of_mass = tibia_body.ipos.copy()

        if side == "_l":
            side_str = "left"
        else:
            side_str = "right"
        amputated_tibia_length = (
            self.amputated_tibia_length.get(side_str, self.amputated_tibia_length)
            if isinstance(self.amputated_tibia_length, dict)
            else self.amputated_tibia_length
        )

        tibia_socket_overlap = (
            self.tibia_socket_overlap.get(side_str, self.tibia_socket_overlap)
            if isinstance(self.tibia_socket_overlap, dict)
            else self.tibia_socket_overlap
        )

        socket_length = original_tibia_length - amputated_tibia_length + tibia_socket_overlap 
        socket_top_offset = original_tibia_length - socket_length
        socket_pos_relative_to_tibia = np.array([
            self.tibia_socket_offset[0],
            -socket_top_offset - tibia_socket_overlap + self.tibia_socket_offset[1],
            self.tibia_socket_offset[2]
        ])

        amputation_ratio = amputated_tibia_length / original_tibia_length
        tibia_body.mass = original_tibia_mass * amputation_ratio
        tibia_radius = self._calculate_cylinder_radius(tibia_body.mass, original_tibia_fullinertia[1])
        tibia_body.fullinertia[1] = original_tibia_fullinertia[1] * amputation_ratio
        tibia_body.fullinertia[0] = self._calculate_cylinder_inertia_xorz(tibia_body.mass, tibia_radius, amputated_tibia_length)
        tibia_body.fullinertia[2] = tibia_body.fullinertia[0]
        tibia_body.ipos[1] = original_tibia_center_of_mass[1] * amputation_ratio

        tibia_body.add_site(
            name=f"tibia_COM{side}",
            pos=tibia_body.ipos,
            size=[0.001, 0.001, 0.001],
            rgba=[0, 1, 0, 1]
        )

        socket_ratio = socket_length / original_tibia_length
        socket_mass = self.original_socket_mass * socket_ratio
        socket_relative_center_of_mass = self.original_socket_relative_center_of_mass.copy()
        socket_relative_center_of_mass[1] *= socket_ratio
        socket_center_of_mass = np.array(socket_relative_center_of_mass)
        socket_radius = self._calculate_cylinder_radius(self.original_socket_mass,self.original_socket_inertia[1])

        socket_params = {
            "socket_length": socket_length,
            "socket_mass": socket_mass,
            "socket_center_of_mass": socket_center_of_mass,
            "socket_pos_relative_to_tibia": socket_pos_relative_to_tibia,
            "socket_radius": socket_radius}

        return socket_params 


    def create_socket(self, spec,tibia_body, socket_params, side): 
        """
        Replaces the biological lower leg and foot geometries with prosthetic components.
        
        This function handles the removal of the biological tibia, fibula, and foot bones 
        for the specified side. It builds the prosthetic shank structure (socket and pylon) 
        and attaches either a standard SACH foot or an ESR foot 
        mesh based on the chosen prosthesis type.
        
        Args:
            spec: The MuJoCo model specification object.
            tibia_body: The biological tibia body object to attach the socket to.
            socket_params (dict): Physical and geometric parameters of the socket.
            side (str): The side of the limb being processed ("_l" or "_r").
       Returns:
            prosthetic_shank_body: The newly created MuJoCo body for the prosthesis.
        """
        if side == "_l":
            side_str = "left"
        else:
            side_str = "right"

        amputated_tibia_length = (
            self.amputated_tibia_length.get(side_str, self.amputated_tibia_length)
            if isinstance(self.amputated_tibia_length, dict)
            else self.amputated_tibia_length
        )

        tibia_socket_overlap = (
            self.tibia_socket_overlap.get(side_str, self.tibia_socket_overlap)
            if isinstance(self.tibia_socket_overlap, dict)
            else self.tibia_socket_overlap
        )
            
        # Add shank body to tibia_body
        prosthetic_shank_body = tibia_body.add_body(
            name=f"pylon_socket{side}",
            pos=socket_params["socket_pos_relative_to_tibia"],
        )

        # Mimic site for sensors to evaluate forces and torques
        prosthetic_shank_body.add_site(
            name=f"pylon_mimic{side}",
            pos=np.array([0, 0, 0]),
            size=[0.001, 0.001, 0.001],
            rgba=[0, 1, 0, 1]
        )

        # Add an attachment point where the prosthetic foot connects to the pylon
        prosthetic_shank_body.add_site(
            name=f"talus_attachment_site_in_pylon{side}",
            pos=[0, -socket_params["socket_length"] + tibia_socket_overlap, 0],
            size=[0.001, 0.001, 0.001],
            rgba=[1, 0, 0, 1]
        )


        if hasattr(self, 'visualize_prosthesis') and self.visualize_prosthesis:
            # Delete sound leg parts to replace it with prosthesis
            for g in tibia_body.geoms:
                if g.name in {f"tibia{side}", f"fibula{side}"}:
                    g.delete()
            self.visualize_prosthesis = True

            # Load 3D CAD files in MUJoCo system
            if hasattr(self, 'visualize_prosthesis') and self.visualize_prosthesis:
                esr_visualization_scaling = self.ESR_visualization_scaling if self.prosthesis_subtype == "ESR" else 0.001
                spec.add_mesh(
                    name="socket",
                    file="/home/naomiklumpf/loco-mujoco/loco_mujoco/models/prosthesis/meshes/socket.stl",
                    scale=[0.001,0.001,0.001]
                )
                spec.add_mesh(
                    name="pylon",
                    file="/home/naomiklumpf/loco-mujoco/loco_mujoco/models/prosthesis/meshes/pylon.stl",
                    scale=[esr_visualization_scaling,0.001,0.001]
                )

                if self.prosthesis_subtype == "SACH":
                    spec.add_mesh(
                        name="sach",
                        file="/home/naomiklumpf/loco-mujoco/loco_mujoco/models/prosthesis/meshes/sach.stl",
                        scale=[0.001,0.001,0.001],
                    )
                    foot_mesh = "sach"
                    foot_geom_name = "sach_geom"

                # TODO: Change stl file to make prosthesis difference visible
                elif self.prosthesis_subtype == "ESR":
                    spec.add_mesh(
                        name="esr",
                        file="/home/naomiklumpf/loco-mujoco/loco_mujoco/models/prosthesis/meshes/sach.stl",
                        scale=[0.001,0.001,0.001],
                    )
                    foot_mesh = "esr"
                    foot_geom_name = "esr_geom"
                
                else:
                    raise ValueError(
                        f"Unknown prosthesis subtype: {self.prosthesis_subtype}"
                    )
                    
                # Add loaded 3D models for socket and pylon
                prosthetic_shank_body.add_geom(
                    name="socket_geom",
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname="socket",
                    rgba=[0.5, 0.5, 0.5, 1],    #[1, 0, 0, 1], # Bright Red
                    euler=[0, 0.0, 1.571],      # Rotate to be vertical if it's currently horizontal
                    pos=[0,-0.05,0],
                    group=0,                    # Ensure it's in a visible group
                    mass = socket_params["socket_mass"]
                )
                # displacement of pylon mesh for visualization such that top edge of pylon is at same height for both prosthesis types (SACH and ESR)
                esr_pylon_displacement = self.ESR_pylon_correction if self.prosthesis_subtype == "ESR" else 0.0
                prosthetic_shank_body.add_geom(
                    name="pylon_geom",
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname="pylon",
                    rgba=[0.5, 0.5, 0.5, 1],    #[1, 0, 0, 1], # Bright Blue
                    euler=[0, 0.0, +1.571],      # Rotate to be vertical if it's currently horizontal
                    pos=[0, -0.05 + esr_pylon_displacement, 0], # displacement of pylon mesh for visualization such that top edge of pylon is at same height for both prosthesis types (SACH and ESR)
                    group=0                      # Ensure it's in a visible group
                )

                # Delete the sound foot
                for side in self.prosthesis_side:
                    talus_name = f"talus{side}"
                    talus_body = spec.find_body(talus_name)
                    calcn_body = spec.find_body(f"calcn{side}")
                    toes_body = spec.find_body(f"toes{side}")
                    
                    # delete talus geoms to avoid visual overlap with pylon mesh
                    for g in talus_body.geoms:
                        g.delete()
                    for g in calcn_body.geoms:
                        g.delete()
                    for g in toes_body.geoms:
                        g.delete()  

                # Add prosthesis
                if self.prosthesis_subtype == "SACH":
                    talus_body.add_geom(
                        name=foot_geom_name,
                        type=mujoco.mjtGeom.mjGEOM_MESH,
                        meshname=foot_mesh,
                        rgba=[0.3,0.250,0.224,1.0],
                        quat=[0.643,0,-0.766,0],
                        pos=[0.145,-0.062,0.02],
                        group=0
                        )
                
                if self.prosthesis_subtype == "ESR":
                    # Use talus body as proximal attachment body for ESR foot (not anatomical talus)
                    # Rearfoot is directly attached to the pylon
                    rearfoot_body = talus_body
                    rearfoot_body.mass = self.ESR_rearfoot_mass
                    geom_pos = np.array([-0.011, 0.0232-0.005, -0.005])          # geom position
                    com_offset_geom = np.array([0.003, -0.001+0.005, 0.0])       # COM is (3,-1,0) mm shifted compared to center of geom
                    q = [0.643, 0, -0.766, 0]                              # rotation
                    # Rotation of COM coordinates
                    r = Rotation.from_quat([q[1], q[2], q[3], q[0]])
                    com_offset_body = r.apply(com_offset_geom)
                    rearfoot_body.ipos = geom_pos + com_offset_body + [0,0,0]

                    # Rearfoot visualization box
                    # COM from heel: (67,84,0)mm. Size: (128,170,85)mm
                    # Connection to pylon ca 75mm from heel
                    rearfoot_body.add_geom(
                        name=f"esr_rearfoot_geom{side}",
                        type=mujoco.mjtGeom.mjGEOM_BOX,
                        size = [0.0425, 0.08, 0.064],    # x: left/right, y: up/down, z: front/back
                        pos=[-0.011, 0.0232-0.005, -0.005],   # x: front/back, - back; y: up/down; # z: left/right
                        rgba=[0.3, 0.3, 0.8, 0.5], 
                        quat=[0.643,0,-0.766,0],        # Rotation same as talus in visualization
                        group=0                         # Ensure it's in a visible group
                    )

                    # Forefoot as new children body
                    forefoot_body = rearfoot_body.add_body(
                        name=f"esr_forefoot{side}", 
                        pos=[0.053,-0.0343,0.0],         # FJC from Lecomte # 53mm in x-direction from Pylon point
                        quat=[0.643, 0, -0.766, 0]
                    )

                    # ESR-Hinge joint
                    forefoot_body.add_joint(
                        name=f"esr_hinge{side}",
                        type=mujoco.mjtJoint.mjJNT_HINGE,
                        axis=[1,0,0],            
                        stiffness=self.ESR_hinge_base_stiffness,           
                        damping=self.ESR_hinge_base_damping,
                        range=[-0.105,0.244] # -6° and 14° in rad (Lecomte)
                    )

                    # Forefoot visualization box
                    # COM from heel: (180,18,0)mm. Size: (147,50,85)mm
                    forefoot_body.add_geom(
                        name=f"esr_forefoot_geom{side}",
                        type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[0.0425,0.025, 0.0735],    # x: left/right, y: up/down, z: front/back
                        pos=[0.005,-0.0025,-0.0735],  
                        rgba=[0.8,0.3,0.2,0.5],
                        group=0                              # Ensure it's in a visible group
                    )
                    forefoot_body.mass = self.ESR_forefoot_mass
                    forefoot_body.ipos = np.array([0.005, -0.0025-0.007, -0.0735+0.0215])  # COM is (0,-7,-21.5)mm shifted compared to geom center

                    rearfoot_body.add_site(
                        name=f"esr_rearfoot_COM{side}",
                        pos=rearfoot_body.ipos,
                        size=[0.01, 0.01, 0.01],
                        rgba=[1, 1, 0, 1]  # gelb
                    )
                    forefoot_body.add_site(
                        name=f"esr_forefoot_COM{side}",
                        pos=forefoot_body.ipos,
                        size=[0.01, 0.01, 0.01],
                        rgba=[1, 1, 0, 1]  # gelb
                    )
  

        socket_joint_offset = (1 / 3) * amputated_tibia_length

        # specify DOF with stiffness, damping and joint range
        for joint_type in ["tx", "ty", "tz", "flexion", "adduction", "rotation"]:
            if f"socket_{joint_type}" in self.socket_joint_dofs:
                joint_name = f"socket_{joint_type}"
                stiffness = self.socket_joint_stiffnesses.get(joint_name)
                damping = self.socket_joint_dampings.get(joint_name)
                joint_range = self.socket_joint_ranges.get(joint_name)

                joint_name = f"socket_{joint_type}{side}"

                # displacement -> slide joint, if rotation -> hinge joint
                is_slide = joint_type in {"tx", "ty", "tz"}
                joint_type_enum = mujoco.mjtJoint.mjJNT_SLIDE if is_slide else mujoco.mjtJoint.mjJNT_HINGE
                
                axis_map = {
                    "tx": [1, 0, 0],
                    "ty": [0, 1, 0],
                    "tz": [0, 0, 1],
                    "flexion": [0, 0, 1],
                    "adduction": [1, 0, 0],
                    "rotation": [0, 1, 0]
                }
                
                # Add joint of prosthesis
                prosthetic_shank_body.add_joint(
                    name=joint_name,
                    type=joint_type_enum,
                    pos=[0, socket_joint_offset, 0],
                    axis=axis_map[joint_type],
                    stiffness=stiffness,
                    damping=damping,
                    range=joint_range,
                )

        return prosthetic_shank_body
    


    def adapt_spec_with_prosthesis_adapter(self, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        """
        Adapts a MuJoCo MjSpec by adding a TRANSTIBIAL prosthesis and reconnecting foot bodies.

        Args:
            spec: The mujoco.MjSpec object to be modified.
        Returns:
            MjSpec: The modified model specification with prosthesis adapter applied.
        """
        def copy_body_recursive(source_body, parent_mjbody, target_pos=None, target_quat=None):
            """Recursively copy a body and its entire subtree."""
            # position and orientation for the new body: use target if provided, otherwise keep original
            pos = target_pos if target_pos is not None else source_body.pos
            quat = target_quat if target_quat is not None else source_body.quat

            # generate new body in the target spec with the same properties as the source body
            new_body = parent_mjbody.add_body(
                name=source_body.name,
                pos=pos,
                quat=quat,
                mocap=source_body.mocap,
                gravcomp=source_body.gravcomp,
            )
            # transfer mass and inertia properties from sound body to prosthesis body
            # mass will be adjusted later to match the appropriate prosthetic parameters
            new_body.mass = source_body.mass
            new_body.ipos = source_body.ipos
            new_body.fullinertia = source_body.fullinertia

            # all physical geoms copied
            for geom in source_body.geoms:
                new_body.add_geom(
                    name=geom.name, type=geom.type, size=geom.size, pos=geom.pos,
                    quat=geom.quat, meshname=geom.meshname, rgba=geom.rgba,
                    contype=geom.contype, conaffinity=geom.conaffinity,
                    condim=geom.condim, group=geom.group, material=geom.material,
                )

            # all biological joints copied
            for joint in source_body.joints:
                new_body.add_joint(
                    name=joint.name, type=joint.type, pos=joint.pos, axis=joint.axis,
                    range=joint.range, stiffness=joint.stiffness, damping=joint.damping,
                    limited=joint.limited, springref=joint.springref,
                )

            # add site for sensoring
            for site in source_body.sites:
                new_body.add_site(
                    name=site.name, pos=site.pos, quat=site.quat, size=site.size,
                    type=site.type, rgba=site.rgba, group=site.group,
                )

            # copy recursively for all child bodies
            for child in source_body.bodies:
                copy_body_recursive(child, new_body)

            return new_body

        def find_talus_body(tibia_body, side):
            """Find talus body by name or ankle joint."""
            # search in biological model for the talus body either by name or by checking for ankle joint connection, to determine where to attach the prosthesis
            for child in tibia_body.bodies:
                if child.name == f"talus{side}":
                    return child
                if any(j.name == f"ankle_angle{side}" for j in child.joints):
                    return child
            return None

        for side in self.prosthesis_side:
            tibia_body = spec.find_body(f"tibia{side}")
            if not tibia_body:
                print(f"Error: Body 'tibia{side}' not found in model spec.")
                continue

            talus_body = find_talus_body(tibia_body, side)
            if not talus_body:
                print(f"Error: Talus body not found as child of 'tibia{side}'.")
                continue
            # calculate socket parameters based on tibia and talus properties
            socket_params= self.calculate_tibia_socket_parameters(
                spec, talus_body.pos, side
            )
            # create the prosthetic shank (socket and pylon) and attach it to the tibia
            prosthetic_shank = self.create_socket(
                spec, tibia_body, socket_params, side
            )

            # search for attachment site on the prosthetic shank where the talus should be connected
            attachment_site = spec.find_site(f"talus_attachment_site_in_pylon{side}")
            if not attachment_site:
                print(f"Error: Attachment site not found for side {side}.")
                continue

            # copy talus body and its subtree to the prosthetic shank at the attachment site
            copy_body_recursive(
                talus_body, prosthetic_shank,
                target_pos=np.array(attachment_site.pos),
                target_quat=talus_body.quat,
            )

            spec.detach_body(talus_body)

        return spec
    
    # NOTE: changed mass and different function name for a more generell use
    def scale_foot_mass_keep_distribution(self, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        """
        Scales the mass and fullinertia of specified foot bodies to a target total mass
        (self.foot_total_mass) while maintaining the original mass distribution proportions
        among the foot segments. It works for SACH and ESR

        Args:
            spec: The MuJoCo MjSpec object representing the model.
        Returns:
            MjSpec: The modified model specification with scaled foot properties.
        """

        mass_dict = {}
        fullinteria_dict = {}

        foot_body_base_names = ['talus', 'calcn', 'toes']
        # run loop for amputed sites
        for side in self.prosthesis_side:
            total_original_mass = 0.0

            for base_name in foot_body_base_names:
                body_name = f"{base_name}{side}"
                body = spec.find_body(body_name)   # search body in model
                # save biological mass and inertia in dictonary
                if body:
                    mass_dict[body_name] = body.mass
                    fullinteria_dict[body_name] = body.fullinertia.copy()
                    total_original_mass += body.mass # add segment mass to total original mass
                else:
                    print(f"Warning: Body '{body_name}' not found in model spec.")

            if total_original_mass == 0:
                print(f"Error: Total original mass for side '{side}' is zero. Cannot scale.")
                continue

            # Calculate the mass ratios for each foot segment
            mass_ratios = {
                body_name: mass / total_original_mass
                for body_name, mass in mass_dict.items()
                if body_name.endswith(side)
            }
            # Go a second time through the segments to write the new scaled values into the model
            for base_name in foot_body_base_names:
                body_name = f"{base_name}{side}"
                body = spec.find_body(body_name)
                # Guarantee the same mass distribution
                new_mass = self.foot_total_mass * mass_ratios[body_name]
                if body:
                    if body_name in fullinteria_dict and mass_dict[body_name] > 0:
                        body.mass = new_mass

                        # Scale the fullinertia. Inertia scales proportionally to mass for a similar shape.
                        # This assumes that the shape and density distribution within each segment remains
                        # similar, only the overall mass changes.
                        # Calculate the scaling factor for inertia
                        inertia_scaling_factor = new_mass / mass_dict[body_name]
                        original_fullinertia = fullinteria_dict[body_name]
                        body.fullinertia = original_fullinertia * inertia_scaling_factor
                elif body_name in fullinteria_dict and mass_dict[body_name] == 0:
                    # If original mass was zero but inertia existed, set new inertia to zero
                    body.fullinertia = [0.0] * len(fullinteria_dict[body_name])
                else:
                    # Handle cases where original fullinertia was not found (e.g., if it was zero or undefined)
                    print(f"Warning: Original fullinertia for {body_name} not found or was zero. Cannot scale inertia proportionally.")

                
                # Add site at mass center
                # Check if a site with this name already exists to avoid duplicates if function is called multiple times
                site_name = f"{body_name}_COM_site"
                site_exists = any(site.name == site_name for site in body.sites)
                # add a point for the center of mass
                if not site_exists:
                    body.add_site(
                        name=site_name,
                        pos=body.ipos,
                        size=[0.001, 0.001, 0.001],
                        rgba=[0, 0, 1, 1],
                    )
                else: 
                    print(f"Site '{site_name}' already exists in body '{body_name}', skipping addition.")
        return spec
    

    def add_prosthesis_properties(self, spec: mujoco.MjSpec) -> mujoco.MjSpec:
        """
        Adapts to foot to be like SACH or ESR foot in the prosthesis adapter in the model specification.

        Args:
            spec: The mujoco.MjSpec object to be modified.
        Returns:
            MjSpec: The modified model specification with SACH or ESR foot added.
        """

        # The SACH Foot has small masses and inertias so adapt the boundmass and boundinertia
        spec.compiler.boundmass = 0.00001
        spec.compiler.boundinertia = 0.00001

        # Talus, Calcn, Toe:  mass, center of mass and inertia --> Scale original mass and inertia down (0.575 for SACH, 0.5833 for ESR)
        if self.prosthesis_subtype == "SACH":
            self.scale_foot_mass_keep_distribution(spec)
        
        # NOTE
        # For ESR prosthesis: delete biological ankle/toe joints
        if self.prosthesis_subtype == "ESR":
            esr_joints_to_remove = ["ankle_angle", "subtalar_angle", "mtp_angle"]
            for side in self.prosthesis_side:
                for joint_name in esr_joints_to_remove:
                    full_name = f"{joint_name}{side}"
                    spec = self.remove_joint(spec, full_name)
                    spec = self.remove_equality(spec, full_name)

        # remove joints and corresponding constraints that are irrelevant for SACH
        if self.remove_joint_names is not None:
            #print(f"Removing joints: {self.remove_joint_names} for sides: {self.prosthesis_side}")
            for side in self.prosthesis_side:
                for joint_name in self.remove_joint_names:
                    print(f'Removing joint for joint: {joint_name}{side}')
                    full_joint_name = f"{joint_name}{side}"
                    spec = self.remove_joint(spec, full_joint_name)
                    spec = self.remove_equality(spec, full_joint_name)
        
        # adapt joint stiffness of remaining joints of the prosthesis
        if self.joint_stiffness is not None:
            #print(f"Adapting joint stiffness for joints: {list(self.joint_stiffness.keys())} for sides: {self.prosthesis_side}")
            for side in self.prosthesis_side:
                for joint_name in self.joint_stiffness.keys():
                    full_joint_name = f"{joint_name}{side}"
                    spec = self.adapt_joint_stiffness(spec, full_joint_name, side)

        # adapt damping of remaining joints of the prosthesis -> higher
        if self.joint_damping is not None:
            #print(f"Adapting joint damping for joints: {list(self.joint_damping.keys())} for sides: {self.prosthesis_side}")
            for side in self.prosthesis_side:
                for joint_name in self.joint_damping.keys():
                    full_joint_name = f"{joint_name}{side}"
                    spec = self.adapt_joint_damping(spec, full_joint_name)

        # remove all muscels and tendons that control the foot
        spec = self.remove_site_actuator_tendon(spec)

        return spec
    

    def compute_ESR_hinge_qfrc(self, data):
        """
        Computes restoring torque at the ESR hinge joint.

        Uses exact kinematics (no small angle approximation):
            z     = L * sin(theta)           # vertical compression [m]
            tau   = -F_z * L * cos(theta)    # joint torque [Nm]

        Force law depends on ESR_model_type:

            linear:
                F_z = k * z
                k selected by sign of theta:
                    theta < 0 → heel loading  → k = k_heel [N/m]
                    theta >= 0 → keel loading → k = k_keel [N/m]

            nonlinear:
                F_z = a * z² + b * z
                a, b selected by sign of theta:
                    theta < 0  → heel: a [N/m²], b [N/m]
                    theta >= 0 → keel: a [N/m²], b [N/m]

        Reference: Lecomte et al., Variflex XC loading curves
        """
        for side in self.prosthesis_side:
            hinge_dof = self.esr_hinge_info[side]["hinge_dof"]
            theta     = data.qpos[self.esr_hinge_info[side]["hinge_qpos"]]
            lever_arm         = self.ESR_lever_arm  # m
            
            # Clip theta for force calculation only (not the actual state)
            # Safety net if MJX joint limits fail
            theta_clipped = jnp.clip(theta,
                          jnp.deg2rad(-6.0),
                          jnp.deg2rad(14.0))
            
            # Vertical compression of the ESR blade (exact, no small angle approx)
            z = lever_arm * jnp.sin(theta_clipped)  # m

            if self.ESR_model_type == "linear":
                # Select stiffness based on loading direction
                # theta < 0: heel strike (dorsiflexion), theta >= 0: keel/push-off (plantarflexion)
                k = jnp.where(
                    theta_clipped < 0,
                    self.linear_params["k_heel"],   # N/m
                    self.linear_params["k_keel"]    # N/m
                )
                F_z = k * z  # N

            elif self.ESR_model_type == "nonlinear":
                # Select nonlinear coefficients based on loading direction
                a = jnp.where(
                    theta_clipped < 0,
                    self.nonlinear_params["heel"]["a"],  # N/m²
                    self.nonlinear_params["keel"]["a"]   # N/m²
                )
                b = jnp.where(
                    theta_clipped < 0,
                    self.nonlinear_params["heel"]["b"],  # N/m
                    self.nonlinear_params["keel"]["b"]   # N/m
                )
                F_z = (a * jnp.abs(z) + b) * z  # N

            # Joint torque (exact, no small angle approximation)
            tau_ESR     = -F_z * lever_arm * jnp.cos(theta_clipped)  # Nm
            tau_mujoco  = -self.ESR_hinge_base_stiffness * theta
            tau         = tau_ESR - tau_mujoco


            # Debug Print
            # jax.debug.print("ESR qfrc called: theta={t:.3f} tau={tau:.2f}",
            #             t=theta, tau=tau)   
            data = data.replace(
                qfrc_applied=data.qfrc_applied.at[hinge_dof].add(tau)
            )
        return data
    

    def _add_box_feet_to_spec(self, spec, alpha_box_feet):
        """
        Adds box feet to Mujoco spec and makes old feet non-collidable.

        Args:
            spec (MjSpec): Mujoco specification.
            alpha_box_feet (float): Alpha parameter of the boxes.

        Returns:
            Modified Mujoco spec.
        """
        # Call parent - build foot_box and toes_box
        spec = super()._add_box_feet_to_spec(spec, alpha_box_feet)

        if self.prosthesis_subtype == "ESR":
            for side in self.prosthesis_side:
                # Old boxes are hidden + no contact (will be deleted in _modify_spec_for_mjx)
                calcn_body = spec.find_body(f"calcn{side}")
                toes_body = spec.find_body(f"toes{side}")

                for body, geom_name in [
                    (calcn_body, f"foot_box{side}"),
                    (toes_body,  f"toes_box{side}")
                ]:
                    if body:
                        for g in body.geoms:
                            if g.name == geom_name:
                                g.contype = 0
                                g.conaffinity = 0
                                g.rgba = [0, 0, 0, 0]

                # New contact boxes for rearfoot and forefoot
                rearfoot_body = spec.find_body(f"talus{side}")
                forefoot_body = spec.find_body(f"esr_forefoot{side}")

                if rearfoot_body:
                    rearfoot_body.add_geom(
                        name=f"esr_heel_contact{side}",
                        type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[0.04, 0.008, 0.02],  # x: left/right, y: up/down, z: front/back
                        pos=[-0.051, -0.0538, -0.011],  # x: front/back, - back; y: up/down; # z: left/right
                        rgba=[0.2, 0.8, 0.2, 0.5],
                        quat=[0.643,0,-0.766,0],        # Rotation same as talus in visualization
                        group=0
                    )
                    
                    rearfoot_body.add_site(
                            name=f"esr_heel_contact_COM{side}",
                            pos=[-0.051, -0.0538, -0.011],
                            size=[0.01, 0.01, 0.01],
                            rgba=[1, 1, 0, 1]  # gelb
                        )

                if forefoot_body:
                    forefoot_body.add_geom(
                        name=f"esr_toe_contact{side}",
                        type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[0.04, 0.008, 0.02],
                        pos=[0.005, -0.0195, -0.102], 
                        rgba=[0.2, 0.8, 0.2, 0.5],
                        group=0
                    )

                    forefoot_body.add_site(
                            name=f"esr_toe_contact_COM{side}",
                            pos=[0.005, -0.0195, -0.102],
                            size=[0.01, 0.01, 0.01],
                            rgba=[1, 1, 0, 1]  # gelb
                        )
        return spec


    def _modify_spec_for_mjx(self, spec):
        # Deactivate all contacts
        for g in spec.geoms:
            g.contype = 0
            g.conaffinity = 0

        if hasattr(self, 'multi_contact_geom_type'):
            if self.multi_contact_geom_type == '2boxes':

                all_sides = ["_l", "_r"]
                sound_sides = [s for s in all_sides if s not in self.prosthesis_side]

                # Sound sites -- normal pairs
                for side in sound_sides:
                    spec.add_pair(geomname1="floor", geomname2=f"foot_box{side}", solref=self.contact_geom_solref)
                    spec.add_pair(geomname1="floor", geomname2=f"toes_box{side}", solref=self.contact_geom_solref)

                if self.prosthesis_subtype == "SACH":
                    for side in self.prosthesis_side:
                        spec.add_pair(geomname1="floor", geomname2=f"foot_box{side}", solref=self.contact_geom_solref)
                        spec.add_pair(geomname1="floor", geomname2=f"toes_box{side}", solref=self.contact_geom_solref)
                        
                # ESR prosthesis sites -- delete biological boxes and add new pairs for ESR contact boxes
                if self.prosthesis_subtype == "ESR":
                    for side in self.prosthesis_side:
                        for body_name, geom_name in [
                            (f"calcn{side}", f"foot_box{side}"),
                            (f"toes{side}",  f"toes_box{side}"),
                        ]:
                            body = spec.find_body(body_name)
                            if body:
                                for g in list(body.geoms):
                                    if g.name == geom_name:
                                        g.delete()

                        # Add ESR Pairs
                        for geom_name in [f"esr_heel_contact{side}", f"esr_toe_contact{side}"]:
                            spec.add_pair(
                                geomname1="floor",
                                geomname2=geom_name,
                                solref=self.contact_geom_solref
                            )
        else:
            spec.add_pair(geomname1="floor", geomname2="foot_box_r", solref=self.contact_geom_solref)
            spec.add_pair(geomname1="floor", geomname2="foot_box_l", solref=self.contact_geom_solref)

        return spec
    

    # NOTE: Overwritten function to apply ESR reaction forces before the simulation step.
    def _mjx_simulation_pre_step(self, model, data, carry):
        """ 
        Injects Rigney ESR forces before each MuJoCo substep.

        Args:
            model (Model): Mujoco model.
            data (Data): Mujoco data structure.
            carry (MjxAdditionalCarry): Additional carry information.

        Returns:
            Tuple[Model, Data, MjxAdditionalCarry]: Updated model, data, and carry.
        """
        model, data, carry = super()._mjx_simulation_pre_step(model, data, carry)
        if self.prosthesis_subtype == "ESR":
            data = self.compute_ESR_hinge_qfrc(data)

        return model, data, carry


    def _simulation_pre_step(self, model, data, carry):
        """
        Overrides parent to inject ESR spring forces in MuJoCo CPU path.
        Equivalent to _mjx_simulation_pre_step for the CPU evaluation path.
        """
        # Call parent first (terrain + domain randomization)
        print("CPU PRE-STEP CALLED")
        model, data, carry = super()._simulation_pre_step(model, data, carry)

        # Apply ESR spring forces (same as MJX path)
        if self.prosthesis_subtype == "ESR":
            print("ESR FORCE APPLIED")
            for side in self.prosthesis_side:
                hinge_dof  = self.esr_hinge_info[side]["hinge_dof"]
                hinge_qpos = self.esr_hinge_info[side]["hinge_qpos"]
                theta      = data.qpos[hinge_qpos]
                L          = self.ESR_lever_arm

                # Clip theta for force calculation
                theta_f = np.clip(theta,
                                np.deg2rad(-6.0),
                                np.deg2rad(14.0))

                z = L * np.sin(theta_f)

                if self.ESR_model_type == "linear":
                    k   = self.linear_params["k_heel"] if theta_f < 0 else self.linear_params["k_keel"]
                    F_z = k * z
                elif self.ESR_model_type == "nonlinear":
                    if theta_f < 0:
                        a, b = self.nonlinear_params["heel"]["a"], self.nonlinear_params["heel"]["b"]
                    else:
                        a, b = self.nonlinear_params["keel"]["a"], self.nonlinear_params["keel"]["b"]
                    F_z = (a * np.abs(z) + b) * z

                tau_ESR    = -F_z * L * np.cos(theta_f)
                tau_mujoco = -self.ESR_hinge_base_stiffness * theta_f
                tau        = tau_ESR - tau_mujoco

                data.qfrc_applied[hinge_dof] += tau
        return model, data, carry 


    def remove_tendons(self, spec, muscle_names):
        """
        Removes tendons associated with the specified muscle names.
        Args:
            spec (MjSpec): The model specification object.
            muscle_names (set): A set of muscle names to match against tendon names.
        """
        for t in spec.tendons:
            if any(m in t.name for m in muscle_names):
                t.delete()

    
    def remove_sites(self, body):
        """
        Removes sites from the specified body that are associated with muscles.
        Args:
            body (MjBody): The body from which to remove sites.
        Returns:
            set: A set of muscle names derived from the removed sites.
        """
        muscle_names = []
        # take end of body name to get side of prosthesis
        side = body.name[-2:]  # Assumes body names end with '_l' or '_r'
        reattach_muscle_names = [name + side for name in self.reattach_muscle_names] if self.reattach_muscle_names else []
        for s in body.sites:  
            if '-P' in s.name:
                site_renamed = s.name[:-3] # Take out -P part of site name
                if site_renamed not in reattach_muscle_names:
                    muscle_names.append(site_renamed)
                    s.delete()     

        return muscle_names #set(muscle_names)


    def remove_actuators(self, spec, muscle_names):
        """
        Removes actuators associated with the specified muscle names.
        Args:
            spec (MjSpec): The model specification object.
            muscle_names (set): A set of muscle names to match against actuator names."""
        for a in spec.actuators:
            if any(m in a.name for m in muscle_names):
                self.actuators_removed.append(a.name)
                a.delete()


    def remove_site_actuator_tendon(self, spec):
        """
        Removes actuators and tendons associated with specific sites in the model specification.

        Args:
            spec: The model specification object.
        Returns:
            MjSpec: The modified model specification with specified actuators and tendons removed.
        """
        for b in self.amputated_body_names:
            body = spec.find_body(b)
            muscle_names = self.remove_sites(body)
            self.remove_tendons(spec, muscle_names)
            self.remove_actuators(spec, muscle_names)

        return spec

    
    def adapt_joint_damping(self, spec, joint_name):        
        """
        Increases the damping of specified joints in the model specification.
        Args:
            spec (MjSpec): The model specification object.
            joint_name (str): The name of the joint to be modified.
        
        Returns:
            MjSpec: The modified model specification with increased joint damping.
        """

        for j in spec.joints:
            if j.name in joint_name:
                # j.damping = self.joint_damping[j.name.replace(self.prosthesis_side,'')]
                # Prüfen, ob prosthesis_side eine Liste ist. Wenn ja, nimm das erste Element.
                side_str = self.prosthesis_side[0] if isinstance(self.prosthesis_side, list) else self.prosthesis_side

                # Jetzt den reinen Text-String für das Ersetzen nutzen
                j.damping = self.joint_damping[j.name.replace(side_str, '')]
        
        return spec


    def adapt_joint_stiffness(self, spec, joint_name,side):
        """
        Increases the stiffness of specified joints in the model specification.
        Args:
            spec (MjSpec): The model specification object.
            joint_name (str): The name of the joint to be modified.
        Returns:
            MjSpec: The modified model specification with increased joint stiffness.
        """    
        for j in spec.joints:
            if j.name == joint_name:
                j.stiffness = self.joint_stiffness[j.name.replace(side,'')]

        return spec


    def remove_equality(self, spec, joint_name):
        """
        Removes equality constraints associated with the specified joint names.

        Args:
            spec: The model specification object.
            joint_name: String with joint names whose equality constraints should be removed.
        """
        for e in spec.equalities:  # Use list to avoid iteration issues during deletion
            if joint_name in e.name:
                e.delete()

        return spec


    def remove_joint(self, spec, joint_name):
        """
        Removes joints specified in self.amputated_joint_names.

        Args:
            spec (MjSpec): The model specification object.
            joint_name (str): The name of the joint to be removed.
        Returns:
            MjSpec: The modified model specification with the specified joint removed.
        """
        for j in spec.joints:
            if j.name in joint_name:
                j.delete()
        return spec
    

    def reattach_muscles_above_amputation(self, spec):
        """
        Reattaches muscles above the amputation point by recalculating attachment sites.
        
        Assumes muscles are attached to the femur and calcn. Needs adaptation for other
        attachment points.
        
        Args:
            spec (MjSpec): The model specification object.
        
        Returns:
            MjSpec: The modified specification with reattached muscles.
        """
        print("Reattaching muscles above amputation...")

        for side in self.prosthesis_side:
            if side == "_l":
                side_str = "left"
            else:
                side_str = "right"

            amputated_tibia_length = (
                self.amputated_tibia_length.get(side_str, self.amputated_tibia_length)
                if isinstance(self.amputated_tibia_length, dict)
                else self.amputated_tibia_length
            )
            
            # full names list of the affected muscles
            reattach_muscle_names = [name + side for name in self.reattach_muscle_names]
            
            # Get body references
            femur_body = spec.find_body(f"femur{side}")
            tibia_body = spec.find_body(f"tibia{side}")
            calcn_body = spec.find_body(f"calcn{side}")
            talus_body = spec.find_body(f"talus{side}")
            pylon_socket = spec.find_body(f"pylon_socket{side}")
            
            site_pos_tibia_P2 = {}
            site_pos_tibia_P3 = {}
            muscle_site_names = []
            
            # Process P2 sites (femur attachment points)
            for s in femur_body.sites:
                if "P2" in s.name:
                    renamed_site = s.name[:-3]
                    if renamed_site in reattach_muscle_names:
                        site_pos_tibia_P2[renamed_site] = s.pos - tibia_body.pos
                        muscle_site_names.append(renamed_site)
                        
            
            # Process P3 sites (calcn attachment points)
            for s in calcn_body.sites:
                renamed_site = s.name[:-3]
                if renamed_site in reattach_muscle_names:
                    site_pos_tibia_P3[renamed_site] = (
                        s.pos + calcn_body.pos + pylon_socket.pos + talus_body.pos
                    )
                    
            # Recalculate muscle attachment sites
            for site_name in muscle_site_names:
                P2 = site_pos_tibia_P2[site_name]
                P3 = site_pos_tibia_P3[site_name]
                
                # Calculate linear interpolation parameters
                dy = P2[1] - P3[1]
                m_x_y = (P2[0] - P3[0]) / dy
                m_z_y = (P2[2] - P3[2]) / dy
                d_x_y = P2[0] - m_x_y * P2[1]
                d_z_y = P2[2] - m_z_y * P2[1]
                
                # Compute new attachment position
                reattach_offset = self.reattach_muscles[site_name.replace(side, "")]
                new_y = -amputated_tibia_length + reattach_offset[1]
                new_x = m_x_y * new_y + d_x_y + reattach_offset[0]
                new_z = m_z_y * new_y + d_z_y + reattach_offset[2]
                new_pos = np.array([new_x, new_y, new_z])
                
                # Update site, delete old attachment point
                old_site = next((s for s in calcn_body.sites if s.name == f"{site_name}-P3"), None)
                if old_site:
                    old_site.delete()
                
                # Reattach the muscle end
                tibia_body.add_site(
                    name=f"{site_name}-P3",
                    pos=new_pos,
                    size=[0.001, 0.001, 0.001],
                    rgba=[0, 1, 0, 1],
                )
                
                # Scale actuator length range
                length_ratio = np.linalg.norm(new_pos - P2) / np.linalg.norm(P3 - P2)
                for a in spec.actuators:
                    if a.name == site_name:
                        a.lengthrange *= length_ratio
        
        return spec
    

    def _get_observation_specification(self, spec: mujoco.MjSpec):
        """
        Getter for the observation space specification.
        Args:
            spec (MjSpec): Specification of the environment.
        Returns:
            List[ObservationType]: List of observation space specification.
        """
        if self.add_pos_ori_to_observation:
            if hasattr(self, 'prosthesis_body_position_range'):
                rand_pos_body_names=[]
                observation_spec_body_pos = []

                # extract keys from the prosthesis_body_position_range dictionary 
                if isinstance(self.prosthesis_body_position_range, dict):
                    rand_pos_body_names = list(self.prosthesis_body_position_range.keys())

                # create names such that prosthesis side is added to body names
                for b in rand_pos_body_names:
                    for side in self.prosthesis_side:
                        full_body_name = f"{b}{side}"  # Append prosthesis side to body names 
                        observation_spec_body_pos.append(ObservationType.ModelBodyPos(f"pos_{full_body_name}", xml_name=full_body_name))

            # same for orientation range
            if hasattr(self, 'prosthesis_body_orientation_range'):
                rand_ori_body_names=[]
                observation_spec_body_quat = []
                if isinstance(self.prosthesis_body_orientation_range, dict):
                    rand_ori_body_names= list(self.prosthesis_body_orientation_range.keys())
                for b in rand_ori_body_names: 
                    for side in self.prosthesis_side:   
                        full_body_name = f"{b}{side}"   # Append prosthesis side to body names
                    observation_spec_body_quat.append(ObservationType.ModelBodyRot(f"quat_{full_body_name}", xml_name=full_body_name))

        # create list of joint names
        joint_names = []
        for j in spec.joints: 
            joint_names.append(j.name)

        # delete root joint
        if 'root' in joint_names: 
            joint_names.remove('root')

        observation_spec_joint_pos = []
        observation_spec_joint_vel = []

        # observation of relative joint position and velocity into list for each joint
        for j in joint_names:
            observation_spec_joint_pos.append(ObservationType.JointPos(f"q_{j}", xml_name=j))
            observation_spec_joint_vel.append(ObservationType.JointVel(f"dq_{j}", xml_name=j))

       
        observation_spec = [  # ------------- JOINT POS -------------
                                ObservationType.FreeJointPosNoXY("q_root", xml_name="root"),

                                ] + observation_spec_joint_pos + observation_spec_joint_vel
        
        if self.add_pos_ori_to_observation:
            if hasattr(self, 'prosthesis_body_position_range'):
                observation_spec += observation_spec_body_pos 
            if hasattr(self, 'prosthesis_body_orientation_range'):
                observation_spec += observation_spec_body_quat
        return observation_spec
        
        

    def _get_action_specification(self, spec: mujoco. MjSpec):
        """
        Getter for the action space specification.

        Args:
            spec (MjSpec): Specification of the environment.

        Returns:
            List[str]: List of action space specification.
        """

        action_spec = []
        for m in spec.actuators:
            action_spec.append(m.name)
        return action_spec
    


    def _calculate_cylinder_inertia_xorz(self, body_mass, radius, height):
        """
        Calculates the mass moment of inertia of a solid cylinder around its 
        transverse axes (X-axis or Z-axis) passing through its center of mass.
        Formula:
            I_x = I_z = (1/12) * m * (3 * r^2 + h^2)
        Args:
            body_mass (float): The mass (m) of the cylinder in kg.
            radius (float): The outer radius (r) of the cylinder in meters.
            height (float): The total longitudinal length/height (h) of the cylinder in meters.
        Returns:
            float: The calculated transverse rotational inertia (I_x or I_z) in kg*m^2.
        """
        inertia = (1/12)*body_mass * (3*radius**2 + height**2)
        return inertia 
    

    def _calculate_cylinder_radius(self, body_mass, cylinder_inertia_y):
        """
        Calculates the physical radius of a solid cylinder based on its mass 
        and its axial mass moment of inertia around the longitudinal axis (Y-axis).
        Formula derivation:
            I_y = 0.5 * m * r^2  -->  r = sqrt(2 * I_y / m)
        Args:
            body_mass (float): The mass (m) of the cylinder in kg.
            cylinder_inertia_y (float): The rotational inertia (I_y) around the Y-axis.
        Returns:
            float: The calculated cylinder radius (r) in meters.
        """
        # NOTE: Corrected formula
        radius = np.sqrt(2*cylinder_inertia_y/body_mass)
        return radius