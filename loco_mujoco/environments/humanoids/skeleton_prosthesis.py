from typing import Union, List, Tuple
import mujoco 
from loco_mujoco.core import ObservationType
from loco_mujoco.environments.humanoids.skeletons import MjxSkeletonMuscle
import numpy as np
from loco_mujoco.core.observations.goals import GoalRandomRootVelocity
from flax import struct
from loco_mujoco.environments.base import  LocoCarry
import jax.numpy as jnp
from loco_mujoco.core.utils import info_property
from collections.abc import Mapping



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

        # NOTE: ESR Data (mass) and ESR_model_type added
        # Define parameters for ESR prosthesis
        if hasattr(self, "prosthesis_subtype") and self.prosthesis_subtype == "ESR":
            self.ESR_total_mass = kwargs.pop("ESR_total_mass", 0.779)  # kg # From Vari-Flex based on Vari-Flex Modular Catalog page (weight with pyramid and foot cover)
            self.foot_total_mass = self.ESR_total_mass
            # Check whether model type is defined which specifies whether der ESR is modeled with the linear elastic model or the distal displacement model as explained in Rigney (2018): Mathematical modelling of energy storage and return prostheses
            # TODO: Define default model_type?
            self.ESR_model_type = kwargs.pop("ESR_model_type", None) # Should be either "linear_elastic" or "distal_displacement"
            if self.ESR_model_type is None:
                raise ValueError("ESR prosthesis requires a 'ESR_model_type':"
                                "'linear_elastic' or 'distal_displacement'")
            if self.ESR_model_type not in {"linear_elastic", "distal_displacement"}:
                raise ValueError(f"Invalid ESR_model_type: '{self.ESR_model_type}'")

            # Linear Elastic Model
            # Stiffness of the linear spring in the linear elastic model of the ESR prosthesis based on Rigney (2018) (Table 6.1, Vari-Flex Modular)
            self.ESR_k = kwargs.pop("ESR_k", {"k_a" : -0.442, "k_b": 35.22}) # N/mm
            # self.ESR_k = kwargs.pop("ESR_k", {"k_a" : 0.229, "k_b": 19.00}) # N/mm # Cheetah Xtreme
            # Distal Displacement Model
            # Coefficients for the functions defining the force displacements relationship in the distal displacement model of the ESR prosthesis based on Rigney (2018) (Table 6.1, Vari-Flex Modular)
            self.ESR_coeffs = kwargs.pop("ESR_coeffs", {"m": 14.47, "n": 0.14, "p": -14.34, "q": -0.84, "r": 7.80, "s": 0.78}) # N/mm, N/mm^2, N/mm, N/mm^2, N/mm, N/mm
            # Cheetah Xtreme (better mass fit?!)
            # self.ESR_coeffs = kwargs.pop("ESR_coeffs", {"m": 53.82, "n": -0.29, "p": -67.41, "q": 0.30, "r": -0.15, "s": 7.54}) # N/mm, N/mm^2, N/mm, N/mm^2, N/mm, N/mm

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

        # For ESR: disable native MuJoCo spring in tx/ty, Rigney force law takes over via qfrc_applied
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

        
        super().__init__(timestep=timestep, n_substeps=n_substeps,
                         spec=spec,
                         **kwargs)
        
        # Cache joint and site IDs for ESR force computation
        # Avoirds repeated mj_name2id calls
        if self.prosthesis_subtype == "ESR":
            self.esr_joint_info = {}
            self.pylon_site_ids = {}
        
            for side in self.prosthesis_side:
                # Joint indices
                ty_id = mujoco.mj_name2id(
                    self._model, mujoco.mjtObj.mjOBJ_JOINT, f"socket_ty{side}"
                )
                tx_id = mujoco.mj_name2id(
                    self._model, mujoco.mjtObj.mjOBJ_JOINT, f"socket_tx{side}"
                )
                self.esr_joint_info[side] = {
                    "ty_qpos": self._model.jnt_qposadr[ty_id],
                    "tx_qpos": self._model.jnt_qposadr[tx_id],
                    "ty_dof": self._model.jnt_dofadr[ty_id],
                    "tx_dof": self._model.jnt_dofadr[tx_id],
                }
                # Site indices for alpha computation
                self.pylon_site_ids[side] = {
                    "proximal": mujoco.mj_name2id(
                        self._model, mujoco.mjtObj.mjOBJ_SITE, f"pylon_mimic{side}"
                    ),
                    "distal": mujoco.mj_name2id(
                        self._model, mujoco.mjtObj.mjOBJ_SITE, f"talus_attachment_site_in_pylon{side}"
                    )
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
                spec.add_mesh(
                    name="socket",
                    file="/home/naomiklumpf/loco-mujoco/loco_mujoco/models/prosthesis/meshes/socket.stl",
                    scale=[0.001,0.001,0.001]
                )
                spec.add_mesh(
                    name="pylon",
                    file="/home/naomiklumpf/loco-mujoco/loco_mujoco/models/prosthesis/meshes/pylon.stl",
                    scale=[0.001,0.001,0.001]
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
                    rgba=[0.5, 0.5, 0.5, 1], #[1, 0, 0, 1], # Bright Red
                    euler=[0, 0.0, 1.571], # Rotate to be vertical if it's currently horizontal
                    pos=[0,-0.05,0],
                    group=0,            # Ensure it's in a visible group
                    mass = socket_params["socket_mass"]
                )
                prosthetic_shank_body.add_geom(
                    name="pylon_geom",
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname="pylon",
                    rgba=[0.5, 0.5, 0.5, 1], #[1, 0, 0, 1], # Bright Blue
                    euler=[0, 0.0, 1.571], # Rotate to be vertical if it's currently horizontal
                    pos=[0, -0.05, 0],
                    group=0            # Ensure it's in a visible group
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
                talus_body.add_geom(
                    name=foot_geom_name,
                    type=mujoco.mjtGeom.mjGEOM_MESH,
                    meshname=foot_mesh,
                    rgba=[0.3,0.250,0.224,1.000],
                    quat=[0.643,0,-0.766,0],
                    pos=[0.145,-0.062,0.02],
                    group=0
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

                # of displacement -> slide joint, if rotation -> hinge joint
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
        self.scale_foot_mass_keep_distribution(spec)

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
    
    # NOTE: New function to compute alpha in Rigney (2018)
    def get_pylon_alpha(self, data, side):
        """ 
        Computes pylon orientation angle alpha relative to vertical (Z-axis).
        Uses proximal/distal sites to define the mechanical pylon axis.

        Coordinate system confirmed:
        - Gravity: [0,0,-9.81] -> Z is vertical axis
        - Pylon longitudinal axis: Z (confirmed from site positions)

        Source: Rigney (2018), Figure 6.1
        """
        proximal_id = self.pylon_site_ids[side]["proximal"]
        distal_id = self.pylon_site_ids[side]["distal"]

        proximal_pos = data.site_xpos[proximal_id]
        distal_pos = data.site_xpos[distal_id]

        # Mechanical pylon axis
        pylon_axis = proximal_pos - distal_pos
        pylon_axis = pylon_axis / jnp.linalg.norm(pylon_axis)

        # TODO: CHECK
        # Sagittal plane angle (X-Z plane)
        alpha = jnp.arctan2(pylon_axis[0], pylon_axis[2])   # rad
        # alpha = jnp.clip(alpha, jnp.deg2rad(-25), jnp.deg2rad(25))
        
        return alpha
    
    # NOTE: New function to implement reaction force
    def compute_ESR_qfrc(self, data):
        """
        Computes generalized forces for ESR compliance DOFs.

        socket_ty = z (vertical displacement, Rigney 2018)
        socket_tx = y (anterior-posterior displacement, Rigney 2018)

        Linear Elastic Model (Rigney 2018, Eq. 6.2? Or 6.3?):
            F_z = k * z
            k   = k_a * alpha + k_b

        Distal Displacement Model (Rigney 2018, Eq. 6.5)
            F_Z = m * Z + n * Z^2 + p * Y + q * Z * Y
            F_Y = r * Y + s * Z
            F_z = F_Z * cos(alpha) + F_Y * sin(alpha)
            F_y = F_Z * sin(alpha) - F_Y * cos(alpha)
        """
        for side in self.prosthesis_side:
            info = self.esr_joint_info[side]

            z = data.qpos[info["ty_qpos"]] # m
            y = data.qpos[info["tx_qpos"]] # m
            z_mm = 1000 * z # mm
            y_mm = 1000 * y # mm
            ty_dof = info["ty_dof"]
            tx_dof = info["tx_dof"]

            alpha = self.get_pylon_alpha(data, side)
            if self.ESR_model_type == "linear_elastic":
                k_a = self.ESR_k["k_a"]
                k_b = self.ESR_k["k_b"]
                k   = k_a * jnp.rad2deg(alpha) + k_b
                F_z = - k * z_mm
                F_y = jnp.zeros_like(F_z)

            elif self.ESR_model_type == "distal_displacement":
                m = self.ESR_coeffs["m"]
                n = self.ESR_coeffs["n"]
                p = self.ESR_coeffs["p"]
                q = self.ESR_coeffs["q"]
                r = self.ESR_coeffs["r"]
                s = self.ESR_coeffs["s"]
                Z_mm =  z_mm * jnp.cos(alpha) + y_mm * jnp.sin(alpha)
                Y_mm =  -z_mm * jnp.sin(alpha) + y_mm * jnp.cos(alpha)
                F_Z = m*Z_mm + n*Z_mm**2 + p*Y_mm + q*Z_mm*Y_mm
                F_Y = r*Y_mm + s*Z_mm
                F_z = - (F_Z * jnp.cos(alpha) + F_Y * jnp.sin(alpha))
                F_y = - (F_Z * jnp.sin(alpha) - F_Y * jnp.cos(alpha))

            # Add reaction force from Rigney
            data = data.replace(
                qfrc_applied=data.qfrc_applied
                .at[ty_dof].add(F_z)
                .at[tx_dof].add(F_y)
            )
        return data

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
            data = self.compute_ESR_qfrc(data)
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