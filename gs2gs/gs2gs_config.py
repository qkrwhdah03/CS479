from __future__ import annotations

from nerfstudio.plugins.types import MethodSpecification
from nerfstudio.engine.optimizers import AdamOptimizerConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig
from nerfstudio.configs.base_config import ViewerConfig

from gs2gs.gs2gs_trainer import Gs2gsTrainerConfig
from gs2gs.gs2gs_datamanager import Gs2gsDataManagerConfig
from gs2gs.gs2gs_dataparser import Gs2gsDataParserConfig
from gs2gs.gs2gs_model import Gs2gsModelConfig
from gs2gs.gs2gs_pipeline import Gs2gsPipelineConfig

gs2gs = MethodSpecification(
    config= Gs2gsTrainerConfig(
        # From TrainerConfig
        steps_per_save= 1000, # Number of steps between saves
        steps_per_eval_batch= 500, # Numebr of steps between randomly sampled batches of rays
        steps_per_eval_image= 500, # Number of steps between single eval images
        steps_per_eval_all_images= 25000, # Number of steps between eval all images
        max_num_iterations= 3000, # Maximum number of iterations to run
        mixed_precision= False, # Whether or not to use mixed precision for training
        save_only_latest_checkpoint= True, # Whether to only save the latest checkpoint or all checkpoints.
        load_dir= None, # Optionally specify a pre-trained model directory to load from.
        load_config= None, # Path to config YAML file.

        # From ExperimentConfig
        method_name = "gs2gs",
        pipeline= Gs2gsPipelineConfig(
            # From VanilaPipelineConfig
            datamanager= Gs2gsDataManagerConfig(
                # From DataManagerConfig
                #data=None, # Source of data, may not be used by all models.

                #From FullImageDataManager
                dataparser= Gs2gsDataParserConfig(
                    # From DataParserConfig
                    #data=None,

                    # From Nerfstudio DataParserConfig
                    #scale_factor= 1.0, # How much to scale the camera origins by.
                    #downscale_factor= None, # Optional[int], How much to downscale images. If not set, images are chosen such that the max dimension is <1600px.
                    #scene_scale= 1.0 , # How much to scale the region of interest by.
                    #orientation_method= "up",  # str, One of ["pca", "up", "vertical", "none"]. Method to use for orientation.
                    #center_method= "poses",  # str, One of ["poses", "focus", "none"]. Method to center the poses.
                    #auto_scale_poses= True,  # Whether to automatically scale the poses to fit in +/- 1 bounding box.

                    #eval_mode= "fraction",  # str, One of ["fraction", "filename", "interval", "all"]. Method to split dataset into train and eval.
                    #train_split_fraction= 0.9,  # float, Only used when eval_mode is "fraction". Fraction of dataset to use for training.
                    #eval_interval= 8,  # int, Only used when eval_mode is "interval". Interval between frames used for evaluation.

                    #depth_unit_scale_factor= 1e-3,  # float, Converts depth values to meters. Default assumes input is in millimeters.
                    #mask_color= None,  # Optional[Tuple[float, float, float]], Replace unknown pixels with this color (used if masks are present).
                    #load_3D_points= False,  # Whether to load 3D points from COLMAP reconstruction.
                    ),
                #camera_res_scale_factor= 1.0, #The scale factor for scaling spatial data such as images, mask, semantics along with relevant information about camera intrinsics
                #eval_num_images_to_sample_from= -1, # Number of images to sample during eval iteration
                #eval_num_times_to_repeat_images= -1, # When not evaluating on all images, number of iterations before picking
                #cache_images= "gpu", # Literal["cpu", "gpu", "disk"]
                #max_thread_workers= None, # The maximum number of threads to use for caching images. If None, uses all available threads.
                #train_cameras_sampling_strategy= "random", #  Literal["random", "fps"] 
                #train_cameras_sampling_seed= 42, # Random seed for sampling train cameras. Fixing seed may help reduce variance of trained models across different runs.
                #fps_reset_every= 100, # The number of iterations before one resets fps sampler repeatly, which is essentially drawing fps_reset_every samples from the pool of all training cameras without replacement before a new round of sampling starts.
                #dataloader_num_workers= 4, # The number of workers performing the dataloading from either disk/RAM
                #prefetch_factor= 4, # Optional[int], The limit number of batches a worker will start loading once an iterator is created. 
                #cache_compressed_images= False, # If True, cache raw image files as byte strings to RAM.
            ),
            model= Gs2gsModelConfig(
                use_l1_loss= False,
                use_lpips_loss= True,
                lpips_loss_weight= 0.1,
                num_random= 50000,
                stop_split_at= 15000,
                refine_every= 100
            ) 
        ),
        optimizers={
           "means": {
            "optimizer": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),
            "scheduler": ExponentialDecaySchedulerConfig(
                lr_final=1.6e-6,
                max_steps=10000,
            ),
            },
            "features_dc": {
                "optimizer": AdamOptimizerConfig(lr=0.0025, eps=1e-15),
                "scheduler": None,
            },
            "features_rest": {
                "optimizer": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),
                "scheduler": None,
            },
            "opacities": {
                "optimizer": AdamOptimizerConfig(lr=0.05, eps=1e-15),
                "scheduler": None,
            },
            "scales": {
                "optimizer": AdamOptimizerConfig(lr=0.005, eps=1e-15),
                "scheduler": None,
            },
            "quats": {"optimizer": AdamOptimizerConfig(lr=0.001, eps=1e-15), "scheduler": None},
            "camera_opt": {
                "optimizer": AdamOptimizerConfig(lr=1e-4, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=5e-7, max_steps=10000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
            "bilateral_grid": {
                "optimizer": AdamOptimizerConfig(lr=2e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(
                    lr_final=1e-4, max_steps=10000, warmup_steps=1000, lr_pre_warmup=0
                ),
            },
        },
        viewer=ViewerConfig(num_rays_per_chunk=1 << 15), 
        vis="viewer"
    ),
    description = "Fine-tunning the 3DGS to convert the style of 3D scene"
)