from __future__ import annotations

from dataclasses import dataclass, field

from typing import Type

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig, Nerfstudio
from nerfstudio.cameras.cameras import Cameras, CAMERA_MODEL_TO_TYPE, CameraType
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json


@dataclass
class Gs2gsDataParserConfig(NerfstudioDataParserConfig):
    _target: Type = field(default_factory=lambda: Gs2gsDataParser)


@dataclass
class Gs2gsDataParser(Nerfstudio):
    config: Gs2gsDataParserConfig

    def _generate_dataparser_outputs(self, split: str = "train") -> DataparserOutputs:
        return super()._generate_dataparser_outputs(split)
