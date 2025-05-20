from __future__ import annotations

from dataclasses import dataclass, field

from typing import Type

from nerfstudio.data.dataparsers.base_dataparser import DataparserOutputs
from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig, Nerfstudio


@dataclass
class Gs2gsDataParserConfig(NerfstudioDataParserConfig):
    _target: Type = field(default_factory=lambda: Gs2gsDataParser)


@dataclass
class Gs2gsDataParser(Nerfstudio):
    config: Gs2gsDataParserConfig

    def _generate_dataparser_outputs(self, split: str = "train") -> DataparserOutputs:
        return super()._generate_dataparser_outputs(split)
