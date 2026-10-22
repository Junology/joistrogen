from .copula import GaussianCopulaSampler
from .normal import JointNormalSampler


class DistributionFactory:
    @staticmethod
    def build(name: str, arguments: str):
        normalized = name.strip().upper()
        if normalized == "N":
            return JointNormalSampler.from_arguments(arguments)
        if normalized == "GC":
            return GaussianCopulaSampler.from_arguments(arguments)
        raise ValueError(f"Unsupported distribution kind: {name!r}")
