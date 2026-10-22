from .copula import GaussianCopulaSampler
from .normal import JointNormalSampler
from ..cli.grammar import parse_distribution


class DistributionFactory:
    @staticmethod
    def build(spec: str):
        parsed = parse_distribution(spec)
        kind = parsed["kind"]
        if kind == "normal":
            return JointNormalSampler(parsed["mean"], parsed["covariance"])
        if kind == "copula":
            return GaussianCopulaSampler(parsed["a"], parsed["b"], parsed["correlation"])
        raise ValueError(f"Unsupported distribution kind: {kind!r}")
