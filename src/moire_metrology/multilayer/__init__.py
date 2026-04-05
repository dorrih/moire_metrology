"""Multi-layer relaxation convenience APIs.

Example:
    from moire_metrology.multilayer import LayerStack
    from moire_metrology import GRAPHENE

    stack = LayerStack(top=GRAPHENE, n_top=5, bottom=GRAPHENE, n_bottom=3, theta_twist=0.5)
    result = stack.solve()
"""

from .stack import LayerStack

__all__ = ["LayerStack"]
