import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True, eq=True, repr=True)
class Point2D():
    x: float
    y: float


@dataclass(frozen=True, repr=True)
class Quadrilateral():
    p1: Point2D
    p2: Point2D
    p3: Point2D
    p4: Point2D

