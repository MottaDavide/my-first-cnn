from classes.first_class import Point2D, Quadrilateral
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == '__main__':
    p1 = Point2D(2,3)
    p2 = Point2D(2,5)
    p3 = Point2D(5,5)
    p4 = Point2D(5,3)

    Q1 = Quadrilateral(p1,p2,p3,p4)

    print(Q1)

    print(pd.__version__)

