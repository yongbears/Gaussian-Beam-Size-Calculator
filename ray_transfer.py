"""
Ray transfer matrix analysis: ABCD matrix for
    1. propagation through homogeneous medium
    2. refraction at planar boundary
    3. refraction at spherical boundary
    4. transimission through thin lens
    5. reflection from planar mirror
    6. reflection from spherical mirror
    7. propagation through radial gradient index (GRIN) medium
"""

import numpy as np

# ABCD Matrix
class ABCDMatrix:
    def __init__(self, A: float, B: float, C: float, D: float):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.matrix = np.array([[A, B], [C, D]])
    
    def __matmul__(self, other):
        """Matrix multiplication for combining optical elements"""
        result = self.matrix @ other.matrix
        return ABCDMatrix(result[0, 0], result[0, 1], result[1, 0], result[1, 1])
    
    def transform_q(self, q: complex) -> complex:
        """q' = (A*q + B)/(C*q + D)"""
        return (self.A * q + self.B) / (self.C * q + self.D)
    
    def __repr__(self):
        return f"ABCD Matrix:\n{self.matrix}"


# ABCD matrices for optics
class Optics:
    @staticmethod
    def propagation_homogeneous(distance: float, n: float = 1.0) -> ABCDMatrix:
        """
        matrix: [1   h/n]
                [0    1 ], h in [mm]
        """
        return ABCDMatrix(1, distance/n, 0, 1)

    @staticmethod
    def refraction_planar(n1: float, n2: float) -> ABCDMatrix:
        """
        matrix: [1      0   ]
                [0   n1/n2  ]
        """
        return ABCDMatrix(1, 0, 0, n1/n2)

    @staticmethod
    def refraction_spherical(n1: float, n2: float, ROC: float) -> ABCDMatrix:
        """
        matrix: [       1              0     ]
                [-(n2-n1)/(n1*ROC)   n1/n2   ], ROC in [mm] (positive for center of curvature on the output side)
        """
        return ABCDMatrix(1, 0, -(n2-n1)/(n1*ROC), n1/n2)

    @staticmethod
    def thin_lens(focal_length: float) -> ABCDMatrix:
        """
        matrix: [1    0]
                [-1/f 1], f in [mm] (positive for converging lens)
        """
        return ABCDMatrix(1, 0, -1/focal_length, 1)

    @staticmethod
    def mirror_planar() -> ABCDMatrix:
        """
        matrix: [1  0]
                [0  1]
        """
        return ABCDMatrix(1, 0, 0, 1)

    @staticmethod
    def mirror_spherical(ROC: float) -> ABCDMatrix:
        """
        matrix: [1      0]
                [2/ROC  1], ROC in [mm] (positive for concave mirror)
        """
        return ABCDMatrix(1, 0, 2/ROC, 1)

    @staticmethod
    def grin_medium(g: float, length: float) -> ABCDMatrix:
        """
        matrix: [  cos(gh)      sin(gh)/g  ]
                [-g*sin(gh)     cos(gh)    ], g, length in [mm]
        """
        gh = g * length
        return ABCDMatrix(np.cos(gh), np.sin(gh)/g, -g*np.sin(gh), np.cos(gh))
    
    @staticmethod
    def thick_lens(R1: float, R2: float, thickness: float, n: float = 1.5, n_env: float = 1.0) -> ABCDMatrix:
        """
        refraction(R1) -> propagation(thickness) -> refraction(R2)
        
        Args:
            R1: Radius of curvature of entrance surface [mm] 
            R2: Radius of curvature of exit surface [mm] 
                * For Plano-Convex (PCX): R1 = R_lens, R2 = infinity
            thickness: center thickness [mm]
        """
        # entrance (air -> lens)
        if np.isinf(R1):
            M1 = Optics.refraction_planar(n_env, n)
        else:
            M1 = Optics.refraction_spherical(n_env, n, R1)
            
        # propagation in the lens
        M2 = Optics.propagation_homogeneous(thickness, n)
        
        # exit (lens -> air)
        if np.isinf(R2):
            M3 = Optics.refraction_planar(n, n_env)
        else:
            M3 = Optics.refraction_spherical(n, n_env, R2)
        return M3 @ M2 @ M1