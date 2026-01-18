import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import pandas as pd
from ray_transfer import ABCDMatrix, Optics

@dataclass
class BeamParameters:
    """ Gaussian beam parameters state container """
    wavelength: float
    q: complex
    
    def __post_init__(self):
        self._update_params()
    
    def _update_params(self):
        """
        Calculates real beam parameters (w, R, w0) from complex q
        """
        if self.q == 0:
            raise ValueError("Invalid q parameter: cannot be zero")

        inv_q = 1 / self.q
        
        # 1. Beam Radius (w)
        imag_part = inv_q.imag
        if imag_part >= 0:
            self.w = np.inf
        else:
            self.w = np.sqrt(-self.wavelength / (np.pi * imag_part))
        
        # 2. Radius of Curvature (R)
        real_part = inv_q.real
        if real_part == 0:
            self.R = np.inf
        else:
            self.R = 1 / real_part
            
        # 3. Rayleigh Range & Waist Size (w0)
        self.z_R = self.q.imag
        self.z_relative = self.q.real
        
        if self.z_R > 0:
            self.w0 = np.sqrt(self.wavelength * self.z_R / np.pi)
        else:
            self.w0 = 0.0
        
    def __repr__(self):
        return (f"BeamParameters(λ={self.wavelength*1e6:.0f}nm, "
                f"w={self.w*1e3:.1f}μm, R={self.R:.1f}mm, w0={self.w0*1e3:.1f}μm)")


class Propagation:
    """
    Propagation of Gaussian beam through optics
    """
    
    def __init__(self, initial_beam: BeamParameters):
        self.initial_beam = initial_beam
        self.elements = []
        
    def add_element(self, matrix: ABCDMatrix, name: str, physical_length: float = 0.0):
        """
        Args:
            matrix: ABCD matrix for optics
            name: label for the element
            physical_length: propagation length IN the optics [mm]
        """
        self.elements.append({
            'matrix': matrix,
            'name': name,
            'length': physical_length
        })
    
    def simulate(self, step_size_mm: float = 1.0) -> pd.DataFrame:
        """
        Args:
            step_size_mm: Step size for spatial slicing in [mm]
        Returns:
            simulation history DataFrame (z, w, R, w0)
        """
        current_q = self.initial_beam.q
        wavelength = self.initial_beam.wavelength
        
        history = []
        current_z = 0.0
        
        b = BeamParameters(wavelength, current_q)
        history.append({
            'z': current_z, 
            'w': b.w, 
            'R': b.R, 
            'w0': b.w0, 
            'label': 'Start'
        })
        
        for elem in self.elements:
            matrix = elem['matrix']
            length = elem['length']
            name = elem['name']
            
            if length > 0:
                steps = int(np.ceil(length / step_size_mm))
                dz = length / steps 
                
                effective_n = length / matrix.B if matrix.B != 0 else 1.0
                
                for _ in range(steps):
                    step_matrix = Optics.propagation_homogeneous(dz, effective_n)
                    current_q = step_matrix.transform_q(current_q)
                    current_z += dz
                    
                    b = BeamParameters(wavelength, current_q)
                    history.append({
                        'z': current_z, 
                        'w': b.w, 
                        'R': b.R, 
                        'w0': b.w0,
                        'label': 'prop'
                    })
                
                history[-1]['label'] = name

            else:
                current_q = matrix.transform_q(current_q)
                b = BeamParameters(wavelength, current_q)
                history.append({
                    'z': current_z, 
                    'w': b.w, 
                    'R': b.R, 
                    'w0': b.w0,
                    'label': name
                })
                
        return pd.DataFrame(history)

    def plot(self, df: pd.DataFrame, title: str = "Beam Propagation"):
        """
        Visualizes the beam propagation profile with left-aligned staggered labels.
        """
        plt.figure(figsize=(12, 7))
        
        z_vals = df['z']
        w_vals = df['w'] # [Unit: mm]
        
        max_w = max(w_vals)
        max_z = max(z_vals)
        
        # Plot Beam Profile
        plt.plot(z_vals, w_vals, 'b-', linewidth=1.5, label='Beam Radius ($w$)')
        plt.plot(z_vals, -w_vals, 'b-', linewidth=1.5)
        plt.fill_between(z_vals, -w_vals, w_vals, color='blue', alpha=0.05)
        
        # Filter Elements for Labeling
        elements_df = df[~df['label'].isin(['Start', 'prop'])].reset_index(drop=True)
        
        # Stagger Levels
        y_levels = [max_w * 1.1, max_w * 1.3]
        text_offset = max_z * 0.01  # Slight offset for text
        
        for i, row in elements_df.iterrows():
            plt.axvline(x=row['z'], color='red', linestyle='--', alpha=0.7)
            
            y_pos = y_levels[i % 2]
            
            # Left-aligned text
            plt.text(row['z'] + text_offset, y_pos, row['label'], 
                     color='black', va='bottom', ha='left', fontsize=10)

        plt.xlabel('Propagation Distance (mm)', fontsize=12)
        plt.ylabel('Beam Radius (mm)', fontsize=12)
        plt.title(title, fontsize=14)
        plt.grid(True, linestyle=':', alpha=0.6)
        
        # Adjust Y-limits to fit labels
        plt.ylim(-max_w*1.6, max_w*1.6)
        plt.tight_layout()
        plt.show()